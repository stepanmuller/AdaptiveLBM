#pragma once

#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"
#include "./markerFunctions.h"
#include "./applyCollision.h"
#include "./boundaryConditions/applyMovingBounceback.h"

// Refill correction: Local streaming and collision only for the uncovered cells, as described by 
// Li Chen, Yang Yu, Jianhua Lu, Guoxiang Hou, 
// A comparative study of lattice Boltzmann methods using bounce-back schemes and immersed boundary ones for flow acoustic problems, 2013
// LI scheme
void runRefillCorrection( GridStruct &Grid, const int &newlyFluidCount )
{
	InfoStruct &Info = Grid.Info;
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto fView  = Grid.fArray.getView();
	auto fBufferView  = Grid.fBufferArray.getView();
	auto newlyFluidIndexView = Grid.newlyFluidIndexArray.getView();
	auto jPlusView = Grid.NBR.jPlusArray.getView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();

	const bool esotwistFlipper = Grid.esotwistFlipper;	

	auto bitPackedMarkerView = Grid.bitPackedMarkerArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{		
		const int cell = newlyFluidIndexView( index );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( kPlusView( cell ) );
		finishNBRPlus( NBR, Info );
		
		const int bitPackedMarkerInt = bitPackedMarkerView( cell );
		bool bitPackedMarkerBits[32];
		intToBools( bitPackedMarkerInt, bitPackedMarkerBits );
		
		int cellReadIndex[27];
		int fReadIndex[27];
		float f[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) f[direction] = fView( fReadIndex[direction], cellReadIndex[direction] );
		
		// run K15 collision
		BCStruct BC;
		BC.collisionLimiter = 0.f;
		applyCollision( f, BC, Info.nu );
		
		// write collided f into *buffer* of our cell
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			fBufferView( direction, index ) = f[direction];
		}	
		
		float rho, ux, uy, uz;
		getRhoUxUyUz(rho, ux, uy, uz, f);
		if (index == 1) printf("%f\n", rho);
		/*
		float fMBB[27];
		for ( int direction = 0; direction < 27; direction++ ) fMBB[direction] = f[direction];
		
		// write the distribution functions that are going to be pulled into our cell next iteration from moving bounceback cells
		getRhoUxUyUz( BC.rho, BC.ux, BC.uy, BC.uz, fMBB );
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		applyMovingBounceback( fMBB, BC );
		for ( int direction = 1; direction < 27; direction++ ) 
		{
			if ( !bitPackedMarkerBits[direction] ) 
			{
				// if we are going to be receiving f from a moving bounceback in this direction,
				// set it to the moving bounceback result
				fView( fReadIndex[direction], cellReadIndex[direction] ) = fMBB[direction];
			}
		}	
		*/
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, cellLambda );
	
	auto bufferLambda = [=] __cuda_callable__ ( const int index ) mutable
	{		
		const int cell = newlyFluidIndexView( index );
		
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		const int bitPackedMarkerInt = bitPackedMarkerView( cell );
		bool bitPackedMarkerBits[32];
		intToBools( bitPackedMarkerInt, bitPackedMarkerBits );
		
		// Load collided f from buffer
		float f[27];
		for ( int direction = 0; direction < 27; direction++ ) f[direction] = fBufferView( direction, index );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( kPlusView( cell ) );
		finishNBRPlus( NBR, Info );
		
		// Write collided f into the refill cell
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreviousPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			fView( fWriteIndex[direction], cellWriteIndex[direction] ) = fBufferView( direction, index );
		}
		
		// Update MBB around the refill cell
		// write the distribution functions that are going to be pulled into our cell next iteration from moving bounceback cells
		int cellNextIndex[27];
		int fNextIndex[27];
		getPreCollisionIndex( cellNextIndex, fNextIndex, NBR, esotwistFlipper, Info );
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCStruct BC;
		getRhoUxUyUz(BC.rho, BC.ux, BC.uy, BC.uz, f);
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		applyMovingBounceback( f, BC );
		for ( int direction = 1; direction < 27; direction++ ) 
		{
			if ( !bitPackedMarkerBits[direction] ) 
			{
				// if we are going to be receiving f from a moving bounceback in this direction,
				// set it to the moving bounceback result
				fView( fNextIndex[direction], cellNextIndex[direction] ) = f[direction];
			}
		}	
		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, bufferLambda );
}

void updateMovingBounceback( GridStruct &Grid, const VoxelizerStruct &Voxelizer )
{
	std::cout << "updating MBB now" << std::endl;
	InfoStruct &Info = Grid.Info;
	BoolArrayType &oldMBBMarkerArray = Grid.markerBuffer;
	
	auto fView  = Grid.fArray.getView();
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto intBuffer1View = Grid.intBuffer1.getView();
	auto newlyFluidIndexView = Grid.newlyFluidIndexArray.getView();
	auto newlyMBBIndexView = Grid.newlyMBBIndexArray.getView();
	auto jPlusView = Grid.NBR.jPlusArray.getView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	auto oldMBBMarkerView = oldMBBMarkerArray.getView();

	const bool esotwistFlipper = Grid.esotwistFlipper;	

	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getView();
	auto deepRefinementMarkerView = Grid.deepRefinementMarkerArray.getConstView();
	auto bitPackedMarkerView = Grid.bitPackedMarkerArray.getView();
	
	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	bool useDeepRefinementMarkerArray = ( Grid.deepRefinementMarkerArray.getSize() > 0 );
	
	// Take copy of the old moving bounceback marker array and update the active array
	oldMBBMarkerArray = Grid.movingBouncebackMarkerArray;
	applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Info.cellCount );
	
	// Update bitPackedMarker, because MBB state of the cells changed
	fillBitPackedMarkerArray( Grid, Info.cellCount );
	
	// Now we need to repair information in cells that were previously moving bounceback and now are fluid = refill algorithm
	// Also, we want to keep track of the torque exerted on the system by removing or adding fluid cells
	// If the cell is newly MBB, momentum of the original fluid cell is being removed
	// If the cell is newly fluid, its momentum is being added 
	// -> track these torque contributions
	// First, identify which cells changed state from MBB to fluid, or from fluid to MBB
	// To gather indexes of those cells we will use intBuffer1 ( = NBR.jMinusArray), we need to repair this later! 
	
	auto newlyFluidMarkerLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool oldMarker = oldMBBMarkerView( cell );
		const bool newMarker = movingBouncebackMarkerView( cell );
		const bool newlyFluid = ( oldMarker && !newMarker );
		if (newlyFluid) intBuffer1View( cell ) = 1;
		else intBuffer1View( cell ) = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, newlyFluidMarkerLambda );
	int lastMarker = Grid.intBuffer1.getElement( Info.cellCount-1 );
	TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer1, 0, Info.cellCount, TNL::Plus{} );
	const int newlyFluidCount = Grid.intBuffer1.getElement( Info.cellCount-1 ) + lastMarker;
	if ( newlyFluidCount > Info.mbbUpdateMemoryCount )
	{
		std::cout << "updateMovingBounceback failed on grid " << Grid.Info.gridID << ", mbbUpdateMemoryCount = " << Info.mbbUpdateMemoryCount << ", newlyFluidCount = " << newlyFluidCount << std::endl;
		throw std::runtime_error("updateMovingBounceback failed, newlyFluidCount exceeded allocated memory. Try increasing MEMORY_MBB_UPDATE_PERCENTAGE in your main file.");
	}
	auto newlyFluidIndexLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool oldMarker = oldMBBMarkerView( cell );
		const bool newMarker = movingBouncebackMarkerView( cell );
		const bool newlyFluid = ( oldMarker && !newMarker );
		if ( newlyFluid ) newlyFluidIndexView( intBuffer1View( cell ) ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, newlyFluidIndexLambda );
	
	auto newlyMBBMarkerLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool oldMarker = oldMBBMarkerView( cell );
		const bool newMarker = movingBouncebackMarkerView( cell );
		const bool newlyMBB = ( !oldMarker && newMarker );
		if (newlyMBB) intBuffer1View( cell ) = 1;
		else intBuffer1View( cell ) = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, newlyMBBMarkerLambda );
	lastMarker = Grid.intBuffer1.getElement( Info.cellCount-1 );
	TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer1, 0, Info.cellCount, TNL::Plus{} );
	const int newlyMBBCount = Grid.intBuffer1.getElement( Info.cellCount-1 ) + lastMarker;
	if ( newlyMBBCount > Info.mbbUpdateMemoryCount )
	{
		std::cout << "updateMovingBounceback failed on grid " << Grid.Info.gridID << ", mbbUpdateMemoryCount = " << Info.mbbUpdateMemoryCount << ", newlyMBBCount = " << newlyMBBCount << std::endl;
		throw std::runtime_error("updateMovingBounceback failed, newlyMBBCount exceeded allocated memory. Try increasing MEMORY_MBB_UPDATE_PERCENTAGE in your main file.");
	}
	auto newlyMBBIndexLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool oldMarker = oldMBBMarkerView( cell );
		const bool newMarker = movingBouncebackMarkerView( cell );
		const bool newlyMBB = ( !oldMarker && newMarker );
		if ( newlyMBB ) newlyMBBIndexView( intBuffer1View( cell ) ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, newlyMBBIndexLambda );
	
	// Now we have assembled our index lists
	// We no longer need intBuffer1 ( = NBR.jMinusArray) -> we repair it
	auto jMinusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jMinusView[ jPlusView[ cell ] ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, jMinusRepairLambda );
	
	// Now, run the refill algorithm
	// For all newly fluid cells, repair the information in them by interpolating from surrounding cells

	//			   		id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
	const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
	const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
	const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
	
	auto newlyFluidLambda = [=] __cuda_callable__ ( const int index ) mutable
	{		
		const int cell = newlyFluidIndexView( index );
	
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( kPlusView( cell ) );
		NBR.jMinus = jMinusView( cell );
		NBR.kMinus = kMinusView( cell );
		finishNBRAll( NBR, Info );
		// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
		// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
		// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
		// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
		int fullNBRList[27];
		// for each direction this holds the neighbour where f[i] will be pulled from in the next iteration
		// 0: Center
		fullNBRList[0]  = cell;
		// 1-6: Straight directions (Faces)
		fullNBRList[1]  = NBR.iMinus; 			// cx=1  -> nx=-1
		fullNBRList[2]  = NBR.iPlus;  			// cx=-1 -> nx=1
		fullNBRList[3]  = NBR.kPlus;  			// cz=-1 -> nz=1
		fullNBRList[4]  = NBR.kMinus; 			// cz=1  -> nz=-1
		fullNBRList[5]  = NBR.jPlus;  			// cy=-1 -> ny=1
		fullNBRList[6]  = NBR.jMinus; 			// cy=1  -> ny=-1
		// 7-18: Diagonal directions (Edges)
		fullNBRList[7]  = kPlusView( NBR.iMinus );	// cx=1,  cz=-1 -> nx=-1, nz=1
		fullNBRList[8]  = kMinusView( NBR.iPlus );	// cx=-1, cz=1  -> nx=1,  nz=-1
		fullNBRList[9]  = kMinusView( NBR.iMinus );	// cx=1,  cz=1  -> nx=-1, nz=-1
		fullNBRList[10] = kPlusView( NBR.iPlus ); 	// cx=-1, cz=-1 -> nx=1,  nz=1
		fullNBRList[11] = jPlusView( NBR.iPlus ); 	// cx=-1, cy=-1 -> nx=1,  ny=1
		fullNBRList[12] = jMinusView( NBR.iMinus );	// cx=1,  cy=1  -> nx=-1, ny=-1
		fullNBRList[13] = kPlusView( NBR.jMinus );	// cy=1,  cz=-1 -> ny=-1, nz=1
		fullNBRList[14] = kMinusView( NBR.jPlus );	// cy=-1, cz=1  -> ny=1,  nz=-1
		fullNBRList[15] = jMinusView( NBR.iPlus );	// cx=-1, cy=1  -> nx=1,  ny=-1
		fullNBRList[16] = jPlusView( NBR.iMinus );	// cx=1,  cy=-1 -> nx=-1, ny=1
		fullNBRList[17] = kMinusView( NBR.jMinus );	// cy=1,  cz=1  -> ny=-1, nz=-1
		fullNBRList[18] = kPlusView( NBR.jPlus ); 	// cy=-1, cz=-1 -> ny=1,  nz=1
		// 19-26: Corner directions (Vertices)
		fullNBRList[19] = kPlusView( jMinusView( NBR.iPlus ) ); 	// cx=-1, cy=1,  cz=-1 -> nx=1,  ny=-1, nz=1
		fullNBRList[20] = kMinusView( jPlusView( NBR.iMinus ) ); 	// cx=1,  cy=-1, cz=1  -> nx=-1, ny=1,  nz=-1
		fullNBRList[21] = kMinusView( jPlusView( NBR.iPlus ) ); 	// cx=-1, cy=-1, cz=1  -> nx=1,  ny=1,  nz=-1
		fullNBRList[22] = kPlusView( jMinusView( NBR.iMinus ) ); 	// cx=1,  cy=1,  cz=-1 -> nx=-1, ny=-1, nz=1
		fullNBRList[23] = kPlusView( jPlusView( NBR.iMinus ) ); 	// cx=1,  cy=-1, cz=-1 -> nx=-1, ny=1,  nz=1
		fullNBRList[24] = kMinusView( jMinusView( NBR.iPlus ) ); 	// cx=-1, cy=1,  cz=1  -> nx=1,  ny=-1, nz=-1
		fullNBRList[25] = kPlusView( jPlusView( NBR.iPlus ) );  	// cx=-1, cy=-1, cz=-1 -> nx=1,  ny=1,  nz=1
		fullNBRList[26] = kMinusView( jMinusView( NBR.iMinus ) );	// cx=1,  cy=1,  cz=1  -> nx=-1, ny=-1, nz=-1
		
		// now look at each neighbour if they are or were MBB or are BB 
		bool isMovingBounceback[27] = {false};
		bool wasMovingBounceback[27] = {false};
		bool isBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isMovingBounceback[direction] = movingBouncebackMarkerView( fullNBRList[direction] );
			wasMovingBounceback[direction] = ( oldMBBMarkerView( fullNBRList[direction] ) );
			if ( useBouncebackMarkerArray ) isBounceback[direction] = bouncebackMarkerView( fullNBRList[direction] );
		}
		// based on which neighbours are or were MBB, we want to find the best outer normal
		float MBBnx = 0.f; float MBBny = 0.f; float MBBnz = 0.f;
		for ( int direction = 1; direction < 27; direction++ )
		{
			if ( isMovingBounceback[direction] || wasMovingBounceback[direction] ) 
			{
				MBBnx += (float)cxArray[direction];
				MBBny += (float)cyArray[direction];
				MBBnz += (float)czArray[direction];
			}
		}
		int bestIndex = 0;
		float bestProduct = 0.f;
		for ( int normalIndex = 1; normalIndex < 27; normalIndex++ )
		{
			float ex = (float)cxArray[normalIndex]; float ey = (float)cyArray[normalIndex]; float ez = (float)czArray[normalIndex];
			const float eLength = std::sqrt( ex*ex + ey*ey + ez*ez );
			ex /= eLength; ey /= eLength; ez /= eLength;
			const float scalarProduct = MBBnx * ex + MBBny * ey + MBBnz * ez;
			if ( scalarProduct > bestProduct )
			{
				bestIndex = normalIndex;
				bestProduct = scalarProduct;
			}
		}
		const int nx = cxArray[bestIndex];
		const int ny = cyArray[bestIndex];
		const int nz = czArray[bestIndex];
		// at this point the normal is identified
		
		// for quadratic extrapolation I need 3 cells in the direction of the normal
		int extrapolatedNbr[3];
		int currentNbr = cell;
		int extrapolatedCount = 0;
		
		if ( nx == 0 && ny == 0 && nz == 0 )
		{
			// this can only ever happen if our cell is completely enclosed in a moving bounceback solid
			// in such case we will set it to equillibrium later
			// do nothing now but skip the else block below
		}
		else
		{
			for ( int step = 0; step < 3; step++ )
			{
				// Step in X direction 
				if ( nx == 1 )       currentNbr += 1;
				else if ( nx == -1 ) currentNbr -= 1;
				// Step in Y direction
				if ( ny == 1 )       currentNbr = jPlusView( currentNbr );
				else if ( ny == -1 ) currentNbr = jMinusView( currentNbr );
				// Step in Z direction
				if ( nz == 1 )       currentNbr = kPlusView( currentNbr );
				else if ( nz == -1 ) currentNbr = kMinusView( currentNbr );
				extrapolatedNbr[step] = currentNbr;
			}
			// check if the extrapolated neighbour is valid = is in the right place and is fluid
			// only use as many extrapolated neighbours as allowed
			for ( int step = 0; step < 3; step++ )
			{
				const int nbr = extrapolatedNbr[step];
				const int iNbr = iView( nbr );
				const int jNbr = jView( nbr );
				const int kNbr = kView( nbr );
				bool valid = true;
				if 		( iNbr != iCell + nx * (step+1) ) valid = false;
				else if ( jNbr != jCell + ny * (step+1) ) valid = false;
				else if ( kNbr != kCell + nz * (step+1) ) valid = false;
				else if ( useMovingBouncebackMarkerArray && ( movingBouncebackMarkerView( nbr ) || oldMBBMarkerView( nbr )) ) valid = false;
				else if ( useBouncebackMarkerArray && bouncebackMarkerView( nbr ) ) valid = false;
				if ( !valid ) break;
				else extrapolatedCount++;
			}
		}
		
		// initialize the distribution functions that we will be inserting into the newly uncovered cell
		float fRepair[27] = {0.f};
		// for a moment pretend we are still moving bounceback, we will need this later
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCStruct BC;
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		BC.rho = 1.f;
		// find fRepair depending on available extrapolation level
		if ( extrapolatedCount < 2 ) // if not even a linear extrapolation is available, fall back to average from all valid neighbour cells
		{
			int averagingCount = 0;
			// Read distribution functions from all valid neighbors
			for ( int nbrIndex = 1; nbrIndex < 27; nbrIndex++ )
			{
				if 		( isMovingBounceback[nbrIndex] ) continue;
				else if ( wasMovingBounceback[nbrIndex] ) continue;
				else if ( isBounceback[nbrIndex] ) continue;
				const int nbr = fullNBRList[nbrIndex];
				NBRStruct NBRofNBR;
				NBRofNBR.self = nbr;
				NBRofNBR.jPlus = jPlusView( nbr );
				NBRofNBR.kPlus = kPlusView( nbr );
				NBRofNBR.jkPlus = jPlusView( kPlusView( nbr ) );
				finishNBRPlus( NBRofNBR, Info );
				int cellReadIndex[27];
				int fReadIndex[27];
				getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
				for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] += fView( fReadIndex[direction], cellReadIndex[direction] );
				averagingCount++;
			}	
			if ( averagingCount == 0 ) // if no neighbour is valid, use equillibrium
			{
				getFeq(	BC.rho, BC.ux, BC.uy, BC.uz, fRepair );
			}
			else for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] /= (float)averagingCount;
		}
		else
		{
			// Read distribution functions from all valid extrapolated neighbors
			float fExtrapolated[3][27];
			for ( int step = 0; step < extrapolatedCount; step++ )
			{
				const int nbr = extrapolatedNbr[step];
				NBRStruct NBRofNBR;
				NBRofNBR.self = nbr;
				NBRofNBR.jPlus = jPlusView( nbr );
				NBRofNBR.kPlus = kPlusView( nbr );
				NBRofNBR.jkPlus = jPlusView( kPlusView( nbr ) );
				finishNBRPlus( NBRofNBR, Info );
				
				int cellReadIndex[27];
				int fReadIndex[27];
				getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
				
				for ( int direction = 0; direction < 27; direction++ )
				{
					fExtrapolated[step][direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
				}
			}
			
			// Apply the appropriate extrapolation formula based on how many valid cells we found
			
			if ( extrapolatedCount == 1 ) // constant extrapolation
			{
				for ( int direction = 0; direction < 27; direction++ )
				{
					fRepair[direction] = fExtrapolated[0][direction];
				}
			}
			else if ( extrapolatedCount == 2 ) // linear extrapolation: f(x) = 2*f(x+1) - f(x+2)
			{
				for ( int direction = 0; direction < 27; direction++ )
				{
					fRepair[direction] = 2.0f * fExtrapolated[0][direction] - 1.0f * fExtrapolated[1][direction];
				}
			}
			else if ( extrapolatedCount == 3 ) // quadratic extrapolation: f(x) = 3*f(x+1) - 3*f(x+2) + f(x+3)
			{
				for ( int direction = 0; direction < 27; direction++ )
				{
					fRepair[direction] = 3.0f * fExtrapolated[0][direction] - 3.0f * fExtrapolated[1][direction] + 1.0f * fExtrapolated[2][direction];
				}
			}
		}
		
		// now, modify the equillibrium to match ux, uy, uz of the MBB
		float rhoAvg, uxAvg, uyAvg, uzAvg;
		getRhoUxUyUz( rhoAvg, uxAvg, uyAvg, uzAvg, fRepair );
		float fEqAvg[27];
		float fEqTarget[27];
		// get equilibrium of the averaged fluid
		getFeq( rhoAvg, uxAvg, uyAvg, uzAvg, fEqAvg );
		// get equilibrium using the target ux, uy, uz (but keep rhoAvg)
		getFeq( rhoAvg, BC.ux, BC.uy, BC.uz, fEqTarget );
		// reconstruct
		for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] = fEqTarget[direction] + ( fRepair[direction] - fEqAvg[direction] );		
		
		// write fRepair into our cell
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreviousPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fView( fWriteIndex[direction], cellWriteIndex[direction] ) = fRepair[direction];
		
		// also repair the distribution functions that are going to be pulled into our cell by previously deep solid moving bounceback cells
		applyMovingBounceback( fRepair, BC );
		int cellNextIndex[27];
		int fNextIndex[27];
		getPreCollisionIndex( cellNextIndex, fNextIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 1; direction < 27; direction++ ) 
		{
			if ( isMovingBounceback[direction] || isBounceback[direction] ) 
			{
				// if we are going to be receiving f from a moving bounceback in this direction,
				// set it to the moving bounceback result
				fView( fNextIndex[direction], cellNextIndex[direction] ) = fRepair[direction];
			}
		}	
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, newlyFluidLambda );
	
	// Now, the above works quite well, but there are still artifacts if K17 collision is used.
	// We will iteratively run local streaming and collision only for the uncovered cells, as described by 
	// Li Chen, Yang Yu, Jianhua Lu, Guoxiang Hou, 
	// A comparative study of lattice Boltzmann methods using bounce-back schemes and immersed boundary ones for flow acoustic problems, 2013
	// LI scheme
	
	for ( int refillCorrectionIteration = 0; refillCorrectionIteration < 100; refillCorrectionIteration++ )
	{
		runRefillCorrection( Grid, newlyFluidCount );
	}
		
	// Last step: Sum the torque contributions

	auto newlyMBBLambda = [=] __cuda_callable__ ( const int index ) mutable
	{		
		const int cell = newlyMBBIndexView( index );
		
		float Tz = 0.f;
		
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( kPlusView( cell ) );
		NBR.jMinus = jMinusView( cell );
		NBR.kMinus = kMinusView( cell );
		finishNBRAll( NBR, Info );
		
		float x, y, z;
		getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
		
		BCStruct BC;
		// load the current state into the boundary condition struct
		getRhoUxUyUz( BC.rho, BC.ux, BC.uy, BC.uz, f );
		float uxOld = BC.ux; float uyOld = BC.uy; float uzOld = BC.uz;
		// pass the current state into the boundary condition function so that BC can also be a function of the current state 
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		
		float gx = BC.rho * ( BC.ux - uxOld );
		float gy = BC.rho * ( BC.uy - uyOld );
		float gz = BC.rho * ( BC.uz - uzOld );
		
		convertToPhysicalForce( gx, gy, gz, Info );
		Tz = - gx * y + gy * x;
		
		return Tz;
	};
	
	auto reduction = [] __cuda_callable__( const float& a, const float& b )
	{
		return a + b;
	};
	
	float TzSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, newlyMBBCount, newlyMBBLambda, reduction, 0.f );
	TzSum = TzSum / 1000.f; // converting from Nmm to Nm
	Grid.Info.torqueReportCumulative += TzSum;	
		
	Info.updatesSinceMovingBouncebackUpdate = 0;
}
