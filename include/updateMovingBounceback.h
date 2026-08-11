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
void runRefillCorrection( GridStruct &Grid, const int &newlyFluidCount, const float &underRelaxation )
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
			fBufferView( direction, index ) = fBufferView( direction, index ) * underRelaxation + f[direction] * (1.f - underRelaxation);
		}	
		
		//float rho, ux, uy, uz;
		//getRhoUxUyUz(rho, ux, uy, uz, f);
		//if (index == 1) printf("rho %f ux %f uy %f uz %f\n", rho, ux, uy, uz);
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
		
		// get MBB velocity
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCStruct BC;
		getRhoUxUyUz(BC.rho, BC.ux, BC.uy, BC.uz, f);
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		/*
		// now, modify the equillibrium to match ux, uy, uz of the MBB
		// get current equilibrium
		float fEq[27];
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, f );
		getFeq( rho, ux, uy, uz, fEq );
		
		// get equilibrium using the target ux, uy, uz (but keep rho)
		float fEqTarget[27];
		getFeq( rho, BC.ux, BC.uy, BC.uz, fEqTarget );
		// reconstruct
		for ( int direction = 0; direction < 27; direction++ ) f[direction] = f[direction] + ( fEqTarget[direction] - fEq[direction] );	
		*/
		
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
	//std::cout << "updating MBB now" << std::endl;
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
		
		// initialize the distribution functions that we will be inserting into the newly uncovered cell
		float fRepair[27] = {0.f};
		// for a moment pretend we are still moving bounceback, we will need this later
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCStruct BC;
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		BC.rho = 1.f;
		// find fRepair as average from all valid neighbour cells. Note: quadratic extrapolation was tested before but sometimes led to crashes.
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
	
	float underRelaxation = 0.f;
	runRefillCorrection( Grid, newlyFluidCount, underRelaxation );
	underRelaxation = 0.25f;
	for ( int refillCorrectionIteration = 0; refillCorrectionIteration < 20; refillCorrectionIteration++ )
	{
		runRefillCorrection( Grid, newlyFluidCount, underRelaxation );
	}
	
	// Last step: Sum the torque contributions
	
	auto reduction = [] __cuda_callable__( const float& a, const float& b )
	{
		return a + b;
	};
	
	auto newlyMBBTorqueLambda = [=] __cuda_callable__ ( const int index ) mutable
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
		
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, f );
		
		// we are removing this fluid cell from the system
		// this is the same as if we slowed it down to zero velocity
		// force = rho * ( new u - old u) ... F = m*a
		float gx = rho * ( 0.f - ux );
		float gy = rho * ( 0.f - uy );
		float gz = rho * ( 0.f - uz );
		
		convertToPhysicalForce( gx, gy, gz, Info );
		Tz = - gx * y + gy * x;
		
		return Tz;
	};
	
	float TzSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, newlyMBBCount, newlyMBBTorqueLambda, reduction, 0.f );
	
	auto newlyFluidTorqueLambda = [=] __cuda_callable__ ( const int index ) mutable
	{		
		const int cell = newlyFluidIndexView( index );
		
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
		
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, f );
		
		// we are adding this fluid cell into the system
		// this is the same as if we accelerated it up from zero velocity
		// force = rho * ( new u - old u) ... F = m*a
		float gx = rho * ( ux - 0.f);
		float gy = rho * ( uy - 0.f );
		float gz = rho * ( uz - 0.f );
		
		convertToPhysicalForce( gx, gy, gz, Info );
		Tz = - gx * y + gy * x;
		
		return Tz;
	};
	
	TzSum += TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, newlyFluidCount, newlyFluidTorqueLambda, reduction, 0.f );
	
	TzSum = TzSum / 1000.f; // converting from Nmm to Nm
	Grid.Info.torqueReportCumulative += TzSum;	
		
	Info.updatesSinceMovingBouncebackUpdate = 0;
}
