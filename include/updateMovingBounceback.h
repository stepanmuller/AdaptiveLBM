#pragma once

#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"
#include "./applyCollision.h"
#include "./boundaryConditions/applyMovingBounceback.h"

void updateMovingBounceback( GridStruct &Grid, const VoxelizerStruct &Voxelizer )
{
	//if ( Grid.Info.iterationsFinished > 310 ) std::cout << "UPDATING MBB" << std::endl;
	InfoStruct &Info = Grid.Info;
	BoolArrayType &oldMovingBouncebackMarkerArray = Grid.markerBuffer;
	
	auto fArrayView  = Grid.fArray.getView();
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto intBuffer1View = Grid.intBuffer1.getView();
	auto intBuffer3View = Grid.intBuffer3.getView();
	auto jPlusView = Grid.NBR.jPlusArray.getView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	const bool &esotwistFlipper = Grid.esotwistFlipper;	
	
	// Take copy of the old moving bounceback marker array and update the active array
	oldMovingBouncebackMarkerArray = Grid.movingBouncebackMarkerArray;
	auto oldMovingBouncebackMarkerView = oldMovingBouncebackMarkerArray.getView();
	applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Info.cellCount );
	
	// Now we need to repair information in cells that were previously moving bounceback and now are fluid
	// First, identify which ones those cells are
	auto markerLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool oldMarker = oldMovingBouncebackMarkerView( cell );
		const bool newMarker = movingBouncebackMarkerView( cell );
		const bool isNewlyFluid = oldMarker && !newMarker;
		oldMovingBouncebackMarkerView( cell ) = isNewlyFluid;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, markerLambda );
	// To gather indexes of those cells we will use intBuffer1 ( = NBR.jMinusArray) and intBuffer3 ( = NBR.jkPlusArray), we need to repair those later! 
	const int newlyFluidCount = countOnesInBoolArray( oldMovingBouncebackMarkerArray, Info.cellCount );
	intArrayFromBoolArray( Grid.intBuffer1, oldMovingBouncebackMarkerArray, Info.cellCount );
	TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer1, 0, Info.cellCount, TNL::Plus{} );
	
	auto indexLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool isNewlyFluid = oldMovingBouncebackMarkerView( cell );
		if ( isNewlyFluid ) intBuffer3View( intBuffer1View( cell ) ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, indexLambda );
	// Now we have our index list in intBuffer3 ( = NBR.jkPlusArray )
	// We no longer need intBuffer1 ( = NBR.jMinusArray) -> we repair it
	auto jMinusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jMinusView[ jPlusView[ cell ] ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, jMinusRepairLambda );
	
	// Now loop through all newly fluid cells and repair the information in them by interpolating from surrounding cells

	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	
	//			   id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
	const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
	const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
	const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
	
	auto newlyFluidLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = intBuffer3View( index );
		
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
			wasMovingBounceback[direction] = oldMovingBouncebackMarkerView( fullNBRList[direction] );
			if ( useBouncebackMarkerArray ) isBounceback[direction] = bouncebackMarkerView( fullNBRList[direction] );
		}
		// based on which neighbours are MBB, we want to find the best outer normal
		float MBBnx = 0.f; float MBBny = 0.f; float MBBnz = 0.f;
		for ( int direction = 1; direction < 27; direction++ )
		{
			if ( isMovingBounceback[direction] ) 
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
			float ex = -(float)cxArray[normalIndex]; float ey = -(float)cyArray[normalIndex]; float ez = -(float)czArray[normalIndex];
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
				else if ( useMovingBouncebackMarkerArray && ( movingBouncebackMarkerView( nbr ) || oldMovingBouncebackMarkerView( nbr )) ) valid = false;
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
		BCRhoUGStruct BCRhoUG;
		getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		BCRhoUG.rho = 1.f;
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
				getPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
				for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] += fArrayView( fReadIndex[direction], cellReadIndex[direction] );
				averagingCount++;
			}	
			if ( averagingCount == 0 ) // if no neighbour is valid, use equillibrium
			{
				getFeq(	BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fRepair );
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
				getPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
				
				for ( int direction = 0; direction < 27; direction++ )
				{
					fExtrapolated[step][direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
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
		getFeq( rhoAvg, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fEqTarget );
		// reconstruct
		for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] = fEqTarget[direction] + ( fRepair[direction] - fEqAvg[direction] );		
		
		// write fRepair into our cell
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = fRepair[direction];
		
		// also repair the distribution functions that are going to be pulled into our cell next iteration from moving bounceback cells
		applyMovingBounceback( fRepair, BCRhoUG );
		int cellNextIndex[27];
		int fNextIndex[27];
		bool inverseEsotwistFlipper = !esotwistFlipper;
		getPreCollisionIndex( cellNextIndex, fNextIndex, NBR, inverseEsotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			if ( isMovingBounceback[direction] || isBounceback[direction] ) 
			{
				// if we are going to be receiving f from a moving bounceback in this direction,
				// set it to the moving bounceback result
				fArrayView( fNextIndex[direction], cellNextIndex[direction] ) = fRepair[direction];
			}
		}			
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, newlyFluidLambda );
	
	// Last step - repair jkPlusArray which we broke previously up to newlyFluidCount
	auto jkPlusView = Grid.NBR.jkPlusArray.getView();
	auto jkPlusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jkPlusView[ cell ] = jPlusView[ kPlusView[ cell ] ];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, jkPlusRepairLambda );
	
	Info.updatesSinceMovingBouncebackUpdate = 0;
}

/*
// Older simplier version that just uses average across all valid neighbour cells.
// The extrapolation method above seems to remove one of the artifacts so we stick with it
void updateMovingBounceback( GridStruct &Grid, const VoxelizerStruct &Voxelizer )
{
	//if ( Grid.Info.iterationsFinished > 310 ) std::cout << "UPDATING MBB" << std::endl;
	InfoStruct &Info = Grid.Info;
	BoolArrayType &oldMovingBouncebackMarkerArray = Grid.markerBuffer;
	
	auto fArrayView  = Grid.fArray.getView();
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto intBuffer1View = Grid.intBuffer1.getView();
	auto intBuffer3View = Grid.intBuffer3.getView();
	auto jPlusView = Grid.NBR.jPlusArray.getView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	const bool &esotwistFlipper = Grid.esotwistFlipper;	
	
	// Take copy of the old moving bounceback marker array and update the active array
	oldMovingBouncebackMarkerArray = Grid.movingBouncebackMarkerArray;
	auto oldMovingBouncebackMarkerView = oldMovingBouncebackMarkerArray.getView();
	applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Info.cellCount );
	
	// Now we need to repair information in cells that were previously moving bounceback and now are fluid
	// First, identify which ones those cells are
	auto markerLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool oldMarker = oldMovingBouncebackMarkerView( cell );
		const bool newMarker = movingBouncebackMarkerView( cell );
		const bool isNewlyFluid = oldMarker && !newMarker;
		oldMovingBouncebackMarkerView( cell ) = isNewlyFluid;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, markerLambda );
	// To gather indexes of those cells we will use intBuffer1 ( = NBR.jMinusArray) and intBuffer3 ( = NBR.jkPlusArray), we need to repair those later! 
	const int newlyFluidCount = countOnesInBoolArray( oldMovingBouncebackMarkerArray, Info.cellCount );
	intArrayFromBoolArray( Grid.intBuffer1, oldMovingBouncebackMarkerArray, Info.cellCount );
	TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer1, 0, Info.cellCount, TNL::Plus{} );
	
	auto indexLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool isNewlyFluid = oldMovingBouncebackMarkerView( cell );
		if ( isNewlyFluid ) intBuffer3View( intBuffer1View( cell ) ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, indexLambda );
	// Now we have our index list in intBuffer3 ( = NBR.jkPlusArray )
	// We no longer need intBuffer1 ( = NBR.jMinusArray) -> we repair it
	auto jMinusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jMinusView[ jPlusView[ cell ] ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, jMinusRepairLambda );
	
	// Now loop through all newly fluid cells and repair the information in them by interpolating from surrounding cells

	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	
	auto newlyFluidLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = intBuffer3View( index );
		
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
			wasMovingBounceback[direction] = oldMovingBouncebackMarkerView( fullNBRList[direction] );
			if ( useBouncebackMarkerArray ) isBounceback[direction] = bouncebackMarkerView( fullNBRList[direction] );
		}
		// initialize the distribution functions that we will be inserting into the newly uncovered cell
		float fRepair[27] = {0.0f};
		int averagingCount = 0;
		// we will still want to set ux, uy, uz to match the moving bounceback
		// for a moment pretend we are still moving bounceback - this is how we get the correct ux, uy, uz
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCRhoUGStruct BCRhoUG;
		getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		BCRhoUG.rho = 1.f;
		
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
			getPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] += fArrayView( fReadIndex[direction], cellReadIndex[direction] );
			averagingCount++;
		}	
		
		if ( averagingCount == 0 ) // if no neighbour is valid, use equillibrium
		{
			getFeq(	BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fRepair );
		}
		else // average and modify equillibrium part for correct ux, uy, uz
		{
			for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] /= (float)averagingCount;
			// now, modify the equillibrium to match ux, uy, uz of the MBB
			float rhoAvg, uxAvg, uyAvg, uzAvg;
			getRhoUxUyUz( rhoAvg, uxAvg, uyAvg, uzAvg, fRepair );
			float fEqAvg[27];
			float fEqTarget[27];
			// get equilibrium of the averaged fluid
			getFeq( rhoAvg, uxAvg, uyAvg, uzAvg, fEqAvg );
			// get equilibrium using the target ux, uy, uz (but keep rhoAvg)
			getFeq( rhoAvg, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fEqTarget );
			// reconstruct
			for ( int direction = 0; direction < 27; direction++ ) fRepair[direction] = fEqTarget[direction] + ( fRepair[direction] - fEqAvg[direction] );		
		}	
		
		// write fRepair into our cell
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = fRepair[direction];
		
		// also repair the distribution functions that are going to be pulled into our cell next iteration from moving bounceback cells
		applyMovingBounceback( fRepair, BCRhoUG );
		int cellNextIndex[27];
		int fNextIndex[27];
		bool inverseEsotwistFlipper = !esotwistFlipper;
		getPreCollisionIndex( cellNextIndex, fNextIndex, NBR, inverseEsotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			if ( isMovingBounceback[direction] || isBounceback[direction] ) 
			{
				// if we are going to be receiving f from a moving bounceback in this direction,
				// set it to the moving bounceback result
				fArrayView( fNextIndex[direction], cellNextIndex[direction] ) = fRepair[direction];
			}
		}			
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, newlyFluidLambda );
	
	// Last step - repair jkPlusArray which we broke previously up to newlyFluidCount
	auto jkPlusView = Grid.NBR.jkPlusArray.getView();
	auto jkPlusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jkPlusView[ cell ] = jPlusView[ kPlusView[ cell ] ];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, newlyFluidCount, jkPlusRepairLambda );
	
	Info.updatesSinceMovingBouncebackUpdate = 0;
}
*/
