#pragma once

#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"
#include "./applyCollision.h"
#include "./boundaryConditions/applyMovingBounceback.h"

void updateMovingBounceback( GridStruct &Grid, const VoxelizerStruct &Voxelizer )
{
	InfoStruct &Info = Grid.Info;
	BoolArrayType &oldMovingBouncebackMarkerArray = Grid.markerBuffer;
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	
	// Take copy of the old moving bounceback marker array and update the active array
	oldMovingBouncebackMarkerArray = Grid.movingBouncebackMarkerArray;
	applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Info.cellCount );
	
	// Now we need to repair information in cells that were previously moving bounceback and now are fluid
	// First, identify which ones those cells are
	auto oldMovingBouncebackMarkerView = oldMovingBouncebackMarkerArray.getView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	
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
	auto intBuffer1View = Grid.intBuffer1.getView();
	auto intBuffer3View = Grid.intBuffer3.getView();
	auto indexLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool isNewlyFluid = oldMovingBouncebackMarkerView( cell );
		if ( isNewlyFluid ) intBuffer3View( intBuffer1View( cell ) ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, indexLambda );
	// Now we have our index list in intBuffer3 ( = NBR.jkPlusArray )
	// We no longer need intBuffer1 ( = NBR.jMinusArray) -> we repair it
	auto jPlusView = Grid.NBR.jPlusArray.getView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto jMinusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jMinusView[ jPlusView[ cell ] ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, jMinusRepairLambda );
	
	// Now loop through all newly fluid cells and repair the information in them by interpolating from surrounding cells
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;	
	auto fArrayView  = Grid.fArray.getView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
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
		// now look at each neighbour if they are MBB 
		bool isMovingBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isMovingBounceback[direction] = movingBouncebackMarkerView( fullNBRList[direction] );
		}
		bool isBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isBounceback[direction] = bouncebackMarkerView( fullNBRList[direction] );
		}
		// next step is to identify the surface normal direction, that is where we will be extrapolating from
		int nx, ny, nz = 0;
		nx = + 1 * isMovingBounceback[1] - 1 * isMovingBounceback[2];
		ny = - 1 * isMovingBounceback[5] + 1 * isMovingBounceback[6];
		nz = - 1 * isMovingBounceback[3] + 1 * isMovingBounceback[4];
		
		if ( nx == 0 && ny == 0 && nz == 0 )
		{
			// very unlikely scenario, if this happens we need to build the normal from the 7-18 or 19-26 neighbours
			// lets work on this later
		}
		// for quadratic extrapolation I need 3 cells in the direction of the normal
		int extrapolatedNbr[3];
		int currentNbr = cell;
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
		int extrapolatedCount = 0;
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
		// initialize the distribution functions that we will be inserting into the newly uncovered cell
		float fRepair[27];
		// for a moment pretend we are still moving bounceback, we will need this later
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCRhoUGStruct BCRhoUG;
		getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		BCRhoUG.rho = 1.f;
		// find fRepair depending on available extrapolation level
		if ( extrapolatedCount == 0 ) // no extrapolation, use equillibrium
		{
			getFeq(BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fRepair);
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
		// now we still want to set ux, uy, uz to match the moving bounceback
		// so we modify the equillibrium part
		
		float rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated;
		getRhoUxUyUz( rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated, fRepair );
		float fEqExtrapolated[27];
		float fEqTarget[27];
		// get equilibrium of the extrapolated fluid
		getFeq( rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated, fEqExtrapolated );
		// get equilibrium using the target wall velocity (but keeping the extrapolated density)
		getFeq( rhoExtrapolated, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fEqTarget );
		// reconstruct: f_new = f_eq(wall_u) + (f_extrap - f_eq(extrap_u))
		for ( int direction = 0; direction < 27; direction++ )
		{
			fRepair[direction] = fEqTarget[direction] + ( fRepair[direction] - fEqExtrapolated[direction] );
		}
		
		// BCRhoUG.rho is needed below for applyMovingBounceback
		BCRhoUG.rho = rhoExtrapolated;
		
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
void updateMovingBouncebackDEBUG( GridStruct &Grid, const VoxelizerStruct &Voxelizer )
{
	std::cout << "UPDATING MBB DEBUG VERSION" << std::endl;
	InfoStruct &Info = Grid.Info;
	BoolArrayType &oldMovingBouncebackMarkerArray = Grid.markerBuffer;
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	
	// Take copy of the old moving bounceback marker array and update the active array
	oldMovingBouncebackMarkerArray = Grid.movingBouncebackMarkerArray;
	applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Info.cellCount );
	
	// Now we need to repair information in cells that were previously moving bounceback and now are fluid
	// First, identify which ones those cells are
	auto oldMovingBouncebackMarkerView = oldMovingBouncebackMarkerArray.getView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	
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
	auto intBuffer1View = Grid.intBuffer1.getView();
	auto intBuffer3View = Grid.intBuffer3.getView();
	auto indexLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const bool isNewlyFluid = oldMovingBouncebackMarkerView( cell );
		if ( isNewlyFluid ) intBuffer3View( intBuffer1View( cell ) ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, indexLambda );
	// Now we have our index list in intBuffer3 ( = NBR.jkPlusArray )
	// We no longer need intBuffer1 ( = NBR.jMinusArray) -> we repair it
	auto jPlusView = Grid.NBR.jPlusArray.getView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto jMinusRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jMinusView[ jPlusView[ cell ] ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, jMinusRepairLambda );
	
	// Now loop through all newly fluid cells and repair the information in them by interpolating from surrounding cells
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;	
	auto fArrayView  = Grid.fArray.getView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	
	FloatArray2DTypeCPU fArrayCPU;
	fArrayCPU = Grid.fArray;
	
	for ( int index = 0; index < newlyFluidCount; index++ )
	{
		const int cell = Grid.intBuffer3.getElement( index );
		
		const int iCell = Grid.IJK.iArray.getElement( cell );
		const int jCell = Grid.IJK.jArray.getElement( cell );
		const int kCell = Grid.IJK.kArray.getElement( cell );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = Grid.NBR.jPlusArray.getElement( cell );
		NBR.kPlus = Grid.NBR.kPlusArray.getElement( cell );
		NBR.jkPlus = Grid.NBR.jPlusArray.getElement( Grid.NBR.kPlusArray.getElement( cell ) );
		NBR.jMinus = Grid.NBR.jMinusArray.getElement( cell );
		NBR.kMinus = Grid.NBR.kMinusArray.getElement( cell );
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
		fullNBRList[7]  = Grid.NBR.kPlusArray.getElement( NBR.iMinus );	// cx=1,  cz=-1 -> nx=-1, nz=1
		fullNBRList[8]  = Grid.NBR.kMinusArray.getElement( NBR.iPlus );	// cx=-1, cz=1  -> nx=1,  nz=-1
		fullNBRList[9]  = Grid.NBR.kMinusArray.getElement( NBR.iMinus );	// cx=1,  cz=1  -> nx=-1, nz=-1
		fullNBRList[10] = Grid.NBR.kPlusArray.getElement( NBR.iPlus ); 	// cx=-1, cz=-1 -> nx=1,  nz=1
		fullNBRList[11] = Grid.NBR.jPlusArray.getElement( NBR.iPlus ); 	// cx=-1, cy=-1 -> nx=1,  ny=1
		fullNBRList[12] = Grid.NBR.jMinusArray.getElement( NBR.iMinus );	// cx=1,  cy=1  -> nx=-1, ny=-1
		fullNBRList[13] = Grid.NBR.kPlusArray.getElement( NBR.jMinus );	// cy=1,  cz=-1 -> ny=-1, nz=1
		fullNBRList[14] = Grid.NBR.kMinusArray.getElement( NBR.jPlus );	// cy=-1, cz=1  -> ny=1,  nz=-1
		fullNBRList[15] = Grid.NBR.jMinusArray.getElement( NBR.iPlus );	// cx=-1, cy=1  -> nx=1,  ny=-1
		fullNBRList[16] = Grid.NBR.jPlusArray.getElement( NBR.iMinus );	// cx=1,  cy=-1 -> nx=-1, ny=1
		fullNBRList[17] = Grid.NBR.kMinusArray.getElement( NBR.jMinus );	// cy=1,  cz=1  -> ny=-1, nz=-1
		fullNBRList[18] = Grid.NBR.kPlusArray.getElement( NBR.jPlus ); 	// cy=-1, cz=-1 -> ny=1,  nz=1
		// 19-26: Corner directions (Vertices)
		fullNBRList[19] = Grid.NBR.kPlusArray.getElement( Grid.NBR.jMinusArray.getElement( NBR.iPlus ) ); 	// cx=-1, cy=1,  cz=-1 -> nx=1,  ny=-1, nz=1
		fullNBRList[20] = Grid.NBR.kMinusArray.getElement( Grid.NBR.jPlusArray.getElement( NBR.iMinus ) ); 	// cx=1,  cy=-1, cz=1  -> nx=-1, ny=1,  nz=-1
		fullNBRList[21] = Grid.NBR.kMinusArray.getElement( Grid.NBR.jPlusArray.getElement( NBR.iPlus ) ); 	// cx=-1, cy=-1, cz=1  -> nx=1,  ny=1,  nz=-1
		fullNBRList[22] = Grid.NBR.kPlusArray.getElement( Grid.NBR.jMinusArray.getElement( NBR.iMinus ) ); 	// cx=1,  cy=1,  cz=-1 -> nx=-1, ny=-1, nz=1
		fullNBRList[23] = Grid.NBR.kPlusArray.getElement( Grid.NBR.jPlusArray.getElement( NBR.iMinus ) ); 	// cx=1,  cy=-1, cz=-1 -> nx=-1, ny=1,  nz=1
		fullNBRList[24] = Grid.NBR.kMinusArray.getElement( Grid.NBR.jMinusArray.getElement( NBR.iPlus ) ); 	// cx=-1, cy=1,  cz=1  -> nx=1,  ny=-1, nz=-1
		fullNBRList[25] = Grid.NBR.kPlusArray.getElement( Grid.NBR.jPlusArray.getElement( NBR.iPlus ) );  	// cx=-1, cy=-1, cz=-1 -> nx=1,  ny=1,  nz=1
		fullNBRList[26] = Grid.NBR.kMinusArray.getElement( Grid.NBR.jMinusArray.getElement( NBR.iMinus ) );	// cx=1,  cy=1,  cz=1  -> nx=-1, ny=-1, nz=-1
		// now look at each neighbour if they are MBB 
		bool isMovingBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isMovingBounceback[direction] = Grid.movingBouncebackMarkerArray.getElement( fullNBRList[direction] );
		}
		bool isBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isBounceback[direction] = Grid.bouncebackMarkerArray.getElement( fullNBRList[direction] );
		}
		// next step is to identify the surface normal direction, that is where we will be extrapolating from
		int nx, ny, nz = 0;
		nx = + 1 * isMovingBounceback[1] - 1 * isMovingBounceback[2];
		ny = - 1 * isMovingBounceback[5] + 1 * isMovingBounceback[6];
		nz = - 1 * isMovingBounceback[3] + 1 * isMovingBounceback[4];
		
		if ( nx == 0 && ny == 0 && nz == 0 )
		{
			// very unlikely scenario, if this happens we need to build the normal from the 7-18 or 19-26 neighbours
			// lets work on this later
		}
		// for quadratic extrapolation I need 3 cells in the direction of the normal
		int extrapolatedNbr[3];
		int currentNbr = cell;
		for ( int step = 0; step < 3; step++ )
		{
			// Step in X direction 
			if ( nx == 1 )       currentNbr += 1;
			else if ( nx == -1 ) currentNbr -= 1;
			// Step in Y direction
			if ( ny == 1 )       currentNbr = Grid.NBR.jPlusArray.getElement( currentNbr );
			else if ( ny == -1 ) currentNbr = Grid.NBR.jMinusArray.getElement( currentNbr );
			// Step in Z direction
			if ( nz == 1 )       currentNbr = Grid.NBR.kPlusArray.getElement( currentNbr );
			else if ( nz == -1 ) currentNbr = Grid.NBR.kMinusArray.getElement( currentNbr );
			extrapolatedNbr[step] = currentNbr;
		}
		// check if the extrapolated neighbour is valid = is in the right place and is fluid
		// only use as many extrapolated neighbours as allowed
		int extrapolatedCount = 0;
		for ( int step = 0; step < 3; step++ )
		{
			const int nbr = extrapolatedNbr[step];
			const int iNbr = Grid.IJK.iArray.getElement( nbr );
			const int jNbr = Grid.IJK.jArray.getElement( nbr );
			const int kNbr = Grid.IJK.kArray.getElement( nbr );
			bool valid = true;
			if 		( iNbr != iCell + nx * (step+1) ) valid = false;
			else if ( jNbr != jCell + ny * (step+1) ) valid = false;
			else if ( kNbr != kCell + nz * (step+1) ) valid = false;
			else if ( useMovingBouncebackMarkerArray && ( Grid.movingBouncebackMarkerArray.getElement( nbr ) || oldMovingBouncebackMarkerArray.getElement( nbr )) ) valid = false;
			else if ( useBouncebackMarkerArray && Grid.bouncebackMarkerArray.getElement( nbr ) ) valid = false;
			if ( !valid ) break;
			else extrapolatedCount++;
		}
		// initialize the distribution functions that we will be inserting into the newly uncovered cell
		float fRepair[27];
		// for a moment pretend we are still moving bounceback, we will need this later
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		BCRhoUGStruct BCRhoUG;
		getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		BCRhoUG.rho = 1.f;
		// find fRepair depending on available extrapolation level
		if ( extrapolatedCount == 0 ) // no extrapolation, use equillibrium
		{
			getFeq(BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fRepair);
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
				NBRofNBR.jPlus = Grid.NBR.jPlusArray.getElement( nbr );
				NBRofNBR.kPlus = Grid.NBR.kPlusArray.getElement( nbr );
				NBRofNBR.jkPlus = Grid.NBR.jPlusArray.getElement( Grid.NBR.kPlusArray.getElement( nbr ) );
				finishNBRPlus( NBRofNBR, Info );
				
				int cellReadIndex[27];
				int fReadIndex[27];
				getPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
				
				for ( int direction = 0; direction < 27; direction++ )
				{
					fExtrapolated[step][direction] = Grid.fArray.getElement(fReadIndex[direction], cellReadIndex[direction]);
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
		// now we still want to set ux, uy, uz to match the moving bounceback
		// so we modify the equillibrium part
		
		float rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated;
		getRhoUxUyUz( rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated, fRepair );
		float fEqExtrapolated[27];
		float fEqTarget[27];
		// get equilibrium of the extrapolated fluid
		getFeq( rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated, fEqExtrapolated );
		// get equilibrium using the target wall velocity (but keeping the extrapolated density)
		getFeq( rhoExtrapolated, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fEqTarget );
		// reconstruct: f_new = f_eq(wall_u) + (f_extrap - f_eq(extrap_u))
		for ( int direction = 0; direction < 27; direction++ )
		{
			fRepair[direction] = fEqTarget[direction] + ( fRepair[direction] - fEqExtrapolated[direction] );
		}
		
		// BCRhoUG.rho is needed below for applyMovingBounceback
		BCRhoUG.rho = rhoExtrapolated;
		
		// write fRepair into our cell
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			if ( cell == 2121100 )
			{
				std::cout << "setting f[" << direction << "] to " << fRepair[direction] << std::endl;
			}
			fArrayCPU( fWriteIndex[direction], cellWriteIndex[direction]) = fRepair[direction];
		}
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
				if ( cell == 2121100 )
				{
					std::cout << "setting incoming f[" << direction << "] to " << fRepair[direction] << std::endl;
				}
				fArrayCPU( fNextIndex[direction], cellNextIndex[direction]) =  fRepair[direction];
			}
			else if ( cell == 2121100 )
			{
				std::cout << "keeping incoming f[" << direction << "] at " << fArrayCPU( fNextIndex[direction], cellNextIndex[direction]) << std::endl;
				if ( direction == 5 ) // the problem is here!
				{
					std::cout << "direction 5 comes from fNextIndex " << fNextIndex[direction] << ", cellNextIndex " << cellNextIndex[direction] << std::endl;
				}
			}
			
		}			
	}
	
	Grid.fArray = fArrayCPU;
	
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
