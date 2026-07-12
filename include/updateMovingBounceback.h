#pragma once

#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"

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
	auto isGeometricMarkerView = Grid.NBR.isGeometricMarkerArray.getConstView(); // order: iPlus, jPlus, ijPlus, kPlus, ikPlus, jkPlus, ijkPlus, iMinus, jMinus, kMinus
	
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	
	auto newlyFluidLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = intBuffer3View( index );
		
		bool isGeometricList[6];
		isGeometricList[0] = isGeometricMarkerView( 0, cell );
		isGeometricList[1] = isGeometricMarkerView( 7, cell );
		isGeometricList[2] = isGeometricMarkerView( 1, cell );
		isGeometricList[3] = isGeometricMarkerView( 8, cell );
		isGeometricList[4] = isGeometricMarkerView( 3, cell );
		isGeometricList[5] = isGeometricMarkerView( 9, cell );
		
		int nbrList[6] = {0};
		if (isGeometricList[0]) nbrList[0] = cell + 1; if ( nbrList[0] >= Info.cellCount ) nbrList[0] = 0;
		if (isGeometricList[1]) nbrList[1] = cell - 1; if ( nbrList[1] < 0 ) nbrList[1] = Info.cellCount - 1;
		nbrList[2] = jPlusView( cell );
		if (isGeometricList[3]) nbrList[3] = jMinusView( cell );
		nbrList[4] = kPlusView( cell );
		if (isGeometricList[5]) nbrList[5] = kMinusView( cell );
		
		float rhoAvg = 0.f;
		float uxAvg = 0.f;
		float uyAvg = 0.f;
		float uzAvg = 0.f;
		float fneqAvg[27] = {0.f};
		int rhoAvgCounter = 0;
		int uAvgCounter = 0;
		int fneqAvgCounter = 0;
		
		for ( int which = 0; which < 6; which++ )
		{
			const int nbr = nbrList[which];
			if ( !isGeometricList[which] ) continue;
			
			const int iCell = iView( nbr );
			const int jCell = jView( nbr );
			const int kCell = kView( nbr );
			
			MarkerStruct Marker;
			if ( useBouncebackMarkerArray ) Marker.bounceback = bouncebackMarkerView( nbr );	
			if ( useMovingBouncebackMarkerArray ) Marker.movingBounceback = movingBouncebackMarkerView( nbr );
			getMarkers( iCell, jCell, kCell, Marker, Info );
			
			if ( Marker.bounceback )
			{
				uAvgCounter++; // we are adding 0 to uAvg
				continue;
			}	
			
			if ( Marker.movingBounceback || oldMovingBouncebackMarkerView( nbr ) )
			{
				Marker.movingBounceback = true;
				BCRhoUGStruct BCRhoUG;
				getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
				uxAvg += BCRhoUG.ux;
				uyAvg += BCRhoUG.uy;
				uzAvg += BCRhoUG.uz;
				uAvgCounter++;
				continue;
			}		
			
			NBRStruct NBR;
			NBR.self = nbr;
			NBR.jPlus = jPlusView( nbr );
			NBR.kPlus = kPlusView( nbr );
			NBR.jkPlus = jPlusView( kPlusView( nbr ) );
			finishNBRPlus( NBR, Info );
			
			float f[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
			
			float rho, ux, uy, uz;
			getRhoUxUyUz( rho, ux, uy, uz, f );
			uxAvg += ux;
			uyAvg += uy;
			uzAvg += uz;
			uAvgCounter++;
			rhoAvg += rho;
			rhoAvgCounter++;
			
			float feq[27];
			getFeq(rho, ux, uy, uz, feq);
			float fneq[27];
			getFneq(f, feq, fneq);
			for (int direction = 0; direction < 27; direction++) fneqAvg[direction] += fneq[direction];	
			fneqAvgCounter++;
		}
		
		if ( uAvgCounter > 0 )
		{
			uxAvg = uxAvg / uAvgCounter;
			uyAvg = uyAvg / uAvgCounter;
			uzAvg = uzAvg / uAvgCounter;
		}
		if ( rhoAvgCounter > 0 ) rhoAvg = rhoAvg / rhoAvgCounter;
		else rhoAvg = 1.f;
		if ( fneqAvgCounter > 0 )
		{
			for (int direction = 0; direction < 27; direction++) fneqAvg[direction] = fneqAvg[direction] / fneqAvgCounter;
		}
		
		float feq[27];
		getFeq(rhoAvg, uxAvg, uyAvg, uzAvg, feq);
		float f[27];
		for (int direction = 0; direction < 27; direction++) f[direction] = feq[direction] + fneqAvg[direction];	
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = nbrList[2];
		NBR.kPlus = nbrList[4];
		NBR.jkPlus = jPlusView( nbrList[4] );
		finishNBRPlus( NBR, Info );
		
		NBR.jMinus = jMinusView( cell );
		NBR.kMinus = kMinusView( cell );
		NBR.iMinus = cell - 1; if ( NBR.iMinus < 0 ) NBR.iMinus = Info.cellCount-1;
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = feq[direction];
		
		// We are not ending yet. We have overwritten the newly uncovered fluid, but the uncovered fluid also has moving bounceback neighbours,
		// whose adjacent side is now newly uncovered. Next iteration we will pull (receive) garbage data from this. Therefore we must also
		// overwrite not only the outcoming PDFs but also those incoming PDFs that are coming from moving bounceback cells. For this, we first
		// need to identify the physical directions where moving bounceback cells lie. So our next step: Find indexes of all 26 neighbours
		// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
		// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
		// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
		// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
		// the neighbour we are looking is located in direction against the PDF direction
		// we can get to the neighbour by sequentially reading neighbours of neighbours in the main directions
		/*
		int fullNBRList[27];
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
		// (note that the ones which were MBB but are no longer MBB are already sorted out by our previous loop)
		bool isMovingBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isMovingBounceback[direction] = movingBouncebackMarkerView( fullNBRList[direction] );
		}
		*/
		int cellNextIndex[27];
		int fNextIndex[27];
		bool inverseEsotwistFlipper = !esotwistFlipper;
		getPreCollisionIndex( cellNextIndex, fNextIndex, NBR, inverseEsotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			/*
			if ( isMovingBounceback[direction] ) 
			{
				// if we are going to be receiving f from a moving bounceback in this direction,
				// just set it to the same f as we currently have in our cell
				fArrayView( fNextIndex[direction], cellNextIndex[direction] ) = feq[direction];
			}
			*/
			fArrayView( fNextIndex[direction], cellNextIndex[direction] ) = feq[direction];
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
