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
			
			if ( Marker.movingBounceback )
			{
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
			finishNBR( NBR, Info );
			
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
		finishNBR( NBR, Info );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
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
