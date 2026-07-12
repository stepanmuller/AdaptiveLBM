#pragma once

#include "../esotwistStreamingFunctions.h"
#include "../cellFunctions.h"
#include "../NBRFunctions.h"

void applyInitialCondition( GridStruct &Grid )
{
	const InfoStruct &Info = Grid.Info;
	
	auto fArrayView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();
	
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	auto deepRefinementMarkerView = Grid.deepRefinementMarkerArray.getConstView();
	
	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	bool useDeepRefinementMarkerArray = ( Grid.deepRefinementMarkerArray.getSize() > 0 );
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jkPlusView( cell );
		finishNBR( NBR, Info );
		
		MarkerStruct Marker;
		if ( useBouncebackMarkerArray ) Marker.bounceback = bouncebackMarkerView( cell );
		if ( useMovingBouncebackMarkerArray ) Marker.movingBounceback = movingBouncebackMarkerView( cell );
		if ( useDeepRefinementMarkerArray ) Marker.deepRefinement = deepRefinementMarkerView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		
		BCRhoUGStruct BCRhoUG;
		getInitialRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		
		float f[27];
		getFeq( BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, f );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
}
