#pragma once

#include "./applyCollision.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"
#include "./boundaryConditions/applyBounceback.h"
#include "./boundaryConditions/applyMovingBounceback.h"
#include "./boundaryConditions/restoreRho.h"
#include "./boundaryConditions/restoreUxUyUz.h"
#include "./boundaryConditions/applyMBBC.h"

void updateGrid( GridStruct &Grid )
{
	applyStreaming( Grid );
	
	InfoStruct &Info = Grid.Info;
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fArrayView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

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
		
		MarkerStruct Marker;
		if ( useBouncebackMarkerArray ) Marker.bounceback = bouncebackMarkerView( cell );
		if ( useMovingBouncebackMarkerArray ) Marker.movingBounceback = movingBouncebackMarkerView( cell );
		if ( useDeepRefinementMarkerArray ) Marker.deepRefinement = deepRefinementMarkerView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		
		if ( Marker.deepRefinement ) return;
		
		if ( Marker.bounceback ) return; // bounceback gets implicitly applied by Esotwist
				
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jkPlusView( cell );
		finishNBR( NBR, Info );
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
		
		BCRhoUGStruct BCRhoUG;
		// load the current state into the boundary condition struct
		getRhoUxUyUz( BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, f );
		// pass the current state into the boundary condition function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, U
		getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		
		if ( Marker.movingBounceback )
		{
			applyMovingBounceback( f, BCRhoUG );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
			return;
		}
		
		if ( Marker.fluid )
		{
			// do nothing, just skip the else block below
		}
		else if ( Marker.BCNonReflectiveOutlet )
		{
			// also do nothing, just skip the else block below
		}
		else
		{
			int outerNormalX, outerNormalY, outerNormalZ;
			getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
			if ( Marker.BCRho )
			{
				restoreUxUyUz( outerNormalX, outerNormalY, outerNormalZ, BCRhoUG, f );
			}
			else if ( Marker.BCU )
			{
				restoreRho( outerNormalX, outerNormalY, outerNormalZ, BCRhoUG, f );
			}
			applyMBBC( outerNormalX, outerNormalY, outerNormalZ, BCRhoUG, f );
		}
		
		applyCollision( f, BCRhoUG, Info.nu );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	Info.updatesSinceRebuild++; 
	Info.updatesSinceMovingBouncebackUpdate++;
	Info.iterationsFinished++;
}
