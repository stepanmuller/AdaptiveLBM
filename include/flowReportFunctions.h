#pragma once

#include "./boundaryConditions/applyMovingBounceback.h"

//  Version that dynamically downsamples finest grids and clips to specified bounds
void getFlowReportGeneral( std::vector<GridStruct> &grids, const int &cutIndex, BoundsStruct &Bounds, FlowReportStruct &FlowReport, PlaneEnum plane )
{
	const int levelCount = grids.size();
	InfoStruct InfoCoarse = grids[0].Info;
	
	// 1. Map the physical Bounds to index bounds on the absolute finest grid FIRST
	InfoStruct finestInfo = grids[levelCount - 1].Info;
	
	int iMin = std::max(0, static_cast<int>(std::floor((Bounds.xMin - finestInfo.ox) / finestInfo.res)));
	int iMax = std::min(finestInfo.cellCountX, static_cast<int>(std::ceil((Bounds.xMax - finestInfo.ox) / finestInfo.res)));
	int jMin = std::max(0, static_cast<int>(std::floor((Bounds.yMin - finestInfo.oy) / finestInfo.res)));
	int jMax = std::min(finestInfo.cellCountY, static_cast<int>(std::ceil((Bounds.yMax - finestInfo.oy) / finestInfo.res)));
	int kMin = std::max(0, static_cast<int>(std::floor((Bounds.zMin - finestInfo.oz) / finestInfo.res)));
	int kMax = std::min(finestInfo.cellCountZ, static_cast<int>(std::ceil((Bounds.zMax - finestInfo.oz) / finestInfo.res)));

	int hMinFinest = 0, hMaxFinest = 0, vMinFinest = 0, vMaxFinest = 0;

	if ( plane == XY ) 
	{
		hMinFinest = iMin; hMaxFinest = iMax;
		vMinFinest = jMin; vMaxFinest = jMax;
	} 
	else if ( plane == ZY ) 
	{
		hMinFinest = kMin; hMaxFinest = kMax;
		vMinFinest = jMin; vMaxFinest = jMax;
	} 
	else // ZX plane
	{
		hMinFinest = kMin; hMaxFinest = kMax;
		vMinFinest = iMin; vMaxFinest = iMax;
	}

	// 2. Find the finest level that fits the CROPPED bounds within the memory limit
	int targetLevelCount = levelCount;
	int targetScale = 1;
	int targetWidth = 0, targetHeight = 0;
	int hMinTarget = 0, hMaxTarget = 0, vMinTarget = 0, vMaxTarget = 0;
	
	while ( targetLevelCount > 0 ) // Evaluate down to 0 to catch the levelCount == 1 case
	{
		targetScale = 1 << (levelCount - targetLevelCount);
		
		hMinTarget = hMinFinest / targetScale;
		hMaxTarget = (hMaxFinest + targetScale - 1) / targetScale; // Ceil division
		vMinTarget = vMinFinest / targetScale;
		vMaxTarget = (vMaxFinest + targetScale - 1) / targetScale; // Ceil division

		targetWidth = std::max(0, hMaxTarget - hMinTarget);
		targetHeight = std::max(0, vMaxTarget - vMinTarget);
		
		// Use the cropped array size, not the global grid size
		long long dataSize = (long long)targetWidth * targetHeight;
		
		// Break if it fits in memory OR if we are forced to use the absolute coarsest grid
		if ( dataSize < 20000000 || targetLevelCount == 1 ) break;
		
		targetLevelCount--;
	}
	
	InfoStruct targetInfo = grids[targetLevelCount - 1].Info;
	
	SectionCutStruct SectionCut;
	SectionCut.rhoArray.setSizes( targetHeight, targetWidth );
	SectionCut.uxArray.setSizes( targetHeight, targetWidth );
	SectionCut.uyArray.setSizes( targetHeight, targetWidth );
	SectionCut.uzArray.setSizes( targetHeight, targetWidth );
	SectionCut.markerArray.setSizes( targetHeight, targetWidth );
	SectionCut.gridIDArray.setSizes( targetHeight, targetWidth );
	
	SectionCut.rhoArray.setValue( 1.f );
	SectionCut.uxArray.setValue( 0.f );
	SectionCut.uyArray.setValue( 0.f );
	SectionCut.uzArray.setValue( 0.f );
	SectionCut.markerArray.setValue( 1 );
	SectionCut.gridIDArray.setValue( 0 );
		
	auto rhoArrayView = SectionCut.rhoArray.getView();
	auto uxArrayView = SectionCut.uxArray.getView();
	auto uyArrayView = SectionCut.uyArray.getView();
	auto uzArrayView = SectionCut.uzArray.getView();
	auto markerArrayView = SectionCut.markerArray.getView();
	auto gridIDArrayView = SectionCut.gridIDArray.getView();
	
	// 3. Loop through ALL grids
	for ( int level = 0; level < levelCount; level++ )
	{
		GridStruct &Grid = grids[level];
		InfoStruct Info = Grid.Info;
		
		const int cellScale = static_cast<int>(pow(2, levelCount - Info.gridID - 1));
		
		auto fArrayView  = Grid.fArray.getConstView();
		bool useBouncebackArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
		bool useMovingBouncebackArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
		bool useRefinementMarkerArray = ( Grid.deepRefinementMarkerArray.getSize() > 0 );
		bool useFineToCoarseMarkerArray = ( Grid.fineToCoarseMarkerArray.getSize() > 0 );
		auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
		auto movingBouncebackMarkerArrayView = Grid.movingBouncebackMarkerArray.getConstView();
		auto deepRefinementMarkerArrayView = Grid.deepRefinementMarkerArray.getConstView();
		auto fineToCoarseMarkerArrayView = Grid.fineToCoarseMarkerArray.getConstView();
		
		auto iView = Grid.IJK.iArray.getConstView();
		auto jView = Grid.IJK.jArray.getConstView();
		auto kView = Grid.IJK.kArray.getConstView();
		
		const bool &esotwistFlipper = Grid.esotwistFlipper;
		
		auto jPlusView = Grid.NBR.jPlusArray.getConstView();
		auto kPlusView = Grid.NBR.kPlusArray.getConstView();

		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			int iCell = iView[ cell ]; 
			int jCell = jView[ cell ];
			int kCell = kView[ cell ];
			int iCellScaled = iCell * cellScale; 
			int jCellScaled = jCell * cellScale;
			int kCellScaled = kCell * cellScale;
			
			int indexHorizontal = 0;
			int indexVertical = 0;
			
			if ( plane == XY ) 
			{
				if ( cutIndex < kCellScaled || cutIndex >= kCellScaled + cellScale ) return;
				indexHorizontal = iCellScaled; 
				indexVertical = jCellScaled; 
			}
			else if ( plane == ZY ) 
			{ 
				if ( cutIndex < iCellScaled || cutIndex >= iCellScaled + cellScale ) return; 
				indexVertical = jCellScaled; 
				indexHorizontal = kCellScaled; 
			}
			else // ZX plane
			{ 
				if ( cutIndex < jCellScaled || cutIndex >= jCellScaled + cellScale ) return; 
				indexVertical = iCellScaled; 
				indexHorizontal = kCellScaled; 
			}
			
			NBRStruct NBR;
			NBR.self = cell;
			NBR.jPlus = jPlusView( cell );
			NBR.kPlus = kPlusView( cell );
			NBR.jkPlus = jPlusView( NBR.kPlus );
			finishNBRPlus( NBR, Info );
			
			float f[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
			
			float rho, ux, uy, uz;
			getRhoUxUyUz(rho, ux, uy, uz, f);

			MarkerStruct Marker;
			if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
			if ( useMovingBouncebackArray ) Marker.movingBounceback = movingBouncebackMarkerArrayView( cell );
			if ( useRefinementMarkerArray ) Marker.deepRefinement = deepRefinementMarkerArrayView( cell );
			if ( useFineToCoarseMarkerArray ) Marker.fineToCoarse = fineToCoarseMarkerArrayView( cell );
			
			if ( Marker.deepRefinement || Marker.fineToCoarse ) return; // there will be fine grid on top so we dont write this
			
			const float marker = Marker.bounceback + Marker.movingBounceback + Marker.deepRefinement;
			
			int outYStart = (indexVertical / targetScale) - vMinTarget;
			int outXStart = (indexHorizontal / targetScale) - hMinTarget;
			
			int spanY = max(1, cellScale / targetScale);
			int spanX = max(1, cellScale / targetScale);
			
			for ( int shiftY = 0; shiftY < spanY; shiftY++ )
			{
				int y = outYStart + shiftY;
				if ( y < 0 || y >= targetHeight ) continue;
				
				for ( int shiftX = 0; shiftX < spanX; shiftX++ )
				{
					int x = outXStart + shiftX;
					if ( x < 0 || x >= targetWidth ) continue;
					
					rhoArrayView( y, x ) = rho;
					uxArrayView( y, x ) = ux;
					uyArrayView( y, x ) = uy;
					uzArrayView( y, x ) = uz;
					markerArrayView( y, x ) = marker;
					gridIDArrayView( y, x ) = Info.gridID;
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	}
	
	const int totalCutCells = targetWidth * targetHeight;

	auto fetchCellCount = [=] __cuda_callable__( const int singleIndex )
	{
		const int y = singleIndex / targetWidth;
		const int x = singleIndex % targetWidth;
		if ( markerArrayView( y, x ) != 0.0f ) return 0; // Explicit check
		else return 1;
	};
	auto reductionCellCount = [] __cuda_callable__( const int& a, const int& b ) { return a + b; };
	const int cellSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetchCellCount, reductionCellCount, 0 );
	if ( cellSum == 0 ) return;

	auto fetch = [=] __cuda_callable__ ( const int singleIndex ) -> FlowReportStruct
	{
		const int y = singleIndex / targetWidth;
		const int x = singleIndex % targetWidth;
		
		float uxPhys = 0.f; float uyPhys = 0.f; float uzPhys = 0.f; float pPhys = 0.f; 
		float massFlowZPhys = 0.f; float momentumFlowZPhys = 0.f; float kineticEnergyFlowZPhys = 0.f;
		
		if ( markerArrayView( y, x ) != 0.0f )
		{
			// we will return the default zeros, skip the else block below
		}
		else 
		{
			uxPhys = uxArrayView( y, x ); uyPhys = uyArrayView( y, x ); uzPhys = uzArrayView( y, x );
			convertToPhysicalVelocity( uxPhys, uyPhys, uzPhys, targetInfo );
			const float areamm2 = targetInfo.res * targetInfo.res;
			const float rho = rhoArrayView( y, x );
			const float rhoPhys = rho * rhoNominalPhys;
			massFlowZPhys = rhoPhys * uzPhys * ( areamm2 / 1000000.f );
			momentumFlowZPhys = massFlowZPhys * uzPhys;
			kineticEnergyFlowZPhys = 0.5f * massFlowZPhys * uzPhys * uzPhys;
			pPhys = rho;
			convertToPhysicalPressure( pPhys );
		}
		return 
		{ 
			uxPhys, uyPhys, uzPhys, pPhys,
			0.f, // we are not calculating areamm2 from this
			massFlowZPhys, momentumFlowZPhys, kineticEnergyFlowZPhys
		}; 
		
	};
	auto reduction = [] __cuda_callable__( const FlowReportStruct& a, const FlowReportStruct& b ) -> FlowReportStruct
	{
		return 
		{ 
			a.uxPhys + b.uxPhys, a.uyPhys + b.uyPhys, a.uzPhys + b.uzPhys, a.pPhys + b.pPhys,
			0.f, // we are not calculating areamm2 from this
			a.massFlowZPhys + b.massFlowZPhys, a.momentumFlowZPhys + b.momentumFlowZPhys, a.kineticEnergyFlowZPhys + b.kineticEnergyFlowZPhys
		}; 
	};
	
	FlowReportStruct zeros;
	
	FlowReport = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetch, reduction, zeros );

	FlowReport.areamm2 = cellSum * ( grids[targetLevelCount-1].Info.res * grids[targetLevelCount-1].Info.res );

	FlowReport.uxPhys /= (float)cellSum;
	FlowReport.uyPhys /= (float)cellSum;
	FlowReport.uzPhys /= (float)cellSum;
	FlowReport.pPhys /= (float)cellSum;
}

void getFlowReportXY( std::vector<GridStruct> &grids, const int &kCell, BoundsStruct &Bounds, FlowReportStruct &FlowReport )
{
	getFlowReportGeneral( grids, kCell, Bounds, FlowReport, XY );
}
void getFlowReportZY( std::vector<GridStruct> &grids, const int &iCell, BoundsStruct &Bounds, FlowReportStruct &FlowReport )
{
	getFlowReportGeneral( grids, iCell, Bounds, FlowReport, ZY );
}
void getFlowReportZX( std::vector<GridStruct> &grids, const int &jCell, BoundsStruct &Bounds, FlowReportStruct &FlowReport )
{
	getFlowReportGeneral( grids, jCell, Bounds, FlowReport, ZX );
}

