// DIAD Version that dynamically downsamples finest grids and clips to specified bounds
void getFlowReportGeneral( std::vector<DIADGridStruct> &grids, const int &cutIndex, XYZBoundsStruct &Bounds, FlowReportStruct &FlowReport, PlaneEnum plane )
{
	const int levelCount = grids.size();
	
	// 1. Map the physical Bounds to index bounds on the absolute finest grid FIRST
	InfoStruct finestInfo = grids[levelCount - 1].Info;
	
	int iMin = std::max(0, static_cast<int>(std::floor((Bounds.xmin - finestInfo.ox) / finestInfo.res)));
	int iMax = std::min(finestInfo.cellCountX, static_cast<int>(std::ceil((Bounds.xmax - finestInfo.ox) / finestInfo.res)));
	int jMin = std::max(0, static_cast<int>(std::floor((Bounds.ymin - finestInfo.oy) / finestInfo.res)));
	int jMax = std::min(finestInfo.cellCountY, static_cast<int>(std::ceil((Bounds.ymax - finestInfo.oy) / finestInfo.res)));
	int kMin = std::max(0, static_cast<int>(std::floor((Bounds.zmin - finestInfo.oz) / finestInfo.res)));
	int kMax = std::min(finestInfo.cellCountZ, static_cast<int>(std::ceil((Bounds.zmax - finestInfo.oz) / finestInfo.res)));

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
		DIADGridStruct &Grid = grids[level];
		InfoStruct Info = Grid.Info;
		
		const int cellScale = static_cast<int>(pow(2, levelCount - Info.gridID - 1));
		
		auto fArrayView  = Grid.fArray.getConstView();
		bool useBouncebackArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
		auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
		bool useForcedVelocityArray = ( Grid.forcedVelocityMarkerArray.getSize() > 0 );
		auto forcedVelocityMarkerArrayView = Grid.forcedVelocityMarkerArray.getConstView();
		
		auto iView = Grid.IJK.iArray.getConstView();
		auto jView = Grid.IJK.jArray.getConstView();
		auto kView = Grid.IJK.kArray.getConstView();
		
		bool esotwistFlipper = Grid.esotwistFlipper;
		auto iNBRView = Grid.EsotwistNBRArray.iNBRArray.getConstView();
		auto jNBRView = Grid.EsotwistNBRArray.jNBRArray.getConstView();
		auto kNBRView = Grid.EsotwistNBRArray.kNBRArray.getConstView();
		auto ijNBRView = Grid.EsotwistNBRArray.ijNBRArray.getConstView();
		auto ikNBRView = Grid.EsotwistNBRArray.ikNBRArray.getConstView();
		auto jkNBRView = Grid.EsotwistNBRArray.jkNBRArray.getConstView();
		auto ijkNBRView = Grid.EsotwistNBRArray.ijkNBRArray.getConstView();

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
			
			DIADEsotwistNBRStruct NBR;
			NBR.i = iNBRView( cell );
			NBR.j = jNBRView( cell );
			NBR.k = kNBRView( cell );
			NBR.ij = ijNBRView( cell );
			NBR.ik = ikNBRView( cell );
			NBR.jk = jkNBRView( cell );
			NBR.ijk = ijkNBRView( cell );
			
			float f[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( cell, cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info ); 
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
			
			float rho, ux, uy, uz;
			getRhoUxUyUz(rho, ux, uy, uz, f);

			MarkerStruct Marker;
			if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
			if ( useForcedVelocityArray ) Marker.forcedVelocity = forcedVelocityMarkerArrayView( cell );
			getMarkers( iCell, jCell, kCell, Marker, Info );
			const float marker = (Marker.bounceback || Marker.forcedVelocity);
			
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
	
	auto fetchUx = [=] __cuda_callable__( const int singleIndex )
	{
		const int y = singleIndex / targetWidth;
		const int x = singleIndex % targetWidth;
		if ( markerArrayView( y, x ) != 0.0f ) return 0.f;
		else return uxArrayView( y, x );
	};
	auto fetchUy = [=] __cuda_callable__( const int singleIndex )
	{
		const int y = singleIndex / targetWidth;
		const int x = singleIndex % targetWidth;
		if ( markerArrayView( y, x ) != 0.0f ) return 0.f;
		else return uyArrayView( y, x );
	};
	auto fetchUz = [=] __cuda_callable__( const int singleIndex )
	{
		const int y = singleIndex / targetWidth;
		const int x = singleIndex % targetWidth;
		if ( markerArrayView( y, x ) != 0.0f ) return 0.f;
		else return uzArrayView( y, x );
	};
	auto fetchRho = [=] __cuda_callable__( const int singleIndex )
	{
		const int y = singleIndex / targetWidth;
		const int x = singleIndex % targetWidth;
		if ( markerArrayView( y, x ) != 0.0f ) return 0.f;
		else return (rhoArrayView( y, x ) - 1.f); // well conditioned
	};
	auto reductionFloat = [] __cuda_callable__( const float& a, const float& b ) { return a + b; };

	const int cellSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetchCellCount, reductionCellCount, 0 );
	float uxSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetchUx, reductionFloat, 0.f );
	float uySum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetchUy, reductionFloat, 0.f );
	float uzSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetchUz, reductionFloat, 0.f );
	float rhoSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, totalCutCells, fetchRho, reductionFloat, 0.f );
	
	// Ensure no division by zero if the cut plane is entirely inside a solid
	if (cellSum > 0) {
		FlowReport.areamm2 = cellSum * ( grids[targetLevelCount-1].Info.res * grids[targetLevelCount-1].Info.res );
		FlowReport.ux = uxSum / (float)cellSum;
		FlowReport.uy = uySum / (float)cellSum;
		FlowReport.uz = uzSum / (float)cellSum;
		FlowReport.rho = (rhoSum / (float)cellSum) + 1.f;
	} else {
		FlowReport.areamm2 = 0.f;
		FlowReport.ux = 0.f; FlowReport.uy = 0.f; FlowReport.uz = 0.f; FlowReport.rho = 1.f;
	}
}

void getFlowReportXY( std::vector<DIADGridStruct> &grids, const int &kCell, XYZBoundsStruct &Bounds, FlowReportStruct &FlowReport )
{
	getFlowReportGeneral( grids, kCell, Bounds, FlowReport, XY );
}
void getFlowReportZY( std::vector<DIADGridStruct> &grids, const int &iCell, XYZBoundsStruct &Bounds, FlowReportStruct &FlowReport )
{
	getFlowReportGeneral( grids, iCell, Bounds, FlowReport, ZY );
}
void getFlowReportZX( std::vector<DIADGridStruct> &grids, const int &jCell, XYZBoundsStruct &Bounds, FlowReportStruct &FlowReport )
{
	getFlowReportGeneral( grids, jCell, Bounds, FlowReport, ZX );
}
