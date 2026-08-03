#pragma once

#include "./boundaryConditions/applyMovingBounceback.h"

//  Version that dynamically downsamples finest grids and clips to specified bounds
void getFlowReportGeneral( std::vector<GridStruct> &grids, const int &cutIndex, BoundsStruct &Bounds, FlowReportStruct &FlowReport, PlaneEnum plane )
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

float getMovingBouncebackTorqueZ( GridStruct &Grid )
{
	if ( Grid.movingBouncebackMarkerArray.getSize() < 1) return 0.f;
	
	InfoStruct &Info = Grid.Info;
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	
	auto fetch = [ = ] __cuda_callable__( const int cell )
	{		
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		if ( iCell == 0 || iCell == Info.cellCountX-1 || jCell == 0 || jCell == Info.cellCountY-1 || kCell == 0 || kCell == Info.cellCountZ-1 ) return 0.f;
		
		if ( !movingBouncebackMarkerView( cell ) ) return 0.f;
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( NBR.kPlus );
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
		bool isNotFluid[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isNotFluid[direction] = movingBouncebackMarkerView( fullNBRList[direction] ) + bouncebackMarkerView( fullNBRList[direction] );
		}
		
		float fIn[27];
		float fOut[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			fIn[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
			fOut[direction] = fIn[direction];
		}
		
		BCRhoUGStruct BCRhoUG;
		getRhoUxUyUz( BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fIn );
		MarkerStruct Marker;
		Marker.movingBounceback = true;
		getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
		
		applyMovingBounceback( fOut, BCRhoUG );
		
		const int cx[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };

		const int cy[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };

		const int cz[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

		const int inverseDirection[27] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25 };

		float gx = 0.f;
		float gy = 0.f;
		float gz = 0.f;

		const float wallUx = BCRhoUG.ux;
		const float wallUy = BCRhoUG.uy;
		const float wallUz = BCRhoUG.uz;
		
		for (int q = 1; q < 27; q++) {
			if (isNotFluid[q]) continue; // we are only interested if the neighbour is fluid
			//const int nx = -cx[q]; const int ny = -cy[q]; const int nz = -cz[q];
			//const int iExpected = iCell + nx; const int jExpected = jCell + ny; const int kExpected = kCell + nz;
			//const int iActual = iView( fullNBRList[q] ); const int jActual = jView( fullNBRList[q] ); const int kActual = kView( fullNBRList[q] );
			//if ( iActual != iExpected || jActual != jExpected || kActual != kExpected ) 
			//{
			//	printf("hey something is wrong here");
			//	continue;
			//}

			gx += (cx[q] - wallUx) * fIn[q] - (cx[inverseDirection[q]] - wallUx) * fOut[inverseDirection[q]];

			gy += (cy[q] - wallUy) * fIn[q] - (cy[inverseDirection[q]] - wallUy) * fOut[inverseDirection[q]];

			gz += (cz[q] - wallUz) * fIn[q] - (cz[inverseDirection[q]] - wallUz) * fOut[inverseDirection[q]];
		}
		
		float x, y, z;
		getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );		
		convertToPhysicalForce( gx, gy, gz, Info );
		float T = - ( - gx * y + gy * x );
		// JUST A TEST START
		// const float r = std::sqrt( x * x + y * y );
		// if ( r > 16.0f ) return 0.f; 
		// if ( r < 11.f ) return 0.f;
		// JUST A TEST END
		return T;
	};
	auto reduction = [] __cuda_callable__( const float& a, const float& b )
	{
		return a + b;
	};
	
	float TSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Info.cellCount, fetch, reduction, 0.f );
	TSum = TSum / 1000.f; // converting from Nmm to Nm
	return TSum;
}

