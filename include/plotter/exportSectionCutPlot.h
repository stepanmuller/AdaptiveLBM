#include "../cellFunctions.h"
#include "../NBRFunctions.h"
#include "../esotwistStreamingFunctions.h"

enum PlaneEnum { XY, ZY, ZX };

void exportSectionCutPlotGeneral( std::vector<GridStruct> &grids, const int &cutIndex, const int &plotNumber, PlaneEnum plane )
{
	const int levelCount = grids.size();
	
	// 1. Find the finest level that fits within the memory limit
	int targetLevelCount = levelCount;
	int targetCellCountHorizontal = 0, targetCellCountVertical = 0;
	
	while ( targetLevelCount > 0 ) // Evaluate down to index 0
	{
		InfoStruct Info = grids[targetLevelCount - 1].Info;
		if ( plane == XY ) 		{ targetCellCountHorizontal = Info.cellCountX; targetCellCountVertical = Info.cellCountY; }
		else if ( plane == ZY ) { targetCellCountHorizontal = Info.cellCountZ; targetCellCountVertical = Info.cellCountY; }
		else 					{ targetCellCountHorizontal = Info.cellCountZ; targetCellCountVertical = Info.cellCountX; }
		
		// Use long long to prevent integer overflow on massive grids
		long long dataSize = (long long)targetCellCountHorizontal * targetCellCountVertical;
		
		// Break if it fits in memory OR if we are forced to use the absolute coarsest grid
		if ( dataSize < 20000000 || targetLevelCount == 1 ) break; 
		
		targetLevelCount--;
	}
	
	// How much smaller the output array is compared to the absolute finest grid
	// Using bitshift (1 << X) which is mathematically identical to std::pow(2, X) but faster/safer
	const int targetScale = 1 << (levelCount - targetLevelCount); 

	SectionCutStruct SectionCut;
	SectionCut.rhoArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uxArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uyArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uzArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.markerArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.gridIDArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	
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
	
	// 2. Loop through ALL grids (none are dropped)
	for ( int level = 0; level < levelCount; level++ )
	{
		GridStruct &Grid = grids[level];
		const InfoStruct &Info = Grid.Info;
		
		// cellScale is relative to the absolute finest grid (levelCount)
		const int cellScale = std::pow(2, (levelCount - Info.gridID - 1) );
		
		auto fArrayView  = Grid.fArray.getConstView();
		bool useBouncebackArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
		bool useMovingBouncebackArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
		bool useRefinementMarkerArray = ( Grid.deepRefinementMarkerArray.getSize() > 0 );
		auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
		auto movingBouncebackMarkerArrayView = Grid.movingBouncebackMarkerArray.getConstView();
		auto deepRefinementMarkerArrayView = Grid.deepRefinementMarkerArray.getConstView();
		
		auto iView = Grid.IJK.iArray.getConstView();
		auto jView = Grid.IJK.jArray.getConstView();
		auto kView = Grid.IJK.kArray.getConstView();
		
		const bool &esotwistFlipper = Grid.esotwistFlipper;
		
		auto jPlusView = Grid.NBR.jPlusArray.getConstView();
		auto kPlusView = Grid.NBR.kPlusArray.getConstView();
		auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();

		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			// Coordinates on the absolute finest grid
			int iCell = iView[ cell ] * cellScale; 
			int jCell = jView[ cell ] * cellScale;
			int kCell = kView[ cell ] * cellScale;
			
			NBRStruct NBR;
			NBR.self = cell;
			NBR.jPlus = jPlusView( cell );
			NBR.kPlus = kPlusView( cell );
			NBR.jkPlus = jkPlusView( cell );
			finishNBR( NBR, Info );
			
			int indexHorizontal = 0;
			int indexVertical = 0;
			
			// cutIndex remains fully accurate because it checks against the absolute finest coordinates
			if ( plane == XY ) 
			{
				if ( cutIndex < kCell || cutIndex >= kCell + cellScale ) return;
				indexHorizontal = iCell; 
				indexVertical = jCell; 
			}
			else if ( plane == ZY ) 
			{ 
				if ( cutIndex < iCell || cutIndex >= iCell + cellScale ) return; 
				indexVertical = jCell; 
				indexHorizontal = kCell; 
			}
			else // ZX plane
			{ 
				if ( cutIndex < jCell || cutIndex >= jCell + cellScale ) return; 
				indexVertical = iCell; 
				indexHorizontal = kCell; 
			}
			
			float f[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
			
			float rho, ux, uy, uz;
			getRhoUxUyUz(rho, ux, uy, uz, f);

			MarkerStruct Marker;
			if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
			if ( useMovingBouncebackArray ) Marker.movingBounceback = movingBouncebackMarkerArrayView( cell );
			if ( useRefinementMarkerArray ) Marker.deepRefinement = deepRefinementMarkerArrayView( cell );
			const float marker = Marker.bounceback + Marker.movingBounceback + Marker.deepRefinement;
			
			// 3. Mapping coordinates to the scaled-down output array
			int outYStart = indexVertical / targetScale;
			int outXStart = indexHorizontal / targetScale;
			
			// How many pixels this cell spans on the output array (minimum 1 for downsampled fine grids)
			int spanY = max(1, cellScale / targetScale);
			int spanX = max(1, cellScale / targetScale);
			
			for ( int shiftY = 0; shiftY < spanY; shiftY++ )
			{
				int y = outYStart + shiftY;
				if (y >= targetCellCountVertical) continue; // Memory safety bound
				
				for ( int shiftX = 0; shiftX < spanX; shiftX++ )
				{
					int x = outXStart + shiftX;
					if (x >= targetCellCountHorizontal) continue; // Memory safety bound
					
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
	
	SectionCutStructCPU SectionCutCPU;
	SectionCutCPU.rhoArray = SectionCut.rhoArray;
	SectionCutCPU.uxArray = SectionCut.uxArray;
	SectionCutCPU.uyArray = SectionCut.uyArray;
	SectionCutCPU.uzArray = SectionCut.uzArray;
	SectionCutCPU.markerArray = SectionCut.markerArray;
	SectionCutCPU.gridIDArray = SectionCut.gridIDArray;
	
	FILE* fp = fopen("/dev/shm/sim_data.bin", "wb");
	int header[4] = {plotNumber, (int)targetCellCountVertical, (int)targetCellCountHorizontal, 6};
	fwrite(header, sizeof(int), 4, fp);
	
	for (int indexVertical = 0; indexVertical < targetCellCountVertical; indexVertical++)
	{
		for (int indexHorizontal = 0; indexHorizontal < targetCellCountHorizontal; indexHorizontal++)
		{
			float rho = SectionCutCPU.rhoArray.getElement(indexVertical, indexHorizontal);
			float ux = SectionCutCPU.uxArray.getElement(indexVertical, indexHorizontal);
			float uy = SectionCutCPU.uyArray.getElement(indexVertical, indexHorizontal);
			float uz = SectionCutCPU.uzArray.getElement(indexVertical, indexHorizontal);
			float marker = SectionCutCPU.markerArray.getElement(indexVertical, indexHorizontal);
			int gridID = SectionCutCPU.gridIDArray.getElement(indexVertical, indexHorizontal);
			float p = rho;
			
			// Use the actual gridID to scale physical parameters properly
			convertToPhysicalVelocity( ux, uy, uz, grids[gridID].Info );
			convertToPhysicalPressure( p, grids[gridID].Info );
			
			float uHorizontal, uVertical, uNormal;
			if ( plane == XY ) 		{ uHorizontal = ux; uVertical = uy; uNormal = uz; }
			else if ( plane == ZY ) { uHorizontal = uz; uVertical = uy; uNormal = ux; }
			else 					{ uHorizontal = uz; uVertical = ux; uNormal = uy; }
			
			float data[6] = {p, uHorizontal, uVertical, uNormal, marker, (float)gridID};
			fwrite(data, sizeof(float), 6, fp);
		}
	}
	fclose(fp);
}

void exportSectionCutPlotXY( std::vector<GridStruct> &grids, const int &kCell, const int &plotNumber )
{
	std::cout << "Exporting DIAD XY section cut plot " << plotNumber << std::endl;
	exportSectionCutPlotGeneral( grids, kCell, plotNumber, XY );
}
void exportSectionCutPlotZY( std::vector<GridStruct> &grids, const int &iCell, const int &plotNumber )
{
	std::cout << "Exporting DIAD ZY section cut plot " << plotNumber << std::endl;
	exportSectionCutPlotGeneral( grids, iCell, plotNumber, ZY );
}
void exportSectionCutPlotZX( std::vector<GridStruct> &grids, const int &jCell, const int &plotNumber )
{
	std::cout << "Exporting DIAD ZX section cut plot " << plotNumber << std::endl;
	exportSectionCutPlotGeneral( grids, jCell, plotNumber, ZX );
}

void exportSectionCutPlotToiletPaperZ( std::vector<GridStruct> &grids, const float &r, const int &plotNumber )
{
	std::cout << "Exporting DIAD Toilet Paper Z section cut plot " << plotNumber << " at radius " << r << " mm" << std::endl;

	const int levelCount = grids.size();
	InfoStruct FinestInfo = grids[levelCount - 1].Info;
	const float PI = 3.14159265359f;
	
	// 1. Find the finest level that fits within the memory limit
	int targetLevelCount = levelCount;
	int targetCellCountHorizontal = 0, targetCellCountVertical = 0;
	
	while ( targetLevelCount > 1 )
	{
		InfoStruct Info = grids[targetLevelCount - 1].Info;
		targetCellCountHorizontal = Info.cellCountZ;
		targetCellCountVertical = (int)( (2.0f * PI * r) / Info.res );
		
		// Use long long to prevent integer overflow on massive grids
		long long dataSize = (long long)targetCellCountHorizontal * targetCellCountVertical;
		if ( dataSize < 20000000 ) break;
		
		targetLevelCount--;
	}
	
	// Update target dimensions based on the found level
	InfoStruct TargetInfo = grids[targetLevelCount - 1].Info;
	targetCellCountHorizontal = TargetInfo.cellCountZ;
	targetCellCountVertical = (int)( (2.0f * PI * r) / TargetInfo.res );
	
	// How much smaller the output array is compared to the absolute finest grid
	const int targetScale = 1 << (levelCount - targetLevelCount); 

	SectionCutStruct SectionCut;
	SectionCut.rhoArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uxArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uyArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uzArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.markerArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.gridIDArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	
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
	
	// Find origin indices to compute physical x,y coordinates relative to (0,0) without needing explicit bounds
	int iOriginFinest, jOriginFinest, kOriginFinest;
	getIJKCellIndexFromXYZ( iOriginFinest, jOriginFinest, kOriginFinest, 0.f, 0.f, 0.f, FinestInfo );

	// 2. Loop through ALL grids (coarse to fine, allowing fine to overwrite coarse)
	for ( int level = 0; level < levelCount; level++ )
	{
		GridStruct &Grid = grids[level];
		InfoStruct Info = Grid.Info;
		
		// cellScale is relative to the absolute finest grid
		const int cellScale = std::pow(2, (levelCount - Info.gridID - 1) );
		
		auto fArrayView  = Grid.fArray.getConstView();
		bool useBouncebackArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
		bool useMovingBouncebackArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
		bool useRefinementMarkerArray = ( Grid.deepRefinementMarkerArray.getSize() > 0 );
		auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
		auto movingBouncebackMarkerArrayView = Grid.movingBouncebackMarkerArray.getConstView();
		auto deepRefinementMarkerArrayView = Grid.deepRefinementMarkerArray.getConstView();
		
		auto iView = Grid.IJK.iArray.getConstView();
		auto jView = Grid.IJK.jArray.getConstView();
		auto kView = Grid.IJK.kArray.getConstView();
		
		const bool &esotwistFlipper = Grid.esotwistFlipper;
		
		auto jPlusView = Grid.NBR.jPlusArray.getConstView();
		auto kPlusView = Grid.NBR.kPlusArray.getConstView();
		auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();

		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			// Coordinates on the absolute finest grid
			int iCell = iView[ cell ] * cellScale; 
			int jCell = jView[ cell ] * cellScale;
			int kCell = kView[ cell ] * cellScale;
			
			NBRStruct NBR;
			NBR.self = cell;
			NBR.jPlus = jPlusView( cell );
			NBR.kPlus = kPlusView( cell );
			NBR.jkPlus = jkPlusView( cell );
			finishNBR( NBR, Info );
			
			// Compute physical center of this cell (x, y) relative to the origin
			float xc = ((float)iCell + 0.5f * cellScale - (float)iOriginFinest) * FinestInfo.res;
			float yc = ((float)jCell + 0.5f * cellScale - (float)jOriginFinest) * FinestInfo.res;
			
			// Distance from origin (cylinder axis)
			float dist = sqrtf(xc*xc + yc*yc);
			float halfWidth = cellScale * FinestInfo.res * 0.5f;

			// Quick Culling: If cell doesn't touch the cylindrical shell, skip it.
			if ( dist < r - halfWidth * 1.5f || dist > r + halfWidth * 1.5f ) return;

			// Calculate polar angle and map to Circumference (S)
			float theta = atan2f(yc, xc);
			if (theta < 0.f) theta += 2.0f * PI;
			
			float s = theta * r;
			int finestVerticalIndex = (int)(s / FinestInfo.res);
			
			float f[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
			
			float rho, ux, uy, uz;
			getRhoUxUyUz(rho, ux, uy, uz, f);

			MarkerStruct Marker;
			if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
			if ( useMovingBouncebackArray ) Marker.movingBounceback = movingBouncebackMarkerArrayView( cell );
			if ( useRefinementMarkerArray ) Marker.deepRefinement = deepRefinementMarkerArrayView( cell );
			const float marker = Marker.bounceback + Marker.movingBounceback + Marker.deepRefinement;
			
			// 3. Mapping coordinates to the scaled-down 2D unrolled array
			int spanY = max(1, cellScale / targetScale);
			int spanX = max(1, cellScale / targetScale);
			
			// Centering the brush stroke for the cell's unrolled footprint
			int outYStart = (finestVerticalIndex - (spanY * targetScale) / 2) / targetScale;
			int outXStart = kCell / targetScale;
			
			for ( int shiftY = 0; shiftY < spanY; shiftY++ )
			{
				int y = outYStart + shiftY;
				
				// Handle wrap-around on the cylinder circumference to hide the seam
				while (y < 0) y += targetCellCountVertical;
				while (y >= targetCellCountVertical) y -= targetCellCountVertical;
				
				for ( int shiftX = 0; shiftX < spanX; shiftX++ )
				{
					int x = outXStart + shiftX;
					if (x >= targetCellCountHorizontal) continue; // Memory safety bound on Z-axis
					
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
	
	SectionCutStructCPU SectionCutCPU;
	SectionCutCPU.rhoArray = SectionCut.rhoArray;
	SectionCutCPU.uxArray = SectionCut.uxArray;
	SectionCutCPU.uyArray = SectionCut.uyArray;
	SectionCutCPU.uzArray = SectionCut.uzArray;
	SectionCutCPU.markerArray = SectionCut.markerArray;
	SectionCutCPU.gridIDArray = SectionCut.gridIDArray;
	
	FILE* fp = fopen("/dev/shm/sim_data.bin", "wb");
	
	// We export 6 fields: PlotNum, cellCountY, cellCountX, NumFields (6)
	int header[4] = {plotNumber, targetCellCountVertical, targetCellCountHorizontal, 6};
	fwrite(header, sizeof(int), 4, fp);
	
	for (int indexVertical = 0; indexVertical < targetCellCountVertical; indexVertical++)
	{
		for (int indexHorizontal = 0; indexHorizontal < targetCellCountHorizontal; indexHorizontal++)
		{
			float rho = SectionCutCPU.rhoArray.getElement(indexVertical, indexHorizontal);
			float ux = SectionCutCPU.uxArray.getElement(indexVertical, indexHorizontal);
			float uy = SectionCutCPU.uyArray.getElement(indexVertical, indexHorizontal);
			float uz = SectionCutCPU.uzArray.getElement(indexVertical, indexHorizontal);
			float marker = SectionCutCPU.markerArray.getElement(indexVertical, indexHorizontal);
			int gridID = SectionCutCPU.gridIDArray.getElement(indexVertical, indexHorizontal);
			
			float p = rho;
			
			// Use the actual gridID to scale physical parameters properly
			convertToPhysicalVelocity( ux, uy, uz, grids[gridID].Info );
			convertToPhysicalPressure( p, grids[gridID].Info );
			
			// Velocity Mapping
			// Recover theta to transform Cartesian ux/uy into uTangential/uRadial
			float theta = ((float)indexVertical * targetScale * FinestInfo.res) / r;
			float uTangential = -ux * sinf(theta) + uy * cosf(theta);
			float uRadial = ux * cosf(theta) + uy * sinf(theta);
			
			// Outputs 6 components matching the typical DIAD generic plot shape
			float data[6] = {p, uz, uTangential, uRadial, marker, (float)gridID};
			fwrite(data, sizeof(float), 6, fp);
		}
	}
	fclose(fp);
}

/*
void exportHistoryData( const std::vector<float>& historyVector, const int &currentIteration ) {
    FILE* fp = fopen("/dev/shm/historyData.bin", "wb");
    if (!fp) return;
    int count = currentIteration + 1;
    fwrite(&count, sizeof(int), 1, fp);
    fwrite(historyVector.data(), sizeof(float), count, fp);
    fclose(fp);
}
*/
