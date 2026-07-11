#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"
#include "./NBRFunctions.h"
#include "./markerFunctions.h"
#include "./buildFinerGrid.h"

void pullSingleFArrayIntoCells( GridStruct &Grid, const int direction, const int postCollisionLocation )
{
	auto fView  = Grid.fArray.getView();
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();
	const int cellCountOld = Grid.Info.cellCountOld;

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		int readIndex;
		if      ( postCollisionLocation == 0 ) readIndex = cell;
		else if ( postCollisionLocation == 1 ) readIndex = cell + 1; 
		else if ( postCollisionLocation == 2 ) readIndex = jPlusView( cell ); 
		else if ( postCollisionLocation == 3 ) readIndex = jPlusView( cell ) + 1;
		else if ( postCollisionLocation == 4 ) readIndex = kPlusView( cell ); 
		else if ( postCollisionLocation == 5 ) readIndex = kPlusView( cell ) + 1; 
		else if ( postCollisionLocation == 6 ) readIndex = jkPlusView( cell ); 
		else    							   readIndex = jkPlusView( cell ) + 1;		
		if ( readIndex >= cellCountOld ) readIndex = 0;
		const float f = fView( direction, readIndex );
		fView( direction + 1, cell ) = f;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountOld, cellLambda );
}

void pullFArrayIntoCells( GridStruct &Grid )
{
	// Because of esotwist streaming, some distribution function of cell i is not always saved in cell i, but in a neighbour cell 
	// To know the neighbour cell we need Grid.NBR
	// This function moves all distribution functions into their own cells so that we can safely overwrite Grid.NBR after this step
	if ( Grid.esotwistFlipper )
	{
		throw std::runtime_error("pullFArrayIntoCells failed, bool esotwistFlipper is 1. This function can only be called when esotwistFlipper is 0.");
	}
	// 						  List of post collision memory locations
	// 						  0 = self
	// 						  1 = iPlus 
	// 						  2 = jPlus
	// 						  3 = ijPlus
	// 						  4 = kPlus
	// 						  5 = ikPlus
	// 						  6 = jkPlus
	// 						  7 = ijkPlus
	//						  direction   { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}
	const int postCollisionLocation[27] = { 0, 1, 0, 0, 4, 0, 2, 1, 4, 5, 0, 0, 3, 2, 4, 2, 1, 6, 0, 2, 5, 4, 3, 1, 6, 0, 7 };	
	for ( int direction = 26; direction >= 0; direction-- ) pullSingleFArrayIntoCells( Grid, direction, postCollisionLocation[ direction ] );
}

void oldToKeepSingleTransform( GridStruct &Grid, const int direction, const int postCollisionLocation )
{
	auto oldToKeepView = Grid.intBuffer3.getConstView();
	auto fView  = Grid.fArray.getView();
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();
	const int cellCountOld = Grid.Info.cellCountOld;
	const int cellCount = Grid.Info.cellCount;

	auto cellLambda = [=] __cuda_callable__ ( const int cellOld ) mutable
	{	
		const int cellNew = oldToKeepView[ cellOld ];
		if ( cellNew < 0 ) return;
		const float f = fView( direction + 1, cellOld );
		int writeIndex;
		if      ( postCollisionLocation == 0 ) writeIndex = cellNew;
		else if ( postCollisionLocation == 1 ) writeIndex = cellNew + 1; 
		else if ( postCollisionLocation == 2 ) writeIndex = jPlusView( cellNew ); 
		else if ( postCollisionLocation == 3 ) writeIndex = jPlusView( cellNew ) + 1;
		else if ( postCollisionLocation == 4 ) writeIndex = kPlusView( cellNew ); 
		else if ( postCollisionLocation == 5 ) writeIndex = kPlusView( cellNew ) + 1; 
		else if ( postCollisionLocation == 6 ) writeIndex = jkPlusView( cellNew ); 
		else    							   writeIndex = jkPlusView( cellNew ) + 1;		
		if ( writeIndex >= cellCount ) writeIndex = 0;
		fView( direction, writeIndex ) = f;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountOld, cellLambda );
}

void oldToKeepTransform( GridStruct &Grid )
{
	// old to keep transformation of the fArray (which we pulled into our cells previously)
	// At the same time, move all distribution functions from their own cells into cells given by esotwist streaming
	if ( Grid.esotwistFlipper )
	{
		throw std::runtime_error("pullFArrayIntoCells failed, bool esotwistFlipper is 1. This function can only be called when esotwistFlipper is 0.");
	}
	// 						  List of post collision memory locations
	// 						  0 = self
	// 						  1 = iPlus 
	// 						  2 = jPlus
	// 						  3 = ijPlus
	// 						  4 = kPlus
	// 						  5 = ikPlus
	// 						  6 = jkPlus
	// 						  7 = ijkPlus
	//						  direction   { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}
	const int postCollisionLocation[27] = { 0, 1, 0, 0, 4, 0, 2, 1, 4, 5, 0, 0, 3, 2, 4, 2, 1, 6, 0, 2, 5, 4, 3, 1, 6, 0, 7 };	
	for ( int direction = 0; direction < 27; direction++ ) oldToKeepSingleTransform( Grid, direction, postCollisionLocation[ direction ] );
}

void fullToKeepTransform( IntArrayType &dataArray, const BoolArrayType &keepCellMarkerArray, const IntArrayType &fullToKeepArray, IntArrayType &intBuffer, const int &upperBound )
{
	auto dataView = dataArray.getView();
	auto keepCellMarkerView = keepCellMarkerArray.getConstView();
	auto fullToKeepView = fullToKeepArray.getConstView();
	auto intBufferView = intBuffer.getView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( keepCellMarkerView[ cell ] )
		{
			const int writeIndex = fullToKeepView[ cell ];
			intBufferView[ writeIndex ] = dataView[ cell ];
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );
	dataArray.swap( intBuffer );
}

void fullToKeepTransformWithIndexRepair( IntArrayType &dataArray, const BoolArrayType &keepCellMarkerArray, const IntArrayType &fullToKeepArray, IntArrayType &intBuffer, const int &upperBound )
{
	auto dataView = dataArray.getView();
	auto keepCellMarkerView = keepCellMarkerArray.getConstView();
	auto fullToKeepView = fullToKeepArray.getConstView();
	auto intBufferView = intBuffer.getView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( keepCellMarkerView[ cell ] )
		{
			const int writeIndex = fullToKeepView[ cell ];
			intBufferView[ writeIndex ] = fullToKeepView[ dataView[ cell ] ];
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );
	dataArray.swap( intBuffer );
}

void rebuildGrids( std::vector<GridStruct> &grids, const VoxelizerStruct &Voxelizer, const int level )
// Consider grids 0, 1, 2, 3 where 3 is the finest. We want to rebuild grids 2, 3 -> we call this function on 2 (level=2) which recursively calls it on all levels below.
{
	TNL::Timer Timer;
	
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];
	InfoStruct &Info = Grid.Info;
	Info.cellCountOld = Info.cellCount;
	
	const bool initPass = ( Grid.fArray.getSizes()[0] < 1 );
	
	SkeletonGridStruct &SkeletonGrid = Grid.SkeletonGrid;
	InfoStruct &SkeletonInfo = SkeletonGrid.Info;
	
	static GridStruct dummyGrid; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridStruct &GridCoarse = iAmCoarsest ? dummyGrid : grids[ level - 1 ];
	InfoStruct &InfoCoarse = GridCoarse.Info;
	
	Timer.reset();
	Timer.start();
	
	// 1) Pull fArray into the correct cells to be able to forget NBR
	if ( !initPass ) pullFArrayIntoCells( Grid );
	
	Timer.stop();
	std::cout << "Step 1 pullFArrayIntoCells on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 2) Mark refinement area
	if ( iAmCoarsest )
	{
		markKeepCells( SkeletonGrid, Voxelizer );
		SkeletonInfo.refinementCount = countOnesInBoolArray( SkeletonGrid.keepCellMarkerArray, SkeletonInfo.cellCount );
		Info.cellCountFull = 8 * countOnesInBoolArray( SkeletonGrid.keepCellMarkerArray, SkeletonInfo.cellCount );
	}
	else
	{
		markRefinementCells( GridCoarse, Voxelizer, GridCoarse.Info.cellCount );
		InfoCoarse.deepRefinementCount = countOnesInBoolArray( GridCoarse.deepRefinementMarkerArray, InfoCoarse.cellCount );
		InfoCoarse.refinementCount = countOnesInBoolArray( GridCoarse.refinementMarkerArray, InfoCoarse.cellCount );
		InfoCoarse.fineToCoarseCount = countOnesInBoolArray( GridCoarse.fineToCoarseMarkerArray, InfoCoarse.cellCount );
		InfoCoarse.coarseToFineCount = InfoCoarse.refinementCount - InfoCoarse.deepRefinementCount - InfoCoarse.fineToCoarseCount;
		Info.cellCountFull = 8 * InfoCoarse.refinementCount;
	}
	
	if ( initPass )
	{
		Info.memoryCountFull = Info.cellCountFull + ( ( Info.cellCountFull * MEMORY_RESERVE_PERCENTAGE ) / 100 );
		//std::cout 	<< "Grid level " << level << " allocated memoryCountFull = " << Info.memoryCountFull << std::endl;
		Grid.IJK.iArray.setSize( Info.memoryCountFull );
		Grid.IJK.jArray.setSize( Info.memoryCountFull );
		Grid.IJK.kArray.setSize( Info.memoryCountFull );
		Grid.NBR.jPlusArray.setSize( Info.memoryCountFull );
		Grid.NBR.kPlusArray.setSize( Info.memoryCountFull );
		Grid.NBR.jkPlusArray.setSize( Info.memoryCountFull );
		Grid.NBR.jMinusArray.setSize( Info.memoryCountFull );
		Grid.NBR.kMinusArray.setSize( Info.memoryCountFull );
		Grid.NBR.isGeometricMarkerArray.setSizes( 10, Info.memoryCountFull );
		Grid.parentMapArray.setSize( Info.memoryCountFull );
		Grid.keepCellMarkerArray.setSize( Info.memoryCountFull );
		Grid.bouncebackMarkerArray.setSize( Info.memoryCountFull );
		Grid.movingBouncebackMarkerArray.setSize( Info.memoryCountFull );
		Grid.markerBuffer.setSize( Info.memoryCountFull );
		if ( !iAmFinest )
		{
			Grid.childMapArray.setSize( Info.memoryCountFull );
			Grid.refinementMarkerArray.setSize( Info.memoryCountFull );
			Grid.deepRefinementMarkerArray.setSize( Info.memoryCountFull );
			Grid.fineToCoarseMarkerArray.setSize( Info.memoryCountFull );
			Grid.coarseToFineMarkerArray.setSize( Info.memoryCountFull );
		}
	}
	else if ( Info.cellCountFull > Info.memoryCountFull )
	{
		std::cout << "rebuildGrid failed on level " << level << ", memoryCountFull = " << Info.memoryCountFull << ", cellCountFull = " << Info.cellCountFull << std::endl;
		throw std::runtime_error("rebuildGrid failed, cellCountFull exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE in your main file.");
	}
	
	Timer.stop();
	std::cout << "Step 2 mark refinement area on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 3) Build our grid (we are the "finer grid" with respect to the grid we are taking spatial information from)
	
	if ( iAmCoarsest ) buildFinerGrid( SkeletonGrid, Grid );
	else buildFinerGrid( GridCoarse, Grid );
	IntArrayType &oldToFullArray = Grid.intBuffer1; // We cannot touch intBuffer1 now!
	
	Timer.stop();
	std::cout << "Step 3 buildFinerGrid on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 4) Get rid of the cells that are deep inside solid. Only keep the necessary ones, mark them in keepCellMarkerArray
	markKeepCells( Grid, Voxelizer, Info.cellCountFull );
	Info.cellCount = countOnesInBoolArray( Grid.keepCellMarkerArray, Info.cellCountFull );
	
	if ( initPass )
	{
		Info.memoryCount = Info.cellCount + ( ( Info.cellCount * MEMORY_RESERVE_PERCENTAGE ) / 100 );
		// std::cout 	<< "Grid level " << level << " allocated memoryCount = " << Info.memoryCount << std::endl;
		Grid.fArray.setSizes( 28, Info.memoryCount );
	}
	else if ( Info.cellCount > Info.memoryCount )
	{
		std::cout << "rebuildGrid failed on level " << level << ", memoryCount = " << Info.memoryCount << ", cellCount = " << Info.cellCount << std::endl;
		throw std::runtime_error("rebuildGrid failed, cellCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE in your main file.");
	}
	
	Timer.stop();
	std::cout << "Step 4 mark keep area on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 5) Now we know which cells to keep, skip the unmarked ones in main NBR arrays
	int jPlus, kPlus;
	jPlus = 1; kPlus = 0;
	skipUnmarkedNBRArray( Grid.NBR.jPlusArray, Grid.keepCellMarkerArray, jPlus, kPlus, Grid, Info.cellCountFull ); 
	jPlus = 0; kPlus = 1;
	skipUnmarkedNBRArray( Grid.NBR.kPlusArray, Grid.keepCellMarkerArray, jPlus, kPlus, Grid, Info.cellCountFull ); 
	
	Timer.stop();
	std::cout << "Step 5 NBR Skip on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 6) We already have oldToFullArray from step 3, now let's build fullToKeepArray map
	IntArrayType &fullToKeepArray = Grid.intBuffer2;
	intArrayFromBoolArray( fullToKeepArray, Grid.keepCellMarkerArray, Info.cellCountFull );
	TNL::Algorithms::inplaceExclusiveScan( fullToKeepArray, 0, Info.cellCountFull, TNL::Plus{} );
	
	Timer.stop();
	std::cout << "Step 6 fullToKeepArray map build on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 7) IJK, NBR full to keep transformation
	fullToKeepTransform( Grid.IJK.iArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.IJK.jArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.IJK.kArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransformWithIndexRepair( Grid.NBR.jPlusArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransformWithIndexRepair( Grid.NBR.kPlusArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.parentMapArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	
	Timer.stop();
	std::cout << "Step 7 full to keep transformation on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 8) transform childMapArray of the coarser grid
	if ( !iAmCoarsest )
	{
		auto childMapView = GridCoarse.childMapArray.getView();
		auto keepCellMarkerView = Grid.keepCellMarkerArray.getConstView();
		auto fullToKeepView = fullToKeepArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cellCoarse ) mutable
		{	
			const int cellFine = childMapView[ cellCoarse ];
			if ( cellFine < 0 ) return;
			if ( keepCellMarkerView[ cellFine ] )
			{
				const int newIndex = fullToKeepView[ cellFine ];
				childMapView[ cellCoarse ] = newIndex;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.Info.cellCount, cellLambda );
	}
	
	Timer.stop();
	std::cout << "Step 8 childMapArray transform on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	if ( !initPass )
	{
		// 9) build oldToKeepArray map
		IntArrayType &oldToKeepArray = Grid.intBuffer3;
		auto oldToFullView = oldToFullArray.getConstView(); // we have been holding this since step 3
		auto fullToKeepView = fullToKeepArray.getConstView();
		auto oldToKeepView = oldToKeepArray.getView();
		auto cellLambda = [=] __cuda_callable__ ( const int cellOld ) mutable
		{	
			oldToKeepView[ cellOld ] = fullToKeepView[ oldToFullView[ cellOld ] ];
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCountOld, cellLambda );
		
		// 10) fArray old to keep transformation
		oldToKeepTransform( Grid );
		
		// 11) we changed our indexes, so we want to repair parentMapArray of the next finer grid
		if ( !iAmFinest )
		{
			auto parentMapView = grids[level+1].parentMapArray.getView();
			auto oldToKeepView = oldToKeepArray.getView();
			auto cellLambda = [=] __cuda_callable__ ( const int cellFine ) mutable
			{	
				const int cellCoarseOld = parentMapView[ cellFine ];
				const int cellCoarseNew = oldToKeepView[ cellCoarseOld ];
				parentMapView[ cellFine ] = cellCoarseNew;
			};
			TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, grids[level+1].Info.cellCount, cellLambda );
		}
	}
	
	Timer.stop();
	std::cout << "Steps 9-11 old to keep transform on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 12) build interface lists of the coarse grid
	if ( !iAmCoarsest )
	{
		// fine to coarse
		GridCoarse.Info.fineToCoarseCount = countOnesInBoolArray( GridCoarse.fineToCoarseMarkerArray, GridCoarse.Info.cellCount );
		if ( initPass )
		{
			GridCoarse.Info.fineToCoarseMemoryCount = GridCoarse.Info.fineToCoarseCount + ( ( GridCoarse.Info.fineToCoarseCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			//std::cout 	<< "Grid level " << level-1 << " allocated fineToCoarseMemoryCount = " 
			//			<< GridCoarse.Info.fineToCoarseMemoryCount << std::endl;
			GridCoarse.fineToCoarseIndexArray.setSize( GridCoarse.Info.fineToCoarseMemoryCount );
		}
		else if ( GridCoarse.Info.fineToCoarseCount > GridCoarse.Info.fineToCoarseMemoryCount )
		{
			std::cout 	<< "rebuildGrid failed on level " << level << ", fineToCoarseMemoryCount = " << GridCoarse.Info.fineToCoarseMemoryCount 
						<< ", fineToCoarseCount = " << GridCoarse.Info.fineToCoarseCount << std::endl;
			throw std::runtime_error("rebuildGrid failed, fineToCoarseCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE_INTERFACE in your main file.");
		}
		intArrayFromBoolArray( GridCoarse.intBuffer1, GridCoarse.fineToCoarseMarkerArray, GridCoarse.Info.cellCount );
		TNL::Algorithms::inplaceExclusiveScan( GridCoarse.intBuffer1, 0, GridCoarse.Info.cellCount, TNL::Plus{} );
		auto fineToCoarseMarkerView = GridCoarse.fineToCoarseMarkerArray.getConstView();
		auto intBuffer1View = GridCoarse.intBuffer1.getView();
		auto fineToCoarseIndexView = GridCoarse.fineToCoarseIndexArray.getView();
		auto cellLambdaFineToCoarse = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( fineToCoarseMarkerView[ cell ] )
			{
				const int index = intBuffer1View[ cell ];
				fineToCoarseIndexView[ index ] = cell;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.Info.cellCount, cellLambdaFineToCoarse );
		
		// coarse to fine
		GridCoarse.Info.coarseToFineCount = countOnesInBoolArray( GridCoarse.coarseToFineMarkerArray, GridCoarse.Info.cellCount );
		if ( initPass )
		{
			GridCoarse.Info.coarseToFineMemoryCount = GridCoarse.Info.coarseToFineCount + ( ( GridCoarse.Info.coarseToFineCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			//std::cout 	<< "Grid level " << level-1 << " allocated coarseToFineMemoryCount = " 
			//			<< GridCoarse.Info.coarseToFineMemoryCount << std::endl;
			GridCoarse.coarseToFineIndexArray.setSize( GridCoarse.Info.coarseToFineMemoryCount );
		}
		else if ( GridCoarse.Info.coarseToFineCount > GridCoarse.Info.coarseToFineMemoryCount )
		{
			std::cout 	<< "rebuildGrid failed on level " << level << ", coarseToFineMemoryCount = " << GridCoarse.Info.coarseToFineMemoryCount 
						<< ", coarseToFineCount = " << GridCoarse.Info.coarseToFineCount << std::endl;
			throw std::runtime_error("rebuildGrid failed, coarseToFineCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE_INTERFACE in your main file.");
		}
		intArrayFromBoolArray( GridCoarse.intBuffer1, GridCoarse.coarseToFineMarkerArray, GridCoarse.Info.cellCount );
		TNL::Algorithms::inplaceExclusiveScan( GridCoarse.intBuffer1, 0, GridCoarse.Info.cellCount, TNL::Plus{} );
		auto coarseToFineMarkerView = GridCoarse.coarseToFineMarkerArray.getConstView();
		// auto intBuffer1View = GridCoarse.intBuffer1.getView(); // already declared
		auto coarseToFineIndexView = GridCoarse.coarseToFineIndexArray.getView();
		auto cellLambdaCoarseToFine = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( coarseToFineMarkerView[ cell ] )
			{
				const int index = intBuffer1View[ cell ];
				coarseToFineIndexView[ index ] = cell;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.Info.cellCount, cellLambdaCoarseToFine );
	}
	
	Timer.stop();
	std::cout << "Step 12 interface list build on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 13) fill jkPlus and NBR minus
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getView();
	auto jMinusView = Grid.NBR.jMinusArray.getView();
	auto kMinusView = Grid.NBR.kMinusArray.getView();
	auto NBRFinishLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jkPlusView[ cell ] = jPlusView[ kPlusView[ cell ] ];
		jMinusView[ jPlusView[ cell ] ] = cell;
		kMinusView[ kPlusView[ cell ] ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, NBRFinishLambda );
	
	Timer.stop();
	std::cout << "Step 13 fill NBR minus on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 14) mark NBR geometric validity
	const bool markNegativeDirectionsToo = true;
	markGeometricNBR( Grid, markNegativeDirectionsToo, Info.cellCount );
	
	Timer.stop();
	std::cout << "Step 14 mark NBR geometric validity on level " << level << " took " << Timer.getRealTime() << " s" << std::endl;
	Timer.reset();
	Timer.start();
	
	// 15) recursion
	if ( !iAmFinest ) rebuildGrids( grids, Voxelizer, level+1 );
}

void initializeGrids( std::vector<GridStruct> &grids, const BoundsStruct &Bounds, const int level )
{
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];
	InfoStruct &Info = Grid.Info;
	
	if ( iAmCoarsest )
	{
		SkeletonGridStruct &SkeletonGrid = Grid.SkeletonGrid;
		InfoStruct &SkeletonInfo = SkeletonGrid.Info;
		SkeletonInfo.res = Info.res * 2.f;
		SkeletonInfo.cellCountX = static_cast<int>((Bounds.xmax - Bounds.xmin) / SkeletonInfo.res);
		SkeletonInfo.cellCountY = static_cast<int>((Bounds.ymax - Bounds.ymin) / SkeletonInfo.res);
		SkeletonInfo.cellCountZ = static_cast<int>((Bounds.zmax - Bounds.zmin) / SkeletonInfo.res);
		SkeletonInfo.cellCount = SkeletonInfo.cellCountX * SkeletonInfo.cellCountY * SkeletonInfo.cellCountZ;
		SkeletonInfo.ox = Bounds.xmin + ((Bounds.xmax - Bounds.xmin) - (SkeletonInfo.cellCountX * SkeletonInfo.res) + SkeletonInfo.res) / 2.0f;
		SkeletonInfo.oy = Bounds.ymin + ((Bounds.ymax - Bounds.ymin) - (SkeletonInfo.cellCountY * SkeletonInfo.res) + SkeletonInfo.res) / 2.0f;
		SkeletonInfo.oz = Bounds.zmin + ((Bounds.zmax - Bounds.zmin) - (SkeletonInfo.cellCountZ * SkeletonInfo.res) + SkeletonInfo.res) / 2.0f;
		SkeletonGrid.intBuffer1.setSize( SkeletonInfo.cellCount );
		SkeletonGrid.intBuffer2.setSize( SkeletonInfo.cellCount );
		SkeletonGrid.intBuffer3.setSize( SkeletonInfo.cellCount );
		SkeletonGrid.keepCellMarkerArray.setSize( SkeletonInfo.cellCount );
		SkeletonGrid.movingBouncebackMarkerArray.setSize( SkeletonInfo.cellCount );
		SkeletonGrid.markerBuffer.setSize( SkeletonInfo.cellCount );
		
		SkeletonGrid.NBRHoleMap.holeStartArray.setSizes( SkeletonInfo.cellCountX, TNL::max( SkeletonInfo.cellCountY, SkeletonInfo.cellCountZ ), RAY_MAP_DEPTH / 2 );
		SkeletonGrid.NBRHoleMap.holeEndArray.setSizes( SkeletonInfo.cellCountX, TNL::max( SkeletonInfo.cellCountY, SkeletonInfo.cellCountZ ), RAY_MAP_DEPTH / 2 );
		SkeletonGrid.NBRHoleMap.startCounterArray.setSizes( SkeletonInfo.cellCountX, TNL::max( SkeletonInfo.cellCountY, SkeletonInfo.cellCountZ ) );
		SkeletonGrid.NBRHoleMap.endCounterArray.setSizes( SkeletonInfo.cellCountX, TNL::max( SkeletonInfo.cellCountY, SkeletonInfo.cellCountZ ) );
			
		Info.cellCountX = SkeletonInfo.cellCountX * 2;
		Info.cellCountY = SkeletonInfo.cellCountY * 2;
		Info.cellCountZ = SkeletonInfo.cellCountZ * 2;
		Info.ox = SkeletonInfo.ox - Info.res * 0.5f;
		Info.oy = SkeletonInfo.oy - Info.res * 0.5f;
		Info.oz = SkeletonInfo.oz - Info.res * 0.5f;
	}
	
	else
	{
		Info.gridID = level;
		GridStruct &GridCoarse = grids[ level-1 ];
		Info.res = GridCoarse.Info.res * 0.5f;
		Info.cellCountX = GridCoarse.Info.cellCountX * 2;
		Info.cellCountY = GridCoarse.Info.cellCountY * 2;
		Info.cellCountZ = GridCoarse.Info.cellCountZ * 2;
		Info.ox = GridCoarse.Info.ox - Info.res * 0.5f;
		Info.oy = GridCoarse.Info.oy - Info.res * 0.5f;
		Info.oz = GridCoarse.Info.oz - Info.res * 0.5f;
	}
	
	Grid.NBRHoleMap.holeStartArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ), RAY_MAP_DEPTH / 2 );
	Grid.NBRHoleMap.holeEndArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ), RAY_MAP_DEPTH / 2 );
	Grid.NBRHoleMap.startCounterArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ) );
	Grid.NBRHoleMap.endCounterArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ) );
	
	if ( !iAmFinest ) initializeGrids( grids, Bounds, level+1 );
}
