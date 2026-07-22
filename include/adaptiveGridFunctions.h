#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"
#include "./NBRFunctions.h"
#include "./markerFunctions.h"
#include "./updateInterface.h"
#include "./boundaryConditions/applyInitialCondition.h"

__host__ __device__ int getFinerGridIndex( 	const int &refinedIndex,
											const int &firstInPlane, const int &lastInPlane, const int &firstInRow, const int &lastInRow, 
											const int &iAdd, const int &jAdd, const int &kAdd )
{
	const int fineIndex = firstInPlane * 8 						// previous Z layers
				+ (lastInPlane - firstInPlane + 1) * 4 * kAdd 	// one more previous Z layer if kAdd == 1
				+ (firstInRow - firstInPlane) * 4 				// previous YZ rows in the same Z plane
				+ (lastInRow - firstInRow + 1) * 2 * jAdd 		// one more previous YZ row if jAdd == 1
				+ (refinedIndex - firstInRow) * 2 				// previous part of our row
				+ iAdd;
	return fineIndex;
}

void buildFinerGrid( GridStruct &GridCoarse, GridStruct &GridFine )
{
	// label stuff for GridCoarse
	const int cellCountCoarse = GridCoarse.Info.cellCount;
	const int refinementCountCoarse = GridCoarse.Info.refinementCount;
	const IntArrayType &iArrayCoarse = GridCoarse.IJK.iArray;
	const IntArrayType &jArrayCoarse = GridCoarse.IJK.jArray;
	const IntArrayType &kArrayCoarse = GridCoarse.IJK.kArray;
	IntArrayType &jPlusArrayCoarse = GridCoarse.NBR.jPlusArray;
	IntArrayType &kPlusArrayCoarse = GridCoarse.NBR.kPlusArray;
	IntArrayType &jkPlusArrayCoarse = GridCoarse.NBR.jkPlusArray;
	IntArrayType &childMapArrayCoarse = GridCoarse.childMapArray;
	IntArrayType &intBuffer1Coarse = GridCoarse.intBuffer1;
	IntArrayType &intBuffer2Coarse = GridCoarse.intBuffer2;
	IntArrayType &intBuffer3Coarse = GridCoarse.intBuffer3;
	BoolArrayType &refinementMarkerArrayCoarse = GridCoarse.refinementMarkerArray;
	// Some GridCoarse Views
	auto iViewCoarse = iArrayCoarse.getConstView();
	auto jViewCoarse = jArrayCoarse.getConstView();
	auto kViewCoarse = kArrayCoarse.getConstView();
	auto jPlusViewCoarse = jPlusArrayCoarse.getConstView();
	auto kPlusViewCoarse = kPlusArrayCoarse.getConstView();
	auto jkPlusViewCoarse = jkPlusArrayCoarse.getView();
	auto childMapViewCoarse = childMapArrayCoarse.getView();
	auto refinementMarkerViewCoarse = refinementMarkerArrayCoarse.getConstView();
	
	// label stuff for GridFine
	const int cellCountOldFine = GridFine.Info.cellCountOld;
	const int cellCountFullFine = GridFine.Info.cellCountFull;
	IntArrayType &iArrayFine = GridFine.IJK.iArray;
	IntArrayType &jArrayFine = GridFine.IJK.jArray;
	IntArrayType &kArrayFine = GridFine.IJK.kArray;
	IntArrayType &jPlusArrayFine = GridFine.NBR.jPlusArray;
	IntArrayType &kPlusArrayFine = GridFine.NBR.kPlusArray;
	IntArrayType &jkPlusArrayFine = GridFine.NBR.jkPlusArray;
	IntArrayType &parentMapArrayFine = GridFine.parentMapArray;
	IntArrayType &intBuffer1Fine = GridFine.intBuffer1;
	IntArrayType &intBuffer2Fine = GridFine.intBuffer2;
	// Some GridFine Views
	auto iViewFine = iArrayFine.getView();
	auto jViewFine = jArrayFine.getView();
	auto kViewFine = kArrayFine.getView();
	auto jPlusViewFine = jPlusArrayFine.getView();
	auto kPlusViewFine = kPlusArrayFine.getView();
	auto jkPlusViewFine = jkPlusArrayFine.getView();
	auto parentMapViewFine = parentMapArrayFine.getView();
	
	// Because fine grid is 8x bigger than count of the refined coarse cells,
	// we will be using its intBuffer as a multiField to temporarily save 5 coarse fields in a row, 
	// i-th field starts from index i*refinementCountCoarse of the multiField
	// multiField will contain:
	// (0-1)*refinementCountCoarse = refinedParentList: Sorted indexes of refined coarse cells
	// (1-2)*refinementCountCoarse = firstInPlane: Index of the first coarse cell that is in the same Z=const plane
	// (2-3)*refinementCountCoarse = lastInPlane: Index of the last coarse cell that is in the same Z=const plane
	// (3-4)*refinementCountCoarse = firstInRow: Index of the first coarse cell that is in the same Z,Y=const row
	// (4-5)*refinementCountCoarse = lastInRow: Index of the last coarse cell that is in the same Z,Y=const row
	IntArrayType &multiField = intBuffer2Fine; // we want to keep intBuffer1Fine for the most important oldToFull index map
	multiField.setValue( 0 );
	auto multiFieldView = multiField.getView();
	
	// 1) fill multiField (0-1) = refinedParentList
	IntArrayType &refinedIndexArray = intBuffer1Coarse;
	intArrayFromBoolArray( refinedIndexArray, refinementMarkerArrayCoarse, cellCountCoarse );
	TNL::Algorithms::inplaceExclusiveScan( refinedIndexArray, 0, cellCountCoarse, TNL::Plus{} );
	auto refinedIndexView = refinedIndexArray.getConstView();
	auto cellLambda1 = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( refinementMarkerViewCoarse[ cell ] )
		{
			const int index = refinedIndexView[ cell ];
			multiFieldView[ index ] = cell;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountCoarse, cellLambda1 );
	
	// 2) fill multiField (1-2) = firstInPlane
	auto cellLambda2 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		if ( index == 0 )
		{
			// the first ever refined cell is definitely the first in its Z plane
			multiFieldView[ 1*refinementCountCoarse + index ] = index;
			return;
		}
		const int cell = multiFieldView[ index ];
		const int previousCell = multiFieldView[ index - 1 ];
		if ( kViewCoarse[ cell ] != kViewCoarse[ previousCell ] )
		{
			// our cell is first in its Z plane
			multiFieldView[ 1*refinementCountCoarse + index ] = index;
			return;
		}
		else multiFieldView[ 1*refinementCountCoarse + index ] = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, cellLambda2 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 1*refinementCountCoarse, 2*refinementCountCoarse, TNL::Max{} );
	
	// 3) fill multiField (2-3) = lastInPlane
	auto cellLambda3 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int firstInPlaneIndex = multiFieldView[ 1*refinementCountCoarse + index ];
		if ( index == refinementCountCoarse - 1 )
		{
			// the last ever refined cell is definitely the last in its Z plane
			multiFieldView[ 2*refinementCountCoarse + firstInPlaneIndex ] = index;
			return;
		}
		const int nextCellfirstInPlaneIndex = multiFieldView[ 1*refinementCountCoarse + index + 1 ];
		if ( firstInPlaneIndex != nextCellfirstInPlaneIndex )
		{
			// our cell is last in its Z plane
			multiFieldView[ 2*refinementCountCoarse + firstInPlaneIndex ] = index;
			return;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, cellLambda3 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 2*refinementCountCoarse, 3*refinementCountCoarse, TNL::Max{} );
	
	// 4) fill multiField (3-4) = firstInRow
	auto cellLambda4 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		if ( index == 0 )
		{
			// the first ever refined cell is definitely the first in its YZ row
			multiFieldView[ 3*refinementCountCoarse + index ] = index;
			return;
		}
		const int firstInPlaneIndex = multiFieldView[ 1*refinementCountCoarse + index ];
		if ( firstInPlaneIndex == index )
		{
			// our cell is the first in its plane, so its definitely also first in its YZ row
			multiFieldView[ 3*refinementCountCoarse + index ] = index;
			return;
		}
		const int cell = multiFieldView[ index ];
		const int previousCell = multiFieldView[ index - 1 ];
		if ( jViewCoarse[ cell ] != jViewCoarse[ previousCell ] )
		{
			// our cell is first in its YZ row
			multiFieldView[ 3*refinementCountCoarse + index ] = index;
			return;
		}
		else multiFieldView[ 3*refinementCountCoarse + index ] = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, cellLambda4 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 3*refinementCountCoarse, 4*refinementCountCoarse, TNL::Max{} );
	
	// 5) fill multiField (4-5) = lastInRow
	auto cellLambda5 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int firstInRowIndex = multiFieldView[ 3*refinementCountCoarse + index ];
		if ( index == refinementCountCoarse - 1 )
		{
			// the last ever refined cell is definitely the last in its YZ row
			multiFieldView[ 4*refinementCountCoarse + firstInRowIndex ] = index;
			return;
		}
		const int nextCellfirstInRowIndex = multiFieldView[ 3*refinementCountCoarse + index + 1 ];
		if ( firstInRowIndex != nextCellfirstInRowIndex )
		{
			// our cell is last in its YZ row
			multiFieldView[ 4*refinementCountCoarse + firstInRowIndex ] = index;
			return;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, cellLambda5 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 4*refinementCountCoarse, 5*refinementCountCoarse, TNL::Max{} );
	
	// 6) build absolutely essential oldToFullArray index convertor on the fine grid, which allows to transfer data into new memory indexes
	// use information from parentMapArray of the fine grid
	// Loop through old fine cells
	// For each old fine cell, find its parent in parentMapArray
	// If the parent cell does not exist, write oldToFull[ oldFineCell ] = -1 (because there won't be any new cell in this place to transfer data to)
	// If the parent cell exists but does not get refined, also write oldToFull[ oldFineCell ] = -1 
	// If the parent cell exists and will get refined, this means there will be some new fine cell in the same position we need to transfer data to
	// We can find the index of the new cell thanks to knowing firstInPlane...lastInRow information
	// We calculate this index and write it to oldToFullArray[ oldFineCell ] = newFineCell
	// The oldToFullArray will be saved to buffer 1 of the fine array, we cannot touch it later!
	IntArrayType &oldToFullArrayFine = intBuffer1Fine;
	auto oldToFullViewFine = oldToFullArrayFine.getView();
	auto cellLambda6 = [=] __cuda_callable__ ( const int cellFineOld ) mutable
	{	
		const int cellCoarse = parentMapViewFine[ cellFineOld ];
		if ( cellCoarse < 0 ) // this means the parent cell doesn't exist anymore
		{
			oldToFullViewFine[ cellFineOld ] = -1; // if the parent cell doesn't exist, the fine cell won't be created again
			return;
		}
		bool refinementMarkerCoarse = refinementMarkerViewCoarse[ cellCoarse ];
		if ( !refinementMarkerCoarse )
		{
			oldToFullViewFine[ cellFineOld ] = -1; // if the parent cell doesn't get refined, the fine cell won't be created again
			return;
		}
		// now we know the coarse cell exists and will get refined -> we need to find the new index our fine cell will receive
		const int index = refinedIndexView[ cellCoarse ];
		const int firstInPlane = multiFieldView[ 1*refinementCountCoarse + index ];
		const int lastInPlane = multiFieldView[ 2*refinementCountCoarse + index ];
		const int firstInRow = multiFieldView[ 3*refinementCountCoarse + index ];
		const int lastInRow = multiFieldView[ 4*refinementCountCoarse + index ];
		const int iFine = iViewFine[ cellFineOld ];
		const int jFine = jViewFine[ cellFineOld ];
		const int kFine = kViewFine[ cellFineOld ];
		const int iAdd = iFine % 2;
		const int jAdd = jFine % 2;
		const int kAdd = kFine % 2;
		const int cellFineNew = getFinerGridIndex( index, firstInPlane, lastInPlane, firstInRow, lastInRow, iAdd, jAdd, kAdd );
		oldToFullViewFine[ cellFineOld ] = cellFineNew;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountOldFine, cellLambda6 );
	
	// 8) Build IJK, jPlus and kPlus for the fine grid in a single big step
	// also fill the parentMapArray of the fine grid (which coarse cell created the fine cell)
	// also fill the childMapArray of the coarse grid (which fine cell is in bottom left corner of the coarse cell)
	// multiField contains:
	// (0-1)*refinementCountCoarse = refinedParentList: Sorted indexes of refined coarse cells
	// (1-2)*refinementCountCoarse = firstInPlane: Index of the first coarse cell that is in the same Z=const plane
	// (2-3)*refinementCountCoarse = lastInPlane: Index of the last coarse cell that is in the same Z=const plane
	// (3-4)*refinementCountCoarse = firstInRow: Index of the first coarse cell that is in the same Z,Y=const row
	// (4-5)*refinementCountCoarse = lastInRow: Index of the last coarse cell that is in the same Z,Y=const row
	// Use coarse buffer 2, 3, then we need to repair the jkPlus array for the coarse grid because thats the same as buffer 3!
	
	childMapArrayCoarse.setValue( -1 );
	
	IntArrayType &jPlusSkippedArray = intBuffer2Coarse;
	IntArrayType &kPlusSkippedArray = intBuffer3Coarse;
	jPlusSkippedArray = GridCoarse.NBR.jPlusArray;
	kPlusSkippedArray = GridCoarse.NBR.kPlusArray;
	int jPlus, kPlus;
	jPlus = 1; kPlus = 0;
	skipUnmarkedNBRArray( jPlusSkippedArray, refinementMarkerArrayCoarse, jPlus, kPlus, GridCoarse, cellCountCoarse ); 
	jPlus = 0; kPlus = 1;
	skipUnmarkedNBRArray( kPlusSkippedArray, refinementMarkerArrayCoarse, jPlus, kPlus, GridCoarse, cellCountCoarse );
	auto jPlusSkippedView = jPlusSkippedArray.getView();
	auto kPlusSkippedView = kPlusSkippedArray.getView();

	auto IJKNBRLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// coarse cell information
		const int cellCoarse = multiFieldView[ index ]; 
		const int iCoarse = iViewCoarse[ cellCoarse ];
		const int jCoarse = jViewCoarse[ cellCoarse ];
		const int kCoarse = kViewCoarse[ cellCoarse ];
		const int fip = multiFieldView[ 1*refinementCountCoarse + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountCoarse + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountCoarse + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountCoarse + index ];	// last in row
		// jPlus coarse neighbour information
		const int jPlusCoarse = jPlusSkippedView[ cellCoarse ];
		const int jPlusIndex = refinedIndexView[ jPlusCoarse ];
		const int jPFip = multiFieldView[ 1*refinementCountCoarse + jPlusIndex ];  // first in plane
		const int jPLip = multiFieldView[ 2*refinementCountCoarse + jPlusIndex ];  // last in plane
		const int jPFir = multiFieldView[ 3*refinementCountCoarse + jPlusIndex ];  // first in row
		const int jPLir = multiFieldView[ 4*refinementCountCoarse + jPlusIndex ];  // last in row
		// kPlus coarse neighbour information
		const int kPlusCoarse = kPlusSkippedView[ cellCoarse ];
		const int kPlusIndex = refinedIndexView[ kPlusCoarse ];
		const int kPFip = multiFieldView[ 1*refinementCountCoarse + kPlusIndex ];  // first in plane
		const int kPLip = multiFieldView[ 2*refinementCountCoarse + kPlusIndex ];  // last in plane
		const int kPFir = multiFieldView[ 3*refinementCountCoarse + kPlusIndex ];  // first in row
		const int kPLir = multiFieldView[ 4*refinementCountCoarse + kPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in coarse cell
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine000 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine100 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine010 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine110 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine001 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine101 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, bottom 4 fine cells (jAdd=0) that lie in jPlus coarse neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine020 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine120 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine021 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine121 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		// Next, left 4 fine cells (kAdd=0) that lie in kPlus coarse neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine002 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine102 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine012 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine112 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		// Fill childMap
		childMapViewCoarse[ cellCoarse ] = fine000;
		// Fill parentMap
		parentMapViewFine[ fine000 ] = cellCoarse;
		parentMapViewFine[ fine100 ] = cellCoarse;
		parentMapViewFine[ fine010 ] = cellCoarse;
		parentMapViewFine[ fine110 ] = cellCoarse;
		parentMapViewFine[ fine001 ] = cellCoarse;
		parentMapViewFine[ fine101 ] = cellCoarse;
		parentMapViewFine[ fine011 ] = cellCoarse;
		parentMapViewFine[ fine111 ] = cellCoarse;
		// Fill IJK
		const int iFine0 = 2 * iCoarse; const int jFine0 = 2 * jCoarse; const int kFine0 = 2 * kCoarse;
		iViewFine[ fine000 ] = iFine0  ;	jViewFine[ fine000 ] = jFine0  ; kViewFine[ fine000 ] = kFine0  ;
		iViewFine[ fine100 ] = iFine0+1;	jViewFine[ fine100 ] = jFine0  ; kViewFine[ fine100 ] = kFine0  ;
		iViewFine[ fine010 ] = iFine0  ;	jViewFine[ fine010 ] = jFine0+1; kViewFine[ fine010 ] = kFine0  ;
		iViewFine[ fine110 ] = iFine0+1;	jViewFine[ fine110 ] = jFine0+1; kViewFine[ fine110 ] = kFine0  ;
		iViewFine[ fine001 ] = iFine0  ;	jViewFine[ fine001 ] = jFine0  ; kViewFine[ fine001 ] = kFine0+1;
		iViewFine[ fine101 ] = iFine0+1;	jViewFine[ fine101 ] = jFine0  ; kViewFine[ fine101 ] = kFine0+1;
		iViewFine[ fine011 ] = iFine0  ;	jViewFine[ fine011 ] = jFine0+1; kViewFine[ fine011 ] = kFine0+1;
		iViewFine[ fine111 ] = iFine0+1;	jViewFine[ fine111 ] = jFine0+1; kViewFine[ fine111 ] = kFine0+1;
		// Fill jPlusFine neighbours
		jPlusViewFine[ fine000 ] = fine010;
		jPlusViewFine[ fine100 ] = fine110;
		jPlusViewFine[ fine010 ] = fine020;
		jPlusViewFine[ fine110 ] = fine120;
		jPlusViewFine[ fine001 ] = fine011;
		jPlusViewFine[ fine101 ] = fine111;
		jPlusViewFine[ fine011 ] = fine021;
		jPlusViewFine[ fine111 ] = fine121;
		// Fill kPlusFine neighbours
		kPlusViewFine[ fine000 ] = fine001;
		kPlusViewFine[ fine100 ] = fine101;
		kPlusViewFine[ fine010 ] = fine011;
		kPlusViewFine[ fine110 ] = fine111;
		kPlusViewFine[ fine001 ] = fine002;
		kPlusViewFine[ fine101 ] = fine102;
		kPlusViewFine[ fine011 ] = fine012;
		kPlusViewFine[ fine111 ] = fine112;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, IJKNBRLambda );
	
	
	// 9) repair coarse jkPlus from jPlus and kPlus, because we broke it using intBuffer3!
	auto jkPlusCoarseLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jkPlusViewCoarse[ cell ] = jPlusViewCoarse[ kPlusViewCoarse[ cell ] ];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountCoarse, jkPlusCoarseLambda );
	
	// 10) fill jkPlus from jPlus and kPlus
	auto jkPlusLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jkPlusViewFine[ cell ] = jPlusViewFine[ kPlusViewFine[ cell ] ];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountFullFine, jkPlusLambda );
	
	// 11) Last little step. We have just filled all fine neighbours. Now mark their geometric validity.
	const bool markNegativeDirectionsToo = false;
	markGeometricNBR( GridFine, markNegativeDirectionsToo, cellCountFullFine );
}

void buildFinerGrid( SkeletonGridStruct &SkeletonGrid, GridStruct &GridFine )
{
	// label stuff for SkeletonGrid
	const int cellCountSkeleton = SkeletonGrid.Info.cellCount;
	const int cellCountXSkeleton = SkeletonGrid.Info.cellCountX;
	const int cellCountYSkeleton = SkeletonGrid.Info.cellCountY;
	// const int cellCountZSkeleton = SkeletonGrid.Info.cellCountZ; // not needed here
	const int cellCountXYSkeleton = cellCountXSkeleton * cellCountYSkeleton;
	const int refinementCountSkeleton = SkeletonGrid.Info.refinementCount;
	IntArrayType &intBuffer1Skeleton = SkeletonGrid.intBuffer1;
	IntArrayType &intBuffer2Skeleton = SkeletonGrid.intBuffer2;
	IntArrayType &intBuffer3Skeleton = SkeletonGrid.intBuffer3;
	BoolArrayType &refinementMarkerArraySkeleton = SkeletonGrid.keepCellMarkerArray;
	// Some SkeletonGrid Views
	auto refinementMarkerViewSkeleton = refinementMarkerArraySkeleton.getConstView();
	
	// label stuff for GridFine
	const int cellCountOldFine = GridFine.Info.cellCountOld;
	const int cellCountFullFine = GridFine.Info.cellCountFull;
	IntArrayType &iArrayFine = GridFine.IJK.iArray;
	IntArrayType &jArrayFine = GridFine.IJK.jArray;
	IntArrayType &kArrayFine = GridFine.IJK.kArray;
	IntArrayType &jPlusArrayFine = GridFine.NBR.jPlusArray;
	IntArrayType &kPlusArrayFine = GridFine.NBR.kPlusArray;
	IntArrayType &jkPlusArrayFine = GridFine.NBR.jkPlusArray;
	IntArrayType &parentMapArrayFine = GridFine.parentMapArray;
	IntArrayType &intBuffer1Fine = GridFine.intBuffer1;
	IntArrayType &intBuffer2Fine = GridFine.intBuffer2;
	// Some GridFine Views
	auto iViewFine = iArrayFine.getView();
	auto jViewFine = jArrayFine.getView();
	auto kViewFine = kArrayFine.getView();
	auto jPlusViewFine = jPlusArrayFine.getView();
	auto kPlusViewFine = kPlusArrayFine.getView();
	auto jkPlusViewFine = jkPlusArrayFine.getView();
	auto parentMapViewFine = parentMapArrayFine.getView();
	
	// Because fine grid is 8x bigger than count of the refined skeleton cells,
	// we will be using its intBuffer as a multiField to temporarily save 5 skeleton fields in a row, 
	// i-th field starts from index i*refinementCountSkeleton of the multiField
	// multiField will contain:
	// (0-1)*refinementCountSkeleton = refinedParentList: Sorted indexes of refined skeleton cells
	// (1-2)*refinementCountSkeleton = firstInPlane: Index of the first skeleton cell that is in the same Z=const plane
	// (2-3)*refinementCountSkeleton = lastInPlane: Index of the last skeleton cell that is in the same Z=const plane
	// (3-4)*refinementCountSkeleton = firstInRow: Index of the first skeleton cell that is in the same Z,Y=const row
	// (4-5)*refinementCountSkeleton = lastInRow: Index of the last skeleton cell that is in the same Z,Y=const row
	IntArrayType &multiField = intBuffer2Fine; // we want to keep intBuffer1Fine for the most important oldToFull index map
	multiField.setValue( 0 );
	auto multiFieldView = multiField.getView();
	
	// 1) fill multiField (0-1) = refinedParentList
	IntArrayType &refinedIndexArray = intBuffer1Skeleton;
	intArrayFromBoolArray( refinedIndexArray, refinementMarkerArraySkeleton, cellCountSkeleton );
	TNL::Algorithms::inplaceExclusiveScan( refinedIndexArray, 0, cellCountSkeleton, TNL::Plus{} );
	auto refinedIndexView = refinedIndexArray.getConstView();
	auto cellLambda1 = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( refinementMarkerViewSkeleton[ cell ] )
		{
			const int index = refinedIndexView[ cell ];
			multiFieldView[ index ] = cell;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountSkeleton, cellLambda1 );
	
	// 2) fill multiField (1-2) = firstInPlane
	auto cellLambda2 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		if ( index == 0 )
		{
			// the first ever refined cell is definitely the first in its Z plane
			multiFieldView[ 1*refinementCountSkeleton + index ] = index;
			return;
		}
		const int cell = multiFieldView[ index ];
		const int previousCell = multiFieldView[ index - 1 ];
		const int kCell = cell / cellCountXYSkeleton;
		const int kPreviousCell = previousCell / cellCountXYSkeleton;
		if ( kCell != kPreviousCell )
		{
			// our cell is first in its Z plane
			multiFieldView[ 1*refinementCountSkeleton + index ] = index;
			return;
		}
		else multiFieldView[ 1*refinementCountSkeleton + index ] = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, cellLambda2 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 1*refinementCountSkeleton, 2*refinementCountSkeleton, TNL::Max{} );
	
	// 3) fill multiField (2-3) = lastInPlane
	auto cellLambda3 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int firstInPlaneIndex = multiFieldView[ 1*refinementCountSkeleton + index ];
		if ( index == refinementCountSkeleton - 1 )
		{
			// the last ever refined cell is definitely the last in its Z plane
			multiFieldView[ 2*refinementCountSkeleton + firstInPlaneIndex ] = index;
			return;
		}
		const int nextCellfirstInPlaneIndex = multiFieldView[ 1*refinementCountSkeleton + index + 1 ];
		if ( firstInPlaneIndex != nextCellfirstInPlaneIndex )
		{
			// our cell is last in its Z plane
			multiFieldView[ 2*refinementCountSkeleton + firstInPlaneIndex ] = index;
			return;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, cellLambda3 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 2*refinementCountSkeleton, 3*refinementCountSkeleton, TNL::Max{} );
	
	// 4) fill multiField (3-4) = firstInRow
	auto cellLambda4 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		if ( index == 0 )
		{
			// the first ever refined cell is definitely the first in its YZ row
			multiFieldView[ 3*refinementCountSkeleton + index ] = index;
			return;
		}
		const int firstInPlaneIndex = multiFieldView[ 1*refinementCountSkeleton + index ];
		if ( firstInPlaneIndex == index )
		{
			// our cell is the first in its plane, so its definitely also first in its YZ row
			multiFieldView[ 3*refinementCountSkeleton + index ] = index;
			return;
		}
		const int cell = multiFieldView[ index ];
		const int previousCell = multiFieldView[ index - 1 ];
		
		const int remainderCell = cell % cellCountXYSkeleton;
		const int remainderPreviousCell = previousCell % cellCountXYSkeleton;
		const int jCell = remainderCell / cellCountXSkeleton;
		const int jPreviousCell = remainderPreviousCell / cellCountXSkeleton;
		
		if ( jCell != jPreviousCell )
		{
			// our cell is first in its YZ row
			multiFieldView[ 3*refinementCountSkeleton + index ] = index;
			return;
		}
		else multiFieldView[ 3*refinementCountSkeleton + index ] = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, cellLambda4 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 3*refinementCountSkeleton, 4*refinementCountSkeleton, TNL::Max{} );
	
	// 5) fill multiField (4-5) = lastInRow
	auto cellLambda5 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int firstInRowIndex = multiFieldView[ 3*refinementCountSkeleton + index ];
		if ( index == refinementCountSkeleton - 1 )
		{
			// the last ever refined cell is definitely the last in its YZ row
			multiFieldView[ 4*refinementCountSkeleton + firstInRowIndex ] = index;
			return;
		}
		const int nextCellfirstInRowIndex = multiFieldView[ 3*refinementCountSkeleton + index + 1 ];
		if ( firstInRowIndex != nextCellfirstInRowIndex )
		{
			// our cell is last in its YZ row
			multiFieldView[ 4*refinementCountSkeleton + firstInRowIndex ] = index;
			return;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, cellLambda5 );
	TNL::Algorithms::inplaceInclusiveScan( multiField, 4*refinementCountSkeleton, 5*refinementCountSkeleton, TNL::Max{} );
	
	// 6) build absolutely essential oldToFullArray index convertor on the fine grid, which allows to transfer data into new memory indexes
	// use information from parentMapArray of the fine grid
	// Loop through old fine cells
	// For each old fine cell, find its parent in parentMapArray
	// If the parent cell does not exist, write oldToFull[ oldFineCell ] = -1 (because there won't be any new cell in this place to transfer data to)
	// If the parent cell exists but does not get refined, also write oldToFull[ oldFineCell ] = -1 
	// If the parent cell exists and will get refined, this means there will be some new fine cell in the same position we need to transfer data to
	// We can find the index of the new cell thanks to knowing firstInPlane...lastInRow information
	// We calculate this index and write it to oldToFullArray[ oldFineCell ] = newFineCell
	// The oldToFullArray will be saved to buffer 1 of the fine array, we cannot touch it later!
	IntArrayType &oldToFullArrayFine = intBuffer1Fine;
	auto oldToFullViewFine = oldToFullArrayFine.getView();
	auto cellLambda6 = [=] __cuda_callable__ ( const int cellFineOld ) mutable
	{	
		const int cellSkeleton = parentMapViewFine[ cellFineOld ];
		/* commenting out: This will never happen for the skeleton grid
		if ( cellSkeleton < 0 ) // this means the parent cell doesn't exist anymore
		{
			oldToFullViewFine[ cellFineOld ] = -1; // if the parent cell doesn't exist, the fine cell won't be created again
			return;
		}
		*/
		bool refinementMarkerSkeleton = refinementMarkerViewSkeleton[ cellSkeleton ];
		if ( !refinementMarkerSkeleton )
		{
			oldToFullViewFine[ cellFineOld ] = -1; // if the parent cell doesn't get refined, the fine cell won't be created again
			return;
		}
		// now we know the skeleton cell exists and will get refined -> we need to find the new index our fine cell will receive
		const int index = refinedIndexView[ cellSkeleton ];
		const int firstInPlane = multiFieldView[ 1*refinementCountSkeleton + index ];
		const int lastInPlane = multiFieldView[ 2*refinementCountSkeleton + index ];
		const int firstInRow = multiFieldView[ 3*refinementCountSkeleton + index ];
		const int lastInRow = multiFieldView[ 4*refinementCountSkeleton + index ];
		const int iFine = iViewFine[ cellFineOld ];
		const int jFine = jViewFine[ cellFineOld ];
		const int kFine = kViewFine[ cellFineOld ];
		const int iAdd = iFine % 2;
		const int jAdd = jFine % 2;
		const int kAdd = kFine % 2;
		const int cellFineNew = getFinerGridIndex( index, firstInPlane, lastInPlane, firstInRow, lastInRow, iAdd, jAdd, kAdd );
		oldToFullViewFine[ cellFineOld ] = cellFineNew;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountOldFine, cellLambda6 );
	
	// 8) Build IJK, jPlus and kPlus for the fine grid in a single big step
	// also fill the parentMapArray of the fine grid (which coarse cell created the fine cell)
	// also fill the childMapArray of the coarse grid (which fine cell is in bottom left corner of the coarse cell)
	// multiField contains:
	// (0-1)*refinementCountCoarse = refinedParentList: Sorted indexes of refined coarse cells
	// (1-2)*refinementCountCoarse = firstInPlane: Index of the first coarse cell that is in the same Z=const plane
	// (2-3)*refinementCountCoarse = lastInPlane: Index of the last coarse cell that is in the same Z=const plane
	// (3-4)*refinementCountCoarse = firstInRow: Index of the first coarse cell that is in the same Z,Y=const row
	// (4-5)*refinementCountCoarse = lastInRow: Index of the last coarse cell that is in the same Z,Y=const row
	// Use skeleton buffer 2, 3
	
	IntArrayType &jPlusSkippedArray = intBuffer2Skeleton;
	IntArrayType &kPlusSkippedArray = intBuffer3Skeleton;
	int jPlus, kPlus;
	jPlus = 1; kPlus = 0;
	getNBRArrayForSkeleton( jPlusSkippedArray, jPlus, kPlus, SkeletonGrid );
	skipUnmarkedNBRArray( jPlusSkippedArray, refinementMarkerArraySkeleton, jPlus, kPlus, SkeletonGrid ); 
	jPlus = 0; kPlus = 1;
	getNBRArrayForSkeleton( kPlusSkippedArray, jPlus, kPlus, SkeletonGrid );
	skipUnmarkedNBRArray( kPlusSkippedArray, refinementMarkerArraySkeleton, jPlus, kPlus, SkeletonGrid );
	auto jPlusSkippedView = jPlusSkippedArray.getView();
	auto kPlusSkippedView = kPlusSkippedArray.getView();

	auto IJKNBRLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// coarse cell information
		const int cellSkeleton = multiFieldView[ index ]; 
		const int kSkeleton = cellSkeleton / cellCountXYSkeleton;
		const int remainder = cellSkeleton % cellCountXYSkeleton;
		const int jSkeleton = remainder / cellCountXSkeleton;
		const int iSkeleton = remainder % cellCountXSkeleton;
		const int fip = multiFieldView[ 1*refinementCountSkeleton + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountSkeleton + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountSkeleton + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountSkeleton + index ];	// last in row
		// jPlus coarse neighbour information
		const int jPlusSkeleton = jPlusSkippedView[ cellSkeleton ];
		const int jPlusIndex = refinedIndexView[ jPlusSkeleton ];
		const int jPFip = multiFieldView[ 1*refinementCountSkeleton + jPlusIndex ];  // first in plane
		const int jPLip = multiFieldView[ 2*refinementCountSkeleton + jPlusIndex ];  // last in plane
		const int jPFir = multiFieldView[ 3*refinementCountSkeleton + jPlusIndex ];  // first in row
		const int jPLir = multiFieldView[ 4*refinementCountSkeleton + jPlusIndex ];  // last in row
		// kPlus coarse neighbour information
		const int kPlusSkeleton = kPlusSkippedView[ cellSkeleton ];
		const int kPlusIndex = refinedIndexView[ kPlusSkeleton ];
		const int kPFip = multiFieldView[ 1*refinementCountSkeleton + kPlusIndex ];  // first in plane
		const int kPLip = multiFieldView[ 2*refinementCountSkeleton + kPlusIndex ];  // last in plane
		const int kPFir = multiFieldView[ 3*refinementCountSkeleton + kPlusIndex ];  // first in row
		const int kPLir = multiFieldView[ 4*refinementCountSkeleton + kPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in coarse cell
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine000 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine100 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine010 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine110 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine001 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine101 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, bottom 4 fine cells (jAdd=0) that lie in jPlus coarse neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine020 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine120 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine021 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine121 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		// Next, left 4 fine cells (kAdd=0) that lie in kPlus coarse neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine002 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine102 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine012 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine112 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		// Fill parentMap
		parentMapViewFine[ fine000 ] = cellSkeleton;
		parentMapViewFine[ fine100 ] = cellSkeleton;
		parentMapViewFine[ fine010 ] = cellSkeleton;
		parentMapViewFine[ fine110 ] = cellSkeleton;
		parentMapViewFine[ fine001 ] = cellSkeleton;
		parentMapViewFine[ fine101 ] = cellSkeleton;
		parentMapViewFine[ fine011 ] = cellSkeleton;
		parentMapViewFine[ fine111 ] = cellSkeleton;
		// Fill IJK
		const int iFine0 = 2 * iSkeleton; const int jFine0 = 2 * jSkeleton; const int kFine0 = 2 * kSkeleton;
		iViewFine[ fine000 ] = iFine0  ;	jViewFine[ fine000 ] = jFine0  ; kViewFine[ fine000 ] = kFine0  ;
		iViewFine[ fine100 ] = iFine0+1;	jViewFine[ fine100 ] = jFine0  ; kViewFine[ fine100 ] = kFine0  ;
		iViewFine[ fine010 ] = iFine0  ;	jViewFine[ fine010 ] = jFine0+1; kViewFine[ fine010 ] = kFine0  ;
		iViewFine[ fine110 ] = iFine0+1;	jViewFine[ fine110 ] = jFine0+1; kViewFine[ fine110 ] = kFine0  ;
		iViewFine[ fine001 ] = iFine0  ;	jViewFine[ fine001 ] = jFine0  ; kViewFine[ fine001 ] = kFine0+1;
		iViewFine[ fine101 ] = iFine0+1;	jViewFine[ fine101 ] = jFine0  ; kViewFine[ fine101 ] = kFine0+1;
		iViewFine[ fine011 ] = iFine0  ;	jViewFine[ fine011 ] = jFine0+1; kViewFine[ fine011 ] = kFine0+1;
		iViewFine[ fine111 ] = iFine0+1;	jViewFine[ fine111 ] = jFine0+1; kViewFine[ fine111 ] = kFine0+1;
		// Fill jPlusFine neighbours
		jPlusViewFine[ fine000 ] = fine010;
		jPlusViewFine[ fine100 ] = fine110;
		jPlusViewFine[ fine010 ] = fine020;
		jPlusViewFine[ fine110 ] = fine120;
		jPlusViewFine[ fine001 ] = fine011;
		jPlusViewFine[ fine101 ] = fine111;
		jPlusViewFine[ fine011 ] = fine021;
		jPlusViewFine[ fine111 ] = fine121;
		// Fill kPlusFine neighbours
		kPlusViewFine[ fine000 ] = fine001;
		kPlusViewFine[ fine100 ] = fine101;
		kPlusViewFine[ fine010 ] = fine011;
		kPlusViewFine[ fine110 ] = fine111;
		kPlusViewFine[ fine001 ] = fine002;
		kPlusViewFine[ fine101 ] = fine102;
		kPlusViewFine[ fine011 ] = fine012;
		kPlusViewFine[ fine111 ] = fine112;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, IJKNBRLambda );
	
	// 8.3) fill jkPlus from jPlus and kPlus
	auto jkPlusLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		jkPlusViewFine[ cell ] = jPlusViewFine[ kPlusViewFine[ cell ] ];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountFullFine, jkPlusLambda );
	
	// 9) Last little step. We have just filled all fine neighbours. Now mark their geometric validity.
	const bool markNegativeDirectionsToo = false;
	markGeometricNBR( GridFine, markNegativeDirectionsToo, cellCountFullFine );
}

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
		else if ( postCollisionLocation == 6 ) writeIndex = jPlusView(kPlusView(cellNew));  
		else    							   writeIndex = jPlusView(kPlusView(cellNew)) + 1;		
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
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];	
	InfoStruct &Info = Grid.Info;
	Info.cellCountOld = Info.cellCount;
	
	Info.updatesSinceRebuild = 0; 
	Info.updatesSinceMovingBouncebackUpdate = 0;
	
	const bool initPass = ( Grid.fArray.getSizes()[0] < 1 );
	
	SkeletonGridStruct &SkeletonGrid = Grid.SkeletonGrid;
	InfoStruct &SkeletonInfo = SkeletonGrid.Info;
	
	static GridStruct dummyGrid; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridStruct &GridCoarse = iAmCoarsest ? dummyGrid : grids[ level - 1 ];
	InfoStruct &InfoCoarse = GridCoarse.Info;
	
	// 1) Pull fArray into the correct cells to be able to forget NBR
	if ( !initPass ) pullFArrayIntoCells( Grid );
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
		Grid.movingBouncebackMarkerArray.setSize( Info.memoryCountFull );
		Grid.markerBuffer.setSize( Info.memoryCountFull );
		Info.gridMemoryMB += (float)(32 * 9 + 13 * 8) * (float)(Info.memoryCountFull) / 8000000.f; // 9 int arrays, 13 bool arrays
		if ( iAmFinest )
		{
			Grid.bouncebackMarkerArray.setSize( Info.memoryCountFull );
			Info.gridMemoryMB += (float)(1 * 8) * (float)(Info.memoryCountFull) / 8000000.f; // 1 bool array
		}
		else
		{
			Grid.childMapArray.setSize( Info.memoryCountFull );
			Grid.refinementMarkerArray.setSize( Info.memoryCountFull );
			Grid.deepRefinementMarkerArray.setSize( Info.memoryCountFull );
			Grid.fineToCoarseMarkerArray.setSize( Info.memoryCountFull );
			Grid.coarseToFineMarkerArray.setSize( Info.memoryCountFull );
			Info.gridMemoryMB += (float)(32 + 4 * 8) * (float)(Info.memoryCountFull) / 8000000.f; // 1 int array, 4 bool arrays
		}
	}
	else if ( Info.cellCountFull > Info.memoryCountFull )
	{
		std::cout << "rebuildGrid failed on level " << level << ", memoryCountFull = " << Info.memoryCountFull << ", cellCountFull = " << Info.cellCountFull << std::endl;
		throw std::runtime_error("rebuildGrid failed, cellCountFull exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE in your main file.");
	}
	// 3) Build our grid (we are the "finer grid" with respect to the grid we are taking spatial information from)
	
	if ( iAmCoarsest ) buildFinerGrid( SkeletonGrid, Grid );
	else buildFinerGrid( GridCoarse, Grid );
	IntArrayType &oldToFullArray = Grid.intBuffer1; // We cannot touch intBuffer1 now!
	// 4) Get rid of the cells that are deep inside solid. Only keep the necessary ones, mark them in keepCellMarkerArray
	markKeepCells( Grid, Voxelizer, Info.cellCountFull );
	Info.cellCount = countOnesInBoolArray( Grid.keepCellMarkerArray, Info.cellCountFull );
	
	if ( initPass )
	{
		Info.memoryCount = Info.cellCount + ( ( Info.cellCount * MEMORY_RESERVE_PERCENTAGE ) / 100 );
		// std::cout 	<< "Grid level " << level << " allocated memoryCount = " << Info.memoryCount << std::endl;
		Grid.fArray.setSizes( 28, Info.memoryCount );
		Info.gridMemoryMB += (float)(32 * 28) * (float)(Info.memoryCount) / 8000000.f; // 28 float arrays
	}
	else if ( Info.cellCount > Info.memoryCount )
	{
		std::cout << "rebuildGrid failed on level " << level << ", memoryCount = " << Info.memoryCount << ", cellCount = " << Info.cellCount << std::endl;
		throw std::runtime_error("rebuildGrid failed, cellCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE in your main file.");
	}
	
	// 5) Now we know which cells to keep, skip the unmarked ones in main NBR arrays
	int jPlus, kPlus;
	jPlus = 1; kPlus = 0;
	skipUnmarkedNBRArray( Grid.NBR.jPlusArray, Grid.keepCellMarkerArray, jPlus, kPlus, Grid, Info.cellCountFull ); 
	jPlus = 0; kPlus = 1;
	skipUnmarkedNBRArray( Grid.NBR.kPlusArray, Grid.keepCellMarkerArray, jPlus, kPlus, Grid, Info.cellCountFull ); 
	
	// 6) We already have oldToFullArray from step 3, now let's build fullToKeepArray map
	IntArrayType &fullToKeepArray = Grid.intBuffer2;
	intArrayFromBoolArray( fullToKeepArray, Grid.keepCellMarkerArray, Info.cellCountFull );
	TNL::Algorithms::inplaceExclusiveScan( fullToKeepArray, 0, Info.cellCountFull, TNL::Plus{} );

	// 7) IJK, NBR full to keep transformation
	fullToKeepTransform( Grid.IJK.iArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.IJK.jArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.IJK.kArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransformWithIndexRepair( Grid.NBR.jPlusArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransformWithIndexRepair( Grid.NBR.kPlusArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.parentMapArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );

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
	
	if ( !initPass )
	{
		// 9) build oldToKeepArray map
		IntArrayType &oldToKeepArray = Grid.intBuffer3;
		auto oldToFullView = oldToFullArray.getConstView(); // we have been holding this since step 3
		auto fullToKeepView = fullToKeepArray.getConstView();
		auto oldToKeepView = oldToKeepArray.getView();
		auto cellLambda = [=] __cuda_callable__ ( const int cellOld ) mutable
		{	
			const int oldToFullIndex = oldToFullView[ cellOld ];
			if ( oldToFullIndex > 0 ) oldToKeepView[ cellOld ] = fullToKeepView[ oldToFullIndex ];
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
	// 12) build interface lists of the coarse grid
	if ( !iAmCoarsest )
	{
		// fine to coarse
		GridCoarse.Info.fineToCoarseCount = countOnesInBoolArray( GridCoarse.fineToCoarseMarkerArray, GridCoarse.Info.cellCount );
		if ( initPass )
		{
			InfoCoarse.fineToCoarseMemoryCount = GridCoarse.Info.fineToCoarseCount + ( ( GridCoarse.Info.fineToCoarseCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			GridCoarse.fineToCoarseIndexArray.setSize( InfoCoarse.fineToCoarseMemoryCount );
			InfoCoarse.gridMemoryMB += (float)(32) * (float)(InfoCoarse.fineToCoarseMemoryCount) / 8000000.f; // 1 int array
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
			InfoCoarse.coarseToFineMemoryCount = GridCoarse.Info.coarseToFineCount + ( ( GridCoarse.Info.coarseToFineCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			GridCoarse.coarseToFineIndexArray.setSize( InfoCoarse.coarseToFineMemoryCount );
			InfoCoarse.gridMemoryMB += (float)(32) * (float)(InfoCoarse.coarseToFineMemoryCount) / 8000000.f; // 1 int array
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
	
	// 14) mark NBR geometric validity
	const bool markNegativeDirectionsToo = true;
	markGeometricNBR( Grid, markNegativeDirectionsToo, Info.cellCount );
	
	// 15) if we are the finest grid, mark moving bounceback and bounceback
	if ( iAmFinest )
	{
		applyMarkersFromRayMap( Grid.bouncebackMarkerArray, Voxelizer.rayMapBounceback, Grid, Grid.Info.cellCount );
		applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Grid.Info.cellCount );
	}
	
	// 15) we apply initial condition for fArray on the coarser grid, because only now we have marked it correctly
	// if we are finest we can apply initial condition on ourselves too
	// at the same time, report memory size
	if ( initPass ) 
	{
		if ( !iAmCoarsest ) 
		{
			applyInitialCondition( grids[ level-1 ] );
			std::cout << "Grid level " << level-1 << " allocated on GPU, it takes " << grids[level-1].Info.gridMemoryMB << " MB" << std::endl;
		}
		if ( iAmFinest ) 
		{
			applyInitialCondition( Grid );
			std::cout << "Grid level " << level << " allocated on GPU, it takes " << Grid.Info.gridMemoryMB << " MB" << std::endl;
		}
	}
	
	// 16 recursion
	if ( !iAmFinest ) rebuildGrids( grids, Voxelizer, level+1 );
	
	// 17
	if ( !iAmCoarsest )
	{
		updateInterface(grids[level - 1], Grid);
	}
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
		
		Info.gridMemoryMB += (float)(3 * 32 + 3 * 8) * (float)SkeletonInfo.cellCount / 8000000.f; // 3 int buffers + 3 marker arrays
		Info.gridMemoryMB += (float)(32) * (float)(SkeletonInfo.cellCountX * TNL::max( SkeletonInfo.cellCountY, SkeletonInfo.cellCountZ ) * (RAY_MAP_DEPTH / 2 * 2 + 2)) / 8000000.f; // NBRHoleMap
			
		Info.cellCountX = SkeletonInfo.cellCountX * 2;
		Info.cellCountY = SkeletonInfo.cellCountY * 2;
		Info.cellCountZ = SkeletonInfo.cellCountZ * 2;
		Info.ox = SkeletonInfo.ox - Info.res * 0.5f;
		Info.oy = SkeletonInfo.oy - Info.res * 0.5f;
		Info.oz = SkeletonInfo.oz - Info.res * 0.5f;
		
		Info.dtPhys = dtPhysGlobal;
		Info.nu = (Info.dtPhys * nuPhys) / ((Info.res/1000.f) * (Info.res/1000.f));
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
		Info.dtPhys = GridCoarse.Info.dtPhys * 0.5f;
		Info.nu = (Info.dtPhys * nuPhys) / ((Info.res/1000.f) * (Info.res/1000.f));
	}
	
	Grid.NBRHoleMap.holeStartArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ), RAY_MAP_DEPTH / 2 );
	Grid.NBRHoleMap.holeEndArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ), RAY_MAP_DEPTH / 2 );
	Grid.NBRHoleMap.startCounterArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ) );
	Grid.NBRHoleMap.endCounterArray.setSizes( Info.cellCountX, TNL::max( Info.cellCountY, Info.cellCountZ ) );
	
	Info.gridMemoryMB += (float)(32) * (float)(Info.cellCountX * TNL::max( Info.cellCountY, Info.cellCountZ ) * (RAY_MAP_DEPTH / 2 * 2 + 2)) / 8000000.f; // NBRHoleMap
	
	if ( !iAmFinest ) initializeGrids( grids, Bounds, level+1 );
}
