#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"

void markGeometricNBR( GridStruct &Grid, const bool markNegativeDirectionsToo, const int &upperBound )
{
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();
	auto jMinusView = Grid.NBR.jMinusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	BoolViewType isGeometricMarkerView[10];
	for ( int i = 0; i < 10; i++ ) isGeometricMarkerView[i] = Grid.NBR.isGeometricMarkerArray[i].getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		
		const int iPlus = (cell + 1 < upperBound) ? cell + 1 : 0;
		const int jPlus = jPlusView[ cell ];
		const int ijPlus = (jPlus + 1 < upperBound) ? jPlus + 1 : 0;
		const int kPlus = kPlusView[ cell ];
		const int ikPlus = (kPlus + 1 < upperBound) ? kPlus + 1 : 0;
		const int jkPlus = jkPlusView[ cell ];		
		const int ijkPlus = (jkPlus + 1 < upperBound) ? jkPlus + 1 : 0;
		
		isGeometricMarkerView[0][cell] = ( iView[iPlus]==iCell+1 && jView[iPlus]==jCell && kView[iPlus]==kCell );
		isGeometricMarkerView[1][cell] = ( iView[jPlus]==iCell && jView[jPlus]==jCell+1 && kView[jPlus]==kCell );
		isGeometricMarkerView[2][cell] = ( iView[ijPlus]==iCell+1 && jView[ijPlus]==jCell+1 && kView[ijPlus]==kCell );
		isGeometricMarkerView[3][cell] = ( iView[kPlus]==iCell && jView[kPlus]==jCell && kView[kPlus]==kCell+1 );
		isGeometricMarkerView[4][cell] = ( iView[ikPlus]==iCell+1 && jView[ikPlus]==jCell && kView[ikPlus]==kCell+1 );
		isGeometricMarkerView[5][cell] = ( iView[jkPlus]==iCell && jView[jkPlus]==jCell+1 && kView[jkPlus]==kCell+1 );
		isGeometricMarkerView[6][cell] = ( iView[ijkPlus]==iCell+1 && jView[ijkPlus]==jCell+1 && kView[ijkPlus]==kCell+1 );
		
		if ( markNegativeDirectionsToo )
		{
			const int jMinus = jMinusView[ cell ];
			const int kMinus = kMinusView[ cell ]; 
			const int iMinus = (cell > 0) ? cell - 1 : upperBound - 1;
			isGeometricMarkerView[7][cell] = ( iView[iMinus]==iCell-1 && jView[iMinus]==jCell && kView[iMinus]==kCell );
			isGeometricMarkerView[8][cell] = ( iView[jMinus]==iCell && jView[jMinus]==jCell-1 && kView[jMinus]==kCell );
			isGeometricMarkerView[9][cell] = ( iView[kMinus]==iCell && jView[kMinus]==jCell && kView[kMinus]==kCell-1 );
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void skipOneUnmarkedNeighbour( IntArrayType &nbrArray, const BoolArrayType &markerArray, const IntArrayType &nbrOldArray, BoolArrayType &finishedMarkerArray, const int &upperBound )
{
	// For each cell, if its neighbour is not marked, travel one neighbour up (set our neighbour to the neighbour's neighbour)
	auto nbrView = nbrArray.getView();
	auto nbrOldView = nbrOldArray.getConstView();
	auto markerView = markerArray.getConstView();
	auto finishedMarkerView = finishedMarkerArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		if ( finishedMarkerView[ cell ] ) return;
		int nbr = nbrOldView[ cell ];
		if ( nbr == cell || markerView[ nbr ] ) // the first condition clicks if we already travelled a full closed unmarked loop
		{
			nbrView[ cell ] = nbr;
			finishedMarkerView[ cell ] = true;
			return;
		}
		int newNbr = nbrOldView[ nbr ]; // neighbour of our neighbour
		nbrView[ cell ] = newNbr;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void skipUnmarkedNeighbours( IntArrayType nbrArray, const BoolArrayType markerArray, IntArrayType intBuffer, BoolArrayType markerBuffer, const int &upperBound )
{
	// Receives nbrArray, which also points on cells that are not marked
	// For each cell, while its neighbour is not marked, we want to travel up the neighbour list
	// to find the first marked neighbour
	IntArrayType &nbrOldArray = intBuffer;
	BoolArrayType &finishedMarkerArray = markerBuffer;
	
	finishedMarkerArray.setValue( false );
	int unfinishedCount = upperBound;
	
	while ( unfinishedCount > 0 )
	{
		nbrArray.swap( nbrOldArray ); // pointer swap without data travel
		skipOneUnmarkedNeighbour( nbrArray, markerArray, nbrOldArray, finishedMarkerArray, upperBound );
		unfinishedCount = countZerosInBoolArray( finishedMarkerArray, upperBound );
	}
}

void spreadMarkers( BoolArrayType &targetMarkerArray, const BoolArrayType &sourceMarkerArray, GridStruct &Grid, const int &upperBound )
{
	// The way this is written creates a race condition, one that is harmless because all threads write the same 1
	auto targetMarkerView = targetMarkerArray.getView();
	auto sourceMarkerView = sourceMarkerArray.getConstView();
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();
	
	BoolViewType isGeometricMarkerView[7];
	for ( int i = 0; i < 7; i++ ) isGeometricMarkerView[i] = Grid.NBR.isGeometricMarkerArray[i].getView();
	
	targetMarkerArray = sourceMarkerArray; // initialize as source

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		int nbrPlus[7];
		nbrPlus[0] = (cell + 1 < upperBound) ? cell + 1 : 0; 					// iPlus
		nbrPlus[1] = jPlusView[ cell ];											// jPlus
		nbrPlus[2] = (nbrPlus[1] + 1 < upperBound) ? nbrPlus[1] + 1 : 0;		// ijPlus
		nbrPlus[3] = kPlusView[ cell ];											// kPlus
		nbrPlus[4] = (nbrPlus[3] + 1 < upperBound) ? nbrPlus[3] + 1 : 0;		// ikPlus
		nbrPlus[5] = jkPlusView[ cell ];										// jkPlus
		nbrPlus[6] = (nbrPlus[5] + 1 < upperBound) ? nbrPlus[5] + 1 : 0;		// ijkPlus
		bool isGeometricMarker[7];
		for ( int q = 0; q < 7; q++ ) isGeometricMarker[q] = isGeometricMarkerView[q][cell];
		
		bool marker = sourceMarkerView[ cell ];
		if ( !marker )
		{
			for ( int q = 0; q < 7; q++ )
			{
				if ( isGeometricMarker[q] )
				{
					if ( sourceMarkerView[nbrPlus[q]] )
					{
						marker = true;
						break;
					}
				}
			}
		}
		if ( marker )
		{
			targetMarkerView[ cell ] = true; // <- race condition here
			for ( int q = 0; q < 7; q++ )
			{
				if ( isGeometricMarker[q] )
				{
					targetMarkerView[nbrPlus[q]] = true; // <- race condition here too <3
				}
			}
		}		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void spreadMarkers( BoolArrayType &targetMarkerArray, const BoolArrayType &sourceMarkerArray, SkeletonGridStruct &SkeletonGrid )
{
	auto targetMarkerView = targetMarkerArray.getView();
	auto sourceMarkerView = sourceMarkerArray.getConstView();
	const int cellCountX = SkeletonGrid.Info.cellCountX;
	const int cellCountY = SkeletonGrid.Info.cellCountY;
	const int cellCountZ = SkeletonGrid.Info.cellCountZ;
	const int cellCount = SkeletonGrid.Info.cellCount;

	targetMarkerArray = sourceMarkerArray; // initialize as source
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		bool marker = sourceMarkerView[ cell ];
		if ( marker ) return; // only continue if the marker is not already 1
		const int kCell = cell / (cellCountX * cellCountY);
		const int remainder = cell % (cellCountX * cellCountY);
		const int jCell = remainder / cellCountX;
		const int iCell = remainder % cellCountX;
		int nbr, iNbr, jNbr, kNbr;
		for ( int kAdd = -1; kAdd <= 1; kAdd++ )
		{
			kNbr = kCell + kAdd;
			if ( kNbr >= 0 && kNbr < cellCountZ )
			{
				for ( int jAdd = -1; jAdd <= 1; jAdd++ )
				{
					jNbr = jCell + jAdd;
					if ( jNbr >= 0 && jNbr < cellCountY )
					{
						for ( int iAdd = -1; iAdd <= 1; iAdd++ )
						{
							if ( kAdd!=0 || jAdd!=0 || iAdd!=0 )
							{
								iNbr = iCell + iAdd;
								if ( iNbr >= 0 && iNbr < cellCountX )
								{
									nbr = kNbr * (cellCountX * cellCountY) + jNbr * cellCountX + iNbr;
									if ( sourceMarkerView[ nbr ] )
									{
										targetMarkerView[ cell ] = true;
										return;
									}
								}
							}
						}
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}


void markKeepCells( GridStruct &Grid, const VoxelizerStruct &Voxelizer, const int &upperBound )
{
	// the marking functions used here are defined in voxelizerFunctions.h
	markFinestFluid( Grid.keepCellMarkerArray, Voxelizer.rayMapTotal, Grid, upperBound );	
	Grid.keepCellMarkerArray.swap( Grid.markerBuffer );
	spreadMarkers( Grid.keepCellMarkerArray, Grid.markerBuffer, Grid, upperBound );
	// the moving bounceback geometry can travel distance up to 2 cells before the grid is fully rebuild
	// so we need to keep a 3-cell thick moving bounceback layer (2 layers may become fluid later)
	// we do this by spreading the keepCell area by 2 more cells
	// then find intersection of the spread with moving bounceback and add the intersection to original keepCell area
	// right now the movingBouncebackMarkerArray is not needed -> we will use it as a second buffer to hold the spread
	Grid.markerBuffer = Grid.keepCellMarkerArray;
	for ( int spread = 0; spread < 2; spread++ )
	{
		Grid.movingBouncebackMarkerArray.swap( Grid.markerBuffer );
		spreadMarkers( Grid.markerBuffer, Grid.movingBouncebackMarkerArray, Grid, upperBound );
	}
	// at this point result of the spread is in markerBuffer
	// now we stop using the movingBouncebackMarkerArray as a buffer and write actual data into it
	markFinestBounceback( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, upperBound );
	Grid.keepCellMarkerArray = Grid.keepCellMarkerArray + Grid.markerBuffer * Grid.movingBouncebackMarkerArray;
}

void markKeepCells( SkeletonGridStruct &SkeletonGrid, const VoxelizerStruct &Voxelizer )
{
	// version for SkeletonGrid, comments for this function are above in version for Grid
	markFinestFluid( SkeletonGrid.keepCellMarkerArray, Voxelizer.rayMapTotal, SkeletonGrid );	
	SkeletonGrid.keepCellMarkerArray.swap( SkeletonGrid.markerBuffer );
	spreadMarkers( SkeletonGrid.keepCellMarkerArray, SkeletonGrid.markerBuffer, SkeletonGrid );
	SkeletonGrid.markerBuffer = SkeletonGrid.keepCellMarkerArray;
	for ( int spread = 0; spread < 2; spread++ )
	{
		SkeletonGrid.movingBouncebackMarkerArray.swap( SkeletonGrid.markerBuffer );
		spreadMarkers( SkeletonGrid.markerBuffer, SkeletonGrid.movingBouncebackMarkerArray, SkeletonGrid );
	}
	markFinestBounceback( SkeletonGrid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, SkeletonGrid );
	SkeletonGrid.keepCellMarkerArray = SkeletonGrid.keepCellMarkerArray + SkeletonGrid.markerBuffer * SkeletonGrid.movingBouncebackMarkerArray;
}

void markRefinementCells( GridStruct &Grid, const VoxelizerStruct &Voxelizer, const int &upperBound )
{
	markKeepCells( Grid, Voxelizer, upperBound );
	// search deep refinement area
	markFinestBounceback( Grid.deepRefinementMarkerArray, Voxelizer.rayMapTotal, Grid, upperBound );
	for ( int spread = 0; spread < WALL_REFINEMENT_COUNT; spread++ )
	{
		Grid.deepRefinementMarkerArray.swap( Grid.markerBuffer );
		spreadMarkers( Grid.deepRefinementMarkerArray, Grid.markerBuffer, Grid, upperBound );
	}
	Grid.deepRefinementMarkerArray = Grid.deepRefinementMarkerArray * Grid.keepCellMarkerArray;
	// search fine to coarse interface
	Grid.fineToCoarseMarkerArray = Grid.deepRefinementMarkerArray;
	Grid.fineToCoarseMarkerArray.swap( Grid.markerBuffer );
	spreadMarkers( Grid.fineToCoarseMarkerArray, Grid.markerBuffer, Grid, upperBound );
	Grid.fineToCoarseMarkerArray = Grid.fineToCoarseMarkerArray * Grid.keepCellMarkerArray * !Grid.deepRefinementMarkerArray;
	// search coarse to fine interface
	Grid.coarseToFineMarkerArray = Grid.fineToCoarseMarkerArray;
	Grid.coarseToFineMarkerArray.swap( Grid.markerBuffer );
	spreadMarkers( Grid.coarseToFineMarkerArray, Grid.markerBuffer, Grid, upperBound );
	Grid.coarseToFineMarkerArray = Grid.coarseToFineMarkerArray * Grid.keepCellMarkerArray * !Grid.deepRefinementMarkerArray * !Grid.fineToCoarseMarkerArray;
	// mark refinement all together
	Grid.refinementMarkerArray = Grid.deepRefinementMarkerArray + Grid.fineToCoarseMarkerArray + Grid.coarseToFineMarkerArray;
}

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
	IntArrayType &intBuffer3Coarse = GridCoarse.intBuffer3;
	const IntArrayType &iArrayCoarse = GridCoarse.IJK.iArray;
	const IntArrayType &jArrayCoarse = GridCoarse.IJK.jArray;
	const IntArrayType &kArrayCoarse = GridCoarse.IJK.kArray;
	IntArrayType &childMapArrayCoarse = GridCoarse.childMapArray;
	auto intBuffer3ViewCoarse = intBuffer3Coarse.getView();
	auto iViewCoarse = iArrayCoarse.getConstView();
	auto jViewCoarse = jArrayCoarse.getConstView();
	auto kViewCoarse = kArrayCoarse.getConstView();
	auto childMapViewCoarse = childMapArrayCoarse.getView();
	auto refinementMarkerViewCoarse = GridCoarse.refinementMarkerArray.getConstView();
	
	// label stuff for GridFine
	const int cellCountOldFine = GridFine.Info.cellCountOld;
	IntArrayType &iArrayFine = GridFine.IJK.iArray;
	IntArrayType &jArrayFine = GridFine.IJK.jArray;
	IntArrayType &kArrayFine = GridFine.IJK.kArray;
	IntArrayType &parentMapArrayFine = GridFine.parentMapArray;
	IntArrayType &oldToFullArrayFine = GridFine.intBuffer2;
	auto iViewFine = iArrayFine.getView();
	auto jViewFine = jArrayFine.getView();
	auto kViewFine = kArrayFine.getView();
	auto parentMapViewFine = parentMapArrayFine.getView();
	auto oldToFullViewFine = oldToFullArrayFine.getView();
	
	// Because fine grid is 8x bigger than count of the refined coarse cells,
	// we will be using its intBuffer as a multiField to temporarily save 5 coarse fields in a row, 
	// i-th field starts from index i*refinementCountCoarse of the multiField
	// multiField will contain:
	// (0-1)*refinementCountCoarse = refinedParentList: Sorted indexes of refined coarse cells
	// (1-2)*refinementCountCoarse = firstInPlane: Index of the first coarse cell that is in the same Z=const plane
	// (2-3)*refinementCountCoarse = lastInPlane: Index of the last coarse cell that is in the same Z=const plane
	// (3-4)*refinementCountCoarse = firstInRow: Index of the first coarse cell that is in the same Z,Y=const row
	// (4-5)*refinementCountCoarse = lastInRow: Index of the last coarse cell that is in the same Z,Y=const row
	IntArrayType &multiField = GridFine.intBuffer3;
	multiField.setValue( 0 );
	auto multiFieldView = multiField.getView();
	
	// 1) fill multiField (0-1) = refinedParentList
	intArrayFromBoolArray( intBuffer3Coarse, GridCoarse.refinementMarkerArray, cellCountCoarse );
	TNL::Algorithms::inplaceExclusiveScan( intBuffer3Coarse, 0, cellCountCoarse, TNL::Plus{} );
	auto cellLambda1 = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( refinementMarkerViewCoarse[ cell ] )
		{
			const int index = intBuffer3ViewCoarse[ cell ];
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
		const int index = intBuffer3ViewCoarse[ cellCoarse ];
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
	
	// 7) write new IJK for the fine grid in already sorted order thanks to knowing firstInPlane...lastInRow information
	// also fill the parentMapArray of the fine grid (which coarse cell created the fine cell)
	// also fill the childMapArray of the coarse grid (which fine cell is in bottom left corner of the coarse cell)
	// multiField contains:
	// (0-1)*refinementCountCoarse = refinedParentList: Sorted indexes of refined coarse cells
	// (1-2)*refinementCountCoarse = firstInPlane: Index of the first coarse cell that is in the same Z=const plane
	// (2-3)*refinementCountCoarse = lastInPlane: Index of the last coarse cell that is in the same Z=const plane
	// (3-4)*refinementCountCoarse = firstInRow: Index of the first coarse cell that is in the same Z,Y=const row
	// (4-5)*refinementCountCoarse = lastInRow: Index of the last coarse cell that is in the same Z,Y=const row
	childMapArrayCoarse.setValue( -1 );
	auto cellLambda7 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int cellCoarse = multiFieldView[ index ]; 
		const int firstInPlane = multiFieldView[ 1*refinementCountCoarse + index ];
		const int lastInPlane = multiFieldView[ 2*refinementCountCoarse + index ];
		const int firstInRow = multiFieldView[ 3*refinementCountCoarse + index ];
		const int lastInRow = multiFieldView[ 4*refinementCountCoarse + index ];
		const int iCoarse = iViewCoarse[ cellCoarse ];
		const int jCoarse = kViewCoarse[ cellCoarse ];
		const int kCoarse = kViewCoarse[ cellCoarse ];
		for ( int kAdd = 0; kAdd <= 1; kAdd++ )
		{
			for ( int jAdd = 0; jAdd <= 1; jAdd++ )
			{
				for ( int iAdd = 0; iAdd <= 1; iAdd++ )
				{
					const int cellFine = getFinerGridIndex( index, firstInPlane, lastInPlane, firstInRow, lastInRow, iAdd, jAdd, kAdd );
					const int iFine = 2 * iCoarse + iAdd;
					const int jFine = 2 * jCoarse + jAdd;
					const int kFine = 2 * kCoarse + kAdd;
					iViewFine[ cellFine ] = iFine;
					jViewFine[ cellFine ] = jFine;
					kViewFine[ cellFine ] = kFine;
					parentMapViewFine[ cellFine ] = cellCoarse;
					if ( iAdd == 0 && jAdd == 0 && kAdd == 0 ) // bottom left fine cell within the coarse cell
						childMapViewCoarse[ cellCoarse ] = cellFine;
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, cellLambda7 );
}
