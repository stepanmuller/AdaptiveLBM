#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"

void getNbrArrayForSkeleton( IntArrayType &nbrArray, const int jPlus, const int kPlus, const SkeletonGridStruct &SkeletonGrid )
{
	const int cellCount = SkeletonGrid.Info.cellCount;
	const int cellCountX = SkeletonGrid.Info.cellCountX;
	const int cellCountY = SkeletonGrid.Info.cellCountY;
	const int cellCountZ = SkeletonGrid.Info.cellCountZ;
	const int cellCountXY = cellCountX * cellCountY;
	auto nbrView = nbrArray.getView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int kCell = cell / cellCountXY;
		const int remainder = cell % cellCountXY;
		const int jCell = remainder / cellCountX;
		const int iCell = remainder % cellCountX;
		const int iNbr = iCell;
		int jNbr = jCell + jPlus;
		int kNbr = kCell + kPlus;
		if ( jNbr >= cellCountY ) jNbr = 0;
		if ( kNbr >= cellCountZ ) kNbr = 0;
		const int nbr = kNbr * cellCountXY + jNbr * cellCountX + iNbr;
		nbrView[ cell ] = nbr;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );
}

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
	auto isGeometricMarkerView = Grid.NBR.isGeometricMarkerArray.getView();
	
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
		
		isGeometricMarkerView(0, cell) = ( iView[iPlus]==iCell+1 && jView[iPlus]==jCell && kView[iPlus]==kCell );
		isGeometricMarkerView(1, cell) = ( iView[jPlus]==iCell && jView[jPlus]==jCell+1 && kView[jPlus]==kCell );
		isGeometricMarkerView(2, cell) = ( iView[ijPlus]==iCell+1 && jView[ijPlus]==jCell+1 && kView[ijPlus]==kCell );
		isGeometricMarkerView(3, cell) = ( iView[kPlus]==iCell && jView[kPlus]==jCell && kView[kPlus]==kCell+1 );
		isGeometricMarkerView(4, cell) = ( iView[ikPlus]==iCell+1 && jView[ikPlus]==jCell && kView[ikPlus]==kCell+1 );
		isGeometricMarkerView(5, cell) = ( iView[jkPlus]==iCell && jView[jkPlus]==jCell+1 && kView[jkPlus]==kCell+1 );
		isGeometricMarkerView(6, cell) = ( iView[ijkPlus]==iCell+1 && jView[ijkPlus]==jCell+1 && kView[ijkPlus]==kCell+1 );
		
		if ( markNegativeDirectionsToo )
		{
			const int jMinus = jMinusView[ cell ];
			const int kMinus = kMinusView[ cell ]; 
			const int iMinus = (cell > 0) ? cell - 1 : upperBound - 1;
			isGeometricMarkerView(7, cell) = ( iView[iMinus]==iCell-1 && jView[iMinus]==jCell && kView[iMinus]==kCell );
			isGeometricMarkerView(8, cell) = ( iView[jMinus]==iCell && jView[jMinus]==jCell-1 && kView[jMinus]==kCell );
			isGeometricMarkerView(9, cell) = ( iView[kMinus]==iCell && jView[kMinus]==jCell && kView[kMinus]==kCell-1 );
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void skipOneUnmarkedNbr( IntArrayType &nbrArray, const BoolArrayType &markerArray, const IntArrayType &nbrOldArray, BoolArrayType &finishedMarkerArray, const int &upperBound )
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

void skipAllUnmarkedNbr( IntArrayType &nbrArray, const BoolArrayType &markerArray, IntArrayType &intBuffer, BoolArrayType &markerBuffer, const int &upperBound )
{
	// Receives nbrArray, which also points on cells that are not marked
	// For each cell, while its neighbour is not marked, we want to travel up the neighbour list
	// to find the first marked neighbour
	IntArrayType &nbrOldArray = intBuffer;
	BoolArrayType &finishedMarkerArray = markerBuffer;
	
	finishedMarkerArray.setValue( false );
	int unfinishedCount = upperBound;
	
	nbrOldArray = nbrArray;
	
	while ( unfinishedCount > 0 )
	{
		nbrArray.swap( nbrOldArray ); // pointer swap without data travel
		skipOneUnmarkedNbr( nbrArray, markerArray, nbrOldArray, finishedMarkerArray, upperBound );
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
	auto isGeometricMarkerView = Grid.NBR.isGeometricMarkerArray.getView();
	
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
		for ( int q = 0; q < 7; q++ ) isGeometricMarker[q] = isGeometricMarkerView(q, cell);
		
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
	const IntArrayType &iArrayCoarse = GridCoarse.IJK.iArray;
	const IntArrayType &jArrayCoarse = GridCoarse.IJK.jArray;
	const IntArrayType &kArrayCoarse = GridCoarse.IJK.kArray;
	IntArrayType &childMapArrayCoarse = GridCoarse.childMapArray;
	IntArrayType &intBuffer1Coarse = GridCoarse.intBuffer1;
	IntArrayType &intBuffer2Coarse = GridCoarse.intBuffer2;
	IntArrayType &intBuffer3Coarse = GridCoarse.intBuffer3;
	BoolArrayType &refinementMarkerArrayCoarse = GridCoarse.refinementMarkerArray;
	BoolArrayType &markerBufferCoarse = GridCoarse.markerBuffer;
	// Some GridCoarse Views
	auto iViewCoarse = iArrayCoarse.getConstView();
	auto jViewCoarse = jArrayCoarse.getConstView();
	auto kViewCoarse = kArrayCoarse.getConstView();
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
		const int jCoarse = jViewCoarse[ cellCoarse ];
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
	
	// 8) Now we want to assemble neighbours of the fine grid
	// For that we will use neighbours on the coarse grid. 
	// However, not all cells of the coarse grid get refined, and so some coarse cells point to neighbours outside of the refinement area
	// To fine cells under those cells we wouldn't be able to assign a neighbour.
	// And so we run a correction. We take copy of each coarse neighbour array and skip unrefined cells so that the refinement cells
	// only point to neighbours that also belong to the refinement area. This way, every refined coarse cell will have a uniquely defined
	// refined neighbour, and on top of this we can assemble neighbours for the fine cells
	// Use coarse buffers 2 and 3
	IntArrayType &nbrSkippedArray = intBuffer2Coarse;
	IntArrayType &nbrBuffer = intBuffer3Coarse;
	auto nbrSkippedView = nbrSkippedArray.getView();
	// 8.1) Direction jPlus
	nbrSkippedArray = GridCoarse.NBR.jPlusArray;
	skipAllUnmarkedNbr( nbrSkippedArray, refinementMarkerArrayCoarse, nbrBuffer, markerBufferCoarse, cellCountCoarse );
	// We use the coarse jPlus neighbour to fill as many fine neighbours as possible
	auto jPlusLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// coarse cell information
		const int cellCoarse = multiFieldView[ index ]; 
		const int fip = multiFieldView[ 1*refinementCountCoarse + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountCoarse + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountCoarse + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountCoarse + index ];	// last in row
		// jPlus coarse neighbour information
		const int jPlusCoarse = nbrSkippedView[ cellCoarse ];
		const int jPlusIndex = refinedIndexView[ jPlusCoarse ];
		const int jPFip = multiFieldView[ 1*refinementCountCoarse + jPlusIndex ];  // first in plane
		const int jPLip = multiFieldView[ 2*refinementCountCoarse + jPlusIndex ];  // last in plane
		const int jPFir = multiFieldView[ 3*refinementCountCoarse + jPlusIndex ];  // first in row
		const int jPLir = multiFieldView[ 4*refinementCountCoarse + jPlusIndex ];  // last in row
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
		// Fill all jPlusFine neighbours we can -> here we can fill all 8
		jPlusViewFine[ fine000 ] = fine010;
		jPlusViewFine[ fine100 ] = fine110;
		jPlusViewFine[ fine010 ] = fine020;
		jPlusViewFine[ fine110 ] = fine120;
		jPlusViewFine[ fine001 ] = fine011;
		jPlusViewFine[ fine101 ] = fine111;
		jPlusViewFine[ fine011 ] = fine021;
		jPlusViewFine[ fine111 ] = fine121;
		// Fill all kPlusFine neighbours we can -> here we can fill 4
		kPlusViewFine[ fine000 ] = fine001;
		kPlusViewFine[ fine100 ] = fine101;
		kPlusViewFine[ fine010 ] = fine011;
		kPlusViewFine[ fine110 ] = fine111;
		// Fill all jkPlusFine neighbours we can -> here we can fill 4
		jkPlusViewFine[ fine000 ] = fine011;
		jkPlusViewFine[ fine100 ] = fine111;
		jkPlusViewFine[ fine010 ] = fine021;
		jkPlusViewFine[ fine110 ] = fine121;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, jPlusLambda );
	
	// 8.2) Direction kPlus
	nbrSkippedArray = GridCoarse.NBR.kPlusArray;
	skipAllUnmarkedNbr( nbrSkippedArray, refinementMarkerArrayCoarse, nbrBuffer, markerBufferCoarse, cellCountCoarse );
	// We use the coarse kPlus neighbour to fill as many fine neighbours as possible
	auto kPlusLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// coarse cell information
		const int cellCoarse = multiFieldView[ index ]; 
		const int fip = multiFieldView[ 1*refinementCountCoarse + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountCoarse + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountCoarse + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountCoarse + index ];	// last in row
		// jPlus coarse neighbour information
		const int kPlusCoarse = nbrSkippedView[ cellCoarse ];
		const int kPlusIndex = refinedIndexView[ kPlusCoarse ];
		const int kPFip = multiFieldView[ 1*refinementCountCoarse + kPlusIndex ];  // first in plane
		const int kPLip = multiFieldView[ 2*refinementCountCoarse + kPlusIndex ];  // last in plane
		const int kPFir = multiFieldView[ 3*refinementCountCoarse + kPlusIndex ];  // first in row
		const int kPLir = multiFieldView[ 4*refinementCountCoarse + kPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in coarse cell
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine001 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine101 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, left 4 fine cells (kAdd=0) that lie in kPlus coarse neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine002 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine102 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine012 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine112 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		// Fill all kPlusFine neighbours we can -> here we can fill 4
		kPlusViewFine[ fine001 ] = fine002;
		kPlusViewFine[ fine101 ] = fine102;
		kPlusViewFine[ fine011 ] = fine012;
		kPlusViewFine[ fine111 ] = fine112;
		// Fill all jkPlusFine neighbours we can -> here we can fill 2
		jkPlusViewFine[ fine001 ] = fine012;
		jkPlusViewFine[ fine101 ] = fine112;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, kPlusLambda );
	
	// 8.3) Direction jkPlus
	nbrSkippedArray = GridCoarse.NBR.jkPlusArray;
	skipAllUnmarkedNbr( nbrSkippedArray, refinementMarkerArrayCoarse, nbrBuffer, markerBufferCoarse, cellCountCoarse );
	// We use the coarse jkPlus neighbour to fill remaining fine neighbours
	auto jkPlusLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// coarse cell information
		const int cellCoarse = multiFieldView[ index ]; 
		const int fip = multiFieldView[ 1*refinementCountCoarse + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountCoarse + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountCoarse + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountCoarse + index ];	// last in row
		// jPlus coarse neighbour information
		const int jkPlusCoarse = nbrSkippedView[ cellCoarse ];
		const int jkPlusIndex = refinedIndexView[ jkPlusCoarse ];
		const int jkPFip = multiFieldView[ 1*refinementCountCoarse + jkPlusIndex ];  // first in plane
		const int jkPLip = multiFieldView[ 2*refinementCountCoarse + jkPlusIndex ];  // last in plane
		const int jkPFir = multiFieldView[ 3*refinementCountCoarse + jkPlusIndex ];  // first in row
		const int jkPLir = multiFieldView[ 4*refinementCountCoarse + jkPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in coarse cell
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, 2 bottom left fine cells (kAdd=0, jAdd=0) that lie in kPlus coarse neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine022 = getFinerGridIndex( jkPlusIndex, jkPFip, jkPLip, jkPFir, jkPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine122 = getFinerGridIndex( jkPlusIndex, jkPFip, jkPLip, jkPFir, jkPLir, iAdd, jAdd, kAdd );
		// Fill 2 last remaining jkPlusFine neighbours
		jkPlusViewFine[ fine011 ] = fine022;
		jkPlusViewFine[ fine111 ] = fine122;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountCoarse, jkPlusLambda );
	
	// 9) Last little step. We have just filled all fine neighbours. Now mark their geometric validity.
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
	BoolArrayType &markerBufferSkeleton = SkeletonGrid.markerBuffer;
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
	
	// 7) write new IJK for the fine grid in already sorted order thanks to knowing firstInPlane...lastInRow information
	// also fill the parentMapArray of the fine grid (which skeleton cell created the fine cell)
	// skeleton grid does not need the childMapArray
	// multiField contains:
	// (0-1)*refinementCountSkeleton = refinedParentList: Sorted indexes of refined skeleton cells
	// (1-2)*refinementCountSkeleton = firstInPlane: Index of the first skeleton cell that is in the same Z=const plane
	// (2-3)*refinementCountSkeleton = lastInPlane: Index of the last skeleton cell that is in the same Z=const plane
	// (3-4)*refinementCountSkeleton = firstInRow: Index of the first skeleton cell that is in the same Z,Y=const row
	// (4-5)*refinementCountSkeleton = lastInRow: Index of the last skeleton cell that is in the same Z,Y=const row
	auto cellLambda7 = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int cellSkeleton = multiFieldView[ index ]; 
		const int firstInPlane = multiFieldView[ 1*refinementCountSkeleton + index ];
		const int lastInPlane = multiFieldView[ 2*refinementCountSkeleton + index ];
		const int firstInRow = multiFieldView[ 3*refinementCountSkeleton + index ];
		const int lastInRow = multiFieldView[ 4*refinementCountSkeleton + index ];
		const int kSkeleton = cellSkeleton / cellCountXYSkeleton;
		const int remainder = cellSkeleton % cellCountXYSkeleton;
		const int jSkeleton = remainder / cellCountXSkeleton;
		const int iSkeleton = remainder % cellCountXSkeleton;
		for ( int kAdd = 0; kAdd <= 1; kAdd++ )
		{
			for ( int jAdd = 0; jAdd <= 1; jAdd++ )
			{
				for ( int iAdd = 0; iAdd <= 1; iAdd++ )
				{
					const int cellFine = getFinerGridIndex( index, firstInPlane, lastInPlane, firstInRow, lastInRow, iAdd, jAdd, kAdd );
					const int iFine = 2 * iSkeleton + iAdd;
					const int jFine = 2 * jSkeleton + jAdd;
					const int kFine = 2 * kSkeleton + kAdd;
					iViewFine[ cellFine ] = iFine;
					jViewFine[ cellFine ] = jFine;
					kViewFine[ cellFine ] = kFine;
					parentMapViewFine[ cellFine ] = cellSkeleton;
					/* commenting out: skeleton grid does not need the childMapArray
					if ( iAdd == 0 && jAdd == 0 && kAdd == 0 ) // bottom left fine cell within the skeleton cell
						childMapViewSkeleton[ cellSkeleton ] = cellFine;
					*/
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, cellLambda7 );
	
	// 8) Now we want to assemble neighbours of the fine grid
	// For that we will use neighbours on the skeleton grid. 
	// However, not all cells of the skeleton grid get refined, and so some skeleton cells point to neighbours outside of the refinement area
	// To fine cells under those cells we wouldn't be able to assign a neighbour.
	// And so we run a correction. We take copy of each skeleton neighbour array and skip unrefined cells so that the refinement cells
	// only point to neighbours that also belong to the refinement area. This way, every refined skeleton cell will have a uniquely defined
	// refined neighbour, and on top of this we can assemble neighbours for the fine cells
	// Use skeleton buffers 2 and 3
	IntArrayType &nbrSkippedArray = intBuffer2Skeleton;
	IntArrayType &nbrBuffer = intBuffer3Skeleton;
	auto nbrSkippedView = nbrSkippedArray.getView();
	// 8.1) Direction jPlus
	// Skeleton does not hold neighbours, so here we must temporarily create jPlus for the Skeleton grid
	int jPlus, kPlus;
	jPlus = 1; kPlus = 0;
	getNbrArrayForSkeleton( nbrSkippedArray, jPlus, kPlus, SkeletonGrid );
	skipAllUnmarkedNbr( nbrSkippedArray, refinementMarkerArraySkeleton, nbrBuffer, markerBufferSkeleton, cellCountSkeleton );
	// We use the skeleton jPlus neighbour to fill as many fine neighbours as possible
	auto jPlusLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// skeleton cell information
		const int cellSkeleton = multiFieldView[ index ]; 
		const int fip = multiFieldView[ 1*refinementCountSkeleton + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountSkeleton + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountSkeleton + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountSkeleton + index ];	// last in row
		// jPlus skeleton neighbour information
		const int jPlusSkeleton = nbrSkippedView[ cellSkeleton ];
		const int jPlusIndex = refinedIndexView[ jPlusSkeleton ];
		const int jPFip = multiFieldView[ 1*refinementCountSkeleton + jPlusIndex ];  // first in plane
		const int jPLip = multiFieldView[ 2*refinementCountSkeleton + jPlusIndex ];  // last in plane
		const int jPFir = multiFieldView[ 3*refinementCountSkeleton + jPlusIndex ];  // first in row
		const int jPLir = multiFieldView[ 4*refinementCountSkeleton + jPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in skeleton cell
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine000 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine100 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine010 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine110 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine001 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine101 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, bottom 4 fine cells (jAdd=0) that lie in jPlus skeleton neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine020 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine120 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine021 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine121 = getFinerGridIndex( jPlusIndex, jPFip, jPLip, jPFir, jPLir, iAdd, jAdd, kAdd );
		// Fill all jPlusFine neighbours we can -> here we can fill all 8
		jPlusViewFine[ fine000 ] = fine010;
		jPlusViewFine[ fine100 ] = fine110;
		jPlusViewFine[ fine010 ] = fine020;
		jPlusViewFine[ fine110 ] = fine120;
		jPlusViewFine[ fine001 ] = fine011;
		jPlusViewFine[ fine101 ] = fine111;
		jPlusViewFine[ fine011 ] = fine021;
		jPlusViewFine[ fine111 ] = fine121;
		// Fill all kPlusFine neighbours we can -> here we can fill 4
		kPlusViewFine[ fine000 ] = fine001;
		kPlusViewFine[ fine100 ] = fine101;
		kPlusViewFine[ fine010 ] = fine011;
		kPlusViewFine[ fine110 ] = fine111;
		// Fill all jkPlusFine neighbours we can -> here we can fill 4
		jkPlusViewFine[ fine000 ] = fine011;
		jkPlusViewFine[ fine100 ] = fine111;
		jkPlusViewFine[ fine010 ] = fine021;
		jkPlusViewFine[ fine110 ] = fine121;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, jPlusLambda );
	
	// 8.2) Direction kPlus
	jPlus = 0; kPlus = 1;
	getNbrArrayForSkeleton( nbrSkippedArray, jPlus, kPlus, SkeletonGrid );
	skipAllUnmarkedNbr( nbrSkippedArray, refinementMarkerArraySkeleton, nbrBuffer, markerBufferSkeleton, cellCountSkeleton );
	// We use the skeleton kPlus neighbour to fill as many fine neighbours as possible
	auto kPlusLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// skeleton cell information
		const int cellSkeleton = multiFieldView[ index ]; 
		const int fip = multiFieldView[ 1*refinementCountSkeleton + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountSkeleton + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountSkeleton + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountSkeleton + index ];	// last in row
		// jPlus skeleton neighbour information
		const int kPlusSkeleton = nbrSkippedView[ cellSkeleton ];
		const int kPlusIndex = refinedIndexView[ kPlusSkeleton ];
		const int kPFip = multiFieldView[ 1*refinementCountSkeleton + kPlusIndex ];  // first in plane
		const int kPLip = multiFieldView[ 2*refinementCountSkeleton + kPlusIndex ];  // last in plane
		const int kPFir = multiFieldView[ 3*refinementCountSkeleton + kPlusIndex ];  // first in row
		const int kPLir = multiFieldView[ 4*refinementCountSkeleton + kPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in skeleton cell
		iAdd = 0; jAdd = 0; kAdd = 1; const int fine001 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 1; const int fine101 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, left 4 fine cells (kAdd=0) that lie in kPlus skeleton neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine002 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine102 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 0; jAdd = 1; kAdd = 0; const int fine012 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 0; const int fine112 = getFinerGridIndex( kPlusIndex, kPFip, kPLip, kPFir, kPLir, iAdd, jAdd, kAdd );
		// Fill all kPlusFine neighbours we can -> here we can fill 4
		kPlusViewFine[ fine001 ] = fine002;
		kPlusViewFine[ fine101 ] = fine102;
		kPlusViewFine[ fine011 ] = fine012;
		kPlusViewFine[ fine111 ] = fine112;
		// Fill all jkPlusFine neighbours we can -> here we can fill 2
		jkPlusViewFine[ fine001 ] = fine012;
		jkPlusViewFine[ fine101 ] = fine112;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, kPlusLambda );
	
	// 8.3) Direction jkPlus
	jPlus = 1; kPlus = 1;
	getNbrArrayForSkeleton( nbrSkippedArray, jPlus, kPlus, SkeletonGrid );
	skipAllUnmarkedNbr( nbrSkippedArray, refinementMarkerArraySkeleton, nbrBuffer, markerBufferSkeleton, cellCountSkeleton );
	// We use the skeleton jkPlus neighbour to fill remaining fine neighbours
	auto jkPlusLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		// skeleton cell information
		const int cellSkeleton = multiFieldView[ index ]; 
		const int fip = multiFieldView[ 1*refinementCountSkeleton + index ];  // first in plane
		const int lip = multiFieldView[ 2*refinementCountSkeleton + index ];  // last in plane
		const int fir = multiFieldView[ 3*refinementCountSkeleton + index ];  // first in row
		const int lir = multiFieldView[ 4*refinementCountSkeleton + index ];	// last in row
		// jPlus skeleton neighbour information
		const int jkPlusSkeleton = nbrSkippedView[ cellSkeleton ];
		const int jkPlusIndex = refinedIndexView[ jkPlusSkeleton ];
		const int jkPFip = multiFieldView[ 1*refinementCountSkeleton + jkPlusIndex ];  // first in plane
		const int jkPLip = multiFieldView[ 2*refinementCountSkeleton + jkPlusIndex ];  // last in plane
		const int jkPFir = multiFieldView[ 3*refinementCountSkeleton + jkPlusIndex ];  // first in row
		const int jkPLir = multiFieldView[ 4*refinementCountSkeleton + jkPlusIndex ];  // last in row
		// Now calculate indexes of all relevant fine cells
		int iAdd, jAdd, kAdd;
		// First, fine cells that lie in skeleton cell
		iAdd = 0; jAdd = 1; kAdd = 1; const int fine011 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 1; kAdd = 1; const int fine111 = getFinerGridIndex( index, fip, lip, fir, lir, iAdd, jAdd, kAdd );
		// Next, 2 bottom left fine cells (kAdd=0, jAdd=0) that lie in kPlus skeleton neighbour
		iAdd = 0; jAdd = 0; kAdd = 0; const int fine022 = getFinerGridIndex( jkPlusIndex, jkPFip, jkPLip, jkPFir, jkPLir, iAdd, jAdd, kAdd );
		iAdd = 1; jAdd = 0; kAdd = 0; const int fine122 = getFinerGridIndex( jkPlusIndex, jkPFip, jkPLip, jkPFir, jkPLir, iAdd, jAdd, kAdd );
		// Fill 2 last remaining jkPlusFine neighbours
		jkPlusViewFine[ fine011 ] = fine022;
		jkPlusViewFine[ fine111 ] = fine122;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, refinementCountSkeleton, jkPlusLambda );
	
	// 9) Last little step. We have just filled all fine neighbours. Now mark their geometric validity.
	const bool markNegativeDirectionsToo = false;
	markGeometricNBR( GridFine, markNegativeDirectionsToo, cellCountFullFine );
}

void rebuildGrid( std::vector<GridStruct> &grids, const VoxelizerStruct &Voxelizer, const int level )
// Consider grids 0, 1, 2, 3 where 3 is the finest. We want to rebuild grids 2, 3 -> we call this function on 2 (level=2) which recursively calls it on all levels below.
{
	GridStruct &Grid = grids[ level ];
	InfoStruct &Info = Grid.Info;
	if ( level == 0 ) SkeletonGridStruct GridCoarse = Grid.SkeletonGrid;
	else GridCoarse = grids[ level - 1 ];
	
	Info.cellCountOld = Info.cellCount;
}
