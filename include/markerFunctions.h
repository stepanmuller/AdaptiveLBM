#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"

void applyMarkersFromRayMap( BoolArrayType &markerArray, const rayMapStruct &rayMap, const GridStruct &Grid, const int &upperBound )
{
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	const IntArray3DType &rayMapArray = rayMap.rayMapArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	
	markerArray.setValue( false );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		int kStart, kEnd;
		for ( int startIndex = 0; startIndex < RAY_MAP_DEPTH; startIndex = startIndex + 2 )
		{
			kStart = rayMapView( iCell, jCell, startIndex );
			if ( kStart > kCell ) break;
			kEnd = rayMapView( iCell, jCell, startIndex + 1 );
			if ( kEnd > kCell )
			{
				markerView[ cell ] = true;
				return;
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void markFinestFluid( BoolArrayType &markerArray, const rayMapStruct &rayMap, const GridStruct &Grid, const int &upperBound )
{
	// marks a coarse grid based on a fine rayMapArray, result is 1 if at least one fine cell is 0 (fluid)
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	const IntArray3DType &rayMapArray = rayMap.rayMapArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	
	const int downsample = rayMapArray.getSizes()[0] / Grid.Info.cellCountX;
	
	markerArray.setValue( true );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCoarse = iView[ cell ];
		const int jCoarse = jView[ cell ];
		const int kCoarse = kView[ cell ];
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				for ( int startIndex = 0; startIndex < RAY_MAP_DEPTH; startIndex = startIndex + 2 )
				{
					kEnd = rayMapView( iFine, jFine, startIndex + 1 );
					if ( kEnd < kFineFirst ) continue;
					else if ( kEnd >= kFineFirst && kEnd <= kFineLast ) return;
					kStart = rayMapView( iFine, jFine, startIndex );
					if ( kStart <= kFineFirst ) break;
					else return;
				}
			}
		}
		markerView[ cell ] = false;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void markFinestFluid( BoolArrayType &markerArray, const rayMapStruct &rayMap, const SkeletonGridStruct &SkeletonGrid )
{
	// marks the top master grid based on a fine rayMapArray, result is 1 if at least one fine cell is 0 (fluid)
	const int cellCountX = SkeletonGrid.Info.cellCountX;
	const int cellCountY = SkeletonGrid.Info.cellCountY;
	// const int cellCountZ = SkeletonGrid.Info.cellCountZ; // this is not needed
	const int cellCount = SkeletonGrid.Info.cellCount;
	const IntArray3DType &rayMapArray = rayMap.rayMapArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	
	const int downsample = rayMapArray.getSizes()[0] / cellCountX;
	
	markerArray.setValue( true );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int kCoarse = cell / (cellCountX * cellCountY);
		const int remainder = cell % (cellCountX * cellCountY);
		const int jCoarse = remainder / cellCountX;
		const int iCoarse = remainder % cellCountX;
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				for ( int startIndex = 0; startIndex < RAY_MAP_DEPTH; startIndex = startIndex + 2 )
				{
					kEnd = rayMapView( iFine, jFine, startIndex + 1 );
					if ( kEnd < kFineFirst ) continue;
					else if ( kEnd >= kFineFirst && kEnd <= kFineLast ) return;
					kStart = rayMapView( iFine, jFine, startIndex );
					if ( kStart <= kFineFirst ) break;
					else return;
				}
			}
		}
		markerView[ cell ] = false;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}

void markFinestBounceback( BoolArrayType &markerArray, const rayMapStruct &rayMap, const GridStruct &Grid, const int &upperBound )
{
	// marks a coarse grid based on a fine rayMapArray, result is 1 if at least one fine cell is 1 (bounceback)
	const int cellCountX = Grid.Info.cellCountX;
	const int cellCountY = Grid.Info.cellCountY;
	const int cellCountZ = Grid.Info.cellCountZ;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	const IntArray3DType &rayMapArray = rayMap.rayMapArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	
	const int downsample = rayMapArray.getSizes()[0] / Grid.Info.cellCountX;
	
	markerArray.setValue( false );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCoarse = iView[ cell ];
		const int jCoarse = jView[ cell ];
		const int kCoarse = kView[ cell ];
		// early exit if we are on the boundary: 
		// we will automatically refine the boundary to prevent having interface cross a boundary condition
		if ( iCoarse == 0 || iCoarse == cellCountX-1 
				|| jCoarse == 0 || jCoarse == cellCountY-1 
				|| kCoarse == 0 || kCoarse == cellCountZ-1 ) 
			{
				markerView[ cell ] = true;
				return;
			}	
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				for ( int startIndex = 0; startIndex < RAY_MAP_DEPTH; startIndex = startIndex + 2 )
				{
					kStart = rayMapView( iFine, jFine, startIndex );
					if ( kStart > kFineLast ) break;
					else if ( kStart >= kFineFirst ) 
					{
						markerView[ cell ] = true;
						return;
					}
					kEnd = rayMapView( iFine, jFine, startIndex + 1 );
					if ( kEnd <= kFineFirst ) continue;
					else 
					{
						markerView[ cell ] = true;
						return;
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void markFinestBounceback( BoolArrayType &markerArray, const rayMapStruct &rayMap, const SkeletonGridStruct &SkeletonGrid )
{
	// marks a coarse grid based on a fine rayMapArray, result is 1 if at least one fine cell is 1 (bounceback)
	const int cellCountX = SkeletonGrid.Info.cellCountX;
	const int cellCountY = SkeletonGrid.Info.cellCountY;
	// const int cellCountZ = SkeletonGrid.Info.cellCountZ; // this is not needed
	const int cellCount = SkeletonGrid.Info.cellCount;
	const IntArray3DType &rayMapArray = rayMap.rayMapArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	
	const int downsample = rayMapArray.getSizes()[0] / cellCountX;
	
	markerArray.setValue( false );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int kCoarse = cell / (cellCountX * cellCountY);
		const int remainder = cell % (cellCountX * cellCountY);
		const int jCoarse = remainder / cellCountX;
		const int iCoarse = remainder % cellCountX;
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				for ( int startIndex = 0; startIndex < RAY_MAP_DEPTH; startIndex = startIndex + 2 )
				{
					kStart = rayMapView( iFine, jFine, startIndex );
					if ( kStart > kFineLast ) break;
					else if ( kStart >= kFineFirst ) 
					{
						markerView[ cell ] = true;
						return;
					}
					kEnd = rayMapView( iFine, jFine, startIndex + 1 );
					if ( kEnd <= kFineFirst ) continue;
					else 
					{
						markerView[ cell ] = true;
						return;
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
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
