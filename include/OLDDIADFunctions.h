#include <TNL/Algorithms/sort.h>
#include "../include/STLFunctions.h"

// Builds IJK from Info by filling the whole domain	
void buildIJKFromInfo( IJKArrayStruct &IJK, InfoStruct &Info )
{
	IJK.iArray.setSize( Info.cellCount );
	IJK.jArray.setSize( Info.cellCount );
	IJK.kArray.setSize( Info.cellCount );
	auto iView = IJK.iArray.getView();
	auto jView = IJK.jArray.getView();
	auto kView = IJK.kArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		int iCell, jCell, kCell;
		getIJKCellIndex( cell, iCell, jCell, kCell, Info );
		iView[ cell ] = iCell;
		jView[ cell ] = jCell;
		kView[ cell ] = kCell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, Info.cellCount, cellLambda );
}

void shiftIJK( IJKArrayStruct &IJK, const int cx, const int cy, const int cz )
{
	auto iView = IJK.iArray.getView();
	auto jView = IJK.jArray.getView();
	auto kView = IJK.kArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		iView[ cell ] = iView[ cell ] + cx;
		jView[ cell ] = jView[ cell ] + cy;
		kView[ cell ] = kView[ cell ] + cz;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, IJK.iArray.getSize(), cellLambda );
}

void shiftIJKPeriodic( IJKArrayStruct &IJK, const int cx, const int cy, const int cz, InfoStruct &Info )
{
	auto iView = IJK.iArray.getView();
	auto jView = IJK.jArray.getView();
	auto kView = IJK.kArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		iView[ cell ] = (iView[ cell ] + cx) % Info.cellCountX;
		jView[ cell ] = (jView[ cell ] + cy) % Info.cellCountY;
		kView[ cell ] = (kView[ cell ] + cz) % Info.cellCountZ;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, IJK.iArray.getSize(), cellLambda );
}

// sort IJK arrays as ascending k, j, i so that k changes the slowest
void sortIJK( IJKArrayStruct &IJK )
{
	const int cellCount = IJK.iArray.getSize();
	auto iView = IJK.iArray.getView();
	auto jView = IJK.jArray.getView();
	auto kView = IJK.kArray.getView();

    auto comparisonLambda = [=] __cuda_callable__ ( const size_t a, const size_t b )
	{
		if ( kView[ a ] < kView[ b ] ) return true;
		else if ( kView[ a ] > kView[ b ] ) return false;
		else
		{
			if ( jView[ a ] < jView[ b ] ) return true;
			else if ( jView[ a ] > jView[ b ] ) return false;
			else
			{
				if ( iView[ a ] < iView[ b ] ) return true;
				else if ( iView[ a ] > iView[ b ] ) return false;
				else return false;
			}
		}
	};
	auto swapLambda = [=] __cuda_callable__ ( const size_t a, const size_t b ) mutable
	{
		TNL::swap( iView[ a ], iView[ b ] );
		TNL::swap( jView[ a ], jView[ b ] );
		TNL::swap( kView[ a ], kView[ b ] );
	};
	TNL::Algorithms::sort<TNL::Devices::Cuda, size_t>( 0, cellCount, comparisonLambda, swapLambda );
}

// sort IJK arrays as ascending k, j, i so that k changes the slowest, also sort the bounceback array
void sortIJK( IJKArrayStruct &IJK, BoolArrayType &markerArray, BoolArrayType &markerArray2 )
{
	const int cellCount = IJK.iArray.getSize();
	auto iView = IJK.iArray.getView();
	auto jView = IJK.jArray.getView();
	auto kView = IJK.kArray.getView();
	auto markerView = markerArray.getView();
	auto markerView2 = markerArray2.getView();

    auto comparisonLambda = [=] __cuda_callable__ ( const size_t a, const size_t b )
	{
		if ( kView[ a ] < kView[ b ] ) return true;
		else if ( kView[ a ] > kView[ b ] ) return false;
		else
		{
			if ( jView[ a ] < jView[ b ] ) return true;
			else if ( jView[ a ] > jView[ b ] ) return false;
			else
			{
				if ( iView[ a ] < iView[ b ] ) return true;
				else if ( iView[ a ] > iView[ b ] ) return false;
				else return false;
			}
		}
	};
	auto swapLambda = [=] __cuda_callable__ ( const size_t a, const size_t b ) mutable
	{
		TNL::swap( iView[ a ], iView[ b ] );
		TNL::swap( jView[ a ], jView[ b ] );
		TNL::swap( kView[ a ], kView[ b ] );
		TNL::swap( markerView[ a ], markerView[ b ] );
		TNL::swap( markerView2[ a ], markerView2[ b ] );
	};
	TNL::Algorithms::sort<TNL::Devices::Cuda, size_t>( 0, cellCount, comparisonLambda, swapLambda );
}

// final sort for finest grid
void sortFinestGrid( DIADGridStruct &Grid, BoolArrayType &keepCellMarkerArray )
{
	const int cellCount = Grid.Info.cellCount;
	auto iView = Grid.IJK.iArray.getView();
	auto jView = Grid.IJK.jArray.getView();
	auto kView = Grid.IJK.kArray.getView();
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getView();
	auto keepCellMarkerView = keepCellMarkerArray.getView();

    auto comparisonLambda = [=] __cuda_callable__ ( const size_t a, const size_t b )
	{
		if ( keepCellMarkerView[ a ] > keepCellMarkerView[ b ] ) return true;
		else if ( keepCellMarkerView[ a ] < keepCellMarkerView[ b ] ) return false;
		else
		{
			if ( kView[ a ] < kView[ b ] ) return true;
			else if ( kView[ a ] > kView[ b ] ) return false;
			else
			{
				if ( jView[ a ] < jView[ b ] ) return true;
				else if ( jView[ a ] > jView[ b ] ) return false;
				else
				{
					if ( iView[ a ] < iView[ b ] ) return true;
					else if ( iView[ a ] > iView[ b ] ) return false;
					else return false;
				}
			}
		}
	};
	auto swapLambda = [=] __cuda_callable__ ( const size_t a, const size_t b ) mutable
	{
		TNL::swap( iView[ a ], iView[ b ] );
		TNL::swap( jView[ a ], jView[ b ] );
		TNL::swap( kView[ a ], kView[ b ] );
		TNL::swap( keepCellMarkerView[ a ], keepCellMarkerView[ b ] );
		TNL::swap( bouncebackMarkerView[ a ], bouncebackMarkerView[ b ] );
	};
	TNL::Algorithms::sort<TNL::Devices::Cuda, size_t>( 0, cellCount, comparisonLambda, swapLambda );
}

// sort for all coarse grid levels
void sortCoarseGrid( DIADGridStruct &Grid, BoolArrayType &keepCellMarkerArray, BoolArrayType &fineToCoarseMarkerArray, BoolArrayType &coarseToFineMarkerArray )
{
	const int cellCount = Grid.Info.cellCount;
	auto iView = Grid.IJK.iArray.getView();
	auto jView = Grid.IJK.jArray.getView();
	auto kView = Grid.IJK.kArray.getView();
	auto keepCellMarkerView = keepCellMarkerArray.getView();
	auto fineToCoarseMarkerView = fineToCoarseMarkerArray.getView();
	auto coarseToFineMarkerView = coarseToFineMarkerArray.getView();
	auto enforceInterfaceBouncebackView = Grid.enforceInterfaceBounceback.getView();
	auto enforceInterfaceFluidView = Grid.enforceInterfaceFluid.getView();
	
	bool useEnforces = false;
	if ( Grid.enforceInterfaceBounceback.getSize() > 0 )
	{
		useEnforces = true;
	}

    auto comparisonLambda = [=] __cuda_callable__ ( const size_t a, const size_t b )
	{
		if ( keepCellMarkerView[ a ] > keepCellMarkerView[ b ] ) return true;
		else if ( keepCellMarkerView[ a ] < keepCellMarkerView[ b ] ) return false;
		else
		{
			if ( kView[ a ] < kView[ b ] ) return true;
			else if ( kView[ a ] > kView[ b ] ) return false;
			else
			{
				if ( jView[ a ] < jView[ b ] ) return true;
				else if ( jView[ a ] > jView[ b ] ) return false;
				else
				{
					if ( iView[ a ] < iView[ b ] ) return true;
					else if ( iView[ a ] > iView[ b ] ) return false;
					else return false;
				}
			}
		}
	};
	auto swapLambda = [=] __cuda_callable__ ( const size_t a, const size_t b ) mutable
	{
		TNL::swap( iView[ a ], iView[ b ] );
		TNL::swap( jView[ a ], jView[ b ] );
		TNL::swap( kView[ a ], kView[ b ] );
		TNL::swap( keepCellMarkerView[ a ], keepCellMarkerView[ b ] );
		TNL::swap( fineToCoarseMarkerView[ a ], fineToCoarseMarkerView[ b ] );
		TNL::swap( coarseToFineMarkerView[ a ], coarseToFineMarkerView[ b ] );
		if ( useEnforces )
		{
			TNL::swap( enforceInterfaceBouncebackView[ a ], enforceInterfaceBouncebackView[ b ] );
			TNL::swap( enforceInterfaceFluidView[ a ], enforceInterfaceFluidView[ b ] );
		}
	};
	TNL::Algorithms::sort<TNL::Devices::Cuda, size_t>( 0, cellCount, comparisonLambda, swapLambda );
}

int countMarkerCells( BoolArrayType &markerArray )
{
	const int cellCount = markerArray.getSize();
	auto markerView = markerArray.getView();
	
	auto fetch = [ = ] __cuda_callable__( const int cell )
		{
			return (int)markerView[ cell ];
		};
		auto reduction = [] __cuda_callable__( const int& a, const int& b )
		{
			return a + b;
		};	
	int result = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, cellCount, fetch, reduction, 0 );
	return result;
}

int countInvalidIndexes( IntArrayType &intArray )
{
	const int cellCount = intArray.getSize();
	auto intView = intArray.getView();
	
	auto fetch = [ = ] __cuda_callable__( const int cell )
		{
			if (intView[ cell ] < 0) return 1;
			else return 0;
		};
		auto reduction = [] __cuda_callable__( const int& a, const int& b )
		{
			return a + b;
		};	
	int result = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, cellCount, fetch, reduction, 0 );
	return result;
}

void buildOutletCellArray( DIADGridStruct &Grid, BoolArrayType &markerArray )
{
	const int cellCount = Grid.Info.cellCount;
	const int outletCellCount = countMarkerCells( markerArray );
	
	Grid.outletCellArray.setSize( Grid.Info.cellCount );
	
	auto markerView = markerArray.getView();
	auto outletCellView = Grid.outletCellArray.getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		outletCellView[ cell ] = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, Grid.Info.cellCount, cellLambda );

    auto comparisonLambda = [=] __cuda_callable__ ( const size_t a, const size_t b )
	{
		if ( markerView[ a ] > markerView[ b ] ) return true;
		else if ( markerView[ a ] < markerView[ b ] ) return false;
		else
		{
			if ( outletCellView[ a ] < outletCellView[ b ] ) return true;
			else return false;
		}
	};
	auto swapLambda = [=] __cuda_callable__ ( const size_t a, const size_t b ) mutable
	{
		TNL::swap( markerView[ a ], markerView[ b ] );
		TNL::swap( outletCellView[ a ], outletCellView[ b ] );
	};
	TNL::Algorithms::sort<TNL::Devices::Cuda, size_t>( 0, Grid.Info.cellCount, comparisonLambda, swapLambda );
	Grid.outletCellArray.resize( outletCellCount );
}

// Receives sorted IJKSource that !must! be already sorted with key k, j, i so that k index changes the slowest
// For each Wanted, find a matching cell in Source. If its found, write its index to resultArray. If there is no such cell, write -1.		
void findMatchingIJKIndex( IJKArrayStruct &Wanted, IJKArrayStruct &Source, IntArrayType &resultArray )
{
	const int wantedCellCount = Wanted.iArray.getSize();
	const int sourceCellCount = Source.iArray.getSize();
	
	auto iWantedView = Wanted.iArray.getConstView();
	auto jWantedView = Wanted.jArray.getConstView();
	auto kWantedView = Wanted.kArray.getConstView();
	auto iSourceView = Source.iArray.getConstView();
	auto jSourceView = Source.jArray.getConstView();
	auto kSourceView = Source.kArray.getConstView();
	auto resultView = resultArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cellWanted ) mutable
	{
		if ( resultView[ cellWanted ] >= 0 ) return; // Do not overwrite the cell if its already valid from before
		
		const int iWanted = iWantedView[ cellWanted ];
		const int jWanted = jWantedView[ cellWanted ];
		const int kWanted = kWantedView[ cellWanted ];
		
		int start = 0;
		int end = sourceCellCount;
		int result = -1;
		
		// Search for k, j, and i in a single binary search
		while ( end > start )
		{
			int half = start + ( end - start ) / 2;
			int kHalf = kSourceView[ half ];
			int jHalf = jSourceView[ half ];
			int iHalf = iSourceView[ half ];
			
			if ( kHalf == kWanted && jHalf == jWanted && iHalf == iWanted )
			{
				result = half;
				break;
			}
			
			// Move the end bound if k is too big, OR if k matches but j is too big, OR if k and j match but i is too big
			if ( kHalf > kWanted || 
			   ( kHalf == kWanted && jHalf > jWanted ) || 
			   ( kHalf == kWanted && jHalf == jWanted && iHalf > iWanted ) ) 
			{
				end = half;
			}
			else 
			{
				start = half + 1;
			}
		}
		if ( result == -1 && start < sourceCellCount )
		{
			if ( kSourceView[ start ] == kWanted && jSourceView[ start ] == jWanted && iSourceView[ start ] == iWanted )
			{
				result = start;
			}
		}
		resultView[ cellWanted ] = result;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, wantedCellCount, cellLambda );
}

// shift by cx, cy, cz from the starter cell until another cell is found
void findPeriodicNeighbourIJKIndex( IJKArrayStruct &IJK, IntArrayType &resultArray, const int cx, const int cy, const int cz, InfoStruct &Info )
{
	const int cellCount = IJK.iArray.getSize();
	
	auto iView = IJK.iArray.getConstView();
	auto jView = IJK.jArray.getConstView();
	auto kView = IJK.kArray.getConstView();
	auto resultView = resultArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		// Starting coordinates for this specific thread
		int targetI = iView[ cell ];
		int targetJ = jView[ cell ];
		int targetK = kView[ cell ];
		
		int finalResultIndex = -1;
		
		// Keep shifting and searching until we find a match
		while ( finalResultIndex == -1 )
		{
			// Shift periodically
			targetI = ((targetI + cx) % Info.cellCountX + Info.cellCountX) % Info.cellCountX;
			targetJ = ((targetJ + cy) % Info.cellCountY + Info.cellCountY) % Info.cellCountY;
			targetK = ((targetK + cz) % Info.cellCountZ + Info.cellCountZ) % Info.cellCountZ;
			
			// Reset binary search bounds for the new target
			int start = 0;
			int end = cellCount;
			
			// Standard binary search
			while ( end > start )
			{
				int half = start + ( end - start ) / 2;
				int kHalf = kView[ half ];
				int jHalf = jView[ half ];
				int iHalf = iView[ half ];
				
				if ( kHalf == targetK && jHalf == targetJ && iHalf == targetI )
				{
					finalResultIndex = half;
					break;
				}
				
				if ( kHalf > targetK || 
				   ( kHalf == targetK && jHalf > targetJ ) || 
				   ( kHalf == targetK && jHalf == targetJ && iHalf > targetI ) ) 
				{
					end = half;
				}
				else 
				{
					start = half + 1;
				}
			}
			
			// Check the final boundary condition of the binary search
			if ( finalResultIndex == -1 && start < cellCount )
			{
				if ( kView[ start ] == targetK && jView[ start ] == targetJ && iView[ start ] == targetI )
				{
					finalResultIndex = start;
				}
			}
		}
		resultView[ cell ] = finalResultIndex;
	};
	
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, cellCount, cellLambda );
}

// Applies the bounceback marker onto an array using the getMarkers function
void applyBouncebackMarkerFromFunction( BoolArrayType &markerArray, IJKArrayStruct &IJK, InfoStruct &Info )
{
	auto markerArrayView = markerArray.getView();
	
	auto iArrayView = IJK.iArray.getView();
	auto jArrayView = IJK.jArray.getView();
	auto kArrayView = IJK.kArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iArrayView[ cell ];
		const int jCell = jArrayView[ cell ];
		const int kCell = kArrayView[ cell ];
		MarkerStruct Marker;
		Marker.bounceback = markerArrayView[ cell ];
		getMarkers( iCell, jCell, kCell, Marker, Info );
		markerArrayView( cell ) = Marker.bounceback || Marker.movingBounceback;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
}

// Applies the refinement marker onto an array using the getMarkers function
void applyRefinementMarkerFromFunction( BoolArrayType &markerArray, IJKArrayStruct &IJK, InfoStruct &Info )
{
	auto markerArrayView = markerArray.getView();
	
	auto iArrayView = IJK.iArray.getView();
	auto jArrayView = IJK.jArray.getView();
	auto kArrayView = IJK.kArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iArrayView[ cell ];
		const int jCell = jArrayView[ cell ];
		const int kCell = kArrayView[ cell ];
		MarkerStruct Marker;
		Marker.refinement = markerArrayView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		markerArrayView( cell ) = Marker.refinement;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
}

// Applies the outlet marker onto an array using the getMarkers function
void applyOutletMarkerFromFunction( BoolArrayType &markerArray, IJKArrayStruct &IJK, InfoStruct &Info )
{
	auto markerArrayView = markerArray.getView();
	
	auto iArrayView = IJK.iArray.getView();
	auto jArrayView = IJK.jArray.getView();
	auto kArrayView = IJK.kArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iArrayView[ cell ];
		const int jCell = jArrayView[ cell ];
		const int kCell = kArrayView[ cell ];
		MarkerStruct Marker;
		getMarkers( iCell, jCell, kCell, Marker, Info );
		markerArrayView( cell ) = Marker.nonReflectiveOutlet;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
}

// Do a repetitive fluid marking in order to find which of our coarse cells could possibly contain 
// at least one fluid cell at ANY finer resolution. 
// Similarly, find which of our coarse cells could possibly contain at least one bounceback cell at ANY finer resolution.
// To do this we will be repeatedly temporarily shifting the origin of our grid, to simulate being on all finer grids.
void markFinestFluidBounceback( BoolArrayType &fluidMarkerArray, BoolArrayType &finestBouncebackMarkerArray, IJKArrayStruct &IJK, std::vector<STLStruct> STLs, InfoStruct &Info )
{
	const float oxOriginal = Info.ox;
	const float oyOriginal = Info.oy;
	const float ozOriginal = Info.oz;
	
	fluidMarkerArray.setValue( 0 );
	finestBouncebackMarkerArray.setValue( 0 );
	
	BoolArrayType markerArray = BoolArrayType( Info.cellCount );
	markerArray.setValue( 0 );
	BoolArrayType markerArraySTL = BoolArrayType( Info.cellCount );
	markerArraySTL.setValue( 0 );

	// Loop through all finer levels
	for ( int targetLevel = Info.gridID; targetLevel < gridLevelCount; targetLevel++ )
	{
		// Calculate shift parameters dynamically for the current target level
		const int shiftCount = std::pow(2, (targetLevel - Info.gridID));
		const float targetRes = Info.res / shiftCount;
		const float shiftStart = - 0.5f * Info.res + 0.5f * targetRes;

		for ( int shiftCountX = 0; shiftCountX < shiftCount; shiftCountX++ )
		{
			for ( int shiftCountY = 0; shiftCountY < shiftCount; shiftCountY++ )
			{
				for ( int shiftCountZ = 0; shiftCountZ < shiftCount; shiftCountZ++ )
				{
					Info.ox = oxOriginal + shiftStart + shiftCountX * targetRes;
					Info.oy = oyOriginal + shiftStart + shiftCountY * targetRes;
					Info.oz = ozOriginal + shiftStart + shiftCountZ * targetRes;	
								
					markerArray.setValue( 0 );
					markerArraySTL.setValue( 0 );
					for ( int STLIndex = 0; STLIndex < (int)STLs.size(); STLIndex++ )
					{
						const bool insideMarkerValue = 1;
						ApplyMarkersInsideSTL( markerArraySTL, IJK, STLs[STLIndex], insideMarkerValue, Info );
						sumBoolArrays( markerArray, markerArraySTL, markerArray );
					}
					applyBouncebackMarkerFromFunction( markerArray, IJK, Info );
					
					// Accumulate the findings into the main arrays
					sumBoolArrays( finestBouncebackMarkerArray, markerArray, finestBouncebackMarkerArray );
					invertBoolArray( markerArray );
					sumBoolArrays( fluidMarkerArray, markerArray, fluidMarkerArray );
				}
			}
		}
	}
	
	// Restore original origin
	Info.ox = oxOriginal;
	Info.oy = oyOriginal;
	Info.oz = ozOriginal;
}

void getDIADNeighbours( IJKArrayStruct &IJK, std::vector<IntArrayType> &nbrArrays )
{
	const int cellCount = IJK.iArray.getSize();
	for ( int i = 0; i < 26; i++ ) { nbrArrays[i].setSize( cellCount ); nbrArrays[i].setValue( -1 ); }
	
	const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
	const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
	const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

	IJKArrayStruct IJKShifted;
	
	for ( int direction = 1; direction < 27; direction++ )
	{
		IJKShifted = IJK;
		shiftIJK( IJKShifted, cxArray[direction], cyArray[direction], czArray[direction] );
		findMatchingIJKIndex( IJKShifted, IJK, nbrArrays[direction-1] );
	}
}

void getDIADEsotwistNBRArray( DIADGridStruct &Grid )
{
	InfoStruct Info = Grid.Info;
	IJKArrayStruct IJK = Grid.IJK;
	const int cellCount = Info.cellCount;
	Grid.EsotwistNBRArray.iNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.jNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.kNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.ijNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.ikNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.jkNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.ijkNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.iMinNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.jMinNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.kMinNBRArray.setSize( cellCount );
	Grid.EsotwistNBRArray.iNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.jNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.kNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.ijNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.ikNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.jkNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.ijkNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.iMinNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.jMinNBRArray.setValue( -1 );
	Grid.EsotwistNBRArray.kMinNBRArray.setValue( -1 );
	
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.iNBRArray,    1, 0, 0, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.jNBRArray,    0, 1, 0, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.kNBRArray,    0, 0, 1, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.ijNBRArray,   1, 1, 0, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.ikNBRArray,   1, 0, 1, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.jkNBRArray,   0, 1, 1, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.ijkNBRArray,  1, 1, 1, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.iMinNBRArray,-1, 0, 0, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.jMinNBRArray, 0,-1, 0, Info );
	findPeriodicNeighbourIJKIndex( IJK, Grid.EsotwistNBRArray.kMinNBRArray, 0, 0,-1, Info );
}

// Mark target cell as 1 if at least one of its neighbours in source is 1
void markDIADNeighbours( std::vector<IntArrayType> &nbrArrays, BoolArrayType &sourceArray, BoolArrayType &targetArray )
{
	const int cellCount = nbrArrays[0].getSize();
		
	auto sourceView = sourceArray.getConstView();
	auto targetView = targetArray.getView();
	
	for ( int i = 0; i < 26; i++ )
	{
		auto nbrView = nbrArrays[i].getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			for ( int i = 0; i < 26; i++ ) 
			{
				if ( nbrView[cell] >=0 && sourceView[ nbrView[cell] ] ) 
				{ 
					targetView[cell] = true; 
					return; 
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, cellCount, cellLambda );
	}
}

void buildDIADGrids( std::vector<DIADGridStruct> &grids, std::vector<STLStruct> STLs, const int level )
{
	DIADGridStruct &Grid = grids[level];
	InfoStruct &Info = Grid.Info;
	std::cout << "Initial cell count on level " << level <<" : " << Info.cellCount << std::endl;
	
	if ( level == 0 ) sortIJK( Grid.IJK );
	else sortIJK( Grid.IJK, Grid.enforceInterfaceBounceback, Grid.enforceInterfaceFluid );
	
	// FINEST GRID BRANCH
	if ( level == gridLevelCount - 1 )
	{
		{ // SCOPE WITH LARGE ALLOCATED NEIGHBOUR ARRAYS
			std::vector<IntArrayType> nbrArrays( 26 );
			getDIADNeighbours( Grid.IJK, nbrArrays );	
				
			BoolArrayType fluidMarkerArray = BoolArrayType( Info.cellCount );
			BoolArrayType bouncebackMarkerArray = BoolArrayType( Info.cellCount );
			BoolArrayType markerArraySTL = BoolArrayType( Info.cellCount );	
			BoolArrayType keepCellMarkerArray = BoolArrayType( Info.cellCount );
			bouncebackMarkerArray.setValue( 0 );
			
			for ( int STLIndex = 0; STLIndex < (int)STLs.size(); STLIndex++ )
			{
				const bool insideMarkerValue = 1;
				ApplyMarkersInsideSTL( markerArraySTL, Grid.IJK, STLs[STLIndex], insideMarkerValue, Info );
				sumBoolArrays( bouncebackMarkerArray, markerArraySTL, bouncebackMarkerArray );
			}
			applyBouncebackMarkerFromFunction( bouncebackMarkerArray, Grid.IJK, Info );
			
			if ( gridLevelCount > 1 ) sumBoolArrays( bouncebackMarkerArray, Grid.enforceInterfaceBounceback, bouncebackMarkerArray );
			fluidMarkerArray = bouncebackMarkerArray;
			invertBoolArray( fluidMarkerArray );
			if ( gridLevelCount > 1 ) sumBoolArrays( fluidMarkerArray, Grid.enforceInterfaceFluid, fluidMarkerArray );
			
			keepCellMarkerArray = fluidMarkerArray;
			markDIADNeighbours( nbrArrays, fluidMarkerArray, keepCellMarkerArray );
			
			Grid.bouncebackMarkerArray.setSize( Info.cellCount );
			Grid.bouncebackMarkerArray.setValue( 1 );
			invertBoolArray( fluidMarkerArray );
			multiplyBoolArrays( fluidMarkerArray, Grid.bouncebackMarkerArray, Grid.bouncebackMarkerArray );
			
			Grid.enforceInterfaceFluid.resize( 0 );
			Grid.enforceInterfaceBounceback.resize( 0 );
			
			sortFinestGrid( Grid, keepCellMarkerArray );
			Info.cellCount = countMarkerCells( keepCellMarkerArray );
		}
		
		Grid.IJK.iArray.resize(Info.cellCount);
		Grid.IJK.jArray.resize(Info.cellCount);
		Grid.IJK.kArray.resize(Info.cellCount);
		Grid.bouncebackMarkerArray.resize(Info.cellCount);
		
		getDIADEsotwistNBRArray( Grid );
		
		// ALLOCATE fArray and initialize
		Grid.fArray.setSizes( 27, Info.cellCount );
		fillEquilibriumFromFunction( Grid );
		
		std::cout << "Final cell count on level " << level <<" : " << Info.cellCount << std::endl;
		return;
	}
	
	// COARSE GRID BRANCH
	BoolArrayType keepCellMarkerArray = BoolArrayType( Info.cellCount );
	BoolArrayType refinementMarkerArray = BoolArrayType( Info.cellCount );
	BoolArrayType coarseToFineMarkerArray = BoolArrayType( Info.cellCount );
	BoolArrayType fineToCoarseMarkerArray = BoolArrayType( Info.cellCount );
	BoolArrayType bouncebackUnderInterface = BoolArrayType( Info.cellCount );
	BoolArrayType fluidUnderInterface = BoolArrayType( Info.cellCount );
	{ // SCOPE WITH LARGE ALLOCATED NEIGHBOUR ARRAYS
		std::vector<IntArrayType> nbrArrays( 26 );
		
		// TIMER
		TNL::Timer lapTimer;
		lapTimer.reset();
		lapTimer.start();
		getDIADNeighbours( Grid.IJK, nbrArrays );	
		lapTimer.stop();
		auto lapTime = lapTimer.getRealTime();
		std::cout << "Binary search for 26 neighbours took " << lapTime << " s" << std::endl;
		
		// MARK FINEST FLUID
		BoolArrayType fluidMarkerArray = BoolArrayType( Info.cellCount );
		BoolArrayType finestBouncebackMarkerArray = BoolArrayType( Info.cellCount );
		markFinestFluidBounceback( fluidMarkerArray, finestBouncebackMarkerArray, Grid.IJK, STLs, Info );	
		// ADD ENFORCED FLUID
		if ( level != 0 )
		{
			sumBoolArrays( fluidMarkerArray, Grid.enforceInterfaceFluid, fluidMarkerArray ); 
		}
		
		// BORDER
		BoolArrayType borderMarkerArray = BoolArrayType( Info.cellCount );
		borderMarkerArray.setValue( 0 );
		markDIADNeighbours( nbrArrays, fluidMarkerArray, borderMarkerArray );
		BoolArrayType fluidInverseMarkerArray;
		fluidInverseMarkerArray = fluidMarkerArray;
		invertBoolArray( fluidInverseMarkerArray );
		multiplyBoolArrays( fluidInverseMarkerArray, borderMarkerArray, borderMarkerArray );
		
		// KEEP CELL ARRAY -> SO FAR KEEP FLUID AND BORDER
		keepCellMarkerArray = fluidMarkerArray;
		sumBoolArrays( keepCellMarkerArray, borderMarkerArray, keepCellMarkerArray );
		
		// THICK REFINEMENT REGION
		refinementMarkerArray = finestBouncebackMarkerArray; // make sure to refine areas where at least one finest bounceback cell is
		sumBoolArrays( refinementMarkerArray, borderMarkerArray, refinementMarkerArray ); // add border.. idk just to make sure
		BoolArrayType newRefinementMarkerArray = BoolArrayType( Info.cellCount );
		newRefinementMarkerArray.setValue( 0 );
		const int thickness = wallRefinementSpan + (gridLevelCount - level - 1);
		for ( int layer = 0; layer < thickness; layer++ )
		{
			markDIADNeighbours( nbrArrays, refinementMarkerArray, newRefinementMarkerArray );
			sumBoolArrays( refinementMarkerArray, newRefinementMarkerArray, newRefinementMarkerArray );
			multiplyBoolArrays( newRefinementMarkerArray, keepCellMarkerArray, refinementMarkerArray );
		}
		
		// MODIFICATION OF THE REFINEMENT REGION USING getMarkers FUNCTION
		applyRefinementMarkerFromFunction( refinementMarkerArray, Grid.IJK, Info );
		
		// FINE TO COARSE COMMUNICATION INTERFACE
		fineToCoarseMarkerArray.setValue( 0 );
		markDIADNeighbours( nbrArrays, refinementMarkerArray, fineToCoarseMarkerArray );
		multiplyBoolArrays( keepCellMarkerArray, fineToCoarseMarkerArray, fineToCoarseMarkerArray );
		BoolArrayType refinementInverseMarkerArray = BoolArrayType( Info.cellCount );
		refinementInverseMarkerArray = refinementMarkerArray;
		invertBoolArray( refinementInverseMarkerArray );
		multiplyBoolArrays( fineToCoarseMarkerArray, refinementInverseMarkerArray, fineToCoarseMarkerArray );
		
		// COARSE TO FINE COMMUNICATION INTERFACE
		coarseToFineMarkerArray.setValue( 0 );
		markDIADNeighbours( nbrArrays, fineToCoarseMarkerArray, coarseToFineMarkerArray );
		multiplyBoolArrays( keepCellMarkerArray, coarseToFineMarkerArray, coarseToFineMarkerArray );
		BoolArrayType fineToCoarseInverseMarkerArray = BoolArrayType( Info.cellCount );
		fineToCoarseInverseMarkerArray = fineToCoarseMarkerArray;
		invertBoolArray( fineToCoarseInverseMarkerArray );
		multiplyBoolArrays( coarseToFineMarkerArray, fineToCoarseInverseMarkerArray, coarseToFineMarkerArray );
		multiplyBoolArrays( coarseToFineMarkerArray, refinementInverseMarkerArray, coarseToFineMarkerArray );
	
		// NOW ALSO ADD THE INTERFACE INTO THE REFINEMENT AREA
		sumBoolArrays( refinementMarkerArray, fineToCoarseMarkerArray, refinementMarkerArray );
		sumBoolArrays( refinementMarkerArray, coarseToFineMarkerArray, refinementMarkerArray );
		refinementInverseMarkerArray = refinementMarkerArray;
		invertBoolArray( refinementInverseMarkerArray );
	
		// IDENTIFY INTERFACE BOUNCEBACK AND FLUID CELLS TO ENFORCE MATCH BETWEEN FINE AND COARSE WALL ON THE INTERFACE
		BoolArrayType coarseBouncebackMarkerArray = BoolArrayType( Info.cellCount );
		coarseBouncebackMarkerArray.setValue( 0 );
		BoolArrayType markerArraySTL = BoolArrayType( Info.cellCount );	
		applyBouncebackMarkerFromFunction( coarseBouncebackMarkerArray, Grid.IJK, Info );
		for ( int STLIndex = 0; STLIndex < (int)STLs.size(); STLIndex++ )
		{
			const bool insideMarkerValue = 1;
			ApplyMarkersInsideSTL( markerArraySTL, Grid.IJK, STLs[STLIndex], insideMarkerValue, Info );
			sumBoolArrays( coarseBouncebackMarkerArray, markerArraySTL, coarseBouncebackMarkerArray );
		}
		
		BoolArrayType bouncebackInverseMarkerArray;
		bouncebackInverseMarkerArray = coarseBouncebackMarkerArray;
		invertBoolArray( bouncebackInverseMarkerArray );
			
		sumBoolArrays( coarseToFineMarkerArray, fineToCoarseMarkerArray, bouncebackUnderInterface );
		multiplyBoolArrays( bouncebackUnderInterface, coarseBouncebackMarkerArray, bouncebackUnderInterface );
		
		sumBoolArrays( coarseToFineMarkerArray, fineToCoarseMarkerArray, fluidUnderInterface );
		multiplyBoolArrays( fluidUnderInterface, bouncebackInverseMarkerArray, fluidUnderInterface );
		
		// MARK KEEP CELLS TO BE ABLE TO DELETE ALL CELLS THAT ARE NOT NEEDED ON OUR COARSE GRID
		keepCellMarkerArray = fluidMarkerArray;
		sumBoolArrays( keepCellMarkerArray, borderMarkerArray, keepCellMarkerArray );
		multiplyBoolArrays( keepCellMarkerArray, refinementInverseMarkerArray, keepCellMarkerArray );
		sumBoolArrays( keepCellMarkerArray, coarseToFineMarkerArray, keepCellMarkerArray );
		sumBoolArrays( keepCellMarkerArray, fineToCoarseMarkerArray, keepCellMarkerArray );
		// ADD ENFORCED FLUID
		if ( level != 0 )
		{
			sumBoolArrays( keepCellMarkerArray, Grid.enforceInterfaceFluid, keepCellMarkerArray ); 
		}
	
	} // CLOSING THE SCOPE DEALOCATES LARGE nbrArrays and most marker arrays
	
	// ASSEMBLE IJK FOR THE NEXT LEVEL FINER GRID BASED ON THE REFINEMENT AREA
	const int cellCountFine = 8 * countMarkerCells( refinementMarkerArray );
	grids[level+1].Info.cellCount = cellCountFine;
	BoolArrayTypeCPU refinementMarkerArrayCPU = BoolArrayTypeCPU( Info.cellCount );
	refinementMarkerArrayCPU = refinementMarkerArray;
	BoolArrayTypeCPU bouncebackUnderInterfaceCPU;
	bouncebackUnderInterfaceCPU = bouncebackUnderInterface;
	BoolArrayTypeCPU fluidUnderInterfaceCPU;
	fluidUnderInterfaceCPU = fluidUnderInterface;
	IJKArrayStructCPU IJKCPU = IJKArrayStructCPU( Grid.IJK );
	BoolArrayTypeCPU fineBouncebackMarkerArrayCPU;
	fineBouncebackMarkerArrayCPU.setSize( cellCountFine );
	BoolArrayTypeCPU fineFluidMarkerArrayCPU;
	fineFluidMarkerArrayCPU.setSize( cellCountFine );
	IJKArrayStructCPU IJKFineCPU;
	IJKFineCPU.iArray.setSize( cellCountFine );
	IJKFineCPU.jArray.setSize( cellCountFine );
	IJKFineCPU.kArray.setSize( cellCountFine );
	int cellFine = 0;
	for ( int cellCoarse = 0; cellCoarse < Info.cellCount; cellCoarse++ )
	{
		if ( refinementMarkerArrayCPU[ cellCoarse ] )
		{
			const int iCoarse = IJKCPU.iArray[ cellCoarse ];
			const int jCoarse = IJKCPU.jArray[ cellCoarse ];
			const int kCoarse = IJKCPU.kArray[ cellCoarse ];
			const bool bouncebackCoarse = bouncebackUnderInterfaceCPU[ cellCoarse ];
			const bool fluidCoarse = fluidUnderInterfaceCPU[ cellCoarse ];
			const int iFine = iCoarse * 2;
			const int jFine = jCoarse * 2;
			const int kFine = kCoarse * 2;
			
			for ( int iPlus = 0; iPlus < 2; iPlus++ )
			{
				for ( int jPlus = 0; jPlus < 2; jPlus++ )
				{
					for ( int kPlus = 0; kPlus < 2; kPlus++ )
					{
						IJKFineCPU.iArray[ cellFine ] = iFine + iPlus;
						IJKFineCPU.jArray[ cellFine ] = jFine + jPlus;
						IJKFineCPU.kArray[ cellFine ] = kFine + kPlus;
						fineBouncebackMarkerArrayCPU[ cellFine ] = bouncebackCoarse;
						fineFluidMarkerArrayCPU[ cellFine ] = fluidCoarse;
						cellFine++;
					}
				}
			}
		}
	}
	grids[level + 1].IJK = IJKArrayStruct( IJKFineCPU );
	grids[level + 1].enforceInterfaceBounceback = fineBouncebackMarkerArrayCPU;
	grids[level + 1].enforceInterfaceFluid = fineFluidMarkerArrayCPU;
	
	// SORT ALL CELLS BY KEY: keepCell, k, j, i
	
	// TIMER
	//TNL::Timer lapTimer;
	//lapTimer.reset();
	//lapTimer.start();
	
	sortCoarseGrid( Grid, keepCellMarkerArray, fineToCoarseMarkerArray, coarseToFineMarkerArray );
	
	//lapTimer.stop();
	//auto lapTime = lapTimer.getRealTime();
	//std::cout << "Sorting took " << lapTime << " s" << std::endl;
	
	Info.cellCount = countMarkerCells( keepCellMarkerArray );
	Grid.IJK.iArray.resize(Info.cellCount);
	Grid.IJK.jArray.resize(Info.cellCount);
	Grid.IJK.kArray.resize(Info.cellCount);
	fineToCoarseMarkerArray.resize(Info.cellCount);
	coarseToFineMarkerArray.resize(Info.cellCount);
	if ( level != 0 )
	{
		Grid.enforceInterfaceBounceback.resize(Info.cellCount);
		Grid.enforceInterfaceFluid.resize(Info.cellCount);
	}
	
	// INTERFACE MARKER FOR LATER USE - OPTIONAL
	Grid.interfaceMarkerArray.setSize(Info.cellCount);
	Grid.interfaceMarkerArray.setValue( 0 );
	sumBoolArrays( Grid.interfaceMarkerArray, fineToCoarseMarkerArray, Grid.interfaceMarkerArray );
	sumBoolArrays( Grid.interfaceMarkerArray, coarseToFineMarkerArray, Grid.interfaceMarkerArray );
	
	std::cout << "Final cell count on level " << level <<" : " << Info.cellCount << std::endl;
	
	// FINAL BOUNCEBACK PASS
	Grid.bouncebackMarkerArray.setSize(Info.cellCount);
	Grid.bouncebackMarkerArray.setValue( 0 );	
	BoolArrayType markerArraySTL = BoolArrayType( Info.cellCount );	
	markerArraySTL.setValue( 0 );
	applyBouncebackMarkerFromFunction( Grid.bouncebackMarkerArray, Grid.IJK, Info );
	for ( int STLIndex = 0; STLIndex < (int)STLs.size(); STLIndex++ )
	{
		const bool insideMarkerValue = 1;
		ApplyMarkersInsideSTL( markerArraySTL, Grid.IJK, STLs[STLIndex], insideMarkerValue, Info );
		sumBoolArrays( Grid.bouncebackMarkerArray, markerArraySTL, Grid.bouncebackMarkerArray );
	}
	if ( level != 0 )
	{
		sumBoolArrays( Grid.bouncebackMarkerArray, Grid.enforceInterfaceBounceback, Grid.bouncebackMarkerArray );
		invertBoolArray( Grid.enforceInterfaceFluid );
		multiplyBoolArrays( Grid.bouncebackMarkerArray, Grid.enforceInterfaceFluid, Grid.bouncebackMarkerArray );
	}
	Grid.enforceInterfaceFluid.resize( 0 );
	Grid.enforceInterfaceBounceback.resize( 0 );
	
	// REMOVE BOUNCEBACK CELLS FROM THE INTERFACE COMMUNICATION
	{
		BoolArrayType bouncebackInverseMarkerArray;
		bouncebackInverseMarkerArray = Grid.bouncebackMarkerArray;
		invertBoolArray( bouncebackInverseMarkerArray );
		multiplyBoolArrays( coarseToFineMarkerArray, bouncebackInverseMarkerArray, coarseToFineMarkerArray );
		multiplyBoolArrays( fineToCoarseMarkerArray, bouncebackInverseMarkerArray, fineToCoarseMarkerArray );
	} // bouncebackInverseMarkerArray runs out of scope
	
	// BUILD ESOTWIST CONNECTIONS
	getDIADEsotwistNBRArray( Grid );
	
	// BUILD OUTLET CELL ARRAY TO STORE INDEXES OF NON REFLECTIVE OUTLET BC CELLS
	{
		int outletCellCount = 0;
		BoolArrayType outletCellMarkerArray;
		outletCellMarkerArray.setSize( Info.cellCount );
		outletCellMarkerArray.setValue( 0 );
		applyOutletMarkerFromFunction( outletCellMarkerArray, Grid.IJK, Info );
		buildOutletCellArray( Grid, outletCellMarkerArray );
	}
	
	// ALLOCATE fArray and initialize
	Grid.fArray.setSizes( 27, Info.cellCount );
	fillEquilibriumFromFunction( Grid );
	
	// CALL THE FUNCTION RECURSIVELY TO BUILD THE NEXT FINER GRID LEVEL
	grids[level+1].Info.gridID = Info.gridID + 1;
	grids[level+1].Info.res = Info.res * 0.5f;
	grids[level+1].Info.dtPhys = Info.dtPhys * 0.5f;
	grids[level+1].Info.nu = (grids[level+1].Info.dtPhys * nuPhys) / ((grids[level+1].Info.res/1000.f) * (grids[level+1].Info.res/1000.f));
	grids[level+1].Info.ox = Info.ox - grids[level+1].Info.res * 0.5f;
	grids[level+1].Info.oy = Info.oy - grids[level+1].Info.res * 0.5f;
	grids[level+1].Info.oz = Info.oz - grids[level+1].Info.res * 0.5f;
	grids[level+1].Info.cellCountX = Info.cellCountX * 2;
	grids[level+1].Info.cellCountY = Info.cellCountY * 2;
	grids[level+1].Info.cellCountZ = Info.cellCountZ * 2;
	buildDIADGrids( grids, STLs, level + 1 );
	
	// MAP INTERFACE COMMUNICATION
	auto mapInterfaceLambda = [&]( BoolArrayType& markerArray, IntArrayType& coarseArray, IntArrayType& fineArray ) 
	{
		const int count = countMarkerCells( markerArray );
		coarseArray.setSize( count );
		fineArray.setSize( count );
		coarseArray.setValue( -1 );
		fineArray.setValue( -1 );

		// CPU copy and compaction
		BoolArrayTypeCPU markerArrayCPU;
		markerArrayCPU = markerArray;
		IntArrayTypeCPU coarseArrayCPU;
		coarseArrayCPU = coarseArray;
		int counter = 0;
		for ( int cell = 0; cell < Info.cellCount; cell++ ) {
			if ( markerArrayCPU[ cell ] ) coarseArrayCPU[ counter++ ] = cell;
		}
		coarseArray = coarseArrayCPU;

		// Generate IJK for the finer level to search for
		IJKArrayStruct fineIJK;
		fineIJK.iArray.setSize( count );
		fineIJK.jArray.setSize( count );
		fineIJK.kArray.setSize( count );

		auto coarseView = coarseArray.getConstView();
		auto iFineView = fineIJK.iArray.getView();
		auto jFineView = fineIJK.jArray.getView();
		auto kFineView = fineIJK.kArray.getView();
		auto iView = Grid.IJK.iArray.getView();
		auto jView = Grid.IJK.jArray.getView();
		auto kView = Grid.IJK.kArray.getView();

		auto cellLambda = [=] __cuda_callable__ ( const int c ) mutable {
			const int cell = coarseView[ c ];
			iFineView[ c ] = iView[ cell ] * 2;
			jFineView[ c ] = jView[ cell ] * 2;
			kFineView[ c ] = kView[ cell ] * 2;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, count, cellLambda );

		// Find the connections on the fine grid
		findMatchingIJKIndex( fineIJK, grids[level+1].IJK, fineArray );
		const int invalidConnections = countInvalidIndexes( fineArray );
		if ( invalidConnections > 0 ) {
			
			// Copy necessary arrays to the CPU
			IntArrayTypeCPU fineArrayCPU;
			fineArrayCPU = fineArray;
			IntArrayTypeCPU coarseArrayCPU;
			coarseArrayCPU = coarseArray;
			IJKArrayStructCPU coarseIJKCPU = IJKArrayStructCPU( Grid.IJK );
			
			// Find and print the faulty coordinates
			for ( int c = 0; c < count; c++ )
			{
				if ( fineArrayCPU[ c ] < 0 )
				{
					const int cellCoarse = coarseArrayCPU[ c ];
					const int iCoarse = coarseIJKCPU.iArray[ cellCoarse ];
					const int jCoarse = coarseIJKCPU.jArray[ cellCoarse ];
					const int kCoarse = coarseIJKCPU.kArray[ cellCoarse ];
					
					// Calculate physical coordinates based on index, origin, and resolution
					
					const float xPhys = iCoarse * Info.res + Info.ox;
					const float yPhys = jCoarse * Info.res + Info.oy;
					const float zPhys = kCoarse * Info.res + Info.oz;
					
					std::cerr << "Invalid interface connection for coarse cell " << cellCoarse 
					          << " | IJK: (" << iCoarse << ", " << jCoarse << ", " << kCoarse << ")"
					          << " | XYZ: (" << xPhys << ", " << yPhys << ", " << zPhys << ")\n";
				}
			}
			
			throw std::runtime_error( 
				"DIAD Grid Generation Error: " + std::to_string(invalidConnections) + 
				" invalid interface connections found on level " + std::to_string(level) 
			);
		}
	};
	mapInterfaceLambda( coarseToFineMarkerArray, Grid.coarseToFineReadArray, Grid.coarseToFineWriteArray );
	mapInterfaceLambda( fineToCoarseMarkerArray, Grid.fineToCoarseWriteArray, Grid.fineToCoarseReadArray );
}
