#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"
#include "./NBRFunctions.h"

// this is used to bit unpack the information from Grid.bitPackedMarkerArray (int array)
__host__ __device__ inline void intToBools( const int &value, bool (&bools)[32] )
{
    for (int i = 0; i < 32; ++i)
    {
        bools[i] = ((value >> i) & 1) != 0;
    }
}

// this is used to bit pack the information into Grid.bitPackedMarkerArray (int array)
__host__ __device__ inline void boolsToInt( int& value, const bool (&bools)[32] )
{
    value = 0;
    for (int i = 0; i < 32; i++ )
    {
        if (bools[i])
        {
            value |= static_cast<int>(1U << i); 
        }
    }
}

void fillBitPackedMarkerArray( GridStruct &Grid, const int &upperBound )
{
	const InfoStruct &Info = Grid.Info;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jMinusView = Grid.NBR.jMinusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto bouncebackMarkerView = Grid.bouncebackMarkerArray.getConstView();
	auto movingBouncebackMarkerView = Grid.movingBouncebackMarkerArray.getConstView();
	auto deepRefinementMarkerView = Grid.deepRefinementMarkerArray.getConstView();
	auto bitPackedMarkerView = Grid.bitPackedMarkerArray.getView();
	
	bool useBouncebackMarkerArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	bool useMovingBouncebackMarkerArray = ( Grid.movingBouncebackMarkerArray.getSize() > 0 );
	bool useDeepRefinementMarkerArray = ( Grid.deepRefinementMarkerArray.getSize() > 0 );
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		bool bitPackedMarkerBits[32] = {false};
		
		if ( useBouncebackMarkerArray ) bitPackedMarkerBits[27] = bouncebackMarkerView( cell );
		if ( useMovingBouncebackMarkerArray ) bitPackedMarkerBits[28] = movingBouncebackMarkerView( cell );
		if ( useDeepRefinementMarkerArray ) bitPackedMarkerBits[29] = deepRefinementMarkerView( cell );
		
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( kPlusView( cell ) );
		NBR.jMinus = jMinusView( cell );
		NBR.kMinus = kMinusView( cell );
		finishNBRAll( NBR, Info );
	
		const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
		const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
		const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
		
		int fullNBRList[27];
		// for each direction this holds the neighbour where f[i] will be pulled from in the next iteration
		// 0: Center
		fullNBRList[0]  = cell;
		// 1-6: Straight directions (Faces)
		fullNBRList[1]  = NBR.iMinus; 			// cxArray=1  -> nx=-1
		fullNBRList[2]  = NBR.iPlus;  			// cxArray=-1 -> nx=1
		fullNBRList[3]  = NBR.kPlus;  			// czArray=-1 -> nz=1
		fullNBRList[4]  = NBR.kMinus; 			// czArray=1  -> nz=-1
		fullNBRList[5]  = NBR.jPlus;  			// cyArray=-1 -> ny=1
		fullNBRList[6]  = NBR.jMinus; 			// cyArray=1  -> ny=-1
		// 7-18: Diagonal directions (Edges)
		fullNBRList[7]  = kPlusView( NBR.iMinus );	// cxArray=1,  czArray=-1 -> nx=-1, nz=1
		fullNBRList[8]  = kMinusView( NBR.iPlus );	// cxArray=-1, czArray=1  -> nx=1,  nz=-1
		fullNBRList[9]  = kMinusView( NBR.iMinus );	// cxArray=1,  czArray=1  -> nx=-1, nz=-1
		fullNBRList[10] = kPlusView( NBR.iPlus ); 	// cxArray=-1, czArray=-1 -> nx=1,  nz=1
		fullNBRList[11] = jPlusView( NBR.iPlus ); 	// cxArray=-1, cyArray=-1 -> nx=1,  ny=1
		fullNBRList[12] = jMinusView( NBR.iMinus );	// cxArray=1,  cyArray=1  -> nx=-1, ny=-1
		fullNBRList[13] = kPlusView( NBR.jMinus );	// cyArray=1,  czArray=-1 -> ny=-1, nz=1
		fullNBRList[14] = kMinusView( NBR.jPlus );	// cyArray=-1, czArray=1  -> ny=1,  nz=-1
		fullNBRList[15] = jMinusView( NBR.iPlus );	// cxArray=-1, cyArray=1  -> nx=1,  ny=-1
		fullNBRList[16] = jPlusView( NBR.iMinus );	// cxArray=1,  cyArray=-1 -> nx=-1, ny=1
		fullNBRList[17] = kMinusView( NBR.jMinus );	// cyArray=1,  czArray=1  -> ny=-1, nz=-1
		fullNBRList[18] = kPlusView( NBR.jPlus ); 	// cyArray=-1, czArray=-1 -> ny=1,  nz=1
		// 19-26: Corner directions (Vertices)
		fullNBRList[19] = kPlusView( jMinusView( NBR.iPlus ) ); 	// cxArray=-1, cyArray=1,  czArray=-1 -> nx=1,  ny=-1, nz=1
		fullNBRList[20] = kMinusView( jPlusView( NBR.iMinus ) ); 	// cxArray=1,  cyArray=-1, czArray=1  -> nx=-1, ny=1,  nz=-1
		fullNBRList[21] = kMinusView( jPlusView( NBR.iPlus ) ); 	// cxArray=-1, cyArray=-1, czArray=1  -> nx=1,  ny=1,  nz=-1
		fullNBRList[22] = kPlusView( jMinusView( NBR.iMinus ) ); 	// cxArray=1,  cyArray=1,  czArray=-1 -> nx=-1, ny=-1, nz=1
		fullNBRList[23] = kPlusView( jPlusView( NBR.iMinus ) ); 	// cxArray=1,  cyArray=-1, czArray=-1 -> nx=-1, ny=1,  nz=1
		fullNBRList[24] = kMinusView( jMinusView( NBR.iPlus ) ); 	// cxArray=-1, cyArray=1,  czArray=1  -> nx=1,  ny=-1, nz=-1
		fullNBRList[25] = kPlusView( jPlusView( NBR.iPlus ) );  	// cxArray=-1, cyArray=-1, czArray=-1 -> nx=1,  ny=1,  nz=1
		fullNBRList[26] = kMinusView( jMinusView( NBR.iMinus ) );	// cxArray=1,  cyArray=1,  czArray=1  -> nx=-1, ny=-1, nz=-1
		
		// now look at each neighbour if they are geometric fluid neighbour
		
		for ( int direction = 1; direction < 27; direction++ )
		{
			MarkerStruct Marker;
			if ( useBouncebackMarkerArray ) Marker.bounceback = bouncebackMarkerView( fullNBRList[direction] );
			if ( useMovingBouncebackMarkerArray ) Marker.movingBounceback = movingBouncebackMarkerView( fullNBRList[direction] );
			if ( useDeepRefinementMarkerArray ) Marker.deepRefinement = deepRefinementMarkerView( fullNBRList[direction] );
			getMarkers( iCell, jCell, kCell, Marker, Info );
			if ( !Marker.bounceback && !Marker.movingBounceback )
			{
				// check position
				const int nx = -cxArray[direction]; const int ny = -cyArray[direction]; const int nz = -czArray[direction];
				const int iExpected = iCell + nx; const int jExpected = jCell + ny; const int kExpected = kCell + nz;
				const int iActual = iView( fullNBRList[direction] ); 
				const int jActual = jView( fullNBRList[direction] ); 
				const int kActual = kView( fullNBRList[direction] );
				if ( iActual == iExpected && jActual == jExpected && kActual == kExpected ) 
				{
					// the cell is neither bounceback, nor moving bounceback AND is geometrically correct -> mark it
					bitPackedMarkerBits[direction] = true;
				}
			}
		}
		
		int bitPackedMarker;
		boolsToInt( bitPackedMarker, bitPackedMarkerBits );
		bitPackedMarkerView( cell ) = bitPackedMarker;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

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
	// const int cellCountZ = SkeletonGrid.Info.cellCountZ; // not needed here
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

void applyUserRefinementModification( BoolArrayType &markerArray, const GridStruct &Grid, const int &upperBound )
{
	// uses the getMarkers function defined in the main file to adjust refinement area. 
	// Beware, the current version isnt able to handle interface on walls or boundaries.
	const InfoStruct &Info = Grid.Info;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto markerView = markerArray.getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		MarkerStruct Marker;
		Marker.refinement = markerView( cell );		
		getMarkers( iCell, jCell, kCell, Marker, Info );
		markerView( cell ) = Marker.refinement;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void applyNonReflectiveOutletMarker( BoolArrayType &markerArray, const GridStruct &Grid, const int &upperBound )
{
	// uses the getMarkers function defined in the main file to mark non reflective outlet cells
	const InfoStruct &Info = Grid.Info;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto markerView = markerArray.getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		MarkerStruct Marker;
		getMarkers( iCell, jCell, kCell, Marker, Info );
		markerView( cell ) = Marker.nonReflectiveOutlet;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void applyNonReflectiveInletMarker( BoolArrayType &markerArray, const GridStruct &Grid, const int &upperBound )
{
	// uses the getMarkers function defined in the main file to mark non reflective inlet cells
	const InfoStruct &Info = Grid.Info;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto markerView = markerArray.getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		MarkerStruct Marker;
		getMarkers( iCell, jCell, kCell, Marker, Info );
		markerView( cell ) = Marker.nonReflectiveInlet;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

void spreadMarkers( BoolArrayType &targetMarkerArray, const BoolArrayType &sourceMarkerArray, GridStruct &Grid, const int &upperBound )
{
	// The way this is written creates a race condition, one that is harmless because all threads write the same 1
	auto targetMarkerView = targetMarkerArray.getView();
	auto sourceMarkerView = sourceMarkerArray.getConstView();
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto isGeometricBitPackedMarkerView = Grid.NBR.isGeometricBitPackedMarkerArray.getView();
	
	targetMarkerArray = sourceMarkerArray; // initialize as source

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		int nbrPlus[7];
		nbrPlus[0] = (cell + 1 < upperBound) ? cell + 1 : 0; 					// iPlus
		nbrPlus[1] = jPlusView[ cell ];											// jPlus
		nbrPlus[2] = (nbrPlus[1] + 1 < upperBound) ? nbrPlus[1] + 1 : 0;		// ijPlus
		nbrPlus[3] = kPlusView[ cell ];											// kPlus
		nbrPlus[4] = (nbrPlus[3] + 1 < upperBound) ? nbrPlus[3] + 1 : 0;		// ikPlus
		nbrPlus[5] = jPlusView[ nbrPlus[3] ];									// jkPlus
		nbrPlus[6] = (nbrPlus[5] + 1 < upperBound) ? nbrPlus[5] + 1 : 0;		// ijkPlus
		
		bool isGeometricMarker[8] = {false};
		const uint8_t isGeometricBitPack = isGeometricBitPackedMarkerView( cell );
		byteToBools( isGeometricBitPack, isGeometricMarker );
		
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
	// the moving bounceback geometry can travel distance up to 2 cells before the grid is fully rebuilt
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
	applyUserRefinementModification( Grid.deepRefinementMarkerArray, Grid, upperBound );
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
