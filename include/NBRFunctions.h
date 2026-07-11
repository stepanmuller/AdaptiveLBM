#pragma once

#include "./types.h"
#include "./genericArrayFunctions.h"

void getNBRArrayForSkeleton( IntArrayType &nbrArray, const int jPlus, const int kPlus, const SkeletonGridStruct &SkeletonGrid )
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

void skipOneUnmarkedNBR( IntArrayType &nbrArray, const BoolArrayType &markerArray, const IntArrayType &nbrOldArray, BoolArrayType &finishedMarkerArray, const int &upperBound )
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
		int newNBR = nbrOldView[ nbr ]; // neighbour of our neighbour
		nbrView[ cell ] = newNBR;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

int countUnfinishedNBR( const BoolArrayType &markerArray, const BoolArrayType &finishedMarkerArray, const int &upperBound )
{
	auto markerView = markerArray.getConstView();
	auto finishedMarkerView = finishedMarkerArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int cell )
	{
		if ( markerView[ cell ] && !finishedMarkerView[ cell ] ) return 1; // sum those who are marked but not finished - we only care about those
		else return 0;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	return TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, upperBound, fetch, reduction, 0 );
}

void skipAllUnmarkedNBR( IntArrayType &nbrArray, const BoolArrayType &markerArray, IntArrayType &intBuffer, BoolArrayType &markerBuffer, const int &upperBound )
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
		skipOneUnmarkedNBR( nbrArray, markerArray, nbrOldArray, finishedMarkerArray, upperBound );
		unfinishedCount = countUnfinishedNBR( markerArray, finishedMarkerArray, upperBound );
	}
}

// Optimized version: On launch build list of unfinished cells and only loop through them

void skipOneUnmarkedNBROptimized( IntArrayType &nbrArray, const BoolArrayType &markerArray, const IntArrayType &nbrOldArray, BoolArrayType &finishedMarkerArray, const int &upperBound )
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
		int newNBR = nbrOldView[ nbr ]; // neighbour of our neighbour
		nbrView[ cell ] = newNBR;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );	
}

int countUnfinishedNBROptimized( const BoolArrayType &markerArray, const BoolArrayType &finishedMarkerArray, const int &upperBound )
{
	auto markerView = markerArray.getConstView();
	auto finishedMarkerView = finishedMarkerArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int cell )
	{
		if ( markerView[ cell ] && !finishedMarkerView[ cell ] ) return 1; // sum those who are marked but not finished - we only care about those
		else return 0;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	return TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, upperBound, fetch, reduction, 0 );
}

void skipAllUnmarkedNBROptimized( IntArrayType &nbrArray, const BoolArrayType &markerArray, IntArrayType &intBuffer, IntArrayType &intBuffer2, BoolArrayType &markerBuffer, const int &upperBound )
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
		skipOneUnmarkedNBR( nbrArray, markerArray, nbrOldArray, finishedMarkerArray, upperBound );
		unfinishedCount = countUnfinishedNBR( markerArray, finishedMarkerArray, upperBound );
	}
}
