#pragma once

#include "./types.h"

void intArrayFromBoolArray( IntArrayType &intArray, const BoolArrayType &boolArray, const int &upperBound )
{
	auto intView = intArray.getView();
	auto boolView = boolArray.getConstView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		if ( boolView[ cell ] ) intView[ cell ] = 1;
		else intView[ cell ] = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, upperBound, cellLambda );
}

int countZerosInBoolArray( const BoolArrayType &boolArray, const int &upperBound )
{
	auto boolView = boolArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int cell )
	{
		if ( !boolView[ cell ] ) return 1;
		else return 0;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	return TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, upperBound, fetch, reduction, 0 );
}

int countOnesInBoolArray( const BoolArrayType &boolArray, const int &upperBound )
{
	auto boolView = boolArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int cell )
	{
		if ( boolView[ cell ] ) return 1;
		else return 0;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	return TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, upperBound, fetch, reduction, 0 );
}

int countOnesInBoolArray2D( const BoolArray2DType &boolArray, const int &firstBound, const int &secondBound )
{
	auto boolView = boolArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int singleIndex )
	{
		const int i = singleIndex % firstBound;
		const int j = singleIndex / firstBound;
		if ( boolView( i, j ) ) return 1;
		else return 0;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	const int start = 0;
	const int end = firstBound * secondBound;
	const int onesCount = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0 );
	return onesCount;
}

float findMaxFloatArray2D( const FloatArray2DType &floatArray, const int &firstBound, const int &secondBound )
{
	auto floatView = floatArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int singleIndex )
	{
		const int i = singleIndex % firstBound;
		const int j = singleIndex / firstBound;
		return std::fabs(floatView( i, j ));
	};
	auto reduction = [] __cuda_callable__( const float& a, const float& b )
	{
		return TNL::max( a, b );
	};
	const int start = 0;
	const int end = firstBound * secondBound;
	const float floatMax = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0.f );
	return floatMax;
}
