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
