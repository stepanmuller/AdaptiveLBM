#pragma once

__host__ __device__ bool getRayHitYesNo( 	const int &i, const int &j, 
											const long long &wab, const long long &wbc, const long long &wca,
											const int &axInt, const int &ayInt, 
											const int &bxInt, const int &byInt,
											const int &cxInt, const int &cyInt,
											const long long &abx, const long long &aby,
											const long long &bcx, const long long &bcy,
											const long long &cax, const long long &cay )
{
	// Identify if triangle is hit by a ray
	if ( wab > 0 && wbc > 0 && wca > 0 ) return true;
    if ( wab < 0 || wbc < 0 || wca < 0 ) return false;
    // If these two checks did not produce a return, it means ray is hitting exactly an edge
	// Translate the triangle so that the ray lies at [0, 0]
	const int rayXInt = i * 100;
	const int rayYInt = j * 100;
    const long long ax0 = axInt - rayXInt;
    const long long ay0 = ayInt - rayYInt;
    const long long bx0 = bxInt - rayXInt;
    const long long by0 = byInt - rayYInt;
    const long long cx0 = cxInt - rayXInt;
    const long long cy0 = cyInt - rayYInt;
            
    if ( wab == 0LL )
    {
		if ( abx == 0LL ) // vertical edge
		{
			if ( cx0 < 0LL ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( bx0 > 0LL && cy0 * bx0 > by0 * cx0 ) return true;
		if ( bx0 < 0LL && cy0 * bx0 < by0 * cx0 ) return true;
	}
	
	if ( wbc == 0LL )
    {
		if ( bcx == 0LL ) // vertical edge
		{
			if ( ax0 < 0LL ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( cx0 > 0LL && ay0 * cx0 > cy0 * ax0 ) return true;
		if ( cx0 < 0LL && ay0 * cx0 < cy0 * ax0 ) return true;
	}
	
	if ( wca == 0LL )
    {
		if ( cax == 0LL ) // vertical edge
		{
			if ( bx0 < 0LL ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( ax0 > 0LL && by0 * ax0 > ay0 * bx0 ) return true;
		if ( ax0 < 0LL && by0 * ax0 < ay0 * bx0 ) return true;
	}

    return false; 
}

void voxelizeSTL( STLStruct &STL, VoxelizerStruct &Voxelizer, IntArray3DType &rayMapArray )
{
	constexpr int rayMapDepth = VoxelizerStruct::rayMapDepth;
	// Result is saved into the rayMap which is a 3D array of size cellCountX * cellCountY * rayMapDepth
	// K indexes of intersections with each I, J ray are saved in ascending order. The rest up to rayMapDepth is filled with int max
	// If rayMapDepth is too low, throw error
	InfoStruct &Info = Voxelizer.Info;
	IntArray2DType &hitsPerRayCounterArray = Voxelizer.hitsPerRayCounterArray;
	
	if ( hitsPerRayCounterArray.getSizes()[0] < 1 )
	{
		hitsPerRayCounterArray.setSizes( Info.cellCountX, Info.cellCountY );
		Voxelizer.rayMapBouncebackArray.setSizes( Info.cellCountX, Info.cellCountY, rayMapDepth );
		Voxelizer.rayMapMovingBouncebackArray.setSizes( Info.cellCountX, Info.cellCountY, rayMapDepth );
		Voxelizer.rayMapTotalArray.setSizes( Info.cellCountX, Info.cellCountY, rayMapDepth );
		const int rayMapElementCount = Info.cellCountX * Info.cellCountY * rayMapDepth;
		const int rayMapMemoryMB = (float)(rayMapElementCount * 4) / 1000000.f;
		std::cout << "Voxelizer GPU memory allocated, a single rayMap now takes " << rayMapMemoryMB << " MB" << std::endl;
	}
	
	hitsPerRayCounterArray.setValue( 0 );
	rayMapArray.setValue( std::numeric_limits<int>::max() );
	
	auto rayMapView = rayMapArray.getView();
	auto hitsPerRayCounterView = hitsPerRayCounterArray.getView();
	
	auto axView = STL.axArray.getConstView();
	auto ayView = STL.ayArray.getConstView();
	auto azView = STL.azArray.getConstView();
	auto bxView = STL.bxArray.getConstView();
	auto byView = STL.byArray.getConstView();
	auto bzView = STL.bzArray.getConstView();
	auto cxView = STL.cxArray.getConstView();
	auto cyView = STL.cyArray.getConstView();
	auto czView = STL.czArray.getConstView();
	
	IntArrayType &raysPerTriangleCounterArray = STL.raysPerTriangleCounterArray;
	auto raysPerTriangleCounterView = raysPerTriangleCounterArray.getView();
	IntArrayType &threadToTriangleMapArray = STL.threadToTriangleMapArray;
	const int threadCountMax = threadToTriangleMapArray.getSize();
	auto threadToTriangleMapView = threadToTriangleMapArray.getView();
	
	auto raysPerTriangleCounterLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		// transform into the coordinate system of the LBM grid
		const float ax = axView[ triangleIndex ] - Info.ox;
		const float ay = ayView[ triangleIndex ] - Info.oy;
		const float az = azView[ triangleIndex ] - Info.oz;
		const float bx = bxView[ triangleIndex ] - Info.ox;
		const float by = byView[ triangleIndex ] - Info.oy;
		const float bz = bzView[ triangleIndex ] - Info.oz;
		const float cx = cxView[ triangleIndex ] - Info.ox;
		const float cy = cyView[ triangleIndex ] - Info.oy;
		const float cz = czView[ triangleIndex ] - Info.oz;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const int axInt = (int)(round( ax * scale )) * 2 + 1;
		const int ayInt = (int)(round( ay * scale )) * 2 + 1;
		const int bxInt = (int)(round( bx * scale )) * 2 + 1;
		const int byInt = (int)(round( by * scale )) * 2 + 1;
		const int cxInt = (int)(round( cx * scale )) * 2 + 1;
		const int cyInt = (int)(round( cy * scale )) * 2 + 1;
		// now figure out which rays can possibly hit the triangle -> get bounds
		const int xIntMin = std::max({ 0, std::min({ axInt, bxInt, cxInt, (int)(Info.cellCountX-1)*100 }) });
		const int xIntMax = std::min({ (int)(Info.cellCountX-1)*100, std::max({ axInt, bxInt, cxInt, 0 }) });
		const int yIntMin = std::max({ 0, std::min({ ayInt, byInt, cyInt, (int)(Info.cellCountY-1)*100 }) });
		const int yIntMax = std::min({ (int)(Info.cellCountY-1)*100, std::max({ ayInt, byInt, cyInt, 0 }) });
		const int iStart = (xIntMin + 99) / 100;
		const int iEnd = xIntMax / 100 + 1;
		const int jStart = (yIntMin + 99) / 100;
		const int jEnd = yIntMax / 100 + 1;
		const int raysPerTriangleCount = ( jEnd - jStart ) * ( iEnd - iStart );
		
		raysPerTriangleCounterView[ triangleIndex ] = raysPerTriangleCount;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, raysPerTriangleCounterLambda );
	
	const int taskCount = TNL::sum( raysPerTriangleCounterArray );
	// set worst case scenario limit for rays per thread so that thread count will never exceed size of the threadToTriangleMapArray
	const int raysPerThreadLimit = std::max({16, std::max({0, taskCount - STL.triangleCount}) / ( threadCountMax - STL.triangleCount )}); 
			
	IntArrayType &threadsPerTriangleScanArray = raysPerTriangleCounterArray;
	threadsPerTriangleScanArray = ( raysPerTriangleCounterArray + raysPerThreadLimit - 1 ) / raysPerThreadLimit;
	TNL::Algorithms::inplaceExclusiveScan( threadsPerTriangleScanArray );
		
	auto rayHitIndexLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		// transform into the coordinate system of the LBM grid
		const float ax = axView[ triangleIndex ] - Info.ox;
		const float ay = ayView[ triangleIndex ] - Info.oy;
		const float az = azView[ triangleIndex ] - Info.oz;
		const float bx = bxView[ triangleIndex ] - Info.ox;
		const float by = byView[ triangleIndex ] - Info.oy;
		const float bz = bzView[ triangleIndex ] - Info.oz;
		const float cx = cxView[ triangleIndex ] - Info.ox;
		const float cy = cyView[ triangleIndex ] - Info.oy;
		const float cz = czView[ triangleIndex ] - Info.oz;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const int axInt = (int)(round( ax * scale )) * 2 + 1;
		const int ayInt = (int)(round( ay * scale )) * 2 + 1;
		const int bxInt = (int)(round( bx * scale )) * 2 + 1;
		const int byInt = (int)(round( by * scale )) * 2 + 1;
		const int cxInt = (int)(round( cx * scale )) * 2 + 1;
		const int cyInt = (int)(round( cy * scale )) * 2 + 1;
		// now figure out which rays can possibly hit the triangle -> get bounds
		const int xIntMin = std::max({ 0, std::min({ axInt, bxInt, cxInt, (int)(Info.cellCountX-1)*100 }) });
		const int xIntMax = std::min({ (int)(Info.cellCountX-1)*100, std::max({ axInt, bxInt, cxInt, 0 }) });
		const int yIntMin = std::max({ 0, std::min({ ayInt, byInt, cyInt, (int)(Info.cellCountY-1)*100 }) });
		const int yIntMax = std::min({ (int)(Info.cellCountY-1)*100, std::max({ ayInt, byInt, cyInt, 0 }) });
		const int iStart = (xIntMin + 99) / 100;
		const int iEnd = xIntMax / 100 + 1;
		const int jStart = (yIntMin + 99) / 100;
		const int jEnd = yIntMax / 100 + 1;
		// Prepare cayIntculation of the intersection yes no detection
		// Here we will have to switch to long long because they get multiplied and an integer could overflow if a triangle is bigger than 300 cells
		// A long long is large enough if the triangle is up to about 20M cells
		// Transform the triangle into coordinate system where the first ray is [iMin, jMin]
		const long long xLongMin = 100LL * (long long)iStart;
		const long long yLongMin = 100LL * (long long)jStart;
		const long long axLong = axInt - xLongMin;
		const long long ayLong = ayInt - yLongMin;
		const long long bxLong = bxInt - xLongMin;
		const long long byLong = byInt - yLongMin;
		const long long cxLong = cxInt - xLongMin;
		const long long cyLong = cyInt - yLongMin;
		const long long abxLong = bxLong - axLong;
		const long long abyLong = byLong - ayLong;
		const long long bcxLong = cxLong - bxLong;
		const long long bcyLong = cyLong - byLong;
		const long long caxLong = axLong - cxLong;
		const long long cayLong = ayLong - cyLong;
		// Projection of the triangle area, if it is zero, there are no intersections
		const long long signedArea = abxLong * bcyLong - abyLong * bcxLong;
		if ( signedArea == 0LL ) return; 
		long long qzLong = 1LL;
		if ( signedArea < 0LL ) qzLong = -1LL; // now ABCQ has positive volume
		// Calculate edge functions for the first ray [iMin, jMin]
		const long long wab0 = qzLong * ( abxLong * ( -byLong ) - abyLong * ( -bxLong ) );
		const long long wbc0 = qzLong * ( bcxLong * ( -cyLong ) - bcyLong * ( -cxLong ) );
		const long long wca0 = qzLong * ( caxLong * ( -ayLong ) - cayLong * ( -axLong ) );
		long long wabJ = wab0; // we will be changing this each time j changes
		long long wbcJ = wbc0;
		long long wcaJ = wca0;
		long long wab;
		long long wbc;
		long long wca;
		// Derivatives of the edge function with respect to i and j
		// We will be adding this each time we do a step in i or j direction
		const long long dwab_di = - qzLong * abyLong * 100LL;
		const long long dwab_dj = qzLong * abxLong * 100LL;
		const long long dwbc_di = - qzLong * bcyLong * 100LL;
		const long long dwbc_dj = qzLong * bcxLong * 100LL;
		const long long dwca_di = - qzLong * cayLong * 100LL;
		const long long dwca_dj = qzLong * caxLong * 100LL;
		// Prepare calculation of the intersection coordinate
		const float v1x = bx - ax;
		const float v1y = by - ay;
		const float v1z = bz - az;
		const float v2x = cx - ax;
		const float v2y = cy - ay;
		const float v2z = cz - az;
		const float nx = v1y * v2z - v1z * v2y;
		const float ny = v1z * v2x - v1x * v2z;
		const float nz = v1x * v2y - v1y * v2x;
		const float maxZ = std::max({az, bz, cz});
		const float minZ = std::min({az, bz, cz});
		const float midZ = 0.5f * (maxZ + minZ);
		
		float rayZ;
		
		for ( int j = jStart; j < jEnd; j++ )
		{
			wab = wabJ;
			wbc = wbcJ;
			wca = wcaJ;
			for ( int i = iStart; i < iEnd; i++ )
			{
				bool rayHit = getRayHitYesNo( i, j, wab, wbc, wca, 
											axInt, ayInt, bxInt, byInt, cxInt, cyInt, 
											abxLong, abyLong, bcxLong, bcyLong, caxLong, cayLong );
				if ( rayHit ) 
				{
					const int writePosition = TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(hitsPerRayCounterView(i, j), 1);
					if ( writePosition < rayMapDepth )
					{
						const float rayX = i * Info.res;
						const float rayY = j * Info.res;
						
						if (nz != 0.0f) 
						{
							rayZ = az - (nx * (rayX - ax) + ny * (rayY - ay)) / nz;
							if ( rayZ > maxZ ) rayZ = maxZ;
							else if ( rayZ < minZ ) rayZ = minZ;
						}
						else rayZ = midZ;
						int k = (int)ceilf(rayZ / Info.res);
						rayMapView(i, j, writePosition) = k;
					}
				}
				// add the increments after ending one i pass - we will be increasing i by 1
				wab = wab + dwab_di;
				wbc = wbc + dwbc_di;
				wca = wca + dwca_di;
			}
			// add the increments after ending one j pass - we will be increasing j by 1
			wabJ = wabJ + dwab_dj;
			wbcJ = wbcJ + dwbc_dj;
			wcaJ = wcaJ + dwca_dj;
		}
		
		raysPerTriangleCounterView[ triangleIndex ] = ( jEnd - jStart ) * ( iEnd - iStart );
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rayHitIndexLambda );
	
	// check if intersection count fits within rayMapDepth
	auto fetch = [ = ] __cuda_callable__( const int singleIndex )
	{
		const int i = singleIndex % Info.cellCountX;
		const int j = singleIndex / Info.cellCountX;
		return hitsPerRayCounterView( i, j );
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return TNL::max( a, b );
	};
	const int start = 0;
	const int end = Info.cellCountX * Info.cellCountY;
	const int maxIntersectionCount = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0 );
	if ( maxIntersectionCount > rayMapDepth ) 
	{
		std::cout << "maxIntersectionCount: " << maxIntersectionCount << std::endl;
		std::cout << "rayMapDepth: " << rayMapDepth << std::endl;
		throw std::runtime_error("Voxelization failed, intersection count exceeded ray map depth. Static constexpr int rayMapDepth can be increased in include/types.h -> VoxelizerStruct");
	}
	
	// sort the intersections in ascending order
	auto rayLambda = [=] __cuda_callable__ ( const IntPairType& doubleIndex ) mutable
	{
		const int iCell = doubleIndex.x();
		const int jCell = doubleIndex.y();
		// load all intersections
		int intersections[rayMapDepth];
		for ( int layer = 0; layer < rayMapDepth; layer++ ) intersections[layer] = rayMapView( iCell, jCell, layer );
		// sort
		for ( int layer = 1; layer < rayMapDepth; layer++ ) 
		{
			int key = intersections[layer];
			int slider = layer - 1;
			while ( slider >= 0 && intersections[slider] > key ) 
			{
				intersections[slider + 1] = intersections[slider];
				slider = slider - 1;
			}
			intersections[slider + 1] = key;
		}
		// write sorted intersections
		for ( int layer = 0; layer < rayMapDepth; layer++ ) rayMapView( iCell, jCell, layer ) = intersections[layer];
	};
	IntPairType startList{ 0, 0 };
	IntPairType endList{ Info.cellCountX, Info.cellCountY };
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(startList, endList, rayLambda );	
}
