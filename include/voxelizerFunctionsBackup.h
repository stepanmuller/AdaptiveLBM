#pragma once

__host__ __device__ bool getRayHitYesNoOLD( 	const long long &ax, const long long &ay, 
											const long long &bx, const long long &by,
											const long long &cx, const long long &cy )
{
	// The triangle is already translated so that the ray lies at [0, 0]
	// The triangle coordinates are already converted to odd long ints to
	//		1) Completely avoid the case of hitting a vertex
	//		2) Avoid any arithmetical error that would be caused by floats
    // First, find how ABC is oriented. A -> B -> C going anti clockwise gives positive signed area
    const long long abx = bx - ax;
    const long long aby = by - ay;
    const long long bcx = cx - bx;
    const long long bcy = cy - by;
    const long long cax = ax - cx;
    const long long cay = ay - cy;
    
    const long long signedArea = abx * bcy - aby * bcx;
    if ( signedArea == 0 ) return false; 
    
    long long qz = 1;
    if ( signedArea < 0 ) qz = -1; // now ABCQ has positive volume
    
    const long long wab = qz * ( abx * ( -by ) - aby * ( -bx ) );
    const long long wbc = qz * ( bcx * ( -cy ) - bcy * ( -cx ) );
    const long long wca = qz * ( cax * ( -ay ) - cay * ( -ax ) );
    if ( wab > 0 && wbc > 0 && wca > 0 ) return true;
    if ( wab < 0 || wbc < 0 || wca < 0 ) return false;
    
    if ( wab == 0 )
    {
		if ( abx == 0 ) // vertical edge
		{
			if ( cx < 0 ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( bx > 0 && cy * bx > by * cx ) return true;
		if ( bx < 0 && cy * bx < by * cx ) return true;
	}
	
	if ( wbc == 0 )
    {
		if ( bcx == 0 ) // vertical edge
		{
			if ( ax < 0 ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( cx > 0 && ay * cx > cy * ax ) return true;
		if ( cx < 0 && ay * cx < cy * ax ) return true;
	}
	
	if ( wca == 0 )
    {
		if ( cax == 0 ) // vertical edge
		{
			if ( bx < 0 ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( ax > 0 && by * ax > ay * bx ) return true;
		if ( ax < 0 && by * ax < ay * bx ) return true;
	}

    return false; 
}

__host__ __device__ float getRayHitZCoordinate(	const float &ax, const float &ay, const float &az,
												const float &bx, const float &by, const float &bz,
												const float &cx, const float &cy, const float &cz,
												const float &rayX, const float &rayY )
{
    float v1x = bx - ax;
    float v1y = by - ay;
    float v1z = bz - az;

    float v2x = cx - ax;
    float v2y = cy - ay;
    float v2z = cz - az;

    float nx = v1y * v2z - v1z * v2y;
    float ny = v1z * v2x - v1x * v2z;
    float nz = v1x * v2y - v1y * v2x;
    
    float rayZ;

    if (nz != 0.0f) 
    {
		rayZ = az - (nx * (rayX - ax) + ny * (rayY - ay)) / nz;
		if ( rayZ > std::max({az, bz, cz}) ) rayZ = std::max({az, bz, cz});
		else if ( rayZ < std::min({az, bz, cz}) ) rayZ = std::min({az, bz, cz});
	}
    else rayZ = 0.5f * ( std::max({az, bz, cz}) + std::min({az, bz, cz}) );
    
    return rayZ;
}

void voxelizeSTL_OLD( const STLStruct &STL, VoxelizerStruct &Voxelizer, IntArray3DType &rayMapArray )
{
	// Result is saved into the rayMap which is a 3D array of size cellCountX * cellCountY * rayMapDepth
	// K indexes of intersections with each I, J ray are saved in ascending order. The rest up to rayMapDepth is filled with int max
	// If rayMapDepth is too low, throw error
	InfoStruct &Info = Voxelizer.Info;
	IntArray2DType &hitsPerRayCounterArray = Voxelizer.hitsPerRayCounterArray;
	constexpr int rayMapDepth = VoxelizerStruct::rayMapDepth;
	
	if ( hitsPerRayCounterArray.getSizes()[0] < 1 )
	{
		hitsPerRayCounterArray.setSizes( Info.cellCountX, Info.cellCountY );
		Voxelizer.rayMapBouncebackArray.setSizes( Info.cellCountX, Info.cellCountY, rayMapDepth );
		Voxelizer.rayMapMovingBouncebackArray.setSizes( Info.cellCountX, Info.cellCountY, rayMapDepth );
		Voxelizer.rayMapTotalArray.setSizes( Info.cellCountX, Info.cellCountY, rayMapDepth );
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
		const long long ak = (long long)(round( ax * scale )) * 2 + 1;
		const long long al = (long long)(round( ay * scale )) * 2 + 1;
		const long long bk = (long long)(round( bx * scale )) * 2 + 1;
		const long long bl = (long long)(round( by * scale )) * 2 + 1;
		const long long ck = (long long)(round( cx * scale )) * 2 + 1;
		const long long cl = (long long)(round( cy * scale )) * 2 + 1;
		// now figure out which rays can possibly hit the triangle -> get bounds
		const long long kmin = std::max({ 0LL, std::min({ ak, bk, ck, (long long)(Info.cellCountX-1)*100 }) });
		const long long kmax = std::min({ (long long)(Info.cellCountX-1)*100, std::max({ ak, bk, ck, 0LL }) });
		const long long lmin = std::max({ 0LL, std::min({ al, bl, cl, (long long)(Info.cellCountY-1)*100 }) });
		const long long lmax = std::min({ (long long)(Info.cellCountY-1)*100, std::max({ al, bl, cl, 0LL }) });
		const long long imin = (kmin + 99) / 100;
		const long long imax = kmax / 100;
		const long long jmin = (lmin + 99) / 100;
		const long long jmax = lmax / 100;
		
		for ( int j = jmin; j <= jmax; j++ )
		{
			for ( int i = imin; i <= imax; i++ )
			{
				const long long rayK = i * 100;
				const long long rayL = j * 100;
				// transform the triangle into coordinate system where ray is [0, 0]
				const long long ak0 = ak - rayK;
				const long long al0 = al - rayL;
				const long long bk0 = bk - rayK;
				const long long bl0 = bl - rayL;
				const long long ck0 = ck - rayK;
				const long long cl0 = cl - rayL;

				const bool rayHit = getRayHitYesNoOLD( ak0, al0, bk0, bl0, ck0, cl0 );

				if (rayHit) 
				{
					const int writePosition = TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(hitsPerRayCounterView(i, j), 1);
					if ( writePosition < rayMapDepth )
					{
						const float rayX = i * Info.res;
						const float rayY = j * Info.res;
						const float rayZ = getRayHitZCoordinate( ax, ay, az, bx, by, bz, cx, cy, cz, rayX, rayY );
						int k = (int)ceilf(rayZ / Info.res);
						rayMapView(i, j, writePosition) = k;
					}
				}
			}
		}
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
		std::cout << "Voxelization failed, intersection count exceeded ray map depth" << std::endl;
		std::cout << "maxIntersectionCount: " << maxIntersectionCount << std::endl;
		std::cout << "rayMapDepth: " << rayMapDepth << std::endl;
		std::cout << "static constexpr int rayMapDepth can be modified in include/types.h -> VoxelizerStruct" << std::endl;
		throw std::runtime_error("Terminating ");
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
