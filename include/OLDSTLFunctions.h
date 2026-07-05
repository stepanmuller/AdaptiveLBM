#pragma once

void readSTL( STLStructCPU &STLCPU, const std::string &filename )
{
	std::cout << "Reading STL: " << filename << std::endl;
	std::ifstream file(filename, std::ios::binary);
	if ( !file.is_open() ) throw std::runtime_error("Failed to open STL file");
	
	 // Skip header
    char header[80];
    file.read( header, 80 );

    uint32_t triangleCount32;
	file.read( reinterpret_cast<char*>(&triangleCount32), sizeof(uint32_t) );
	int triangleCount = static_cast<int>( triangleCount32 );
	STLCPU.triangleCount = triangleCount;
    
    std::cout<<"	triangleCount: "<< triangleCount << std::endl;

    STLCPU.axArray = FloatArrayTypeCPU( triangleCount );
    STLCPU.ayArray = FloatArrayTypeCPU( triangleCount );
    STLCPU.azArray = FloatArrayTypeCPU( triangleCount );

    STLCPU.bxArray = FloatArrayTypeCPU( triangleCount );
    STLCPU.byArray = FloatArrayTypeCPU( triangleCount );
    STLCPU.bzArray = FloatArrayTypeCPU( triangleCount );

    STLCPU.cxArray = FloatArrayTypeCPU( triangleCount );
    STLCPU.cyArray = FloatArrayTypeCPU( triangleCount );
    STLCPU.czArray = FloatArrayTypeCPU( triangleCount );
    
    // Initialize minmax
    STLCPU.xmin = std::numeric_limits<float>::max();
	STLCPU.ymin = std::numeric_limits<float>::max();
	STLCPU.zmin = std::numeric_limits<float>::max();

	STLCPU.xmax = std::numeric_limits<float>::lowest();
	STLCPU.ymax = std::numeric_limits<float>::lowest();
	STLCPU.zmax = std::numeric_limits<float>::lowest();

    for ( int triangle = 0; triangle < triangleCount; triangle++ )
    {
        float ax, ay, az;
        float bx, by, bz;
        float cx, cy, cz;
        uint16_t attr;

		// Skip normal (nx, ny, nz) = 3 floats = 12 bytes
		file.seekg(12, std::ios::cur);

        file.read( reinterpret_cast<char*>(&ax), 4 );
        file.read( reinterpret_cast<char*>(&ay), 4 );
        file.read( reinterpret_cast<char*>(&az), 4 );

        file.read( reinterpret_cast<char*>(&bx), 4 );
        file.read( reinterpret_cast<char*>(&by), 4 );
        file.read( reinterpret_cast<char*>(&bz), 4 );

        file.read( reinterpret_cast<char*>(&cx), 4 );
        file.read( reinterpret_cast<char*>(&cy), 4 );
        file.read( reinterpret_cast<char*>(&cz), 4 );

        file.read( reinterpret_cast<char*>(&attr), 2 );

        STLCPU.axArray[triangle] = ax;
        STLCPU.ayArray[triangle] = ay;
        STLCPU.azArray[triangle] = az;

        STLCPU.bxArray[triangle] = bx;
        STLCPU.byArray[triangle] = by;
        STLCPU.bzArray[triangle] = bz;

        STLCPU.cxArray[triangle] = cx;
        STLCPU.cyArray[triangle] = cy;
        STLCPU.czArray[triangle] = cz;
        
         // Update bounding box (vertices only)
        STLCPU.xmin = std::min(STLCPU.xmin, std::min({ax, bx, cx}));
        STLCPU.ymin = std::min(STLCPU.ymin, std::min({ay, by, cy}));
        STLCPU.zmin = std::min(STLCPU.zmin, std::min({az, bz, cz}));

        STLCPU.xmax = std::max(STLCPU.xmax, std::max({ax, bx, cx}));
        STLCPU.ymax = std::max(STLCPU.ymax, std::max({ay, by, cy}));
        STLCPU.zmax = std::max(STLCPU.zmax, std::max({az, bz, cz}));
    }
    std::cout << "	xmin xmax: " << STLCPU.xmin << " " << STLCPU.xmax << "\n";
    std::cout << "	ymin ymax: " << STLCPU.ymin << " " << STLCPU.ymax << "\n";
    std::cout << "	zmin zmax: " << STLCPU.zmin << " " << STLCPU.zmax << "\n";
}


void checkSTLEdges( STLStruct &STL )
// For every edge, counts number of triangles that share it. Must be always 2 for a closed STL.
{
	std::cout << "Starting STL check for shared edges" << std::endl;
	auto axArrayView = STL.axArray.getConstView();
	auto ayArrayView = STL.ayArray.getConstView();
	auto azArrayView = STL.azArray.getConstView();
	auto bxArrayView = STL.bxArray.getConstView();
	auto byArrayView = STL.byArray.getConstView();
	auto bzArrayView = STL.bzArray.getConstView();
	auto cxArrayView = STL.cxArray.getConstView();
	auto cyArrayView = STL.cyArray.getConstView();
	auto czArrayView = STL.czArray.getConstView();
	
	IntArray2DType edgeCounterArray;
	edgeCounterArray.setSizes( STL.triangleCount, 3 );
	edgeCounterArray.setValue(0);
	auto edgeCounterArrayView = edgeCounterArray.getView();

    auto counterLambda = [ = ] __cuda_callable__( const int triangle1Index ) mutable
    {		
		float triangle1[9];
		triangle1[0] = axArrayView[ triangle1Index ];
		triangle1[1] = ayArrayView[ triangle1Index ];
		triangle1[2] = azArrayView[ triangle1Index ];
		triangle1[3] = bxArrayView[ triangle1Index ];
		triangle1[4] = byArrayView[ triangle1Index ];
		triangle1[5] = bzArrayView[ triangle1Index ];
		triangle1[6] = cxArrayView[ triangle1Index ];
		triangle1[7] = cyArrayView[ triangle1Index ];
		triangle1[8] = czArrayView[ triangle1Index ];
		
		for ( int triangle2Index = 0; triangle2Index < STL.triangleCount; triangle2Index++ ) 
		{
			float triangle2[9];
			triangle2[0] = axArrayView[ triangle2Index ];
			triangle2[1] = ayArrayView[ triangle2Index ];
			triangle2[2] = azArrayView[ triangle2Index ];
			triangle2[3] = bxArrayView[ triangle2Index ];
			triangle2[4] = byArrayView[ triangle2Index ];
			triangle2[5] = bzArrayView[ triangle2Index ];
			triangle2[6] = cxArrayView[ triangle2Index ];
			triangle2[7] = cyArrayView[ triangle2Index ];
			triangle2[8] = czArrayView[ triangle2Index ];
			
			for ( int edge1 = 0; edge1 < 3; edge1++ )
			{
				for ( int edge2 = 0; edge2 < 3; edge2++ )
				{
					float ax1 = triangle1[3*edge1];
					float ay1 = triangle1[3*edge1+1];
					float az1 = triangle1[3*edge1+2];
					float bx1 = triangle1[3*((edge1+1)%3)];
					float by1 = triangle1[3*((edge1+1)%3)+1];
					float bz1 = triangle1[3*((edge1+1)%3)+2];
					
					float ax2 = triangle2[3*edge2];
					float ay2 = triangle2[3*edge2+1];
					float az2 = triangle2[3*edge2+2];
					float bx2 = triangle2[3*((edge2+1)%3)];
					float by2 = triangle2[3*((edge2+1)%3)+1];
					float bz2 = triangle2[3*((edge2+1)%3)+2];
					// testing same orientation of the edge
					if (ax1 == ax2 && ay1 == ay2 && az1 == az2 && bx1 == bx2 && by1 == by2 && bz1 == bz2) 
					{
						TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(edgeCounterArrayView(triangle1Index, edge1), 1);
					}
					// testing reverse orientation of the edge
					if (ax1 == bx2 && ay1 == by2 && az1 == bz2 && bx1 == ax2 && by1 == ay2 && bz1 == az2) 
					{
						TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(edgeCounterArrayView(triangle1Index, edge1), 1);
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, counterLambda );
	int errorCounter = 0;
	for ( int triangleIndex = 0; triangleIndex < STL.triangleCount; triangleIndex++ )
    {
		int ABcount = edgeCounterArray.getElement( triangleIndex, 0 );
		int BCcount = edgeCounterArray.getElement( triangleIndex, 1 );
		int CAcount = edgeCounterArray.getElement( triangleIndex, 2 );
		if (ABcount != 2 || BCcount != 2 || CAcount != 2)
		{
			errorCounter++;
			std::cout << "	Edge problem on triangle " << triangleIndex << ", ABcount: " << ABcount << ", BCcount: " << BCcount << ", CAcount: " << CAcount << std::endl;
		}
	}    
	std::cout<< "	Total shared edge problems: " << errorCounter << std::endl; 
}

__host__ __device__ bool getRayHitYesNo( 	const long long &ax, const long long &ay, 
											const long long &bx, const long long &by,
											const long long &cx, const long long &cy )
{
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

// DIAD Version. DIAD stands for directly adressed. This means, IJK arrays are provided that capture positions of existing cells (not all cells exist in the block)
void ApplyMarkersInsideSTL( BoolArrayType &markerArray, IJKArrayStruct &IJK, STLStruct &STL, const bool &insideMarkerValue, InfoStruct &Info )
{
	auto markerArrayView = markerArray.getView();
	
	auto axArrayView = STL.axArray.getConstView();
	auto ayArrayView = STL.ayArray.getConstView();
	auto azArrayView = STL.azArray.getConstView();
	auto bxArrayView = STL.bxArray.getConstView();
	auto byArrayView = STL.byArray.getConstView();
	auto bzArrayView = STL.bzArray.getConstView();
	auto cxArrayView = STL.cxArray.getConstView();
	auto cyArrayView = STL.cyArray.getConstView();
	auto czArrayView = STL.czArray.getConstView();
	
	IntArray2DType intersectionCounterArray;
	intersectionCounterArray.setSizes( Info.cellCountX, Info.cellCountY );
	intersectionCounterArray.setValue( 0 );
	auto intersectionCounterArrayView = intersectionCounterArray.getView();

    auto counterLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		// transform into the coordinate system of the LBM grid
		const float ax = axArrayView[ triangleIndex ] + STL.ox - Info.ox;
		const float ay = ayArrayView[ triangleIndex ] + STL.oy - Info.oy;
		const float bx = bxArrayView[ triangleIndex ] + STL.ox - Info.ox;
		const float by = byArrayView[ triangleIndex ] + STL.oy - Info.oy;
		const float cx = cxArrayView[ triangleIndex ] + STL.ox - Info.ox;
		const float cy = cyArrayView[ triangleIndex ] + STL.oy - Info.oy;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const long long ak = (long long)(round( ax * scale )) * 2 + 1;
		const long long al = (long long)(round( ay * scale )) * 2 + 1;
		const long long bk = (long long)(round( bx * scale )) * 2 + 1;
		const long long bl = (long long)(round( by * scale )) * 2 + 1;
		const long long ck = (long long)(round( cx * scale )) * 2 + 1;
		const long long cl = (long long)(round( cy * scale )) * 2 + 1;
		
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

				const bool rayHit = getRayHitYesNo( ak0, al0, bk0, bl0, ck0, cl0 );

				if (rayHit) 
				{
					TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(intersectionCounterArrayView(i, j), 1);
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, counterLambda );
	
	auto fetch = [ = ] __cuda_callable__( const int singleIndex )
	{
		const int i = singleIndex % Info.cellCountX;
		const int j = singleIndex / Info.cellCountX;
		return intersectionCounterArrayView( i, j );
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return TNL::max( a, b );
	};
	const int start = 0;
	const int end = Info.cellCountX * Info.cellCountY;
	int intersectionCountMax = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0 );
	
	if ( intersectionCountMax < 1 ) intersectionCountMax = 1;
	// extracting the intersection indexes
	IntArray3DType intersectionIndexArray;
	intersectionIndexArray.setSizes( intersectionCountMax, Info.cellCountX, Info.cellCountY );
	intersectionIndexArray.setValue( Info.cellCountZ );
	auto intersectionIndexArrayView = intersectionIndexArray.getView();
	intersectionCounterArray.setValue(0);
	
	auto rayHitIndexLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		// transform into the coordinate system of the LBM grid
		const float ax = axArrayView[ triangleIndex ] + STL.ox - Info.ox;
		const float ay = ayArrayView[ triangleIndex ] + STL.oy - Info.oy;
		const float az = azArrayView[ triangleIndex ] + STL.oz - Info.oz;
		const float bx = bxArrayView[ triangleIndex ] + STL.ox - Info.ox;
		const float by = byArrayView[ triangleIndex ] + STL.oy - Info.oy;
		const float bz = bzArrayView[ triangleIndex ] + STL.oz - Info.oz;
		const float cx = cxArrayView[ triangleIndex ] + STL.ox - Info.ox;
		const float cy = cyArrayView[ triangleIndex ] + STL.oy - Info.oy;
		const float cz = czArrayView[ triangleIndex ] + STL.oz - Info.oz;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const long long ak = (long long)(round( ax * scale )) * 2 + 1;
		const long long al = (long long)(round( ay * scale )) * 2 + 1;
		const long long bk = (long long)(round( bx * scale )) * 2 + 1;
		const long long bl = (long long)(round( by * scale )) * 2 + 1;
		const long long ck = (long long)(round( cx * scale )) * 2 + 1;
		const long long cl = (long long)(round( cy * scale )) * 2 + 1;
		
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

				const bool rayHit = getRayHitYesNo( ak0, al0, bk0, bl0, ck0, cl0 );

				if (rayHit) 
				{
					const int writePosition = TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(intersectionCounterArrayView(i, j), 1);
					const float rayX = i * Info.res;
					const float rayY = j * Info.res;
					const float rayZ = getRayHitZCoordinate( ax, ay, az, bx, by, bz, cx, cy, cz, rayX, rayY );
					int k = (int)std::max( 0.0f, ceilf(rayZ / Info.res) );
					k = std::min( k, Info.cellCountZ );
					intersectionIndexArrayView(writePosition, i, j) = k;
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rayHitIndexLambda );
	
	auto rayLambda = [=] __cuda_callable__ ( const IntPairType& doubleIndex ) mutable
	{
		const int iCell = doubleIndex.x();
		const int jCell = doubleIndex.y();
		
		for ( int layer = 1; layer < intersectionCountMax; layer++ ) // sort
		{
			int key = intersectionIndexArrayView( layer, iCell, jCell );
			int slider = layer - 1;
			while ( slider >= 0 && intersectionIndexArrayView( slider, iCell, jCell ) > key ) 
			{
				intersectionIndexArrayView( slider + 1, iCell, jCell ) = intersectionIndexArrayView( slider, iCell, jCell );
				slider = slider - 1;
			}
			intersectionIndexArrayView( slider + 1, iCell, jCell ) = key;
		}
	};
	IntPairType startList{ 0, 0 };
	IntPairType endList{ Info.cellCountX, Info.cellCountY };
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(startList, endList, rayLambda );	
	
	auto iArrayView = IJK.iArray.getView();
	auto jArrayView = IJK.jArray.getView();
	auto kArrayView = IJK.kArray.getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iArrayView[ cell ];
		const int jCell = jArrayView[ cell ];
		const int kCell = kArrayView[ cell ];
		
		bool markerValue = !insideMarkerValue; // we are always starting outside
		for ( int interval = 0; interval <= intersectionCountMax; interval++ )
		{
			int start = 0;
			int end = 0;
			if ( interval == 0 ) end = intersectionIndexArrayView( 0, iCell, jCell );
			else if ( interval == intersectionCountMax ) 
			{
				start = intersectionIndexArrayView( intersectionCountMax-1, iCell, jCell );
				end = Info.cellCountZ;
			}
			else
			{
				start = intersectionIndexArrayView( interval-1, iCell, jCell );
				end = intersectionIndexArrayView( interval, iCell, jCell );
			}
			if ( (kCell >= start) && (kCell < end) ) 
			{
				markerArrayView( cell ) = markerValue;
				return;
			}
			markerValue = !markerValue;
		}		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
}

void rotateSTLAlongX( STLStruct &STL, float &radians )
{
	auto ayArrayView = STL.ayArray.getView();
	auto azArrayView = STL.azArray.getView();
	auto byArrayView = STL.byArray.getView();
	auto bzArrayView = STL.bzArray.getView();
	auto cyArrayView = STL.cyArray.getView();
	auto czArrayView = STL.czArray.getView();
	
	auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
	{
		const float ay = ayArrayView[ triangleIndex ];
		const float az = azArrayView[ triangleIndex ];
		const float by = byArrayView[ triangleIndex ];
		const float bz = bzArrayView[ triangleIndex ];
		const float cy = cyArrayView[ triangleIndex ];
		const float cz = czArrayView[ triangleIndex ];
		
		const float s = sinf(radians);
		const float c = cosf(radians);
		const float newAy = ay * c - az * s;
		const float newAz = ay * s + az * c;
		const float newBy = by * c - bz * s;
		const float newBz = by * s + bz * c;
		const float newCy = cy * c - cz * s;
		const float newCz = cy * s + cz * c;
		
		ayArrayView[ triangleIndex ] = newAy;
		azArrayView[ triangleIndex ] = newAz;
		byArrayView[ triangleIndex ] = newBy;
		bzArrayView[ triangleIndex ] = newBz;
		cyArrayView[ triangleIndex ] = newCy;
		czArrayView[ triangleIndex ] = newCz;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rotateLambda );
}

void rotateSTLAlongY( STLStruct &STL, float &radians )
{
	auto axArrayView = STL.axArray.getView();
	auto azArrayView = STL.azArray.getView();
	auto bxArrayView = STL.bxArray.getView();
	auto bzArrayView = STL.bzArray.getView();
	auto cxArrayView = STL.cxArray.getView();
	auto czArrayView = STL.czArray.getView();
	
	auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
	{
		const float ax = axArrayView[ triangleIndex ];
		const float az = azArrayView[ triangleIndex ];
		const float bx = bxArrayView[ triangleIndex ];
		const float bz = bzArrayView[ triangleIndex ];
		const float cx = cxArrayView[ triangleIndex ];
		const float cz = czArrayView[ triangleIndex ];
		
		const float s = sinf(radians);
		const float c = cosf(radians);
		
		// Standard right-handed rotation around the Y axis
		const float newAx = ax * c + az * s;
		const float newAz = -ax * s + az * c;
		const float newBx = bx * c + bz * s;
		const float newBz = -bx * s + bz * c;
		const float newCx = cx * c + cz * s;
		const float newCz = -cx * s + cz * c;
		
		axArrayView[ triangleIndex ] = newAx;
		azArrayView[ triangleIndex ] = newAz;
		bxArrayView[ triangleIndex ] = newBx;
		bzArrayView[ triangleIndex ] = newBz;
		cxArrayView[ triangleIndex ] = newCx;
		czArrayView[ triangleIndex ] = newCz;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rotateLambda );
}

void rotateSTLAlongZ( STLStruct &STL, const float &radians )
{
	auto axArrayView = STL.axArray.getView();
	auto ayArrayView = STL.ayArray.getView();
	auto bxArrayView = STL.bxArray.getView();
	auto byArrayView = STL.byArray.getView();
	auto cxArrayView = STL.cxArray.getView();
	auto cyArrayView = STL.cyArray.getView();
	
    auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		const float ax = axArrayView[ triangleIndex ];
		const float ay = ayArrayView[ triangleIndex ];
		const float bx = bxArrayView[ triangleIndex ];
		const float by = byArrayView[ triangleIndex ];
		const float cx = cxArrayView[ triangleIndex ];
		const float cy = cyArrayView[ triangleIndex ];
		
		const float s = sinf(radians);
		const float c = cosf(radians);
		const float newAx = ax * c - ay * s;
		const float newAy = ax * s + ay * c;
		const float newBx = bx * c - by * s;
		const float newBy = bx * s + by * c;
		const float newCx = cx * c - cy * s;
		const float newCy = cx * s + cy * c;
		
		axArrayView[ triangleIndex ] = newAx;
		ayArrayView[ triangleIndex ] = newAy;
		bxArrayView[ triangleIndex ] = newBx;
		byArrayView[ triangleIndex ] = newBy;
		cxArrayView[ triangleIndex ] = newCx;
		cyArrayView[ triangleIndex ] = newCy;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rotateLambda );
}

void multiplyBoolArrays( BoolArrayType &markerArray1, BoolArrayType &markerArray2, BoolArrayType &resultArray )
{
	const int size = resultArray.getSize();
	auto markerArrayView1 = markerArray1.getView();
	auto markerArrayView2 = markerArray2.getView();
	auto resultArrayView = resultArray.getView();
    auto cellLambda = [ = ] __cuda_callable__( const int cell ) mutable
    {
		resultArrayView( cell ) = markerArrayView1( cell ) && markerArrayView2( cell );
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, size, cellLambda );
}

void sumBoolArrays( BoolArrayType &markerArray1, BoolArrayType &markerArray2, BoolArrayType &resultArray )
{
	const int size = resultArray.getSize();
	auto markerArrayView1 = markerArray1.getView();
	auto markerArrayView2 = markerArray2.getView();
	auto resultArrayView = resultArray.getView();
    auto cellLambda = [ = ] __cuda_callable__( const int cell ) mutable
    {
		resultArrayView( cell ) = markerArrayView1( cell ) || markerArrayView2( cell );
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, size, cellLambda );
}

void invertBoolArray( BoolArrayType &resultArray )
{
	const int size = resultArray.getSize();
	auto resultArrayView = resultArray.getView();
    auto cellLambda = [ = ] __cuda_callable__( const int cell ) mutable
    {
		resultArrayView( cell ) = !resultArrayView( cell );
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, size, cellLambda );
}
