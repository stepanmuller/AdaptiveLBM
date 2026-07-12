#pragma once

#include <iostream>
#include <sstream>
#include <cmath>
#include <fstream> 
#include <cstdlib>
#include <limits>
#include <climits>
#include <string>
#include <vector>
#include <algorithm>
#include <new>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Timer.h>

//------------------------------------------------------------------------------------
//------------------------------- CONSTANTS ------------------------------------------
//------------------------------------------------------------------------------------



//------------------------------------------------------------------------------------
//--------------------------- ARRAYS, VECTORS  ---------------------------------------
//------------------------------------------------------------------------------------

using BoolArrayType = TNL::Containers::Vector< bool, TNL::Devices::Cuda, size_t >;
using BoolArrayTypeCPU = TNL::Containers::Vector< bool, TNL::Devices::Host, size_t >;
												
using IntArrayType = TNL::Containers::Vector< int, TNL::Devices::Cuda, size_t >;
using IntConstViewType = TNL::Containers::VectorView< const int, TNL::Devices::Cuda, size_t >;
using IntArrayTypeCPU = TNL::Containers::Vector< int, TNL::Devices::Host, size_t >;

using IntArray2DType = TNL::Containers::NDArray< int, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Cuda >;
using IntArray2DTypeCPU = TNL::Containers::NDArray< int, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Host >;
												
using IntArray3DType = TNL::Containers::NDArray< int, 
												TNL::Containers::SizesHolder< size_t, 0, 0, 0 >,
												std::index_sequence< 0, 1, 2 >,
												TNL::Devices::Cuda >;

using FloatArrayType = TNL::Containers::Vector< float, TNL::Devices::Cuda, size_t >;
using FloatArrayTypeCPU = TNL::Containers::Vector< float, TNL::Devices::Host, size_t >;

using BoolArray2DType = TNL::Containers::NDArray< bool, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Cuda >;

using FloatArray2DType = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Cuda >;
using FloatArray2DTypeCPU = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Host >;
												
using FloatArray3DType = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0, 0 >,
												std::index_sequence< 0, 1, 2 >,
												TNL::Devices::Cuda >;
using FloatArray3DTypeCPU = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0, 0 >,
												std::index_sequence< 0, 1, 2 >,
												TNL::Devices::Host >;

using IntPairType = TNL::Containers::StaticArray< 2, int >;											
using IntTripleType = TNL::Containers::StaticArray< 3, int >;

//------------------------------------------------------------------------------------
//--------------------------------- STRUCTS  -----------------------------------------
//------------------------------------------------------------------------------------

struct InfoStruct { float gridID = 0; float gridMemoryMB; int iterationsFinished = 0;
					float res = 1.f; float ox = 0.f; float oy = 0.f; float oz = 0.f; 
					float nu = 1.f; float dtPhys = 1.f; 
					int cellCountX = 0; int cellCountY = 0; int cellCountZ = 0; 
					int cellCount = 0; int cellCountFull = 0; int cellCountOld = 0;
					int refinementCount = 0; int deepRefinementCount = 0; 
					int fineToCoarseCount = 0; int coarseToFineCount = 0; 
					int memoryCount = 0; int memoryCountFull = 0; 
					int fineToCoarseMemoryCount = 0; int coarseToFineMemoryCount = 0;
					bool esotwistFlipper = 0; int updatesSinceRebuild = 0; int updatesSinceMovingBouncebackUpdate = 0;
					float pRegulator = 0.f; float iRegulator = 0.f; };

struct MarkerStruct { 	bool fluid = 0; bool bounceback = 0; bool movingBounceback = 0;
						bool BCRho = 0; bool BCU = 0; bool BCNonReflectiveOutlet = 0; 
						bool refinement = 0; bool deepRefinement = 0; };
						
struct BCRhoUGStruct { float rho = 1.f; float ux = 0.f; float uy = 0.f; float uz = 0.f; float gx = 0.f; float gy = 0.f; float gz = 0.f; };

struct BoundsStruct { float xmin; float ymin; float zmin; float xmax; float ymax; float zmax; }; 
					
// IJK holds cell indexes on X, Y, Z axes within the Grid that owns it
struct IJKArrayStructCPU; // just declaring first
struct IJKArrayStruct { IntArrayType iArray; IntArrayType jArray; IntArrayType kArray; 
						IJKArrayStruct() = default;
						// Constructor copies data from IJKArrayStructCPU
						IJKArrayStruct( const IJKArrayStructCPU& IJKCPU );
					};

struct IJKArrayStructCPU { 	IntArrayTypeCPU iArray; IntArrayTypeCPU jArray; IntArrayTypeCPU kArray; 
							IJKArrayStructCPU() = default;
							// Constructor copies data from IJKArrayStruct (GPU)
							IJKArrayStructCPU( const IJKArrayStruct& IJK )
							{
								iArray = IJK.iArray; 
								jArray = IJK.jArray;
								kArray = IJK.kArray;
							}
						};
inline IJKArrayStruct::IJKArrayStruct(const IJKArrayStructCPU& IJKCPU) {
    iArray = IJKCPU.iArray;
    jArray = IJKCPU.jArray;
    kArray = IJKCPU.kArray;
}

struct rayMapStruct { IntArray3DType rayMapArray; IntArray2DType hitCounterArray; };

struct VoxelizerStruct { 	InfoStruct Info; 
							rayMapStruct rayMapBounceback;
							rayMapStruct rayMapMovingBounceback;
							rayMapStruct rayMapTotal; };

// NBR holds:
// Connectivity for Esotwist: indexes of 3 neighbours in the positive direction jPlus, kPlus, jkPlus. 
// Thanks to cell sorting where X runs the fastest, the indexes for remaining 4 neighbours iPlus, ijPlus, ikPlus, ijkPlus are just +1 to self, jPlus, kPlus, jkPlus.
// Then it holds 2 more neighbour indexes in the main negative directions jMinus, kMinus (iMinus would be self-1)
// In addition to that it holds a list of 10 bool vectors which are 1 if the respective neighbour is also truly geometric neighbour (there is no gap between)
// These 10 vectors are ordered as is iPlus, jPlus, ijPlus, kPlus, ikPlus, jkPlus, ijkPlus, iMinus, jMinus, kMinus
struct NBRArrayStruct { IntArrayType jPlusArray; IntArrayType kPlusArray; IntArrayType jkPlusArray; 
						IntArrayType jMinusArray; IntArrayType kMinusArray; 
						BoolArray2DType isGeometricMarkerArray; }; 
										
struct NBRStruct { 	int self;
					int iPlus; int jPlus; int kPlus; int ijPlus; int ikPlus; int jkPlus; int ijkPlus; 
					int iMinus; int jMinus; int kMinus;
					int isGeometricMarker[10]; }; 
					
struct NBRHoleMapStruct { IntArray3DType holeStartArray; IntArray2DType startCounterArray; IntArray3DType holeEndArray; IntArray2DType endCounterArray; };

struct SkeletonGridStruct { InfoStruct Info; 
							IntArrayType intBuffer1; IntArrayType intBuffer2; IntArrayType intBuffer3;
							NBRHoleMapStruct NBRHoleMap;
							BoolArrayType keepCellMarkerArray; BoolArrayType movingBouncebackMarkerArray; BoolArrayType markerBuffer; };	

struct GridStruct { InfoStruct Info; IJKArrayStruct IJK; NBRArrayStruct NBR; 
					FloatArray2DType fArray; bool esotwistFlipper = false;
					IntArrayType parentMapArray; IntArrayType childMapArray; 
					IntArrayType fineToCoarseIndexArray; IntArrayType coarseToFineIndexArray;
					IntArrayType &intBuffer1; // This will point to NBR.jMinusArray which we temporarily use as a buffer and then refill it correctly
					IntArrayType &intBuffer2; // This will point to NBR.kMinusArray which we temporarily use as a buffer and then refill it correctly
					IntArrayType &intBuffer3; // This will point to NBR.jkPlusArray which we temporarily use as a buffer and then refill it correctly
					NBRHoleMapStruct NBRHoleMap;
					BoolArrayType keepCellMarkerArray; 
					BoolArrayType bouncebackMarkerArray; BoolArrayType movingBouncebackMarkerArray; 
					BoolArrayType refinementMarkerArray; BoolArrayType deepRefinementMarkerArray;
					BoolArrayType fineToCoarseMarkerArray; BoolArrayType coarseToFineMarkerArray;
					BoolArrayType markerBuffer;
					SkeletonGridStruct SkeletonGrid;
					GridStruct()
						: intBuffer1(NBR.jMinusArray),
						  intBuffer2(NBR.kMinusArray),
						  intBuffer3(NBR.jkPlusArray)
					{}
					}; 		
					
struct STLStructCPU { 	int triangleCount;
						FloatArrayTypeCPU axArray; FloatArrayTypeCPU ayArray; FloatArrayTypeCPU azArray; 
						FloatArrayTypeCPU bxArray; FloatArrayTypeCPU byArray; FloatArrayTypeCPU bzArray; 
						FloatArrayTypeCPU cxArray; FloatArrayTypeCPU cyArray; FloatArrayTypeCPU czArray; 
						BoundsStruct Bounds; }; 

struct STLStruct { 	static constexpr int threadsToTrianglesRatio = 4;
					int triangleCount;
					FloatArrayType axArray; FloatArrayType ayArray; FloatArrayType azArray; 
					FloatArrayType bxArray; FloatArrayType byArray; FloatArrayType bzArray; 
					FloatArrayType cxArray; FloatArrayType cyArray; FloatArrayType czArray; 
					BoundsStruct Bounds; 
					IntArrayType raysPerTriangleCounterArray;
					IntArrayType threadToTriangleMapArray;
					STLStruct() = default;
					// Constructor copies data from STLStructCPU
					STLStruct( const STLStructCPU& STLCPU )
					{
						triangleCount = STLCPU.triangleCount;
						axArray = STLCPU.axArray; ayArray = STLCPU.ayArray;	azArray = STLCPU.azArray; 
						bxArray = STLCPU.bxArray; byArray = STLCPU.byArray;	bzArray = STLCPU.bzArray; 
						cxArray = STLCPU.cxArray; cyArray = STLCPU.cyArray;	czArray = STLCPU.czArray; 
						Bounds = STLCPU.Bounds;	
					}
				};

struct FlowReportStruct { float ux = 0.f; float uy = 0.f; float uz = 0.f; float rho = 1.f; float areamm2 = 0.f; }; 

struct LocalDuStruct { float duxdx = 0.f; float duydy = 0.f; float duzdz = 0.f; float duXY = 0.f; float duYZ = 0.f; float duXZ = 0.f; };

struct SectionCutStruct { 	FloatArray2DType rhoArray; FloatArray2DType uxArray; FloatArray2DType uyArray; FloatArray2DType uzArray; 
							FloatArray2DType markerArray; FloatArray2DType scalarTransportArray; IntArray2DType gridIDArray; };
							
struct SectionCutStructCPU { 	FloatArray2DTypeCPU rhoArray; FloatArray2DTypeCPU uxArray; FloatArray2DTypeCPU uyArray; FloatArray2DTypeCPU uzArray; 
								FloatArray2DTypeCPU markerArray; FloatArray2DTypeCPU scalarTransportArray; IntArray2DTypeCPU gridIDArray; };
