#include <iostream>
#include <cmath>
#include <fstream> 
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Timer.h>

using BoolArrayType = TNL::Containers::Array< bool, TNL::Devices::Cuda, size_t >;
using BoolArrayTypeCPU = TNL::Containers::Array< bool, TNL::Devices::Host, size_t >;
												
using IntArrayType = TNL::Containers::Array< int, TNL::Devices::Cuda, size_t >;
using IntArrayConstViewType = TNL::Containers::ArrayView< const int, TNL::Devices::Cuda, size_t >;
using IntArrayTypeCPU = TNL::Containers::Array< int, TNL::Devices::Host, size_t >;

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

using FloatArrayType = TNL::Containers::Array< float, TNL::Devices::Cuda, size_t >;
using FloatArrayTypeCPU = TNL::Containers::Array< float, TNL::Devices::Host, size_t >;

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

struct FStructCPU { IntArrayTypeCPU shifter; FloatArray2DTypeCPU fArray; };

struct SectionCutStruct { 	FloatArray2DType rhoArray; FloatArray2DType uxArray; FloatArray2DType uyArray; FloatArray2DType uzArray; 
							FloatArray2DType markerArray; FloatArray2DType scalarTransportArray; IntArray2DType gridIDArray; };
struct SectionCutStructCPU { 	FloatArray2DTypeCPU rhoArray; FloatArray2DTypeCPU uxArray; FloatArray2DTypeCPU uyArray; FloatArray2DTypeCPU uzArray; 
								FloatArray2DTypeCPU markerArray; FloatArray2DTypeCPU scalarTransportArray; IntArray2DTypeCPU gridIDArray; };

struct Section3DStruct { FloatArray3DType rhoArray; FloatArray3DType uxArray; FloatArray3DType uyArray; FloatArray3DType uzArray; FloatArray3DType markerArray; };
struct Section3DStructCPU { FloatArray3DTypeCPU rhoArray; FloatArray3DTypeCPU uxArray; FloatArray3DTypeCPU uyArray; FloatArray3DTypeCPU uzArray; FloatArray3DTypeCPU markerArray; };	

struct InfoStruct { float gridID = 0; float res = 1.f; int cellCountX; int cellCountY; int cellCountZ; int cellCount; float ox = 0.f; float oy = 0.f; float oz = 0.f; float nu = 1.f; float dtPhys = 1.f; 
					int iSubgridStart = 0; int jSubgridStart = 0; int kSubgridStart = 0; int iSubgridEnd = 0; int jSubgridEnd = 0; int kSubgridEnd = 0; int cellCountFineToCoarse = 0; int cellCountCoarseToFine = 0; 
					float pRegulator = 0.f; float iRegulator = 0.f; }; 
								
struct GridStruct { InfoStruct Info; FloatArray2DType fArray; IntArrayType shifter; BoolArrayType bouncebackMarkerArray; }; 

struct MarkerStruct { bool fluid = 0; bool bounceback = 0; bool movingBounceback = 0; bool forcedVelocity = 0;
	bool periodicX = 0; bool periodicY = 0; bool periodicZ = 0; bool givenRho = 0; bool givenUxUyUz = 0; bool nonReflectiveOutlet = 0; bool refinement = 0; bool ghost = 0; };

struct ScalarTransportMarkerStruct { bool givenT = 0; };
					
struct STLStructCPU { 	FloatArrayTypeCPU axArray; FloatArrayTypeCPU ayArray; FloatArrayTypeCPU azArray; 
						FloatArrayTypeCPU bxArray; FloatArrayTypeCPU byArray; FloatArrayTypeCPU bzArray; 
						FloatArrayTypeCPU cxArray; FloatArrayTypeCPU cyArray; FloatArrayTypeCPU czArray; 
						float xmin; float ymin; float zmin; float xmax; float ymax; float zmax; 
						float ox = 0.f; float oy = 0.f; float oz = 0.f; int triangleCount; }; // ox, oy, oz is the position of STL origin in global coordinates

struct STLStruct { 	FloatArrayType axArray; FloatArrayType ayArray; FloatArrayType azArray; 
					FloatArrayType bxArray; FloatArrayType byArray; FloatArrayType bzArray; 
					FloatArrayType cxArray; FloatArrayType cyArray; FloatArrayType czArray; 
					float xmin; float ymin; float zmin; float xmax; float ymax; float zmax; 
					float ox; float oy; float oz; int triangleCount; // ox, oy, oz is the position of STL origin in global coordinates
					STLStruct() = default;
					// Constructor copies data from STLStructCPU
					STLStruct( const STLStructCPU& STLCPU )
					{
						axArray = STLCPU.axArray;
						ayArray = STLCPU.ayArray;
						azArray = STLCPU.azArray; 
						bxArray = STLCPU.bxArray;
						byArray = STLCPU.byArray;
						bzArray = STLCPU.bzArray; 
						cxArray = STLCPU.cxArray;
						cyArray = STLCPU.cyArray;
						czArray = STLCPU.czArray; 
						xmin = STLCPU.xmin;
						ymin = STLCPU.ymin;
						zmin = STLCPU.zmin; 
						xmax = STLCPU.xmax;
						ymax = STLCPU.ymax;
						zmax = STLCPU.zmax; 
						ox = STLCPU.ox;
						oy = STLCPU.oy;
						oz = STLCPU.oz;
						triangleCount = STLCPU.triangleCount;
					}
				};

struct ScalarTransportStruct { FloatArray2DType TArray; float tauT = 1.f; }; 

struct FlowReportStruct { float ux = 0.f; float uy = 0.f; float uz = 0.f; float rho = 1.f; float areamm2 = 0.f; }; 	
struct XYZBoundsStruct { float xmin; float ymin; float zmin; float xmax; float ymax; float zmax; }; 
struct LocalDuStruct { float duxdx = 0.f; float duydy = 0.f; float duzdz = 0.f; float duXY = 0.f; float duYZ = 0.f; float duXZ = 0.f; };

//------------------------------------------------------------------------------------
//--------------------------------- DIAD SECTION -------------------------------------
//------------------------------------------------------------------------------------
// DIAD stands for directly adressed grid. On such grid, not all cells within the block exist. The cells that exist are saved using IJK coordinates. 
// Neighbour mapping must be kept in memory, because implicit indexing no longer works.

// This holds i, j, k cell indexes for a DIAD grid
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

// This holds cell indexes of the neighbour cells in positive i,j,k directions for Esotwist streaming and negative directions for the main axes only
struct DIADEsotwistNbrArrayStruct { 	IntArrayType iNbrArray; IntArrayType jNbrArray; IntArrayType kNbrArray; 
										IntArrayType ijNbrArray; IntArrayType ikNbrArray; IntArrayType jkNbrArray; 
										IntArrayType ijkNbrArray; 
										IntArrayType iMinNbrArray; IntArrayType jMinNbrArray; IntArrayType kMinNbrArray; }; 
										
// This holds cell indexes of the neighbour cells in only positive i,j,k directions for Esotwist streaming for a single cell
struct DIADEsotwistNbrStruct { int i; int j; int k; int ij; int ik; int jk; int ijk; }; 

struct DIADGridStruct { InfoStruct Info; FloatArray2DType fArray; BoolArrayType bouncebackMarkerArray; BoolArrayType forcedVelocityMarkerArray; BoolArrayType interfaceMarkerArray;
						IJKArrayStruct IJK; DIADEsotwistNbrArrayStruct EsotwistNbrArray; 
						IntArrayType fineToCoarseWriteArray; IntArrayType fineToCoarseReadArray; 
						IntArrayType coarseToFineWriteArray; IntArrayType coarseToFineReadArray; 
						BoolArrayType enforceInterfaceFluid; BoolArrayType enforceInterfaceBounceback;
						bool esotwistFlipper = 0; IntArrayType outletCellArray; }; 
						
