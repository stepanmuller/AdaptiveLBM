static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 20;
static constexpr int GRID_LEVEL_COUNT = 3;

#include "../../include/types.h"
#include "../../include/adaptiveGridFunctions.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"

std::string STLPathMain = "M-Jet_35_pump_main.STL";
std::string STLPathImpeller = "M-Jet_35_impeller.STL";

int main(int argc, char **argv)
{
	// BuildFinerGrid test
	GridStruct GridCoarse;
	GridCoarse.Info.cellCount = 4;
	GridCoarse.Info.refinementCount = 3;
	GridCoarse.IJK.iArray = std::vector<int>{ 0, 1, 2, 1 };
	GridCoarse.IJK.jArray = std::vector<int>{ 0, 0, 0, 1 };
	GridCoarse.IJK.kArray = std::vector<int>{ 0, 0, 0, 0 };
	GridCoarse.NBR.jPlusArray = std::vector<int>{ 0, 3, 2, 1 };
	GridCoarse.NBR.kPlusArray = std::vector<int>{ 0, 1, 2, 3 };
	GridCoarse.NBR.jkPlusArray = std::vector<int>{ 0, 1, 2, 3 };
	GridCoarse.childMapArray = std::vector<int>{ 0, -1, -1, 4 };
	GridCoarse.intBuffer1.setSize( 4 );
	GridCoarse.intBuffer2.setSize( 4 );
	GridCoarse.intBuffer3.setSize( 4 );
	GridCoarse.refinementMarkerArray.setSize( 4 ); 
	GridCoarse.refinementMarkerArray.setElement( 0, true );
	GridCoarse.refinementMarkerArray.setElement( 1, false );
	GridCoarse.refinementMarkerArray.setElement( 2, true );
	GridCoarse.refinementMarkerArray.setElement( 3, true );
	GridCoarse.markerBuffer.setSize( 4 );
	
	GridStruct GridFine;
	GridFine.Info.cellCountOld = 16;
	GridFine.Info.cellCountFull = 24;
	GridFine.IJK.iArray = std::vector<int>{ 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1 };
	GridFine.IJK.jArray = std::vector<int>{ 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1 };
	GridFine.IJK.kArray = std::vector<int>{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1 };
	GridFine.NBR.jPlusArray.setSize( 24 );
	GridFine.NBR.kPlusArray.setSize( 24 );
	GridFine.NBR.jkPlusArray.setSize( 24 );
	GridFine.parentMapArray = std::vector<int>{ 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1 };
	GridFine.intBuffer1.setSize( 24 );
	GridFine.intBuffer2.setSize( 24 );
	GridFine.intBuffer3.setSize( 24 );
	GridFine.NBR.isGeometricMarkerArray.setSizes( 10, 24 );
	
	buildFinerGrid( GridCoarse, GridFine );
	
	std::cout << "iArrayFine " << GridFine.IJK.iArray << std::endl;
	std::cout << "jArrayFine " << GridFine.IJK.jArray << std::endl;
	std::cout << "kArrayFine " << GridFine.IJK.kArray << std::endl;
	std::cout << "jPlusArrayFine " << GridFine.NBR.jPlusArray << std::endl;
	std::cout << "kPlusArrayFine " << GridFine.NBR.kPlusArray << std::endl;
	std::cout << "jkPlusArrayFine " << GridFine.NBR.jkPlusArray << std::endl;
	std::cout << "childMapArrayCoarse " << GridCoarse.childMapArray << std::endl;
	std::cout << "parentMapArrayFine " << GridFine.parentMapArray << std::endl;
	std::cout << "oldToFull " << GridFine.intBuffer1 << std::endl;

	// STL tests
	STLStruct STLMain;
	readSTL( STLMain, STLPathMain );
	STLStruct STLImpeller;
	readSTL( STLImpeller, STLPathImpeller );
	
	// Voxelizer tests
	VoxelizerStruct Voxelizer;
	Voxelizer.Info.ox = -18.f;
	Voxelizer.Info.oy = -18.f;
	Voxelizer.Info.res = 0.05f;
	Voxelizer.Info.cellCountX = (int)(36.f / Voxelizer.Info.res);
	Voxelizer.Info.cellCountY = Voxelizer.Info.cellCountX;
	
	TNL::Timer Timer;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapBounceback, STLMain, Voxelizer );
	Timer.stop();
	std::cout << "Voxelization 1 took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpeller, Voxelizer );
	Timer.stop();
	std::cout << "Voxelization 2 took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
	Timer.stop();
	std::cout << "Summing ray maps took " << Timer.getRealTime() << " s" << std::endl;
	
	// STL tests
	
	return EXIT_SUCCESS;
}
