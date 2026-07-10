static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 20;
static constexpr int GRID_LEVEL_COUNT = 2;

#include "../../include/types.h"
#include "../../include/adaptiveGridFunctions.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"

std::string STLPathMain = "M-Jet_35_pump_main.STL";
std::string STLPathImpeller = "M-Jet_35_impeller.STL";

int main(int argc, char **argv)
{
	STLStruct STLMain;
	readSTL( STLMain, STLPathMain );
	STLStruct STLImpeller;
	readSTL( STLImpeller, STLPathImpeller );
	
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	grids[ 0 ].Info.res = 10.0f;
	initializeGrids( grids, STLMain.Bounds, 0 );
	std::cout << "Skeleton cellCount = " << grids[0].SkeletonGrid.Info.cellCount << std::endl;
	
	// Voxelizer tests
	VoxelizerStruct Voxelizer;
	Voxelizer.Info = grids[ GRID_LEVEL_COUNT-1 ].Info;
	
	TNL::Timer Timer;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapBounceback, STLMain, Voxelizer );
	Timer.stop();
	std::cout << "Voxelizing Bounceback took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpeller, Voxelizer );
	Timer.stop();
	std::cout << "Voxelizing Moving Bounceback took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
	Timer.stop();
	std::cout << "Summing ray maps took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	rebuildGrids( grids, Voxelizer, 0 );
	Timer.stop();
	std::cout << "Initial rebuild grids took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	rebuildGrids( grids, Voxelizer, 0 );
	Timer.stop();
	std::cout << "Second rebuild grids took " << Timer.getRealTime() << " s" << std::endl;
	
	return EXIT_SUCCESS;
}
