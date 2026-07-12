static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 5;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;
static constexpr int GRID_LEVEL_COUNT = 3;
static constexpr float SMAGORINSKY_CONSTANT = 0.1;


constexpr float resGlobal = 0.3f; 
constexpr float uzInlet = 0.01f; 														// also works as nominal LBM Mach number	
constexpr float nuPhys = 1e-6;															// m2/s water
constexpr float rhoNominalPhys = 1000.0f;												// kg/m3 water
constexpr float uzInletPhys = 4.5986f; 													// m/s
constexpr float dtPhysGlobal = (uzInlet / uzInletPhys) * (resGlobal/1000); 				// s
constexpr float invSqrt3 = 0.577350269f; 
constexpr float soundspeedPhys = invSqrt3 * (resGlobal/1000) / dtPhysGlobal; 			// m/s


#include "../../include/types.h"
#include "../../include/adaptiveGridFunctions.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/plotter/OLDexportSectionCutPlot.h"

std::string STLPathMain = "M-Jet_35_pump_main.STL";
std::string STLPathImpeller = "M-Jet_35_impeller.STL";

int main(int argc, char **argv)
{
	STLStruct STLMain;
	readSTL( STLMain, STLPathMain );
	STLStruct STLImpeller;
	readSTL( STLImpeller, STLPathImpeller );
	
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	grids[ 0 ].Info.res = resGlobal;
	initializeGrids( grids, STLMain.Bounds, 0 );
	
	// Voxelizer tests
	VoxelizerStruct Voxelizer;
	Voxelizer.Info = grids[ GRID_LEVEL_COUNT-1 ].Info;
	
	TNL::Timer Timer;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapBounceback, STLMain, Voxelizer );
	Timer.stop();
	std::cout << "Voxelizing Bounceback took " << Timer.getRealTime() << " s" << std::endl;
	std::cout << std::endl;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpeller, Voxelizer );
	Timer.stop();
	std::cout << "Voxelizing Moving Bounceback took " << Timer.getRealTime() << " s" << std::endl;
	std::cout << std::endl;
	
	Timer.reset();
	Timer.start();
	Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
	Timer.stop();
	std::cout << "Summing ray maps took " << Timer.getRealTime() << " s" << std::endl;
	std::cout << std::endl;
	
	Timer.reset();
	Timer.start();
	rebuildGrids( grids, Voxelizer, 0 );
	Timer.stop();
	std::cout << "Initial rebuild grids took " << Timer.getRealTime() << " s" << std::endl;
	std::cout << std::endl;	
	
	std::cout << "Total cell count " << grids[0].Info.cellCount + grids[1].Info.cellCount + grids[2].Info.cellCount << std::endl; 
	
	applyMarkersFromRayMap( grids[GRID_LEVEL_COUNT-1].bouncebackMarkerArray, Voxelizer.rayMapBounceback, grids[GRID_LEVEL_COUNT-1], grids[GRID_LEVEL_COUNT-1].Info.cellCount );
	applyMarkersFromRayMap( grids[GRID_LEVEL_COUNT-1].movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, grids[GRID_LEVEL_COUNT-1], grids[GRID_LEVEL_COUNT-1].Info.cellCount );
	
	const int iCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountX / 2;
	exportSectionCutPlotZY( grids, iCut, 0 );
	system("python3 ../../include/plotter/plotterGridID.py");
	
	/*
	Timer.reset();
	Timer.start();
	rebuildGrids( grids, Voxelizer, 0 );
	Timer.stop();
	std::cout << "Second rebuild grids from level 0 took " << Timer.getRealTime() << " s" << std::endl;
	std::cout << std::endl;
	*/
	
	return EXIT_SUCCESS;
}
