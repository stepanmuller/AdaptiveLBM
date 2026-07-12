static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 4;
static constexpr int GRID_REBUILD_PERIOD = 16;

static constexpr int GRID_LEVEL_COUNT = 3;
static constexpr float SMAGORINSKY_CONSTANT = 0.1;


constexpr float resGlobal = 0.30f; 														// mm

constexpr float uzInlet = 0.01f; 														// also works as nominal LBM Mach number	
constexpr float nuPhys = 1e-6;															// m2/s water
constexpr float rhoNominalPhys = 1000.0f;												// kg/m3 water
constexpr float uzInletPhys = 4.5986f; 													// m/s
constexpr float dtPhysGlobal = (uzInlet / uzInletPhys) * (resGlobal/1000); 				// s
constexpr float invSqrt3 = 0.577350269f; 
constexpr float soundspeedPhys = invSqrt3 * (resGlobal/1000) / dtPhysGlobal; 			// m/s
constexpr float RIn = 3.75f;															// mm
constexpr float ROut = 16.5f;															// mm
constexpr float angularVelocity = 2000.f;												// rad/s
const float boundaryLayerThickness = 0.2f;												// mm

#include "../../include/types.h"
#include "../../include/cellFunctions.h"

std::string STLPathMain = "M-Jet_35_pump_main.STL";
std::string STLPathImpeller = "M-Jet_35_impeller.STL";

__cuda_callable__ void getMarkers( 	const int& iCell, const int& jCell, const int& kCell, 
									MarkerStruct &Marker, const InfoStruct& Info )
{
	if ( Marker.bounceback ) return;
	if ( Marker.movingBounceback ) return;
	if ( kCell == 0 ) Marker.BCU = 1;
	else if ( kCell == Info.cellCountZ-1 ) Marker.BCRho = 1;
	else Marker.fluid = 1;
}

__cuda_callable__ void getInitialRhoUG( BCRhoUGStruct &BCRhoUG,
										const int& iCell, const int& jCell, const int& kCell, 
										const InfoStruct& Info, MarkerStruct &Marker )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = angularVelocity * (r / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( Marker.movingBounceback )
	{
		BCRhoUG.ux = 0.f; //- vt * (y / r);
		BCRhoUG.uy = 0.f; //vt * (x / r);
		BCRhoUG.uz = 0.f;
	}
	else if ( Marker.bounceback )
	{
		BCRhoUG.ux = - vt * (y / r);
		BCRhoUG.uy = vt * (x / r);
		BCRhoUG.uz = 0.f;
	}
	else
	{
		BCRhoUG.ux = 0.f;
		BCRhoUG.uy = 0.f;
		BCRhoUG.uz = uzInlet;
	}
	BCRhoUG.rho = 1.f;
}

__cuda_callable__ void getBCRhoUG( 	BCRhoUGStruct &BCRhoUG,
									const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info, MarkerStruct &Marker )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = angularVelocity * (r / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	const float wallDistancePhys = std::max(0.f, std::min(r - RIn, ROut - r));
	const float delta = std::max( 0.f, std::min( 1.f, wallDistancePhys / boundaryLayerThickness ));
	const float velocityMultiplier = delta * delta * (3.0f - 2.0f * delta);
	if ( Marker.movingBounceback )
	{
		BCRhoUG.ux = - vt * (y / r);
		BCRhoUG.uy = vt * (x / r);
		BCRhoUG.uz = 0.f;
	}
	else
	{
		BCRhoUG.ux = 0.f;
		BCRhoUG.uy = 0.f;
		BCRhoUG.uz = uzInlet * velocityMultiplier;
	}
	BCRhoUG.rho = 1.f;
}

#include "../../include/adaptiveGridFunctions.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/updateInterface.h"
#include "../../include/updateMovingBounceback.h"
#include "../../include/plotter/exportSectionCutPlot.h"

void applyGlobalUpdate( std::vector<GridStruct>& grids, int level, VoxelizerStruct &Voxelizer, STLStruct &STLImpellerStationary, STLStruct &STLImpellerMoving ) 
{
	//applyNonReflectiveOutletZ(grids[level]);
    updateGrid(grids[level]);
    if (level < GRID_LEVEL_COUNT - 1) // I am not the finest grid
    {
        for ( int i = 0; i < 2; i++) applyGlobalUpdate(grids, level + 1, Voxelizer, STLImpellerStationary, STLImpellerMoving );
        updateInterface(grids[level], grids[level + 1]);
    }
    if ( level == GRID_LEVEL_COUNT - 1 ) // I am the finest grid
    {
		if (grids[level].Info.updatesSinceMovingBouncebackUpdate >= MOVING_BOUNCEBACK_UPDATE_PERIOD )
		{
			const float radians = grids[level].Info.iterationsFinished * grids[level].Info.dtPhys * angularVelocity;
			rotateSTLAlongZ( STLImpellerMoving, STLImpellerStationary, radians );
			Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
			voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpellerMoving, Voxelizer );
			sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
			updateMovingBounceback( grids[level], Voxelizer );
		}
	}
    if ( grids[level].Info.updatesSinceRebuild >= GRID_REBUILD_PERIOD )
    {
		const float radians = grids[level].Info.iterationsFinished * grids[level].Info.dtPhys * angularVelocity;
		rotateSTLAlongZ( STLImpellerMoving, STLImpellerStationary, radians );
		Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
		voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpellerMoving, Voxelizer );
		sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
		std::cout << "trying to do rebuildGrids on level " << level << std::endl;
		rebuildGrids( grids, Voxelizer, level );
		std::cout << "rebuildGrids on level " << level << " succeeded" << std::endl;
		if (level < GRID_LEVEL_COUNT - 1) updateInterface(grids[level], grids[level + 1]);
	}
    
}

int main(int argc, char **argv)
{
	// STLs
	STLStruct STLMain;
	readSTL( STLMain, STLPathMain );
	STLStruct STLImpellerStationary;
	readSTL( STLImpellerStationary, STLPathImpeller );
	STLStruct STLImpellerMoving;
	STLImpellerMoving = STLImpellerStationary;
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	grids[ 0 ].Info.res = resGlobal;
	initializeGrids( grids, STLMain.Bounds, 0 );
	
	// Voxelizer 
	VoxelizerStruct Voxelizer;
	Voxelizer.Info = grids[ GRID_LEVEL_COUNT-1 ].Info;
	voxelizeSTL( Voxelizer.rayMapBounceback, STLMain, Voxelizer );
	voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpellerMoving, Voxelizer );
	Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
	
	// first rebuildGrids
	rebuildGrids( grids, Voxelizer, 0 );
	
	int totalCellCount = 0;
	int usefulCellUpdatesPerIteration = 0;
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
	{
		totalCellCount += grids[level].Info.cellCount;
		usefulCellUpdatesPerIteration += ( grids[level].Info.cellCount - grids[level].Info.deepRefinementCount ) * std::pow( 2, level );
	}
	std::cout << "Total cell count " << totalCellCount << std::endl; 
	std::cout << "Useful cell updates per iteration " << usefulCellUpdatesPerIteration << std::endl; 	
	std::cout << std::endl;
	
	for ( int iteration = 0; iteration < 100000; iteration++ )
	{
		if ( iteration % 1 == 0 )
		{
			std::cout << "Finished iteration " << iteration << std::endl;
			const int r = 14.f;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration );
			if (system("python3 ../../include/plotter/plotterGridID.py") != 0) {}
		}
		applyGlobalUpdate(grids, 0, Voxelizer, STLImpellerStationary, STLImpellerMoving );
	}
	
	
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
