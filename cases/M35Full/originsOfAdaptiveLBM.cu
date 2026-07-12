static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 8;
static constexpr int GRID_REBUILD_PERIOD = 24;

static constexpr int GRID_LEVEL_COUNT = 3;
static constexpr float SMAGORINSKY_CONSTANT = 0.1;

constexpr int iterationChunk = 8;
constexpr int iterationCount = 100000;

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
		rebuildGrids( grids, Voxelizer, level );
	}
	//applyNonReflectiveOutletZ(grids[level]);
    updateGrid(grids[level]);
    if (level < GRID_LEVEL_COUNT - 1) // I am not the finest grid
    {
        for ( int i = 0; i < 2; i++) applyGlobalUpdate(grids, level + 1, Voxelizer, STLImpellerStationary, STLImpellerMoving );
        updateInterface(grids[level], grids[level + 1]);
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
	
	std::cout << "Maximum cells travelled by MBB per iteration: " << dtPhysGlobal * angularVelocity * 17.5f / resGlobal << std::endl;
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iteration = 0; iteration < iterationCount; iteration++ )
	{
		if ( iteration % iterationChunk == 0 )
		{
			std::cout << "Finished iteration " << iteration << std::endl;
			
			lapTimer.stop();
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)usefulCellUpdatesPerIteration * (float)iterationChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			/*
			const float fMax = findMaxFloatArray2D( grids[0].fArray, 27, grids[0].Info.cellCount );
			std::cout << "fMax " << fMax << std::endl;
			if ( iteration >= 69 )
			{
				GridStruct &Grid = grids[0];
				InfoStruct &Info = Grid.Info;
				FloatArray2DTypeCPU fArrayCPU;
				fArrayCPU = Grid.fArray;
				
				IntArrayTypeCPU jPlusArrayCPU;
				IntArrayTypeCPU kPlusArrayCPU;
				IntArrayTypeCPU jkPlusArrayCPU;
				jPlusArrayCPU = Grid.NBR.jPlusArray;
				kPlusArrayCPU = Grid.NBR.kPlusArray;
				jkPlusArrayCPU = Grid.NBR.jkPlusArray;
				
				BoolArrayTypeCPU movingBouncebackMarkerCPU;
				movingBouncebackMarkerCPU = Grid.movingBouncebackMarkerArray;
				
				for ( int cell = 0; cell < Grid.Info.cellCount; cell++ )
				{
					NBRStruct NBR;
					NBR.self = cell;
					NBR.jPlus = jPlusArrayCPU[ cell ];
					NBR.kPlus = kPlusArrayCPU[ cell ];
					NBR.jkPlus = jkPlusArrayCPU[ cell ];
					finishNBRPlus( NBR, Grid.Info );
					float f[27];
					int cellReadIndex[27];
					int fReadIndex[27];
					getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, Grid.esotwistFlipper, Grid.Info );
					for ( int direction = 0; direction < 27; direction++ )
					{
						if ( cellReadIndex[direction] >= Info.cellCount )
						{
							std::cout << "Faulty cellReadIndex for cell " << cell << std::endl;
						}
					}
					for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayCPU.getElement(fReadIndex[direction], cellReadIndex[direction]);
					for ( int direction = 0; direction < 27; direction++ )
					{
						if ( std::fabs( f[direction] ) > 1.f ) 
						{
							std::cout << "Maximum exceeded in cell " << cell << std::endl;
							std::cout << "Direction " << direction << std::endl;
							std::cout << "cellReadIndex " << cellReadIndex[direction] << std::endl;
							std::cout << "cell moving bounceback marker " << movingBouncebackMarkerCPU[cell] << std::endl;
						}
					}
				}
			}
				*/
			
			const int r = 14.f;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration / iterationChunk );
			if (system("python3 ../../include/plotter/plotterGridID.py") != 0) {}
			/*
			const int iCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountX/2;
			exportSectionCutPlotZY( grids, iCut, iteration+1 );
			if (system("python3 ../../include/plotter/plotterGridID.py") != 0) {}
			*/
			lapTimer.reset();
			lapTimer.start();
		}
		applyGlobalUpdate(grids, 0, Voxelizer, STLImpellerStationary, STLImpellerMoving );
	}
		
	return EXIT_SUCCESS;
}
