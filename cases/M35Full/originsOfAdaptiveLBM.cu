static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 8;
static constexpr int GRID_REBUILD_PERIOD = 24;

static constexpr int GRID_LEVEL_COUNT = 2;
static constexpr float SMAGORINSKY_CONSTANT = 0.0f;

int iterationChunk = 1000;
constexpr int iterationCount = 5000;

constexpr float resGlobal = 0.30f; 														// mm

constexpr float uzInlet = 0.008f; 														// also works as nominal LBM Mach number	
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

void findCellState( GridStruct &Grid, const int iTarget, const int jTarget, const int kTarget )
{
	std::cout << "LAUNCHING CELL REPORT " << std::endl;
	std::cout << "Searching state of cell i " << iTarget << " j " << jTarget << " k " << kTarget << std::endl;
	InfoStruct &Info = Grid.Info;
	FloatArray2DTypeCPU fArrayCPU;
	fArrayCPU = Grid.fArray;
	
	IntArrayTypeCPU iArrayCPU;
	IntArrayTypeCPU jArrayCPU;
	IntArrayTypeCPU kArrayCPU;
	iArrayCPU = Grid.IJK.iArray;
	jArrayCPU = Grid.IJK.jArray;
	kArrayCPU = Grid.IJK.kArray;
	
	IntArrayTypeCPU jPlusArrayCPU;
	IntArrayTypeCPU kPlusArrayCPU;
	IntArrayTypeCPU jkPlusArrayCPU;
	IntArrayTypeCPU jMinusArrayCPU;
	IntArrayTypeCPU kMinusArrayCPU;
	jPlusArrayCPU = Grid.NBR.jPlusArray;
	kPlusArrayCPU = Grid.NBR.kPlusArray;
	jkPlusArrayCPU = Grid.NBR.jkPlusArray;
	jMinusArrayCPU = Grid.NBR.jMinusArray;
	kMinusArrayCPU = Grid.NBR.kMinusArray;
	
	BoolArrayTypeCPU movingBouncebackMarkerArrayCPU;
	movingBouncebackMarkerArrayCPU = Grid.movingBouncebackMarkerArray;
	BoolArrayTypeCPU bouncebackMarkerArrayCPU;
	bouncebackMarkerArrayCPU = Grid.bouncebackMarkerArray;
	
	for ( int cell = 0; cell < Info.cellCount; cell++ )
	{				
		const int iCell = iArrayCPU( cell );
		const int jCell = jArrayCPU( cell );
		const int kCell = kArrayCPU( cell );
		
		if ( iCell != iTarget || jCell != jTarget || kCell != kTarget ) continue;
	
		std::cout << "Cell found, its index is now " << cell << std::endl;		
		std::cout << "moving bounceback state of cell " << cell << ": " << movingBouncebackMarkerArrayCPU[ cell ] << std::endl;		
		
		std::cout << "iCell " << iCell << " jCell " << jCell << " kCell " << kCell << std::endl;
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusArrayCPU( cell );
		NBR.kPlus = kPlusArrayCPU( cell );
		NBR.jkPlus = jPlusArrayCPU( kPlusArrayCPU( cell ) );
		NBR.jMinus = jMinusArrayCPU( cell );
		NBR.kMinus = kMinusArrayCPU( cell );
		finishNBRAll( NBR, Info );
		
		// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
		// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
		// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
		// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
		
		int fullNBRList[27];
		// for each direction this holds the neighbour where f[i] will be pulled from in the next iteration
		// 0: Center
		fullNBRList[0]  = cell;
		// 1-6: Straight directions (Faces)
		fullNBRList[1]  = NBR.iMinus; 			// cx=1  -> nx=-1
		fullNBRList[2]  = NBR.iPlus;  			// cx=-1 -> nx=1
		fullNBRList[3]  = NBR.kPlus;  			// cz=-1 -> nz=1
		fullNBRList[4]  = NBR.kMinus; 			// cz=1  -> nz=-1
		fullNBRList[5]  = NBR.jPlus;  			// cy=-1 -> ny=1
		fullNBRList[6]  = NBR.jMinus; 			// cy=1  -> ny=-1
		// 7-18: Diagonal directions (Edges)
		fullNBRList[7]  = kPlusArrayCPU( NBR.iMinus );	// cx=1,  cz=-1 -> nx=-1, nz=1
		fullNBRList[8]  = kMinusArrayCPU( NBR.iPlus );	// cx=-1, cz=1  -> nx=1,  nz=-1
		fullNBRList[9]  = kMinusArrayCPU( NBR.iMinus );	// cx=1,  cz=1  -> nx=-1, nz=-1
		fullNBRList[10] = kPlusArrayCPU( NBR.iPlus ); 	// cx=-1, cz=-1 -> nx=1,  nz=1
		fullNBRList[11] = jPlusArrayCPU( NBR.iPlus ); 	// cx=-1, cy=-1 -> nx=1,  ny=1
		fullNBRList[12] = jMinusArrayCPU( NBR.iMinus );	// cx=1,  cy=1  -> nx=-1, ny=-1
		fullNBRList[13] = kPlusArrayCPU( NBR.jMinus );	// cy=1,  cz=-1 -> ny=-1, nz=1
		fullNBRList[14] = kMinusArrayCPU( NBR.jPlus );	// cy=-1, cz=1  -> ny=1,  nz=-1
		fullNBRList[15] = jMinusArrayCPU( NBR.iPlus );	// cx=-1, cy=1  -> nx=1,  ny=-1
		fullNBRList[16] = jPlusArrayCPU( NBR.iMinus );	// cx=1,  cy=-1 -> nx=-1, ny=1
		fullNBRList[17] = kMinusArrayCPU( NBR.jMinus );	// cy=1,  cz=1  -> ny=-1, nz=-1
		fullNBRList[18] = kPlusArrayCPU( NBR.jPlus ); 	// cy=-1, cz=-1 -> ny=1,  nz=1
		// 19-26: Corner directions (Vertices)
		fullNBRList[19] = kPlusArrayCPU( jMinusArrayCPU( NBR.iPlus ) ); 	// cx=-1, cy=1,  cz=-1 -> nx=1,  ny=-1, nz=1
		fullNBRList[20] = kMinusArrayCPU( jPlusArrayCPU( NBR.iMinus ) ); 	// cx=1,  cy=-1, cz=1  -> nx=-1, ny=1,  nz=-1
		fullNBRList[21] = kMinusArrayCPU( jPlusArrayCPU( NBR.iPlus ) ); 	// cx=-1, cy=-1, cz=1  -> nx=1,  ny=1,  nz=-1
		fullNBRList[22] = kPlusArrayCPU( jMinusArrayCPU( NBR.iMinus ) ); 	// cx=1,  cy=1,  cz=-1 -> nx=-1, ny=-1, nz=1
		fullNBRList[23] = kPlusArrayCPU( jPlusArrayCPU( NBR.iMinus ) ); 	// cx=1,  cy=-1, cz=-1 -> nx=-1, ny=1,  nz=1
		fullNBRList[24] = kMinusArrayCPU( jMinusArrayCPU( NBR.iPlus ) ); 	// cx=-1, cy=1,  cz=1  -> nx=1,  ny=-1, nz=-1
		fullNBRList[25] = kPlusArrayCPU( jPlusArrayCPU( NBR.iPlus ) );  	// cx=-1, cy=-1, cz=-1 -> nx=1,  ny=1,  nz=1
		fullNBRList[26] = kMinusArrayCPU( jMinusArrayCPU( NBR.iMinus ) );	// cx=1,  cy=1,  cz=1  -> nx=-1, ny=-1, nz=-1
		// now look at each neighbour if they are or were MBB or are BB 
		bool isMovingBounceback[27] = {false};
		bool isBounceback[27] = {false};
		for ( int direction = 1; direction < 27; direction++ )
		{
			isMovingBounceback[direction] = movingBouncebackMarkerArrayCPU( fullNBRList[direction] );
			isBounceback[direction] = bouncebackMarkerArrayCPU( fullNBRList[direction] );
		}
		
		std::cout << "MBB nbrs: ";
		for (int i = 0; i < 27; ++i)
		{
			std::cout << isMovingBounceback[i] << " ";
		}
		std::cout << '\n';

		std::cout << " BB nbrs: ";
		for (int i = 0; i < 27; ++i)
		{
			std::cout << isBounceback[i] << " ";
		}
		std::cout << '\n';
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, Grid.esotwistFlipper, Grid.Info );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayCPU.getElement(fReadIndex[direction], cellReadIndex[direction]);
		
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, f );
		float u = std::sqrt(ux * ux + uy * uy + uz * uz);
		
		std::cout << "rho " << rho << std::endl;
		std::cout << "u " << u << std::endl;
	}
}

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
			//if ( grids[level].Info.iterationsFinished < 55 ) updateMovingBounceback( grids[level], Voxelizer );
			//else updateMovingBouncebackDEBUG( grids[level], Voxelizer );
			updateMovingBounceback( grids[level], Voxelizer );
		}
	}
	if ( grids[level].Info.updatesSinceRebuild >= GRID_REBUILD_PERIOD )
    {
		for ( int sublevel = std::max(1, level); sublevel < GRID_LEVEL_COUNT; sublevel++) updateInterface(grids[sublevel-1], grids[sublevel]);
		rebuildGrids( grids, Voxelizer, level );
	}
	//applyNonReflectiveOutletZ(grids[level]);
	//std::cout << "COLLISION " << std::endl;
    updateGrid(grids[level]);
    //if ( grids[level].Info.iterationsFinished >= 312 ) findCellState( grids[0], 95, 173, 97 );
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
	
	for ( int iteration = 0; iteration <= iterationCount; iteration++ )
	{
		if ( iteration % iterationChunk == 0 || iteration > 31000000 )
		{
			std::cout << std::endl;
			std::cout << "Finished iteration " << iteration << std::endl;
			
			lapTimer.stop();
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)usefulCellUpdatesPerIteration * (float)iterationChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			const int r = 14.f;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration );
			const float rotatingFrameUy = - ( r / 1000.f ) * angularVelocity;
			if (system(("python3 ../../include/plotter/plotterRotatingFrame.py " + std::to_string(rotatingFrameUy)).c_str()) != 0) {}
			
			
			const int iCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountX/2 + 75;
			exportSectionCutPlotZY( grids, iCut, iteration+1 );
			if (system("python3 ../../include/plotter/plotterGridID.py") != 0) {}
			
			lapTimer.reset();
			lapTimer.start();
		}
		applyGlobalUpdate(grids, 0, Voxelizer, STLImpellerStationary, STLImpellerMoving );
	}
		
	return EXIT_SUCCESS;
}
