static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 8;
static constexpr int GRID_REBUILD_PERIOD = 24;

static constexpr int GRID_LEVEL_COUNT = 2;
static constexpr float SMAGORINSKY_CONSTANT = 0.f;

int reportChunk = 31;
int plotterChunk = 1;
constexpr int iterationCount = 1502;

constexpr float resGlobal = 0.16f; 														// mm

constexpr float uzInlet = 0.01f; 														// also works as nominal LBM Mach number	
constexpr float nuPhys = 1e-6;															// m2/s water
constexpr float rhoNominalPhys = 1000.0f;												// kg/m3 water
constexpr float massFlowTargetPhys = 3.04f;												// kg/s
constexpr float RIn = 3.75f;															// mm
constexpr float ROut = 16.5f;															// mm
constexpr float inletAreamm2 = 3.14159f * ( ROut * ROut - RIn * RIn);					// mm2
constexpr float uzInletPhys = massFlowTargetPhys / ( rhoNominalPhys * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float dtPhysGlobal = (uzInlet / uzInletPhys) * (resGlobal/1000); 				// s
constexpr float soundspeedPhys = 0.577350269f * (resGlobal/1000) / dtPhysGlobal; 		// m/s (0.577350269f is 1/sqrt(3))
constexpr float angularVelocity = 2000.f;												// rad/s
const float boundaryLayerThickness = 0.2f;												// mm
const float shaftRotationStartDistance = 10.f;											// mm

constexpr float targetInletPower = 0.f;													// W
constexpr float iRegulatorInletStrength = 0.0048f;

#include "../../include/types.h"
#include "../../include/cellFunctions.h"

std::string STLPathMain = "M-Jet_35_pump_main.STL";
std::string STLPathImpeller = "M-Jet_35_impeller.STL";

__cuda_callable__ void getMarkers( 	const int& iCell, const int& jCell, const int& kCell, 
									MarkerStruct &Marker, const InfoStruct& Info )
{
	if ( Marker.bounceback ) return;
	if ( Marker.movingBounceback ) return;
	
	if ( kCell == 0 ) Marker.refinement = 1;
	if ( kCell > Info.cellCountZ - 50 ) Marker.refinement = 1;
	
	if ( kCell == 0 ) Marker.nonReflectiveInlet = 1; // Marker.BCU = 1;
	else if ( kCell == Info.cellCountZ-1 ) Marker.nonReflectiveOutlet = 1; //Marker.BCRho = 1;
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
		BCRhoUG.ux = - vt * (y / r);
		BCRhoUG.uy = vt * (x / r);
		BCRhoUG.uz = 0.f;
	}
	else if ( Marker.bounceback )
	{
		BCRhoUG.ux = 0.f;
		BCRhoUG.uy = 0.f;
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
		
	const float inletDistancePhys = z - Info.oz;
	const float rotationDelta = std::max( 0.f, std::min( 1.f, inletDistancePhys / shaftRotationStartDistance ));
	const float rotationMultiplier = rotationDelta * rotationDelta * (3.0f - 2.0f * rotationDelta);
	
	if ( Marker.movingBounceback )
	{
		BCRhoUG.ux = - vt * (y / r) * rotationMultiplier;
		BCRhoUG.uy = vt * (x / r) * rotationMultiplier;
		BCRhoUG.uz = 0.f;
	}
	else
	{
		BCRhoUG.ux = 0.f;
		BCRhoUG.uy = 0.f;
		BCRhoUG.uz = ( uzInlet + Info.iRegulatorInlet ) * velocityMultiplier;
	}
	if ( Marker.BCRho || Marker.nonReflectiveOutlet ) BCRhoUG.rho = 1.f;
}

#include "../../include/adaptiveGridFunctions.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/updateInterface.h"
#include "../../include/updateMovingBounceback.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/flowReportFunctions.h"

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
		for ( int sublevel = std::max(1, level); sublevel < GRID_LEVEL_COUNT; sublevel++) updateInterface(grids[sublevel-1], grids[sublevel]);
		rebuildGrids( grids, Voxelizer, level );
	}
    updateGrid(grids[level]);
    if (level < GRID_LEVEL_COUNT - 1) // I am not the finest grid
    {
        for ( int i = 0; i < 2; i++) applyGlobalUpdate(grids, level + 1, Voxelizer, STLImpellerStationary, STLImpellerMoving );
        updateInterface(grids[level], grids[level + 1]);
    }
}

void exportHistoryData( const std::vector<float>& historyInletPower, 
                        const std::vector<float>& historyMassFlow, 
                        const std::vector<float>& historyTorque, 
                        const int &currentIteration, int fileNumber ) {
    FILE* fp = fopen("/dev/shm/historyData.bin", "wb");
    if (!fp) return;
    
    int count = std::min({iterationCount-1, currentIteration}) + 1;
    fwrite(&count, sizeof(int), 1, fp);
    
    // Write all three vectors sequentially
    fwrite(historyInletPower.data(), sizeof(float), count, fp);
    fwrite(historyMassFlow.data(), sizeof(float), count, fp);
    fwrite(historyTorque.data(), sizeof(float), count, fp);
    
    fclose(fp);
    
    // Construct the command string to pass the number as an argument
    std::string cmd = "python3 historyPlotter.py " + std::to_string(fileNumber) + " &";
    if (system(cmd.c_str()) != 0) {}
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
	
	std::vector<float> historyInletPower( iterationCount, 0.f );
	std::vector<float> historyMassFlow( iterationCount, 0.f );
	std::vector<float> historyTorque( iterationCount, 0.f );
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iteration = 0; iteration <= iterationCount; iteration++ )
	{
		if ( iteration % reportChunk == 0 )
		{
			BoundsStruct Bounds;
			Bounds = STLMain.Bounds;
			
			// get inlet data
			FlowReportStruct FlowReportIn;
			int kCut = 0;
			getFlowReportXY( grids, kCut, Bounds, FlowReportIn );
			float pIn = FlowReportIn.rho;
			float uzIn = FlowReportIn.uz;
			convertToPhysicalPressure( pIn );
			float uTemp = 0.f;
			convertToPhysicalVelocity( uzIn, uTemp, uTemp, grids[0].Info );
			
			// get outlet data
			FlowReportStruct FlowReportOut;
			kCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountZ-1;
			getFlowReportXY( grids, kCut, Bounds, FlowReportOut );
			float pOut = FlowReportOut.rho;
			convertToPhysicalPressure( pOut );
			
			const float pDiff = pIn - pOut;
			
			const float pTotalIn = 0.5f * rhoNominalPhys * uzIn * uzIn + pDiff;
			const float massFlow = uzIn * ( FlowReportIn.areamm2 / 1000000.f ) * FlowReportIn.rho * rhoNominalPhys;
			const float inletPower = pTotalIn * massFlow / rhoNominalPhys;
			
			// regulate inlet
			grids[0].Info.iRegulatorInlet -= (inletPower - targetInletPower) * iRegulatorInletStrength * (float)reportChunk * dtPhysGlobal;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorInlet = grids[0].Info.iRegulatorInlet;
			
			// sum torque contributions and reset
			float torque = 0.f;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
			{
				auto fView  = grids[level].fArray.getView();
				auto fetch = [ = ] __cuda_callable__( const int cell ) mutable { const float T = fView( 27, cell ); fView( 27, cell ) = 0.f; return T; };
				auto reduction = [] __cuda_callable__( const float& a, const float& b ) { return a + b; };
				float TzSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, grids[level].Info.cellCount, fetch, reduction, 0.f );
				grids[level].Info.torqueReportCumulative += ( TzSum / 1000.f ); // converting from Nmm to Nm
				torque += ( grids[level].Info.torqueReportCumulative / (float)( reportChunk * std::pow( 2, level ) ) );
				grids[level].Info.torqueReportCumulative = 0.f;
			}
			
			for ( int shifter = 0; shifter <= reportChunk; shifter++ )
			{
				if ( iteration+shifter < iterationCount )
				{
					historyInletPower[iteration+shifter] = inletPower;
					historyMassFlow[iteration+shifter] = massFlow;
					historyTorque[iteration+shifter] = torque;
				}
			}
		}
		
		if ( iteration % plotterChunk == 0 && iteration >= 1499 )
		{
			std::cout << std::endl;
			std::cout << "Finished iteration " << iteration << std::endl;
			
			lapTimer.stop();
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)usefulCellUpdatesPerIteration * (float)plotterChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			if ( iteration > 0 ) exportHistoryData( historyInletPower, historyMassFlow, historyTorque, iteration, 0 );
			
			float r = 14.8f;
			exportSectionCutPlotToiletPaperZ( grids, r, 10*iteration );
			float rotatingFrameUy = - ( r / 1000.f ) * angularVelocity;
			if (system(("python3 ../../include/plotter/plotterRotatingFrame.py " + std::to_string(rotatingFrameUy)).c_str()) != 0) {}
			
			r = 15.f;
			exportSectionCutPlotToiletPaperZ( grids, r, 10*iteration+1 );
			rotatingFrameUy = - ( r / 1000.f ) * angularVelocity;
			if (system(("python3 ../../include/plotter/plotterRotatingFrame.py " + std::to_string(rotatingFrameUy)).c_str()) != 0) {}
			
			r = 15.2f;
			exportSectionCutPlotToiletPaperZ( grids, r, 10*iteration+2 );
			rotatingFrameUy = - ( r / 1000.f ) * angularVelocity;
			if (system(("python3 ../../include/plotter/plotterRotatingFrame.py " + std::to_string(rotatingFrameUy)).c_str()) != 0) {}
			
			/*
			const int iCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountX/2;
			exportSectionCutPlotZY( grids, iCut, iteration+1 );
			if (system("python3 ../../include/plotter/plotter.py") != 0) {}
			*/
			
			lapTimer.reset();
			lapTimer.start();
		}
		applyGlobalUpdate(grids, 0, Voxelizer, STLImpellerStationary, STLImpellerMoving );
	}
		
	return EXIT_SUCCESS;
}
