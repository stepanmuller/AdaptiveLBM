static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2; // min 2
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;
static constexpr int MEMORY_MBB_UPDATE_PERCENTAGE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 8;
static constexpr int GRID_REBUILD_PERIOD = 24;

static constexpr int GRID_LEVEL_COUNT = 1;

int reportChunk = 31;
int plotterChunk = 2000;
constexpr int iterationCount = 80000;

constexpr float resGlobal = 0.1f; 														// mm

constexpr float angularVelocity = 2000.f;												// rad/s
constexpr float targetInletPower = 0.f;													// W
constexpr float iRegulatorInletStrength = 0.003f;
constexpr float massFlowInitPhys = 2.5f;												// kg/s
constexpr float RIn = 4.f;																// mm
constexpr float ROut = 14.25f;															// mm
const float boundaryLayerThickness = 0.2f;												// mm
const float shaftRotationStartDistance = 10.f;											// mm

constexpr float uImpMax = 0.07f;														// also works as nominal LBM Mach number
constexpr float rzImpMax = 17.1f;														// mm
constexpr float uImpPhys = angularVelocity * rzImpMax / 1000.f;							// m/s
constexpr float dtPhysGlobal = (uImpMax / uImpPhys) * (resGlobal/1000.f); 				// s	

constexpr float nuPhys = 1e-6;															// m2/s water
constexpr float rhoNominalPhys = 1000.0f;												// kg/m3 water
constexpr float inletAreamm2 = 3.14159f * ( ROut * ROut - RIn * RIn);					// mm2
constexpr float uzInletPhys = massFlowInitPhys / ( rhoNominalPhys * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float uzInletBase = uzInletPhys * dtPhysGlobal / (resGlobal/1000);				
constexpr float soundspeedPhys = 0.577350269f * (resGlobal/1000) / dtPhysGlobal; 		// m/s (0.577350269f is 1/sqrt(3))

#include "../../include/types.h"
#include "../../include/cellFunctions.h"

std::string STLPathMain = "M-Jet_40_pump_main.STL";
std::string STLPathImpeller = "M-Jet_40_impeller.STL";

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

__cuda_callable__ void getInitialRhoUG( BCStruct &BC,
										const int& iCell, const int& jCell, const int& kCell, 
										const InfoStruct& Info, MarkerStruct &Marker )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = angularVelocity * (r / 1000.f);
	const float vt = vtPhys * ( uzInletBase / uzInletPhys );
	if ( Marker.movingBounceback )
	{
		BC.ux = - vt * (y / r);
		BC.uy = vt * (x / r);
		BC.uz = 0.f;
	}
	else if ( Marker.bounceback )
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	else
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzInletBase;
	}
	BC.rho = 1.f;
}

__cuda_callable__ void getBC( 	BCStruct &BC,
									const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info, MarkerStruct &Marker )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = angularVelocity * (r / 1000.f);
	const float vt = vtPhys * ( uzInletBase / uzInletPhys );
	const float wallDistancePhys = std::max(0.f, std::min(r - RIn, ROut - r));
	const float delta = std::max( 0.f, std::min( 1.f, wallDistancePhys / boundaryLayerThickness ));
	const float velocityMultiplier = delta * delta * (3.0f - 2.0f * delta);
		
	const float inletDistancePhys = z - Info.oz;
	const float rotationDelta = std::max( 0.f, std::min( 1.f, inletDistancePhys / shaftRotationStartDistance ));
	const float rotationMultiplier = rotationDelta * rotationDelta * (3.0f - 2.0f * rotationDelta);
	
	if ( Marker.movingBounceback )
	{
		BC.ux = - vt * (y / r) * rotationMultiplier;
		BC.uy = vt * (x / r) * rotationMultiplier;
		BC.uz = 0.f;
	}
	else
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = ( uzInletBase + Info.iRegulatorInlet ) * velocityMultiplier; // uzInletBase * velocityMultiplier; // 
	}
	if ( Marker.BCRho || Marker.nonReflectiveOutlet ) BC.rho = 1.f;
	if ( z > 15.41f ) BC.collisionLimiter = 0.f; //BC.nuMultiplier = ( 28.f - z ) * 0.1f * 100.f;
}

#include "../../include/adaptiveGridFunctions.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/updateInterface.h"
#include "../../include/updateMovingBounceback.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/flowReportFunctions.h"

void applyGlobalUpdate( std::vector<GridStruct>& grids, int level, VoxelizerStruct &Voxelizer, STLStruct &STLRotorStationary, STLStruct &STLRotorMoving ) 
{
	if ( level == GRID_LEVEL_COUNT - 1 ) // I am the finest grid
    {
		if (grids[level].Info.updatesSinceMovingBouncebackUpdate >= MOVING_BOUNCEBACK_UPDATE_PERIOD )
		{
			const float radians = grids[level].Info.iterationsFinished * grids[level].Info.dtPhys * angularVelocity;
			rotateSTLAlongZ( STLRotorMoving, STLRotorStationary, radians );
			Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
			voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLRotorMoving, Voxelizer );
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
        for ( int i = 0; i < 2; i++) applyGlobalUpdate(grids, level + 1, Voxelizer, STLRotorStationary, STLRotorMoving );
        updateInterface(grids[level], grids[level + 1]);
    }
}

void exportHistoryData( const std::vector<float>& historyInletPower, 
                        const std::vector<float>& historyThrust, 
                        const std::vector<float>& historyTorque, 
                        const int &currentIteration, int fileNumber ) {
    FILE* fp = fopen("/dev/shm/historyData.bin", "wb");
    if (!fp) return;
    
    int count = std::min({iterationCount-1, currentIteration}) + 1;
    fwrite(&count, sizeof(int), 1, fp);
    
    // Write all three vectors sequentially
    fwrite(historyInletPower.data(), sizeof(float), count, fp);
    fwrite(historyThrust.data(), sizeof(float), count, fp);
    fwrite(historyTorque.data(), sizeof(float), count, fp);
    
    fclose(fp);
    
    // Construct the command string to pass the number as an argument
    std::string cmd = "python3 historyPlotter.py " + std::to_string(fileNumber) + " &";
    if (system(cmd.c_str()) != 0) {}
}

int main(int argc, char **argv)
{
	// STLs
	STLStruct STLStator;
	readSTL( STLStator, STLPathMain );
	STLStruct STLRotorStationary;
	readSTL( STLRotorStationary, STLPathImpeller );
	STLStruct STLRotorMoving;
	STLRotorMoving = STLRotorStationary;
	
	std::cout << "Cells travelled by MBB per iteration: " 
		<< dtPhysGlobal * angularVelocity * STLRotorStationary.Bounds.rzMax / resGlobal << std::endl;
	std::cout << "Cells travelled by MBB per MBB update: " 
		<< (float)MOVING_BOUNCEBACK_UPDATE_PERIOD * dtPhysGlobal * angularVelocity * STLRotorStationary.Bounds.rzMax / resGlobal << std::endl;
	std::cout << "Cells travelled by MBB per grid rebuild: " 
		<< (float)GRID_REBUILD_PERIOD * dtPhysGlobal * angularVelocity * STLRotorStationary.Bounds.rzMax / resGlobal << std::endl;
	std::cout << std::endl;
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	grids[ 0 ].Info.res = resGlobal;
	BoundsStruct DomainBounds;
	DomainBounds = STLStator.Bounds;
	DomainBounds.zMin = -60.f;
	initializeGrids( grids, DomainBounds, 0 );
	
	// Voxelizer 
	VoxelizerStruct Voxelizer;
	Voxelizer.Info = grids[ GRID_LEVEL_COUNT-1 ].Info;
	voxelizeSTL( Voxelizer.rayMapBounceback, STLStator, Voxelizer );
	voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLRotorMoving, Voxelizer );
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
	
	std::vector<float> historyInletPower( iterationCount, 0.f );
	std::vector<float> historyThrust( iterationCount, 0.f );
	std::vector<float> historyTorque( iterationCount, 0.f );
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iteration = 0; iteration <= iterationCount; iteration++ )
	{
		if ( iteration % reportChunk == 0 )
		{
			BoundsStruct Bounds;
			Bounds = STLStator.Bounds;
			
			// get inlet data
			FlowReportStruct FlowReportIn;
			int kCut = 0;
			getFlowReportXY( grids, kCut, Bounds, FlowReportIn );
			float pIn = FlowReportIn.pPhys;
			float uzIn = FlowReportIn.uzPhys;
			
			// get outlet data
			FlowReportStruct FlowReportOut;
			int iTemp, jTemp;
			float xTemp = 0.f; float yTemp = 0.f;
			float z = 15.41f;
			getIJKCellIndexFromXYZ( iTemp, jTemp, kCut, xTemp, yTemp, z, grids[GRID_LEVEL_COUNT-1].Info);
			getFlowReportXY( grids, kCut, Bounds, FlowReportOut );
			float pOut = FlowReportOut.pPhys;
			
			const float pDiff = pIn - pOut;
			
			const float inletPower = FlowReportIn.kineticEnergyFlowZPhys + FlowReportIn.pPhys * uzIn * ( FlowReportIn.areamm2 / 1000000.f );
			
			// regulate inlet
			grids[0].Info.iRegulatorInlet -= (inletPower - targetInletPower) * iRegulatorInletStrength * (float)reportChunk * dtPhysGlobal;
			grids[0].Info.iRegulatorInlet = std::clamp( grids[0].Info.iRegulatorInlet, -0.8f * uzInletBase, 2.f * uzInletBase );
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
					historyThrust[iteration+shifter] = FlowReportOut.momentumFlowZPhys;
					historyTorque[iteration+shifter] = torque;
				}
			}
		}
		
		if ( iteration % plotterChunk == 0 )
		{
			std::cout << std::endl;
			std::cout << "Finished iteration " << iteration << std::endl;
			
			lapTimer.stop();
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)usefulCellUpdatesPerIteration * (float)plotterChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			if ( iteration > 0 ) exportHistoryData( historyInletPower, historyThrust, historyTorque, iteration, 0 );
			
			float r = 12.0f;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration );
			float rotatingFrameUy = - ( r / 1000.f ) * angularVelocity;
			if (system(("python3 ../../include/plotter/plotterRotatingFrame.py " + std::to_string(rotatingFrameUy)).c_str()) != 0) {}

			const int iCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountX/2;
			exportSectionCutPlotZY( grids, iCut, iteration+1 );
			if (system("python3 ../../include/plotter/plotter.py") != 0) {}
			
			lapTimer.reset();
			lapTimer.start();
		}
		applyGlobalUpdate(grids, 0, Voxelizer, STLRotorStationary, STLRotorMoving );
	}
		
	return EXIT_SUCCESS;
}
