static constexpr int RAY_MAP_DEPTH = 128;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 8;
static constexpr int GRID_REBUILD_PERIOD = 24;

static constexpr int GRID_LEVEL_COUNT = 2;
static constexpr float SMAGORINSKY_CONSTANT = 0.05f;

int reportChunk = 31;
int plotterChunk = 1000;
constexpr int iterationCount = 100000;

constexpr float resGlobal = 3.0f; 														// mm

constexpr float uzInlet = 0.01f; 														// also works as nominal LBM Mach number	
constexpr float nuPhys = 1e-6;															// m2/s water
constexpr float rhoNominalPhys = 997.0f;												// kg/m3 water
constexpr float massFlowPhys = 335.f;													// kg/s
constexpr float RInlet = 150.f;															// mm
// constexpr float ROutlet = 175.f;														// mm
constexpr float inletAreamm2 = 3.14159f * RInlet * RInlet;								// mm2
constexpr float uzInletPhys = massFlowPhys / ( rhoNominalPhys * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float dtPhysGlobal = (uzInlet / uzInletPhys) * (resGlobal/1000.f); 			// s
constexpr float soundspeedPhys = 0.577350269f * (resGlobal/1000.f) / dtPhysGlobal; 		// m/s (0.577350269f is 1/sqrt(3))
constexpr float angularVelocity = -198.967f;											// rad/s
const float boundaryLayerThickness = 2.f;												// mm

#include "../../include/types.h"
#include "../../include/cellFunctions.h"

std::string STLPathStator = "../../../../ns300/ns300_STATOR_ENLARGED_TIP_GAP_STL.STL";
// std::string STLPathStator = "../../../../ns300/ns300_STATOR_DEFAULT_STL.STL";
std::string STLPathRotor = "../../../../ns300/ns300_ROTOR_STL.STL";

__cuda_callable__ void getMarkers( 	const int& iCell, const int& jCell, const int& kCell, 
									MarkerStruct &Marker, const InfoStruct& Info )
{
	if ( Marker.bounceback ) return;
	if ( Marker.movingBounceback ) return;
	
	if ( kCell == Info.cellCountZ-1 ) Marker.refinement = 1;
	if ( jCell == Info.cellCountY-1 ) Marker.refinement = 1;
	
	if ( kCell == Info.cellCountZ-1 ) Marker.nonReflectiveInlet = 1; // Marker.BCU = 1;
	else if ( jCell == Info.cellCountY-1 ) Marker.nonReflectiveOutlet = 1; //Marker.BCRho = 1;
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
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( Marker.movingBounceback )
	{
		BC.ux = - vt * (y / r);
		BC.uy = vt * (x / r);
		BC.uz = 0.f;
	}
	else if ( Marker.bounceback )
	{
		BC.ux = 0.f; //- vt * (y / r);
		BC.uy = 0.f; //vt * (x / r);
		BC.uz = 0.f;
	}
	else
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
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
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	const float wallDistancePhys = std::max(0.f, RInlet-r);
	const float delta = std::max( 0.f, std::min( 1.f, wallDistancePhys / boundaryLayerThickness ));
	const float velocityMultiplier = delta * delta * (3.0f - 2.0f * delta);
	
	if ( Marker.movingBounceback )
	{
		BC.ux = - vt * (y / r);
		BC.uy = vt * (x / r);
		BC.uz = 0.f;
	}
	else
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = - ( uzInlet ) * velocityMultiplier;
	}
	if ( Marker.BCRho || Marker.nonReflectiveOutlet ) { BC.rho = 1.f; }
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

void exportHistoryData( const std::vector<float>& historyInletPressure, 
						const std::vector<float>& historyOutletPressure, 
                        const std::vector<float>& historyInletMassFlow, 
                        const std::vector<float>& historyTorque, 
                        const int &currentIteration, int fileNumber ) {
    FILE* fp = fopen("/dev/shm/historyData.bin", "wb");
    if (!fp) return;
    
    int count = std::min({iterationCount-1, currentIteration}) + 1;
    fwrite(&count, sizeof(int), 1, fp);
    
    // Write all three vectors sequentially
    fwrite(historyInletPressure.data(), sizeof(float), count, fp);
    fwrite(historyOutletPressure.data(), sizeof(float), count, fp);
    fwrite(historyInletMassFlow.data(), sizeof(float), count, fp);
    fwrite(historyTorque.data(), sizeof(float), count, fp);
    
    fclose(fp);
    
    // Construct the command string to pass the number as an argument
    std::string cmd = "python3 historyPlotter.py " + std::to_string(fileNumber) + " &";
    if (system(cmd.c_str()) != 0) {}
}

int main(int argc, char **argv)
{
	std::cout << uzInletPhys << std::endl;
	// STLs
	STLStruct STLStator;
	readSTL( STLStator, STLPathStator );
	STLStruct STLRotorStationary;
	readSTL( STLRotorStationary, STLPathRotor );
	STLStruct STLRotorMoving;
	STLRotorMoving = STLRotorStationary;
	
	std::cout << "Cells travelled by MBB per iteration: " 
		<< dtPhysGlobal * angularVelocity * STLImpellerStationary.Bounds.rzMax / resGlobal << std::endl;
	std::cout << "Cells travelled by MBB per MBB update: " 
		<< (float)MOVING_BOUNCEBACK_UPDATE_PERIOD * dtPhysGlobal * angularVelocity * STLImpellerStationary.Bounds.rzMax / resGlobal << std::endl;
	std::cout << "Cells travelled by MBB per grid rebuild: " 
		<< (float)GRID_REBUILD_PERIOD * dtPhysGlobal * angularVelocity * STLImpellerStationary.Bounds.rzMax / resGlobal << std::endl;
	std::cout << std::endl;
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	grids[ 0 ].Info.res = resGlobal;
	BoundsStruct DomainBounds;
	DomainBounds = STLStator.Bounds;
	DomainBounds.zMax = 400.f;
	DomainBounds.yMax = 1000.f;
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
	
	std::cout << "Maximum cells travelled by MBB per iteration: " << dtPhysGlobal * std::abs(angularVelocity) * 164.5f / resGlobal << std::endl;
	
	std::vector<float> historyInletPressure( iterationCount, 0.f );
	std::vector<float> historyOutletPressure( iterationCount, 0.f );
	std::vector<float> historyInletMassFlow( iterationCount, 0.f );
	std::vector<float> historyTorque( iterationCount, 0.f );
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iteration = 0; iteration <= iterationCount; iteration++ )
	{
		if ( iteration % reportChunk == 0 )
		{
			BoundsStruct InletBounds;
			InletBounds.xMin = -155.f;
			InletBounds.xMax = 155.f;
			InletBounds.yMin = -155.f;
			InletBounds.yMax = 155.f;
			// get inlet data
			FlowReportStruct FlowReportIn;
			int kCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountZ-1;
			getFlowReportXY( grids, kCut, InletBounds, FlowReportIn );
			float pIn = FlowReportIn.rho;
			float uzIn = FlowReportIn.uz;
			convertToPhysicalPressure( pIn );
			float uTemp = 0.f;
			convertToPhysicalVelocity( uzIn, uTemp, uTemp, grids[0].Info );
			const float massFlowIn = uzIn * ( FlowReportIn.areamm2 / 1000000.f ) * FlowReportIn.rho * rhoNominalPhys;
			
			BoundsStruct OutletBounds;
			OutletBounds.xMin = -535.f;
			OutletBounds.xMax = -175.f;
			OutletBounds.zMin = -210.f;
			OutletBounds.zMax = 145.f;
			// get inlet data
			FlowReportStruct FlowReportOut;
			int jCut = grids[GRID_LEVEL_COUNT-1].Info.cellCountY-1;
			getFlowReportZX( grids, jCut, OutletBounds, FlowReportOut );
			float pOut = FlowReportOut.rho;
			//float uzOut = FlowReportIn.uz;
			convertToPhysicalPressure( pOut );
			//float uTemp = 0.f;
			//convertToPhysicalVelocity( uzIn, uTemp, uTemp, grids[0].Info );
			//const float massFlow = uzIn * ( FlowReportIn.areamm2 / 1000000.f ) * FlowReportIn.rho * rhoNominalPhys;
			
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
					historyInletPressure[iteration+shifter] = pIn;
					historyOutletPressure[iteration+shifter] = pOut;
					historyInletMassFlow[iteration+shifter] = -massFlowIn;
					historyTorque[iteration+shifter] = -torque;
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
			
			if ( iteration > 0 ) exportHistoryData( historyInletPressure, historyOutletPressure, historyInletMassFlow, historyTorque, iteration, 0 );
			
			// prepare section cuts
			int iCut, jCut, kCut;
			const float xTemp = 0.f; const float yTemp = 0.f; const float zTemp = 0.f;
			
			// ZY section cut shows the inlet pipe
			float xCut = 0.f;
			getIJKCellIndexFromXYZ( iCut, jCut, kCut, xCut, yTemp, zTemp, grids[GRID_LEVEL_COUNT-1].Info);
			exportSectionCutPlotZY( grids, iCut, iteration );
			if (system("python3 ../../include/plotter/plotter.py") != 0) {}
			
			// XY section cut shows the rotor and the outlet pipe
			float zCut = 32.5f;
			getIJKCellIndexFromXYZ( iCut, jCut, kCut, xTemp, yTemp, zCut, grids[GRID_LEVEL_COUNT-1].Info);
			exportSectionCutPlotXY( grids, kCut, iteration+1 );
			if (system("python3 ../../include/plotter/plotter.py") != 0) {}
			
			lapTimer.reset();
			lapTimer.start();
		}
		applyGlobalUpdate(grids, 0, Voxelizer, STLRotorStationary, STLRotorMoving );
	}
		
	return EXIT_SUCCESS;
}
