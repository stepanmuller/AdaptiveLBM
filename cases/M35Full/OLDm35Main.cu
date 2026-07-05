constexpr float resGlobal = 0.40f; 														// mm
constexpr int gridLevelCount = 3;
constexpr int wallRefinementSpan = 1;

constexpr int iterationCount = 500000;
constexpr int iterationChunk = 5000;
constexpr int historyChunk = 19;

constexpr float SmagorinskyConstantGlobal = 0.1f; 										// set to zero to turn off LES

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

constexpr float targetInletPower = 0.f;													// W
constexpr float iRegulatorStrength = 0.25f * 1e-7f;

#include "../include/types.h"

#include <sstream>

#include "../include/cellFunctions.h"
#include "../include/applyStreaming.h"
//#include "../include/applyCollision.h"
#include "./applyCollisionCustomized.h"

#include "../include/boundaryConditions/applyBounceback.h"
#include "../include/boundaryConditions/applyMovingBounceback.h"
#include "../include/boundaryConditions/restoreRho.h"
#include "../include/boundaryConditions/restoreUxUyUz.h"
#include "../include/boundaryConditions/restoreRhoUxUyUz.h"
#include "../include/boundaryConditions/applyMBBC.h"
#include "../include/boundaryConditions/applyNonReflectiveOutlet.h"

#include "../include/STLFunctions.h"
std::string STLPathMain = "M-Jet_35_pump_main.STL";
std::string STLPathShaft = "M-Jet_35_shaft.STL";
std::string STLPathImpeller = "M-Jet_35_impeller.STL";

__cuda_callable__ void getMarkers( 	const int& iCell, const int& jCell, const int& kCell, 
									MarkerStruct &Marker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	//if ( r < 17.f ) Marker.refinement = 1;
	if ( kCell < 5 ) Marker.refinement = 1;
	if ( Info.cellCountZ - kCell < 5 ) Marker.refinement = 1;
	if ( Marker.bounceback ) 
	{
		if ( r < 10.f && z < -0.5f ) 
		{
			Marker.bounceback = 0;
			Marker.movingBounceback = 1;
		}
		else return;
	}
	if ( Marker.forcedVelocity ) return;
	if ( kCell == 0 ) Marker.givenUxUyUz = 1;
	else if ( kCell == Info.cellCountZ-1 ) Marker.givenRho = 1;
	else Marker.fluid = 1;
}

__cuda_callable__ void getGivenRhoUxUyUz( 	const int& iCell, const int& jCell, const int& kCell, 
											float& rho, float& ux, float& uy, float& uz,
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
	if ( Marker.forcedVelocity || Marker.movingBounceback )
	{
		ux = - vt * (y / r);
		uy = vt * (x / r);
		uz = 0.f;
	}
	else
	{
		ux = 0.f;
		uy = 0.f;
		uz = Info.iRegulator * velocityMultiplier;
	}
	rho = 1.f;
}

__cuda_callable__ void getForcing( 	const int& iCell, const int& jCell, const int& kCell, 
									const float (&f)[27], 
									float& gx, float& gy, float& gz,
									const InfoStruct& Info )
{
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = angularVelocity * (r / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	const float uxTarget = - vt * (y / r);
	const float uyTarget = vt * (x / r);
	const float uzTarget = 0.f;
	gx = rho * ( uxTarget - ux );
	gy = rho * ( uyTarget - uy );
	gz = rho * ( uzTarget - uz );
}

__cuda_callable__ float getSmagorinskyConstant( const int  &iCell, const int &jCell, const int &kCell, const InfoStruct &Info  )
{
	return SmagorinskyConstantGlobal;
}

__cuda_callable__ void getInitialRhoUxUyUz( const int &iCell, const int &jCell, const int &kCell, float &rho, float &ux, float &uy, float &uz, const MarkerStruct &Marker, const InfoStruct &Info )
{
	ux = 0.f;
	uy = 0.f;
	uz = uzInlet;
	rho = 1.f;
}

//#include "../include/applyLocalCellUpdate.h"
#include "./applyLocalCellUpdateCustomized.h"
#include "./exportSectionCutPlotCustomized.h"
#include "../include/fillEquilibrium.h"
#include "../include/gridRefinementFunctions.h"
#include "../include/DIADFunctions.h"
#include "../include/flowReportFunctions.h"

void updateGrid( std::vector<DIADGridStruct>& grids, int level ) 
{
	//applyNonReflectiveOutletZ(grids[level]);
    applyStreaming(grids[level]);
    applyLocalCellUpdate(grids[level]);
    if (level < gridLevelCount - 1) 
    {
        for (int i = 0; i < 2; i++) updateGrid(grids, level + 1);
        applyCoarseFineGridCommunication(grids[level], grids[level + 1]);
    }
}

void exportHistoryData( const std::vector<float>& historyInletPower, 
                        const std::vector<float>& historyMassFlow, 
                        const std::vector<float>& historyTorque, 
                        const int &currentIteration, int fileNumber ) {
    FILE* fp = fopen("/dev/shm/historyData.bin", "wb");
    if (!fp) return;
    
    int count = currentIteration + 1;
    fwrite(&count, sizeof(int), 1, fp);
    
    // Write all three vectors sequentially
    fwrite(historyInletPower.data(), sizeof(float), count, fp);
    fwrite(historyMassFlow.data(), sizeof(float), count, fp);
    fwrite(historyTorque.data(), sizeof(float), count, fp);
    
    fclose(fp);
    
    // Construct the command string to pass the number as an argument
    std::string cmd = "python3 historyPlotter.py " + std::to_string(fileNumber) + " &";
    system(cmd.c_str());
}

float getTorque( DIADGridStruct &Grid )
{
	auto fArrayView  = Grid.fArray.getConstView();
	InfoStruct Info = Grid.Info;
	
	bool esotwistFlipper = Grid.esotwistFlipper;
	auto iNbrView = Grid.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrView = Grid.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrView = Grid.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrView = Grid.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrView = Grid.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrView = Grid.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrView = Grid.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	
	bool useBouncebackArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
	bool useForcedVelocityArray = ( Grid.forcedVelocityMarkerArray.getSize() > 0 );
	auto forcedVelocityMarkerArrayView = Grid.forcedVelocityMarkerArray.getConstView();
	bool useInterfaceMarkerArray = ( Grid.interfaceMarkerArray.getSize() > 0 );
	auto interfaceMarkerArrayView = Grid.interfaceMarkerArray.getConstView();
	
	auto fetch = [ = ] __cuda_callable__( const int cell )
	{		
		if ( useInterfaceMarkerArray ) 
		{
			if ( interfaceMarkerArrayView( cell ) ) return 0.f;
		}
		
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		if ( kCell < 5 ) return 0.f;
		
		MarkerStruct Marker;
		if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
		if ( useForcedVelocityArray ) Marker.forcedVelocity = forcedVelocityMarkerArrayView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		
		if ( Marker.movingBounceback ) return 0.f;
		
		if ( !Marker.forcedVelocity ) return 0.f;
		
		float x, y, z;
		getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
		const float r = std::sqrt( x * x + y * y );
		
		DIADEsotwistNbrStruct Nbr;
		Nbr.i = iNbrView( cell );
		Nbr.j = jNbrView( cell );
		Nbr.k = kNbrView( cell );
		Nbr.ij = ijNbrView( cell );
		Nbr.ik = ikNbrView( cell );
		Nbr.jk = jkNbrView( cell );
		Nbr.ijk = ijkNbrView( cell );
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		const bool flippedFlipper = !esotwistFlipper;
		getEsotwistReadIndex( cell, cellReadIndex, fReadIndex, Nbr, flippedFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
		
		float gx = 0.f;
		float gy = 0.f;
		float gz = 0.f;
		
		getForcing( iCell, jCell, kCell, f, gx, gy, gz, Info );
		convertToPhysicalForce( gx, gy, gz, Info );
		float T = - gx * y + gy * x;
		return T;
	};
	auto reduction = [] __cuda_callable__( const float& a, const float& b )
	{
		return a + b;
	};
	
	float TSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Info.cellCount, fetch, reduction, 0.f );
	TSum = TSum / 1000.f; // converting from Nmm to Nm
	return TSum;
}

int main(int argc, char **argv)
{
	int caseID = 0;
    if (argc > 1) 
    {
		caseID = std::stoi(argv[1]);
	}

	STLStructCPU STLCPUMain;
	readSTL( STLCPUMain, STLPathMain );
	STLStruct STLMain( STLCPUMain );
	checkSTLEdges( STLMain );
	
	STLStructCPU STLCPUShaft;
	readSTL( STLCPUShaft, STLPathShaft );
	STLStruct STLShaft( STLCPUShaft );
	checkSTLEdges( STLShaft );
	
	std::vector<STLStruct> STLs = { STLMain, STLShaft };
	
	std::vector<DIADGridStruct> grids(gridLevelCount);
	// Coarse grid: Grid0
	grids[0].Info.res = resGlobal;
	grids[0].Info.dtPhys = dtPhysGlobal;
	grids[0].Info.nu = (grids[0].Info.dtPhys * nuPhys) / ((grids[0].Info.res/1000) * (grids[0].Info.res/1000));
	grids[0].Info.ox = STLMain.xmin + 0.5f * grids[0].Info.res;
	grids[0].Info.oy = STLMain.ymin + 0.5f * grids[0].Info.res;
	grids[0].Info.oz = STLMain.zmin + 0.5f * grids[0].Info.res;
	grids[0].Info.cellCountX = (int)( (STLMain.xmax - STLMain.xmin) / grids[0].Info.res );
	grids[0].Info.cellCountY = (int)( (STLMain.ymax - STLMain.ymin) / grids[0].Info.res );
	grids[0].Info.cellCountZ = (int)( (STLMain.zmax - STLMain.zmin) / grids[0].Info.res );
	grids[0].Info.cellCount = grids[0].Info.cellCountX * grids[0].Info.cellCountY * grids[0].Info.cellCountZ;
	buildIJKFromInfo( grids[0].IJK, grids[0].Info );
	
	buildDIADGrids( grids, STLs, 0 );
	
	STLStructCPU STLCPUImpeller;
	readSTL( STLCPUImpeller, STLPathImpeller );
	STLStruct STLImpeller( STLCPUImpeller );
	checkSTLEdges( STLImpeller );
	
	for ( int level = 0; level < gridLevelCount; level++ )
	{
		grids[level].forcedVelocityMarkerArray.setSize( grids[level].Info.cellCount );
		const bool insideMarkerValue = 1;
		ApplyMarkersInsideSTL( grids[level].forcedVelocityMarkerArray, grids[level].IJK, STLImpeller, insideMarkerValue, grids[level].Info );
		grids[level].Info.iRegulator = uzInlet;
	}
	
	int cellCountTotal = 0;
	long int cellUpdatesPerIteration = 0;
	for ( int level = 0; level < gridLevelCount; level++ )
	{
		const int cellCountLevel = grids[level].Info.cellCount;
		cellCountTotal += cellCountLevel; 
		cellUpdatesPerIteration += cellCountLevel * std::pow(2, level);
	}
	std::cout << "Cell count total: " << cellCountTotal << std::endl;
	std::cout << "Cell updates per iteration: " << cellUpdatesPerIteration << std::endl;	
	
	std::vector<float> historyInletPower( iterationCount, 0.f );
	std::vector<float> historyMassFlow( iterationCount, 0.f );
	std::vector<float> historyTorque( iterationCount, 0.f );
	
	std::cout << "Starting simulation" << std::endl;
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for (int iteration=0; iteration<=iterationCount; iteration++)
	{
		if (iteration%2 == 0)
		{
			const float rotationAngle = angularVelocity * grids[0].Info.dtPhys * iteration;
			STLStruct STLImpellerRotated;
			STLImpellerRotated = STLImpeller;
			rotateSTLAlongZ( STLImpellerRotated, rotationAngle );
			for ( int level = 0; level < gridLevelCount; level++ )
			{
				const bool insideMarkerValue = 1;
				ApplyMarkersInsideSTL( grids[level].forcedVelocityMarkerArray, grids[level].IJK, STLImpellerRotated, insideMarkerValue, grids[level].Info );
			}
		}
		updateGrid( grids, 0 );
		if (iteration%historyChunk == 0)
		{
			XYZBoundsStruct Bounds;
			Bounds.xmin = -ROut;
			Bounds.xmax = ROut;
			Bounds.ymin = -ROut;
			Bounds.ymax = ROut;
			
			FlowReportStruct FlowReportIn;
			int kCut = 0;
			getFlowReportXY( grids, kCut, Bounds, FlowReportIn );
			float pIn = FlowReportIn.rho;
			float uzIn = FlowReportIn.uz;
			convertToPhysicalPressure( pIn );
			float uTemp = 0.f;
			convertToPhysicalVelocity( uzIn, uTemp, uTemp, grids[0].Info );
			
			const float pTotalIn = 0.5f * rhoNominalPhys * uzIn * uzIn + pIn;
			
			const float massFlow = uzIn * ( FlowReportIn.areamm2 / 1000000.f ) * FlowReportIn.rho * rhoNominalPhys;
			
			const float inletPower = pTotalIn * massFlow / rhoNominalPhys;
			
			grids[0].Info.iRegulator -= (inletPower - targetInletPower) * iRegulatorStrength;
			for ( int level = 0; level < gridLevelCount; level++ )
			{
				grids[level].Info.iRegulator = grids[0].Info.iRegulator;
			}
			
			float torque = 0.f;
			for ( int level = 0; level < gridLevelCount; level++ )
			{
				torque += getTorque( grids[level] );
			}
			
			for ( int shifter = 0; shifter < historyChunk; shifter++ )
			{
				if ( iteration+shifter < iterationCount )
				{
					historyInletPower[iteration+shifter] = inletPower;
					historyMassFlow[iteration+shifter] = massFlow;
					historyTorque[iteration+shifter] = torque;
				}
			}
		}
		
		if ( iteration % iterationChunk == 0) 
		{
			lapTimer.stop();
			auto lapTime = lapTimer.getRealTime();
			std::cout << "Finished iteration " << iteration << std::endl;
			const float updateCount = (float)cellUpdatesPerIteration * (float)iterationChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			std::cout << "GLUPS: " << glups << std::endl;
			
			exportHistoryData( historyInletPower, historyMassFlow, historyTorque, iteration, 0 );
			
			int counter = 1;
			for (float r = RIn + 1.f; r < ROut; r = r + 1.f) 
			{
				bool rotatingFrameOfReference = 0;
				exportSectionCutPlotToiletPaperZ( grids, r, iteration + counter, rotatingFrameOfReference );
				system("python3 ../include/plotter/plotterGridID.py");
				counter++;
				rotatingFrameOfReference = 1;
				exportSectionCutPlotToiletPaperZ( grids, r, iteration + counter, rotatingFrameOfReference );
				system("python3 ../include/plotter/plotterGridID.py");
				counter++;
			}	
			counter = 1;
			for (float z = grids[gridLevelCount-1].Info.oz; z < ((grids[gridLevelCount-1].Info.cellCountZ-1) * grids[gridLevelCount-1].Info.res + grids[gridLevelCount-1].Info.oz); z = z + 5.f) 
			{
				int iTemp, jTemp, kCut;
				const float xTemp = 0;
				const float yTemp = 0;
				getIJKCellIndexFromXYZ( iTemp, jTemp, kCut, xTemp, yTemp, z, grids[gridLevelCount-1].Info);
				exportSectionCutPlotXY( grids, kCut, iteration + 30 + counter );
				system("python3 ../include/plotter/plotterGridID.py");
				counter++;
			}	
			const int kCut = grids[gridLevelCount-1].Info.cellCountZ - 1;
			exportSectionCutPlotXY( grids, kCut, iteration + 30 + counter );
			system("python3 ../include/plotter/plotterGridID.py");
			
			const int iCut = grids[gridLevelCount-1].Info.cellCountX / 2;
			exportSectionCutPlotZY( grids, iCut, iteration + 60 );
			system("python3 ../include/plotter/plotterGridID.py");
			
			/*
			const float r = 13.25f;
			bool rotatingFrameOfReference = 0;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration, rotatingFrameOfReference );
			system("python3 ../include/plotter/plotterGridID.py");
			rotatingFrameOfReference = 1;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration + 1, rotatingFrameOfReference );
			system("python3 ../include/plotter/plotterGridID.py");
			const int iCut = grids[gridLevelCount-1].Info.cellCountX / 2;
			exportSectionCutPlotZY( grids, iCut, iteration + 2 );
			system("python3 ../include/plotter/plotterGridID.py");			
			*/	
			
			lapTimer.reset();
			lapTimer.start();
		}
	}
	return EXIT_SUCCESS;
}
