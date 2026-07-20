static constexpr int RAY_MAP_DEPTH = 32;
static constexpr int WALL_REFINEMENT_COUNT = 2;
static constexpr int MEMORY_RESERVE_PERCENTAGE = 10;
static constexpr int MEMORY_RESERVE_PERCENTAGE_INTERFACE = 10;

static constexpr int MOVING_BOUNCEBACK_UPDATE_PERIOD = 8;
static constexpr int GRID_REBUILD_PERIOD = 24;

static constexpr int GRID_LEVEL_COUNT = 1;
static constexpr float SMAGORINSKY_CONSTANT = 0.1;

constexpr int iterationChunk = 1;
constexpr int iterationCount = 58;

constexpr float resGlobal = 0.20f; 														// mm

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

void FIND_WHATS_GOING_ON( GridStruct &Grid )
{
	std::cout << "WE ARE HERE" << std::endl;
	
	const bool esotwistFlipper = Grid.esotwistFlipper;
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
	
	std::cout << "moving bounceback state of cell 2121100: " << movingBouncebackMarkerArrayCPU[ 2121100 ] << std::endl;
	
	
	// INVESTIGATING CELL 2121100

	const int cell = 2121100;
	
	const int iCell = iArrayCPU( cell );
	const int jCell = jArrayCPU( cell );
	const int kCell = kArrayCPU( cell );
	
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
	// now look at each neighbour if they are MBB 
	bool isMovingBounceback[27] = {false};
	for ( int direction = 1; direction < 27; direction++ )
	{
		isMovingBounceback[direction] = movingBouncebackMarkerArrayCPU( fullNBRList[direction] );
	}
	// next step is to identify the surface normal direction, that is where we will be extrapolating from
	int nx, ny, nz = 0;
	nx = + 1 * isMovingBounceback[1] - 1 * isMovingBounceback[2];
	ny = - 1 * isMovingBounceback[5] + 1 * isMovingBounceback[6];
	nz = - 1 * isMovingBounceback[3] + 1 * isMovingBounceback[4];
	
	std::cout << "nx " << nx << " ny " << ny << " nz " << nz << std::endl;
	
	if ( nx == 0 && ny == 0 && nz == 0 )
	{
		// very unlikely scenario, if this happens we need to build the normal from the 7-18 or 19-26 neighbours
		// lets work on this later
	}
	// for quadratic extrapolation I need 3 cells in the direction of the normal
	int extrapolatedNbr[3];
	int currentNbr = cell;
	for ( int step = 0; step < 3; step++ )
	{
		// Step in X direction 
		if ( nx == 1 )       currentNbr += 1;
		else if ( nx == -1 ) currentNbr -= 1;
		// Step in Y direction
		if ( ny == 1 )       currentNbr = jPlusArrayCPU( currentNbr );
		else if ( ny == -1 ) currentNbr = jMinusArrayCPU( currentNbr );
		// Step in Z direction
		if ( nz == 1 )       currentNbr = kPlusArrayCPU( currentNbr );
		else if ( nz == -1 ) currentNbr = kMinusArrayCPU( currentNbr );
		extrapolatedNbr[step] = currentNbr;
	}
	
	// check if the extrapolated neighbour is valid = is in the right place and is fluid
	// only use as many extrapolated neighbours as allowed
	int extrapolatedCount = 0;
	for ( int step = 0; step < 3; step++ )
	{
		const int nbr = extrapolatedNbr[step];
		const int iNbr = iArrayCPU( nbr );
		const int jNbr = jArrayCPU( nbr );
		const int kNbr = kArrayCPU( nbr );
		bool valid = true;
		if 		( iNbr != iCell + nx * (step+1) ) valid = false;
		else if ( jNbr != jCell + ny * (step+1) ) valid = false;
		else if ( kNbr != kCell + nz * (step+1) ) valid = false;
		else if ( true && ( movingBouncebackMarkerArrayCPU( nbr ) || movingBouncebackMarkerArrayCPU( nbr ) ) ) valid = false;
		else if ( true && bouncebackMarkerArrayCPU( nbr ) ) valid = false;
		if ( !valid ) break;
		else extrapolatedCount++;
	}
	std::cout << "extrapolatedCount " << extrapolatedCount << std::endl;
	// initialize the distribution functions that we will be inserting into the newly uncovered cell
	float fRepair[27];
	// for a moment pretend we are still moving bounceback, we will need this later
	MarkerStruct Marker;
	Marker.movingBounceback = true;
	BCRhoUGStruct BCRhoUG;
	getBCRhoUG( BCRhoUG, iCell, jCell, kCell, Info, Marker ); 
	BCRhoUG.rho = 1.f;
	// find fRepair depending on available extrapolation level
	if ( extrapolatedCount == 0 ) // no extrapolation, use equillibrium
	{
		getFeq(BCRhoUG.rho, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fRepair);
	}
	else
	{
		// Read distribution functions from all valid extrapolated neighbors
		float fExtrapolated[3][27];
		for ( int step = 0; step < extrapolatedCount; step++ )
		{
			const int nbr = extrapolatedNbr[step];
			NBRStruct NBRofNBR;
			NBRofNBR.self = nbr;
			NBRofNBR.jPlus = jPlusArrayCPU( nbr );
			NBRofNBR.kPlus = kPlusArrayCPU( nbr );
			NBRofNBR.jkPlus = jPlusArrayCPU( kPlusArrayCPU( nbr ) );
			finishNBRPlus( NBRofNBR, Info );
			
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( cellReadIndex, fReadIndex, NBRofNBR, esotwistFlipper, Info );
			
			for ( int direction = 0; direction < 27; direction++ )
			{
				fExtrapolated[step][direction] = fArrayCPU(fReadIndex[direction], cellReadIndex[direction]);
			}
		}
		
		// Apply the appropriate extrapolation formula based on how many valid cells we found
		if ( extrapolatedCount == 1 ) // constant extrapolation
		{
			for ( int direction = 0; direction < 27; direction++ )
			{
				fRepair[direction] = fExtrapolated[0][direction];
			}
		}
		else if ( extrapolatedCount == 2 ) // linear extrapolation: f(x) = 2*f(x+1) - f(x+2)
		{
			for ( int direction = 0; direction < 27; direction++ )
			{
				fRepair[direction] = 2.0f * fExtrapolated[0][direction] - 1.0f * fExtrapolated[1][direction];
			}
		}
		else if ( extrapolatedCount == 3 ) // quadratic extrapolation: f(x) = 3*f(x+1) - 3*f(x+2) + f(x+3)
		{
			for ( int direction = 0; direction < 27; direction++ )
			{
				fRepair[direction] = 3.0f * fExtrapolated[0][direction] - 3.0f * fExtrapolated[1][direction] + 1.0f * fExtrapolated[2][direction];
			}
		}
	}		
	// now we still want to set ux, uy, uz to match the moving bounceback
	// so we modify the equillibrium part
	
	float rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated;
	getRhoUxUyUz( rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated, fRepair );
	float fEqExtrapolated[27];
	float fEqTarget[27];
	// get equilibrium of the extrapolated fluid
	getFeq( rhoExtrapolated, uxExtrapolated, uyExtrapolated, uzExtrapolated, fEqExtrapolated );
	// get equilibrium using the target wall velocity (but keeping the extrapolated density)
	getFeq( rhoExtrapolated, BCRhoUG.ux, BCRhoUG.uy, BCRhoUG.uz, fEqTarget );
	// reconstruct: f_new = f_eq(wall_u) + (f_extrap - f_eq(extrap_u))
	for ( int direction = 0; direction < 27; direction++ )
	{
		fRepair[direction] = fEqTarget[direction] + ( fRepair[direction] - fEqExtrapolated[direction] );
	}
	
	// BCRhoUG.rho is needed below for applyMovingBounceback
	BCRhoUG.rho = rhoExtrapolated;
	
	// write fRepair into our cell
	int cellWriteIndex[27];
	int fWriteIndex[27];
	getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
	std::cout << "Overwriting cell now" << std::endl;
	for ( int direction = 0; direction < 27; direction++ ) fArrayCPU( fWriteIndex[direction], cellWriteIndex[direction] ) = fRepair[direction];
	
	// also repair the distribution functions that are going to be pulled into our cell next iteration from moving bounceback cells
	applyMovingBounceback( fRepair, BCRhoUG );
	int cellNextIndex[27];
	int fNextIndex[27];
	bool inverseEsotwistFlipper = !esotwistFlipper;
	getPreCollisionIndex( cellNextIndex, fNextIndex, NBR, inverseEsotwistFlipper, Info );
	std::cout << "Overwriting incoming populations now" << std::endl;	
	for ( int direction = 0; direction < 27; direction++ ) 
	{
		if ( isMovingBounceback[direction] ) 
		{
			// if we are going to be receiving f from a moving bounceback in this direction,
			// set it to the moving bounceback result
			fArrayCPU( fNextIndex[direction], cellNextIndex[direction] ) = fRepair[direction];
		}
	}	

	
	
	// END
	
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
			updateMovingBounceback( grids[level], Voxelizer );
			if ( grids[level].Info.iterationsFinished > 54 ) FIND_WHATS_GOING_ON( grids[level]);
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
	
	float fMaxTotal = 0.f;
	
	for ( int iteration = 0; iteration < iterationCount; iteration++ )
	{
		if ( iteration % iterationChunk == 0 )
		{
			std::cout << std::endl;
			std::cout << "Finished iteration " << iteration << std::endl;
			
			lapTimer.stop();
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)usefulCellUpdatesPerIteration * (float)iterationChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			InfoStruct &Info = grids[0].Info;
			const bool &esotwistFlipper = grids[0].esotwistFlipper;
			auto fView = grids[0].fArray.getConstView();
			auto movingBouncebackMarkerView = grids[0].movingBouncebackMarkerArray.getConstView();
			auto bouncebackMarkerView = grids[0].bouncebackMarkerArray.getConstView();
			auto jPlusView = grids[0].NBR.jPlusArray.getConstView();
			auto kPlusView = grids[0].NBR.kPlusArray.getConstView();
			auto jkPlusView = grids[0].NBR.jkPlusArray.getConstView();
			auto fetch = [ = ] __cuda_callable__( const int singleIndex )
			{
				const int direction = singleIndex % 27;
				const int cell = singleIndex / 27;
				const bool MBBmarker = movingBouncebackMarkerView( cell );
				if ( MBBmarker ) return 0.f;
				const bool BBmarker = bouncebackMarkerView( cell );
				if ( BBmarker ) return 0.f;
				
				NBRStruct NBR;
				NBR.self = cell;
				NBR.jPlus = jPlusView( cell );
				NBR.kPlus = kPlusView( cell );
				NBR.jkPlus = jkPlusView( cell );
				finishNBRPlus( NBR, Info );
				
				int cellReadIndex[27];
				int fReadIndex[27];
				getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
				const float f = fView(fReadIndex[direction], cellReadIndex[direction]);
				
				return std::fabs(f);
			};
			auto reduction = [] __cuda_callable__( const float& a, const float& b )
			{
				return TNL::max( a, b );
			};
			const int start = 0;
			const int end = 27 * grids[0].Info.cellCount;
			const float fMax = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0.f );
			
			//const float fMax = findMaxFloatArray2D( grids[0].fArray, 27, grids[0].Info.cellCount );
			std::cout << "fMax " << fMax << std::endl;
			if ( fMax > fMaxTotal ) fMaxTotal = fMax;
			std::cout << "fMaxTotal " << fMaxTotal << std::endl;
			if ( iteration >= 57 )
			{
				GridStruct &Grid = grids[0];
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
				
				std::cout << "moving bounceback state of cell 2121100: " << movingBouncebackMarkerArrayCPU[ 2121100 ] << std::endl;				
				
				for ( int cell = 0; cell < Info.cellCount; cell++ )
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
					for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayCPU.getElement(fReadIndex[direction], cellReadIndex[direction]);
					for ( int direction = 0; direction < 27; direction++ )
					{
						if ( std::fabs( f[direction] ) > 0.336127f ) 
						{
							if ( !movingBouncebackMarkerArrayCPU[cell] && !bouncebackMarkerArrayCPU[cell] )
							{
								std::cout << "Maximum exceeded in cell " << cell << std::endl;
								std::cout << "Direction " << direction << std::endl;
							}
						}
					}
				}
			}
			
			/*
			const int r = 14.f;
			exportSectionCutPlotToiletPaperZ( grids, r, iteration );
			if (system("python3 ../../include/plotter/plotterGridID.py") != 0) {}
			*/
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
