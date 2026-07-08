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
	
	VoxelizerStruct Voxelizer;
	Voxelizer.Info.ox = -18.f;
	Voxelizer.Info.oy = -18.f;
	Voxelizer.Info.res = 0.05f;
	Voxelizer.Info.cellCountX = (int)(36.f / Voxelizer.Info.res);
	Voxelizer.Info.cellCountY = Voxelizer.Info.cellCountX;
	
	TNL::Timer Timer;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapBounceback, STLMain, Voxelizer );
	Timer.stop();
	std::cout << "Voxelization 1 took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( Voxelizer.rayMapMovingBounceback, STLImpeller, Voxelizer );
	Timer.stop();
	std::cout << "Voxelization 2 took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	Voxelizer.rayMapTotal = Voxelizer.rayMapBounceback;
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMapMovingBounceback );
	Timer.stop();
	std::cout << "Summing ray maps took " << Timer.getRealTime() << " s" << std::endl;
	
	return EXIT_SUCCESS;
}
