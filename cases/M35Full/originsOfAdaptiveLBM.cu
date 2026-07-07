#include "../../include/types.h"

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
	voxelizeSTL( STLMain, Voxelizer, Voxelizer.rayMapBouncebackArray );
	Timer.stop();
	std::cout << "Voxelization 1 took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( STLImpeller, Voxelizer, Voxelizer.rayMapMovingBouncebackArray );
	Timer.stop();
	std::cout << "Voxelization 2 took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	Voxelizer.rayMapTotalArray = Voxelizer.rayMapBouncebackArray;
	sumRayMaps( Voxelizer.rayMapTotalArray, Voxelizer.rayMapMovingBouncebackArray, Voxelizer.counterArray );
	Timer.stop();
	std::cout << "Summing ray maps took " << Timer.getRealTime() << " s" << std::endl;
	
	return EXIT_SUCCESS;
}
