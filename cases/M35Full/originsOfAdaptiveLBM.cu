#include "../../include/types.h"

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/voxelizerFunctionsBackup.h"

std::string STLPath = "M-Jet_35_pump_main.STL";

int main(int argc, char **argv)
{
	STLStruct STL;
	readSTL( STL, STLPath );
	
	VoxelizerStruct Voxelizer;
	Voxelizer.Info.ox = -18.f;
	Voxelizer.Info.oy = -18.f;
	Voxelizer.Info.res = 0.01f;
	Voxelizer.Info.cellCountX = (int)(36.f / Voxelizer.Info.res);
	Voxelizer.Info.cellCountY = Voxelizer.Info.cellCountX;
	
	TNL::Timer Timer;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL( STL, Voxelizer, Voxelizer.rayMapBouncebackArray );
	Timer.stop();
	std::cout << "New version took " << Timer.getRealTime() << " s" << std::endl;
	
	Timer.reset();
	Timer.start();
	voxelizeSTL_OLD( STL, Voxelizer, Voxelizer.rayMapMovingBouncebackArray );
	Timer.stop();
	std::cout << "Old version took " << Timer.getRealTime() << " s" << std::endl;
	
	// check if they are completely the same
	bool error = false;
	auto rayMapView1 = Voxelizer.rayMapBouncebackArray.getView();
	auto rayMapView2 = Voxelizer.rayMapMovingBouncebackArray.getView();
	auto fetch = [ = ] __cuda_callable__( const int singleIndex )
		{
			const int i = singleIndex % Voxelizer.Info.cellCountX;
			const int j = singleIndex / Voxelizer.Info.cellCountX;
			return std::abs( rayMapView1(i, j, layer) - rayMapView2(i, j, layer) ); // <- need to fix this line now
		};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
		{
			return TNL::max( a, b );
		};
	const int start = 0;
	const int end = Voxelizer.Info.cellCountX * Voxelizer.Info.cellCountY;
	for ( int layer = 0; layer < 32; layer++ )
	{
		const int maxError = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0 );
		if (maxError > 0) error = true;
	}
	if (error) std::cout << "ERROR! Old and new voxelizer don't match" << std::endl;
	else  std::cout << "OK! Old and new voxelizer match in every single index" << std::endl;
	
	return EXIT_SUCCESS;
}
