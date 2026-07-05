#include "../../include/types.h"

#include "../../include/STLFunctions.h"

std::string STLPath = "M-Jet_35_pump_main.STL";

int main(int argc, char **argv)
{
	STLStruct STL;
	readSTL( STL, STLPath );
	return EXIT_SUCCESS;
}
