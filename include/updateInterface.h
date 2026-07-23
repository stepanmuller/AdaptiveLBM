#pragma once

#include "./applyCollision.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"

void updateFineToCoarseInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	const InfoStruct &InfoCoarse = GridCoarse.Info;
	auto fViewCoarse = GridCoarse.fArray.getView();
	const bool &esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto jPlusViewCoarse = GridCoarse.NBR.jPlusArray.getConstView();
	auto kPlusViewCoarse = GridCoarse.NBR.kPlusArray.getConstView();
	auto jkPlusViewCoarse = GridCoarse.NBR.jkPlusArray.getConstView();
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto jPlusViewFine = GridFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.NBR.kPlusArray.getConstView();
	auto jkPlusViewFine = GridFine.NBR.jkPlusArray.getConstView();
	
	auto fineToCoarseIndexView = GridCoarse.fineToCoarseIndexArray.getConstView();
	auto childMapView = GridCoarse.childMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cellCoarse = fineToCoarseIndexView( index );
		const int cellFine0 = childMapView( cellCoarse );
		
		// Initialize accumulation variables
		// linear interpolation for rho -> this is a simple average from all fine cells
		float rho = 0.f;
		// quadratic interpolation for u -> use coefficients found in Pavel Eichler's disertation, Appendix C
		// we only need the value at [0, 0, 0] so all we need is a0, b0, c0 and the helper coefficients
		float A11 = 0.f, A12 = 0.f, A13 = 0.f, B22 = 0.f, B12 = 0.f, B23 = 0.f, C33 = 0.f, C13 = 0.f, C23 = 0.f;
		float ux = 0.f, uy = 0.f, uz = 0.f; // in our case this equals a0, b0, c0
		// Linear interpolation for f, from which we will calculate central moments
		float fAvg[27] = {0.f};

		int stencilCell[8];
		stencilCell[0] = cellFine0;
		stencilCell[1] = cellFine0 + 1; if ( stencilCell[1] >= InfoFine.cellCount ) stencilCell[1] = 0;
		stencilCell[2] = jPlusViewFine( cellFine0 );
		stencilCell[3] = stencilCell[2] + 1; if ( stencilCell[3] >= InfoFine.cellCount ) stencilCell[3] = 0;
		stencilCell[4] = kPlusViewFine( cellFine0 );
		stencilCell[5] = stencilCell[4] + 1; if ( stencilCell[5] >= InfoFine.cellCount ) stencilCell[5] = 0;
		stencilCell[6] = jkPlusViewFine( cellFine0 );
		stencilCell[7] = stencilCell[6] + 1; if ( stencilCell[7] >= InfoFine.cellCount ) stencilCell[7] = 0;
		
		// Accumulate contributions for each neighbour
		for ( int i = 0; i < 8; i++ )
		{
			const int nbr = stencilCell[i];
			NBRStruct NBRofNBR;
			NBRofNBR.self = nbr;
			NBRofNBR.jPlus = jPlusViewFine( nbr );
			NBRofNBR.kPlus = kPlusViewFine( nbr );
			NBRofNBR.jkPlus = jkPlusViewFine( nbr );
			finishNBRPlus( NBRofNBR, InfoFine );
			
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPostCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperFine, InfoFine );
			
			float fNbr[27];
			float rhoNbr, uxNbr, uyNbr, uzNbr;
			LocalDuStruct LocalDuNbr;

			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewFine( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			getRhoUxUyUz( rhoNbr, uxNbr, uyNbr, uzNbr, fNbr );
			getLocalDu( fNbr, InfoFine.nu, LocalDuNbr );
			
			// Add contributions to rho
			rho += rhoNbr;
			// Add contributions to f
			for ( int direction = 0; direction < 27; direction++ ) fAvg[direction] += fNbr[direction];
			
			//ux += 2.f * uxNbr; // LINEAR VERSION
			//uy += 2.f * uyNbr;
			//uz += 2.f * uzNbr;
			// Add this neighbor's specific contribution to ux, uy, uz and the helper coefficients
			switch(i) {
				case 0: // (-1/2, -1/2, -1/2) -> x1
					ux += 2.f * uxNbr + uzNbr + uyNbr;
					uy += uxNbr + 2.f * uyNbr + uzNbr;
					uz += uxNbr + uyNbr + 2.f * uzNbr;
					A11 -= LocalDuNbr.duxdx; 
					A12 -= LocalDuNbr.duxdyCross;
					A13 -= LocalDuNbr.duxdzCross;
					B22 -= LocalDuNbr.duydy; 
					B12 -= LocalDuNbr.duxdyCross;
					B23 -= LocalDuNbr.duydzCross;
					C33 -= LocalDuNbr.duzdz; 
					C13 -= LocalDuNbr.duxdzCross;
					C23 -= LocalDuNbr.duydzCross;
					break;
				case 1: // (+1/2, -1/2, -1/2) -> x2
					ux += 2.f * uxNbr - uyNbr - uzNbr;
					uy += -uxNbr + 2.f * uyNbr + uzNbr;
					uz += -uxNbr + uyNbr + 2.f * uzNbr;
					A11 += LocalDuNbr.duxdx; 
					A12 += LocalDuNbr.duxdyCross;
					A13 += LocalDuNbr.duxdzCross;
					B22 -= LocalDuNbr.duydy; 
					B12 -= LocalDuNbr.duxdyCross;
					B23 -= LocalDuNbr.duydzCross;
					C33 -= LocalDuNbr.duzdz; 
					C13 -= LocalDuNbr.duxdzCross;
					C23 -= LocalDuNbr.duydzCross;
					break;
				case 2: // (-1/2, +1/2, -1/2) -> x5
					ux += 2.f * uxNbr - uyNbr + uzNbr;
					uy += -uxNbr + 2.f * uyNbr - uzNbr;
					uz += uxNbr - uyNbr + 2.f * uzNbr;
					A11 -= LocalDuNbr.duxdx; 
					A12 -= LocalDuNbr.duxdyCross;
					A13 -= LocalDuNbr.duxdzCross;
					B22 += LocalDuNbr.duydy; 
					B12 += LocalDuNbr.duxdyCross;
					B23 += LocalDuNbr.duydzCross;
					C33 -= LocalDuNbr.duzdz; 
					C13 -= LocalDuNbr.duxdzCross;
					C23 -= LocalDuNbr.duydzCross;
					break;
				case 3: // (+1/2, +1/2, -1/2) -> x6
					ux += 2.f * uxNbr + uyNbr - uzNbr;
					uy += uxNbr + 2.f * uyNbr - uzNbr;
					uz += -uxNbr - uyNbr + 2.f * uzNbr;
					A11 += LocalDuNbr.duxdx; 
					A12 += LocalDuNbr.duxdyCross;
					A13 += LocalDuNbr.duxdzCross;
					B22 += LocalDuNbr.duydy; 
					B12 += LocalDuNbr.duxdyCross;
					B23 += LocalDuNbr.duydzCross;
					C33 -= LocalDuNbr.duzdz; 
					C13 -= LocalDuNbr.duxdzCross;
					C23 -= LocalDuNbr.duydzCross;
					break;
				case 4: // (-1/2, -1/2, +1/2) -> x4 
					ux += 2.f * uxNbr + uyNbr - uzNbr;
					uy += uxNbr + 2.f * uyNbr - uzNbr;
					uz += -uxNbr - uyNbr + 2.f * uzNbr;
					A11 -= LocalDuNbr.duxdx; 
					A12 -= LocalDuNbr.duxdyCross;
					A13 -= LocalDuNbr.duxdzCross;
					B22 -= LocalDuNbr.duydy; 
					B12 -= LocalDuNbr.duxdyCross;
					B23 -= LocalDuNbr.duydzCross;
					C33 += LocalDuNbr.duzdz; 
					C13 += LocalDuNbr.duxdzCross;
					C23 += LocalDuNbr.duydzCross;
					break;
				case 5: // (+1/2, -1/2, +1/2) -> x3
					ux += 2.f * uxNbr - uyNbr + uzNbr;
					uy += -uxNbr + 2.f * uyNbr - uzNbr;
					uz += uxNbr - uyNbr + 2.f * uzNbr;
					A11 += LocalDuNbr.duxdx; 
					A12 += LocalDuNbr.duxdyCross;
					A13 += LocalDuNbr.duxdzCross;
					B22 -= LocalDuNbr.duydy; 
					B12 -= LocalDuNbr.duxdyCross;
					B23 -= LocalDuNbr.duydzCross;
					C33 += LocalDuNbr.duzdz; 
					C13 += LocalDuNbr.duxdzCross;
					C23 += LocalDuNbr.duydzCross;
					break;
				case 6: // (-1/2, +1/2, +1/2) -> x8 
					ux += 2.f * uxNbr - uyNbr - uzNbr;
					uy += -uxNbr + 2.f * uyNbr + uzNbr;
					uz += -uxNbr + uyNbr + 2.f * uzNbr;
					A11 -= LocalDuNbr.duxdx; 
					A12 -= LocalDuNbr.duxdyCross;
					A13 -= LocalDuNbr.duxdzCross;
					B22 += LocalDuNbr.duydy; 
					B12 += LocalDuNbr.duxdyCross;
					B23 += LocalDuNbr.duydzCross;
					C33 += LocalDuNbr.duzdz; 
					C13 += LocalDuNbr.duxdzCross;
					C23 += LocalDuNbr.duydzCross;
					break;
				case 7: // (+1/2, +1/2, +1/2) -> x7 
					ux += 2.f * uxNbr + uyNbr + uzNbr;
					uy += uxNbr + 2.f * uyNbr + uzNbr;
					uz += uxNbr + uyNbr + 2.f * uzNbr;
					A11 += LocalDuNbr.duxdx; 
					A12 += LocalDuNbr.duxdyCross;
					A13 += LocalDuNbr.duxdzCross;
					B22 += LocalDuNbr.duydy; 
					B12 += LocalDuNbr.duxdyCross;
					B23 += LocalDuNbr.duydzCross;
					C33 += LocalDuNbr.duzdz; 
					C13 += LocalDuNbr.duxdzCross;
					C23 += LocalDuNbr.duydzCross;
					break;
			}
		}
		
		// divide rho by 8
		rho *= 0.125f;
		// divide f by 8
		for ( int direction = 0; direction < 27; direction++ ) fAvg[direction] *= 0.125f;
		
		// divide A, B, C by 4
		A11 *= 0.25f; A12 *= 0.25f; A13 *= 0.25f; B22 *= 0.25f; B12 *= 0.25f; B23 *= 0.25f; C33 *= 0.25f; C13 *= 0.25f; C23 *= 0.25f;
		
		// add the helper coefficients to find final ux, uy, uz 
		ux += -2.f * A11 - 2.f * C13 - 2 * B12;
		uy += -2.f * B22 - 2.f * C23 - 2 * A12;
		uz += -2.f * C33 - 2.f * B23 - 2 * A13;
		
		// divide ux, uy, uz by 16
		ux *= 0.0625f; uy *= 0.0625f; uz *= 0.0625f;
		
		// get central moments
		float rhoTemp, uxTemp, uyTemp, uzTemp;
		float k_000; 
		float k_100; float k_010; float k_001; 
		float k_200; float k_020; float k_002;
		float k_011; float k_101; float k_110;
		LocalDuStruct LocalDuTemp;
		getInterpolationVariables( 	rhoTemp, uxTemp, uyTemp, uzTemp, 
									k_000,
									k_100, k_010, k_001, 
									k_200, k_020, k_002, 
									k_011, k_101, k_110,
									LocalDuTemp, 
									fAvg, InfoFine.nu );
		
		// rescale central moments
		const bool coarseToFine = false;
		rescaleCentralMoments( 	k_200, k_020, k_002, 
								k_011, k_101, k_110,
								coarseToFine );
		
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jkPlusViewCoarse( cellCoarse );
		finishNBRPlus( NBR, InfoCoarse );
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		
		float f[27];
		useInterpolatedVariables( 	rho, ux, uy, uz, 
									k_000,
									k_100, k_010, k_001, 
									k_200, k_020, k_002, 
									k_011, k_101, k_110, 
									f );
		
		for ( int direction = 0; direction < 27; direction++ ) fViewCoarse( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
	};
	
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, InfoCoarse.fineToCoarseCount, cellLambda );
}

void updateCoarseToFineInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	const InfoStruct &InfoCoarse = GridCoarse.Info;
	auto fViewCoarse = GridCoarse.fArray.getView();
	const bool &esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto jPlusViewCoarse = GridCoarse.NBR.jPlusArray.getConstView();
	auto kPlusViewCoarse = GridCoarse.NBR.kPlusArray.getConstView();
	auto jkPlusViewCoarse = GridCoarse.NBR.jkPlusArray.getConstView();
	auto jMinusViewCoarse = GridCoarse.NBR.jMinusArray.getConstView();
	auto kMinusViewCoarse = GridCoarse.NBR.kMinusArray.getConstView();
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto jPlusViewFine = GridFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.NBR.kPlusArray.getConstView();
	auto jkPlusViewFine = GridFine.NBR.jkPlusArray.getConstView();
	
	auto coarseToFineIndexView = GridCoarse.coarseToFineIndexArray.getConstView();
	auto childMapView = GridCoarse.childMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cellCoarse = coarseToFineIndexView( index );
		
		// Get base data = center cell
		float fBase[27];
		float rhoBase, uxBase, uyBase, uzBase;
		float k_000Base; 
		float k_100Base; float k_010Base; float k_001Base; 
		float k_200Base; float k_020Base; float k_002Base;
		float k_011Base; float k_101Base; float k_110Base;
		LocalDuStruct LocalDuBase;
		
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jkPlusViewCoarse( cellCoarse );
		finishNBRPlus( NBR, InfoCoarse );
		int cellReadIndex[27], fReadIndex[27];
		getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		for ( int direction = 0; direction < 27; direction++ ) {
			fBase[direction] = fViewCoarse(fReadIndex[direction], cellReadIndex[direction]);
		}
		
		getInterpolationVariables( 	rhoBase, uxBase, uyBase, uzBase, 
									k_000Base,
									k_100Base, k_010Base, k_001Base, 
									k_200Base, k_020Base, k_002Base, 
									k_011Base, k_101Base, k_110Base,
									LocalDuBase, 
									fBase, InfoCoarse.nu );

		// Initialize accumulation variables
		// linear interpolation for rho
		float dRhodx = 0.f, dRhody = 0.f, dRhodz = 0.f;
		// quadratic interpolation for u -> prepare all coefficients
		// u_x(x,y,z) = a_0 + a_x x + a_y y + a_z z + a_{xy} xy + a_{xz} xz + a_{yz} yz + a_{xx} x^2 + a_{yy} y^2 + a_{zz} z^2
		// u_y(x,y,z) = b_0 + b_x x + b_y y + b_z z + b_{xy} xy + b_{xz} xz + b_{yz} yz + b_{xx} x^2 + b_{yy} y^2 + b_{zz} z^2
		// u_z(x,y,z) = c_0 + c_x x + c_y y + c_z z + c_{xy} xy + c_{xz} xz + c_{yz} yz + c_{xx} x^2 + c_{yy} y^2 + c_{zz} z^2
		float ax = 0.f, bx = 0.f, cx = 0.f;
		float ay = 0.f, by = 0.f, cy = 0.f;
		float az = 0.f, bz = 0.f, cz = 0.f;
		float axy = 0.f, axz = 0.f, bxy = 0.f, byz = 0.f, cxz = 0.f, cyz = 0.f;
		float K1 = 0.f, K2 = 0.f, K3 = 0.f;

		// Second derivatives start with the center cell contribution
		float axx = -uxBase, bxx = -uyBase, cxx = -uzBase;
		float ayy = -uxBase, byy = -uyBase, cyy = -uzBase;
		float azz = -uxBase, bzz = -uyBase, czz = -uzBase;

		const int stencilCell[6] = { 
			cellCoarse+1, cellCoarse-1, 
			jPlusViewCoarse(cellCoarse), jMinusViewCoarse(cellCoarse), 
			kPlusViewCoarse(cellCoarse), kMinusViewCoarse(cellCoarse) 
		};
		
		// Accumulate contributions for each neighbour
		for ( int i = 0; i < 6; i++ )
		{
			const int nbr = stencilCell[i];
			NBRStruct NBRofNBR;
			NBRofNBR.self = nbr;
			NBRofNBR.jPlus = jPlusViewCoarse( nbr );
			NBRofNBR.kPlus = kPlusViewCoarse( nbr );
			NBRofNBR.jkPlus = jkPlusViewCoarse( nbr );
			finishNBRPlus( NBRofNBR, InfoCoarse );
			
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPostCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperCoarse, InfoCoarse );
			
			float fNbr[27];
			float rhoNbr, uxNbr, uyNbr, uzNbr;
			float k_000Nbr; 
			float k_100Nbr; float k_010Nbr; float k_001Nbr; 
			float k_200Nbr; float k_020Nbr; float k_002Nbr;
			float k_011Nbr; float k_101Nbr; float k_110Nbr;
			LocalDuStruct LocalDuNbr;

			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewCoarse( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			
			getInterpolationVariables( 	rhoNbr, uxNbr, uyNbr, uzNbr, 
										k_000Nbr,
										k_100Nbr, k_010Nbr, k_001Nbr, 
										k_200Nbr, k_020Nbr, k_002Nbr, 
										k_011Nbr, k_101Nbr, k_110Nbr,
										LocalDuNbr, 
										fNbr, InfoCoarse.nu );
			
			// Add this neighbor's specific contribution based on its position
			switch(i) {
				case 0: // X+
					//for(int d=0; d<27; d++) dfNeqdx[d] += 0.5f * fNeqNbr[d];
					dRhodx += 0.5f * rhoNbr; 
					ax += 0.5f * uxNbr; bx += 0.5f * uyNbr; cx += 0.5f * uzNbr;
					axx += 0.5f * uxNbr; bxx += 0.5f * uyNbr; cxx += 0.5f * uzNbr;
					bxy += 0.5f * LocalDuNbr.duydy; cxz += 0.5f * LocalDuNbr.duzdz;
					K3 += 0.5f * LocalDuNbr.duydzCross;
					break;
				case 1: // X-
					//for(int d=0; d<27; d++) dfNeqdx[d] -= 0.5f * fNeqNbr[d];
					dRhodx -= 0.5f * rhoNbr; 
					ax -= 0.5f * uxNbr; bx -= 0.5f * uyNbr; cx -= 0.5f * uzNbr;
					axx += 0.5f * uxNbr; bxx += 0.5f * uyNbr; cxx += 0.5f * uzNbr;
					bxy -= 0.5f * LocalDuNbr.duydy; cxz -= 0.5f * LocalDuNbr.duzdz;
					K3 -= 0.5f * LocalDuNbr.duydzCross;
					break;
				case 2: // Y+
					//for(int d=0; d<27; d++) dfNeqdy[d] += 0.5f * fNeqNbr[d];
					dRhody += 0.5f * rhoNbr;
					ay += 0.5f * uxNbr; by += 0.5f * uyNbr; cy += 0.5f * uzNbr;
					ayy += 0.5f * uxNbr; byy += 0.5f * uyNbr; cyy += 0.5f * uzNbr;
					axy += 0.5f * LocalDuNbr.duxdx; cyz += 0.5f * LocalDuNbr.duzdz;
					K2 += 0.5f * LocalDuNbr.duxdzCross;
					break;
				case 3: // Y-
					//for(int d=0; d<27; d++) dfNeqdy[d] -= 0.5f * fNeqNbr[d];
					dRhody -= 0.5f * rhoNbr;
					ay -= 0.5f * uxNbr; by -= 0.5f * uyNbr; cy -= 0.5f * uzNbr;
					ayy += 0.5f * uxNbr; byy += 0.5f * uyNbr; cyy += 0.5f * uzNbr;
					axy -= 0.5f * LocalDuNbr.duxdx; cyz -= 0.5f * LocalDuNbr.duzdz;
					K2 -= 0.5f * LocalDuNbr.duxdzCross;
					break;
				case 4: // Z+
					//for(int d=0; d<27; d++) dfNeqdz[d] += 0.5f * fNeqNbr[d];
					dRhodz += 0.5f * rhoNbr;
					az += 0.5f * uxNbr; bz += 0.5f * uyNbr; cz += 0.5f * uzNbr;
					azz += 0.5f * uxNbr; bzz += 0.5f * uyNbr; czz += 0.5f * uzNbr;
					axz += 0.5f * LocalDuNbr.duxdx; byz += 0.5f * LocalDuNbr.duydy;
					K1 += 0.5f * LocalDuNbr.duxdyCross;
					break;
				case 5: // Z-
					//for(int d=0; d<27; d++) dfNeqdz[d] -= 0.5f * fNeqNbr[d];
					dRhodz -= 0.5f * rhoNbr;
					az -= 0.5f * uxNbr; bz -= 0.5f * uyNbr; cz -= 0.5f * uzNbr;
					azz += 0.5f * uxNbr; bzz += 0.5f * uyNbr; czz += 0.5f * uzNbr;
					axz -= 0.5f * LocalDuNbr.duxdx; byz -= 0.5f * LocalDuNbr.duydy;
					K1 -= 0.5f * LocalDuNbr.duxdyCross;
					break;
			}
		}

		// Final calculations for the cross terms
		float ayz = 0.5f * ( K1 + K2 - K3 );
		float bxz = 0.5f * ( K1 - K2 + K3 );
		float cxy = 0.5f * ( -K1 + K2 + K3 );
		
		// rescale central moments
		const bool coarseToFine = true;
		rescaleCentralMoments( 	k_200Base, k_020Base, k_002Base, 
								k_011Base, k_101Base, k_110Base,
								coarseToFine );
		
		const int cellFine0 = childMapView( cellCoarse );

		int cellFineList[8];
		cellFineList[0] = cellFine0;
		cellFineList[1] = cellFine0 + 1; if ( cellFineList[1] >= InfoFine.cellCount ) cellFineList[1] = 0;
		cellFineList[2] = jPlusViewFine( cellFine0 );
		cellFineList[3] = cellFineList[2] + 1; if ( cellFineList[3] >= InfoFine.cellCount ) cellFineList[3] = 0;
		cellFineList[4] = kPlusViewFine( cellFine0 );
		cellFineList[5] = cellFineList[4] + 1; if ( cellFineList[5] >= InfoFine.cellCount ) cellFineList[5] = 0;
		cellFineList[6] = jkPlusViewFine( cellFine0 );
		cellFineList[7] = cellFineList[6] + 1; if ( cellFineList[7] >= InfoFine.cellCount ) cellFineList[7] = 0;
		const float cellFineDx[8] = {-0.25f, 0.25f,-0.25f, 0.25f,-0.25f, 0.25f,-0.25f, 0.25f};
		const float cellFineDy[8] = {-0.25f,-0.25f, 0.25f, 0.25f,-0.25f,-0.25f, 0.25f, 0.25f};
		const float cellFineDz[8] = {-0.25f,-0.25f,-0.25f,-0.25f, 0.25f, 0.25f, 0.25f, 0.25f};		
		
		for ( int which = 0; which < 8; which++ )
		{
			const int cellFine = cellFineList[which];
			NBR.self = cellFine;
			NBR.jPlus = jPlusViewFine( cellFine );
			NBR.kPlus = kPlusViewFine( cellFine );
			NBR.jkPlus = jkPlusViewFine( cellFine );
			finishNBRPlus( NBR, InfoFine );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperFine, InfoFine );
			
			const float dx = cellFineDx[which];
			const float dy = cellFineDy[which];
			const float dz = cellFineDz[which];
			const float rho = rhoBase + dRhodx * dx + dRhody * dy + dRhodz * dz;
			const float ux = uxBase + ax * dx + ay * dy + az * dz + axy * dx * dy + axz * dx * dz + ayz * dy * dz + axx * dx * dx + ayy * dy * dy + azz * dz * dz;
			const float uy = uyBase + bx * dx + by * dy + bz * dz + bxy * dx * dy + bxz * dx * dz + byz * dy * dz + bxx * dx * dx + byy * dy * dy + bzz * dz * dz;
			const float uz = uzBase + cx * dx + cy * dy + cz * dz + cxy * dx * dy + cxz * dx * dz + cyz * dy * dz + cxx * dx * dx + cyy * dy * dy + czz * dz * dz;
			
			float f[27];
			useInterpolatedVariables( 	rho, ux, uy, uz, 
										k_000Base,
										k_100Base, k_010Base, k_001Base, 
										k_200Base, k_020Base, k_002Base, 
										k_011Base, k_101Base, k_110Base, 
										f );
			
			for ( int direction = 0; direction < 27; direction++ ) fViewFine( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, InfoCoarse.coarseToFineCount, cellLambda );
}

void updateInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	updateFineToCoarseInterface( GridCoarse, GridFine );
	updateCoarseToFineInterface( GridCoarse, GridFine );
}
