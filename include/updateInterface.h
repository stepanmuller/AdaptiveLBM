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
		// We want linear interpolation for fNeq -> this is a simple average from all fine cells
		float fNeq[27] = {0.f};
		// linear interpolation for rho -> this is also a simple average
		float rho = 0.f;
		// quadratic interpolation for u -> use coefficients found in Pavel Eichler's disertation, Appendix C
		// we only need the value at [0, 0, 0] so all we need is a0, b0, c0 and the helper coefficients
		float A11 = 0.f, A12 = 0.f, A13 = 0.f, B22 = 0.f, B12 = 0.f, B23 = 0.f, C33 = 0.f, C13 = 0.f, C23 = 0.f;
		float ux = 0.f, uy = 0.f, uz = 0.f; // in our case this equals a0, b0, c0

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
			
			// reused variables for this neighbor only
			float fNeqNbr[27];
			float rhoNbr, uxNbr, uyNbr, uzNbr;
			LocalDuStruct LocalDuNbr;

			for ( int direction = 0; direction < 27; direction++ ) fNeqNbr[direction] = fViewFine( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			getRhoUxUyUz( rhoNbr, uxNbr, uyNbr, uzNbr, fNeqNbr );
			getLocalDu( fNeqNbr, InfoCoarse.nu, LocalDuNbr );
			float feqNbr[27];
			getFeq(rhoNbr, uxNbr, uyNbr, uzNbr, feqNbr);
			for ( int direction = 0; direction < 27; direction++ ) fNeqNbr[direction] = fNeqNbr[direction] - feqNbr[direction];
			
			// Add contributions to rho and fNeq
			rho += rhoNbr;
			for (int direction = 0; direction < 27; direction++) fNeq[direction] += fNeqNbr[direction];
			
			ux += 2.f * uxNbr; // LINEAR VERSION
			uy += 2.f * uyNbr;
			uz += 2.f * uzNbr;
			
			// Add this neighbor's specific contribution to ux, uy, uz and the helper coefficients
			switch(i) {
				case 0: // (-1/2, -1/2, -1/2) -> x1
					//ux += 2.f * uxNbr + uzNbr + uyNbr;
					//uy += uxNbr + 2.f * uyNbr + uzNbr;
					//uz += uxNbr + uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr - uyNbr - uzNbr;
					//uy += -uxNbr + 2.f * uyNbr + uzNbr;
					//uz += -uxNbr + uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr - uyNbr + uzNbr;
					//uy += -uxNbr + 2.f * uyNbr - uzNbr;
					//uz += uxNbr - uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr + uyNbr - uzNbr;
					//uy += uxNbr + 2.f * uyNbr - uzNbr;
					//uz += -uxNbr - uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr + uyNbr - uzNbr;
					//uy += uxNbr + 2.f * uyNbr - uzNbr;
					//uz += -uxNbr - uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr - uyNbr + uzNbr;
					//uy += -uxNbr + 2.f * uyNbr - uzNbr;
					//uz += uxNbr + uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr - uyNbr - uzNbr;
					//uy += -uxNbr + 2.f * uyNbr + uzNbr;
					//uz += -uxNbr + uyNbr + 2.f * uzNbr;
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
					//ux += 2.f * uxNbr + uyNbr + uzNbr;
					//uy += uxNbr + 2.f * uyNbr + uzNbr;
					//uz += uxNbr + uyNbr + 2.f * uzNbr;
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
		
		// divide rho and fNeq by 8
		rho *= 0.125f;
		for (int direction = 0; direction < 27; direction++) fNeq[direction] *= 0.125f;
		// divide A, B, C by 4
		A11 *= 0.25f; A12 *= 0.25f; A13 *= 0.25f; B22 *= 0.25f; B12 *= 0.25f; B23 *= 0.25f; C33 *= 0.25f; C13 *= 0.25f; C23 *= 0.25f;
		
		// add the helper coefficients to find final ux, uy, uz -> this is not working correctly
		//ux += -2.f * A11 - 2.f * C13 - 2 * B12;
		//uy += -2.f * B22 - 2.f * C23 - 2 * A12;
		//uz += -2.f * C33 - 2.f * B23 - 2 * A13;
		
		// divide ux, uy, uz by 16
		ux *= 0.0625f; uy *= 0.0625f; uz *= 0.0625f;
		
		const float scale = ( 6.f * InfoCoarse.nu + 1.f) / ( 3.f * InfoFine.nu + 0.5f );
		
		for ( int direction = 0; direction < 27; direction++ ) fNeq[direction] *= scale;
		
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jkPlusViewCoarse( cellCoarse );
		finishNBRPlus( NBR, InfoCoarse );
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		
		float feq[27];
		getFeq( rho, ux, uy, uz, feq );
		
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			fViewCoarse( fWriteIndex[direction], cellWriteIndex[direction] ) = feq[direction] + fNeq[direction];
		}
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
		float fNeqBase[27];
		float rhoBase, uxBase, uyBase, uzBase;
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jkPlusViewCoarse( cellCoarse );
		finishNBRPlus( NBR, InfoCoarse );
		int cellReadIndex[27], fReadIndex[27];
		getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		for ( int direction = 0; direction < 27; direction++ ) {
			fNeqBase[direction] = fViewCoarse(fReadIndex[direction], cellReadIndex[direction]);
		}
		getRhoUxUyUz( rhoBase, uxBase, uyBase, uzBase, fNeqBase );
		float feqBase[27];
		getFeq(rhoBase, uxBase, uyBase, uzBase, feqBase);
		for ( int direction = 0; direction < 27; direction++ ) fNeqBase[direction] = fNeqBase[direction] - feqBase[direction];

		// Initialize accumulation variables
		// We want linear interpolation for fNeq
		float dfNeqdx[27] = {0.f}, dfNeqdy[27] = {0.f}, dfNeqdz[27] = {0.f};
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
			
			// reused variables for this neighbor only
			float fNeqNbr[27];
			float rhoNbr, uxNbr, uyNbr, uzNbr;
			LocalDuStruct localDuNbr;

			for ( int direction = 0; direction < 27; direction++ ) fNeqNbr[direction] = fViewCoarse( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			getRhoUxUyUz( rhoNbr, uxNbr, uyNbr, uzNbr, fNeqNbr );
			getLocalDu( fNeqNbr, InfoCoarse.nu, localDuNbr );
			float feqNbr[27];
			getFeq(rhoNbr, uxNbr, uyNbr, uzNbr, feqNbr);
			for ( int direction = 0; direction < 27; direction++ ) fNeqNbr[direction] = fNeqNbr[direction] - feqNbr[direction];
			
			// Add this neighbor's specific contribution based on its position
			switch(i) {
				case 0: // X+
					for(int d=0; d<27; d++) dfNeqdx[d] += 0.5f * fNeqNbr[d];
					dRhodx += 0.5f * rhoNbr; 
					ax += 0.5f * uxNbr; bx += 0.5f * uyNbr; cx += 0.5f * uzNbr;
					axx += 0.5f * uxNbr; bxx += 0.5f * uyNbr; cxx += 0.5f * uzNbr;
					bxy += 0.5f * localDuNbr.duydy; cxz += 0.5f * localDuNbr.duzdz;
					K3 += 0.5f * localDuNbr.duydzCross;
					break;
				case 1: // X-
					for(int d=0; d<27; d++) dfNeqdx[d] -= 0.5f * fNeqNbr[d];
					dRhodx -= 0.5f * rhoNbr; 
					ax -= 0.5f * uxNbr; bx -= 0.5f * uyNbr; cx -= 0.5f * uzNbr;
					axx += 0.5f * uxNbr; bxx += 0.5f * uyNbr; cxx += 0.5f * uzNbr;
					bxy -= 0.5f * localDuNbr.duydy; cxz -= 0.5f * localDuNbr.duzdz;
					K3 -= 0.5f * localDuNbr.duydzCross;
					break;
				case 2: // Y+
					for(int d=0; d<27; d++) dfNeqdy[d] += 0.5f * fNeqNbr[d];
					dRhody += 0.5f * rhoNbr;
					ay += 0.5f * uxNbr; by += 0.5f * uyNbr; cy += 0.5f * uzNbr;
					ayy += 0.5f * uxNbr; byy += 0.5f * uyNbr; cyy += 0.5f * uzNbr;
					axy += 0.5f * localDuNbr.duxdx; cyz += 0.5f * localDuNbr.duzdz;
					K2 += 0.5f * localDuNbr.duxdzCross;
					break;
				case 3: // Y-
					for(int d=0; d<27; d++) dfNeqdy[d] -= 0.5f * fNeqNbr[d];
					dRhody -= 0.5f * rhoNbr;
					ay -= 0.5f * uxNbr; by -= 0.5f * uyNbr; cy -= 0.5f * uzNbr;
					ayy += 0.5f * uxNbr; byy += 0.5f * uyNbr; cyy += 0.5f * uzNbr;
					axy -= 0.5f * localDuNbr.duxdx; cyz -= 0.5f * localDuNbr.duzdz;
					K2 -= 0.5f * localDuNbr.duxdzCross;
					break;
				case 4: // Z+
					for(int d=0; d<27; d++) dfNeqdz[d] += 0.5f * fNeqNbr[d];
					dRhodz += 0.5f * rhoNbr;
					az += 0.5f * uxNbr; bz += 0.5f * uyNbr; cz += 0.5f * uzNbr;
					azz += 0.5f * uxNbr; bzz += 0.5f * uyNbr; czz += 0.5f * uzNbr;
					axz += 0.5f * localDuNbr.duxdx; byz += 0.5f * localDuNbr.duydy;
					K1 += 0.5f * localDuNbr.duxdyCross;
					break;
				case 5: // Z-
					for(int d=0; d<27; d++) dfNeqdz[d] -= 0.5f * fNeqNbr[d];
					dRhodz -= 0.5f * rhoNbr;
					az -= 0.5f * uxNbr; bz -= 0.5f * uyNbr; cz -= 0.5f * uzNbr;
					azz += 0.5f * uxNbr; bzz += 0.5f * uyNbr; czz += 0.5f * uzNbr;
					axz -= 0.5f * localDuNbr.duxdx; byz -= 0.5f * localDuNbr.duydy;
					K1 -= 0.5f * localDuNbr.duxdyCross;
					break;
			}
		}

		// Final calculations for the cross terms
		float ayz = 0.5f * ( K1 + K2 - K3 );
		float bxz = 0.5f * ( K1 - K2 + K3 );
		float cxy = 0.5f * ( -K1 + K2 + K3 );
		
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
		
		const float scale = ( 3.f * InfoFine.nu + 0.5f ) / ( 6.f * InfoCoarse.nu + 1.f);
		for ( int direction = 0; direction < 27; direction++ )
		{
			fNeqBase[direction] *= scale;
			dfNeqdx[direction] *= scale;
			dfNeqdy[direction] *= scale;
			dfNeqdz[direction] *= scale;
		}
		
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
			float fNeq[27];
			for ( int direction = 0; direction < 27; direction++ ) fNeq[direction] = fNeqBase[direction] + dfNeqdx[direction] * dx + dfNeqdy[direction] * dy + dfNeqdz[direction] * dz;
			float feq[27];
			getFeq( rho, ux, uy, uz, feq );
			
			for ( int direction = 0; direction < 27; direction++ ) 
			{
				fViewFine( fWriteIndex[direction], cellWriteIndex[direction] ) = feq[direction] + fNeq[direction];
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, InfoCoarse.coarseToFineCount, cellLambda );
}

void updateInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	updateFineToCoarseInterface( GridCoarse, GridFine );
	updateCoarseToFineInterface( GridCoarse, GridFine );
}


/*
__host__ __device__ void rescaleFOLD( float (&f)[27], const bool &coarseToFine )
{
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
	//
	//	We are using cummulant collision, which involves transformation to central moments. Therefore to
	//	perform the transformation of the distribution functions when travelling between grids, we can perform
	//	the central moment transformation, scale the relevant moments, set the high order cummulants to zero,
	//	skip the relaxation part and perform the backwards transformation to receive the rescaled distribution
	//	functions on the target grid.

	//-------------------------- CUMMULANT COLLISION EQUATIONS ---------------------------
	//------------------------------------------------------------------------------------
	//--------------------------- TRANSFORM TO CENTRAL MOMENTS ---------------------------
	//------------------------------------------------------------------------------------

	//Eq Geiger 2015(43)
	//first part of the central moments transformation
	const float k_aa0 = (f[21] + f[25]) + f[11];
	const float k_ab0 = (f[8] + f[10]) + f[2];
	const float k_ac0 = (f[24] + f[19]) + f[15];
	const float k_ba0 = (f[14] + f[18]) + f[5];
	const float k_bb0 = (f[4] + f[3]) + f[0];
	const float k_bc0 = (f[17] + f[13]) + f[6];
	const float k_ca0 = (f[20] + f[23]) + f[16];
	const float k_cb0 = (f[9] + f[7]) + f[1];
	const float k_cc0 = (f[26] + f[22]) + f[12];

	const float k_aa1 = (f[21] - f[25]) - uz * k_aa0;
	const float k_ab1 = (f[8] - f[10]) - uz * k_ab0;
	const float k_ac1 = (f[24] - f[19]) - uz * k_ac0;
	const float k_ba1 = (f[14] - f[18]) - uz * k_ba0;
	const float k_bb1 = (f[4] - f[3]) - uz * k_bb0;
	const float k_bc1 = (f[17] - f[13]) - uz * k_bc0;
	const float k_ca1 = (f[20] - f[23]) - uz * k_ca0;
	const float k_cb1 = (f[9] - f[7]) - uz * k_cb0;
	const float k_cc1 = (f[26] - f[22]) - uz * k_cc0;

	const float k_aa2 = (f[21] + f[25]) - 2.f * uz * (f[21] - f[25]) + uz * uz * k_aa0;
	const float k_ab2 = (f[8] + f[10]) - 2.f * uz * (f[8] - f[10]) + uz * uz * k_ab0;
	const float k_ac2 = (f[24] + f[19]) - 2.f * uz * (f[24] - f[19]) + uz * uz * k_ac0;
	const float k_ba2 = (f[14] + f[18]) - 2.f * uz * (f[14] - f[18]) + uz * uz * k_ba0;
	const float k_bb2 = (f[4] + f[3]) - 2.f * uz * (f[4] - f[3]) + uz * uz * k_bb0;
	const float k_bc2 = (f[17] + f[13]) - 2.f * uz * (f[17] - f[13]) + uz * uz * k_bc0;
	const float k_ca2 = (f[20] + f[23]) - 2.f * uz * (f[20] - f[23]) + uz * uz * k_ca0;
	const float k_cb2 = (f[9] + f[7]) - 2.f * uz * (f[9] - f[7]) + uz * uz * k_cb0;
	const float k_cc2 = (f[26] + f[22]) - 2.f * uz * (f[26] - f[22]) + uz * uz * k_cc0;

	//Eq Geiger 2015(44)
	//second part of the central moments transformation
	const float k_a00 = (k_ac0 + k_aa0) + k_ab0;
	const float k_b00 = (k_bc0 + k_ba0) + k_bb0;
	const float k_c00 = (k_cc0 + k_ca0) + k_cb0;
	const float k_a01 = (k_ac1 + k_aa1) + k_ab1;
	const float k_b01 = (k_bc1 + k_ba1) + k_bb1;
	const float k_c01 = (k_cc1 + k_ca1) + k_cb1;
	const float k_a02 = (k_ac2 + k_aa2) + k_ab2;
	const float k_b02 = (k_bc2 + k_ba2) + k_bb2;
	const float k_c02 = (k_cc2 + k_ca2) + k_cb2;

	const float k_a10 = (k_ac0 - k_aa0) - uy * k_a00;
	const float k_b10 = (k_bc0 - k_ba0) - uy * k_b00;
	const float k_c10 = (k_cc0 - k_ca0) - uy * k_c00;

	const float k_a11 = (k_ac1 - k_aa1) - uy * k_a01;
	const float k_b11 = (k_bc1 - k_ba1) - uy * k_b01;
	const float k_c11 = (k_cc1 - k_ca1) - uy * k_c01;

	const float k_a20 = (k_ac0 + k_aa0) - 2.f * uy * (k_ac0 - k_aa0) + uy * uy * k_a00;
	const float k_b20 = (k_bc0 + k_ba0) - 2.f * uy * (k_bc0 - k_ba0) + uy * uy * k_b00;
	const float k_c20 = (k_cc0 + k_ca0) - 2.f * uy * (k_cc0 - k_ca0) + uy * uy * k_c00;

	//Eq Geiger 2015(45)
	// third part of the central moments transformation
	const float k_000 = (k_c00 + k_a00) + k_b00;
	const float k_001 = (k_c01 + k_a01) + k_b01;
	const float k_002prev = (k_c02 + k_a02) + k_b02;
	const float k_010 = (k_c10 + k_a10) + k_b10;
	const float k_011prev = (k_c11 + k_a11) + k_b11;
	const float k_020prev = (k_c20 + k_a20) + k_b20;

	const float k_100 = (k_c00 - k_a00) - ux * k_000;
	const float k_101prev = (k_c01 - k_a01) - ux * k_001;
	const float k_110prev = (k_c10 - k_a10) - ux * k_010;

	const float k_200prev = (k_c00 + k_a00) - 2.f * ux * (k_c00 - k_a00) + ux * ux * k_000;
	
	//------------------------------------------------------------------------------------
	//------------------------------ RESCALE CENTRAL MOMENTS -----------------------------
	//------------------------------------------------------------------------------------
	
	float k_200, k_020, k_002, k_011, k_101, k_110;
	
	if ( coarseToFine )
	{
		k_200 = (2.f / 3.f) * k_200prev + (1.f / 6.f) * k_020prev + (1.f / 6.f) * k_002prev;
		k_020 = (1.f / 6.f) * k_200prev + (2.f / 3.f) * k_020prev + (1.f / 6.f) * k_002prev;
		k_002 = (1.f / 6.f) * k_200prev + (1.f / 6.f) * k_020prev + (2.f / 3.f) * k_002prev;
		k_011 = 0.5f * k_011prev;
		k_101 = 0.5f * k_101prev;
		k_110 = 0.5f * k_110prev;
	}
	
	else
	{
		k_200 = (5.f / 3.f) * k_200prev - (1.f / 3.f) * k_020prev - (1.f / 3.f) * k_002prev;
		k_020 = - (1.f / 3.f) * k_200prev + (5.f / 3.f) * k_020prev - (1.f / 3.f) * k_002prev;
		k_002 = - (1.f / 3.f) * k_200prev - (1.f / 3.f) * k_020prev + (5.f / 3.f) * k_002prev;
		k_011 = 2.f * k_011prev;
		k_101 = 2.f * k_101prev;
		k_110 = 2.f * k_110prev;
	}
	
	//------------------------------------------------------------------------------------
	//------------------------------ CENTRAL MOM. TO CUMULANTS ---------------------------
	//------------------------------------------------------------------------------------

	//Eq Geiger 2015(47)
	const float C_110 = k_110;
	const float C_101 = k_101;
	const float C_011 = k_011;

	//Eq Geiger 2015(48)
	const float C_200 = k_200;
	const float C_020 = k_020;
	const float C_002 = k_002;

	// higher order cummulants all get relaxed to zero so they dont have to be calculated
	// relaxation of the low order cummulants is skipped

	//------------------------------------------------------------------------------------
	//------------------------------ CUMULANTS TO CENTRAL MOM. ---------------------------
	//------------------------------------------------------------------------------------

	const float ks_000 = k_000;

	// Permutation again

	//Eq Geiger 2015(47) backwards
	const float ks_110 = C_110;
	const float ks_101 = C_101;
	const float ks_011 = C_011;

	//Eq Geiger 2015(48) backwards
	const float ks_200 = C_200;
	const float ks_020 = C_020;
	const float ks_002 = C_002;

	//Eq. Geiger 2015(85, 86, 87)
	const float ks_100 = -k_100;
	const float ks_010 = -k_010;
	const float ks_001 = -k_001;

	//Eq. Geiger 2015(81)
	const float ks_211 = (ks_200 * ks_011 + 2.f * ks_101 * ks_110) / rho;
	const float ks_121 = (ks_020 * ks_101 + 2.f * ks_110 * ks_011) / rho;
	const float ks_112 = (ks_002 * ks_110 + 2.f * ks_011 * ks_101) / rho;

	//Eq. Geiger 2015(82)
	const float ks_220 = (ks_020 * ks_200 + 2.f * ks_110 * ks_110) / rho;
	const float ks_022 = (ks_002 * ks_020 + 2.f * ks_011 * ks_011) / rho;
	const float ks_202 = (ks_200 * ks_002 + 2.f * ks_101 * ks_101) / rho;

	// Eq. Geiger 2015(84)
	const float ks_222 = (
		(ks_200 * ks_022 + ks_020 * ks_202 + ks_002 * ks_220 +
		4.f * (ks_011 * ks_211 + ks_101 * ks_121 + ks_110 * ks_112)) / rho
		- (16.0 * ks_110 * ks_101 * ks_011 + 4.f * (ks_101 * ks_101 * ks_020 +
				ks_011 * ks_011 * ks_200 +
				ks_110 * ks_110 * ks_002) +
		2.f * ks_200 * ks_020 * ks_002) / rho / rho
		);

	//------------------------------------------------------------------------------------
	//----------------------- TRANSFORM TO DISTRIBUTION FUNCTION -------------------------
	//------------------------------------------------------------------------------------

	//Eq Geiger 2015(88)
	const float ks_b00 = ks_000 * (1.f - ux * ux) - 2.f * ux * ks_100 - ks_200;
	const float ks_b01 = ks_001 * (1.f - ux * ux) - 2.f * ux * ks_101;
	const float ks_b02 = ks_002 * (1.f - ux * ux) - ks_202;
	const float ks_b10 = ks_010 * (1.f - ux * ux) - 2.f * ux * ks_110;
	const float ks_b11 = ks_011 * (1.f - ux * ux) - ks_211;
	const float ks_b12 = - 2.f * ux * ks_112;
	const float ks_b20 = ks_020 * (1.f - ux * ux) - ks_220;
	const float ks_b21 = - 2.f * ux * ks_121;
	const float ks_b22 = ks_022 * (1.f - ux * ux) - ks_222;

	//Eq  Geiger 2015(89)
	const float ks_a00 = (ks_000 * (ux * ux - ux) + ks_100 * (2.f * ux - 1.f) + ks_200) * 0.5f;
	const float ks_a01 = (ks_001 * (ux * ux - ux) + ks_101 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a02 = (ks_002 * (ux * ux - ux) + ks_202) * 0.5f;
	const float ks_a10 = (ks_010 * (ux * ux - ux) + ks_110 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a11 = (ks_011 * (ux * ux - ux) + ks_211) * 0.5f;
	const float ks_a12 = (ks_112 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a20 = (ks_020 * (ux * ux - ux) + ks_220) * 0.5f;
	const float ks_a21 = (ks_121 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a22 = (ks_022 * (ux * ux - ux) + ks_222) * 0.5f;

	//Eq  Geiger 2015(90)
	const float ks_c00 = (ks_000 * (ux * ux + ux) + ks_100 * (2.f * ux + 1.f) + ks_200) * 0.5f;
	const float ks_c01 = (ks_001 * (ux * ux + ux) + ks_101 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c02 = (ks_002 * (ux * ux + ux) + ks_202) * 0.5f;
	const float ks_c10 = (ks_010 * (ux * ux + ux) + ks_110 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c11 = (ks_011 * (ux * ux + ux) + ks_211) * 0.5f;
	const float ks_c12 = (ks_112 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c20 = (ks_020 * (ux * ux + ux) + ks_220) * 0.5f;
	const float ks_c21 = (ks_121 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c22 = (ks_022 * (ux * ux + ux) + ks_222) * 0.5f;

	//Eq Geiger 2015(91)
	const float ks_ab0 = ks_a00 * (1.f - uy * uy) - 2.f * uy * ks_a10 - ks_a20;
	const float ks_ab1 = ks_a01 * (1.f - uy * uy) - 2.f * uy * ks_a11 - ks_a21;
	const float ks_ab2 = ks_a02 * (1.f - uy * uy) - 2.f * uy * ks_a12 - ks_a22;
	const float ks_bb0 = ks_b00 * (1.f - uy * uy) - 2.f * uy * ks_b10 - ks_b20;
	const float ks_bb1 = ks_b01 * (1.f - uy * uy) - 2.f * uy * ks_b11 - ks_b21;
	const float ks_bb2 = ks_b02 * (1.f - uy * uy) - 2.f * uy * ks_b12 - ks_b22;
	const float ks_cb0 = ks_c00 * (1.f - uy * uy) - 2.f * uy * ks_c10 - ks_c20;
	const float ks_cb1 = ks_c01 * (1.f - uy * uy) - 2.f * uy * ks_c11 - ks_c21;
	const float ks_cb2 = ks_c02 * (1.f - uy * uy) - 2.f * uy * ks_c12 - ks_c22;

	//Eq  Geiger 2015(92)
	const float ks_aa0 = (ks_a00 * (uy * uy - uy) + ks_a10 * (2.f * uy - 1.f) + ks_a20) * 0.5f;
	const float ks_aa1 = (ks_a01 * (uy * uy - uy) + ks_a11 * (2.f * uy - 1.f) + ks_a21) * 0.5f;
	const float ks_aa2 = (ks_a02 * (uy * uy - uy) + ks_a12 * (2.f * uy - 1.f) + ks_a22) * 0.5f;
	const float ks_ba0 = (ks_b00 * (uy * uy - uy) + ks_b10 * (2.f * uy - 1.f) + ks_b20) * 0.5f;
	const float ks_ba1 = (ks_b01 * (uy * uy - uy) + ks_b11 * (2.f * uy - 1.f) + ks_b21) * 0.5f;
	const float ks_ba2 = (ks_b02 * (uy * uy - uy) + ks_b12 * (2.f * uy - 1.f) + ks_b22) * 0.5f;
	const float ks_ca0 = (ks_c00 * (uy * uy - uy) + ks_c10 * (2.f * uy - 1.f) + ks_c20) * 0.5f;
	const float ks_ca1 = (ks_c01 * (uy * uy - uy) + ks_c11 * (2.f * uy - 1.f) + ks_c21) * 0.5f;
	const float ks_ca2 = (ks_c02 * (uy * uy - uy) + ks_c12 * (2.f * uy - 1.f) + ks_c22) * 0.5f;

	//Eq Geiger 2015(93)
	const float ks_ac0 = (ks_a00 * (uy * uy + uy) + ks_a10 * (2.f * uy + 1.f) + ks_a20) * 0.5f;
	const float ks_ac1 = (ks_a01 * (uy * uy + uy) + ks_a11 * (2.f * uy + 1.f) + ks_a21) * 0.5f;
	const float ks_ac2 = (ks_a02 * (uy * uy + uy) + ks_a12 * (2.f * uy + 1.f) + ks_a22) * 0.5f;
	const float ks_bc0 = (ks_b00 * (uy * uy + uy) + ks_b10 * (2.f * uy + 1.f) + ks_b20) * 0.5f;
	const float ks_bc1 = (ks_b01 * (uy * uy + uy) + ks_b11 * (2.f * uy + 1.f) + ks_b21) * 0.5f;
	const float ks_bc2 = (ks_b02 * (uy * uy + uy) + ks_b12 * (2.f * uy + 1.f) + ks_b22) * 0.5f;
	const float ks_cc0 = (ks_c00 * (uy * uy + uy) + ks_c10 * (2.f * uy + 1.f) + ks_c20) * 0.5f;
	const float ks_cc1 = (ks_c01 * (uy * uy + uy) + ks_c11 * (2.f * uy + 1.f) + ks_c21) * 0.5f;
	const float ks_cc2 = (ks_c02 * (uy * uy + uy) + ks_c12 * (2.f * uy + 1.f) + ks_c22) * 0.5f;

	//Eq Geiger 2015(94)
	f[11] = ks_aa0 * (1.f - uz * uz) - 2.f * uz * ks_aa1 - ks_aa2;
	f[2] = ks_ab0 * (1.f - uz * uz) - 2.f * uz * ks_ab1 - ks_ab2;
	f[15] = ks_ac0 * (1.f - uz * uz) - 2.f * uz * ks_ac1 - ks_ac2;
	f[5] = ks_ba0 * (1.f - uz * uz) - 2.f * uz * ks_ba1 - ks_ba2;
	f[0] = ks_bb0 * (1.f - uz * uz) - 2.f * uz * ks_bb1 - ks_bb2;
	f[6] = ks_bc0 * (1.f - uz * uz) - 2.f * uz * ks_bc1 - ks_bc2;
	f[16] = ks_ca0 * (1.f - uz * uz) - 2.f * uz * ks_ca1 - ks_ca2;
	f[1] = ks_cb0 * (1.f - uz * uz) - 2.f * uz * ks_cb1 - ks_cb2;
	f[12] = ks_cc0 * (1.f - uz * uz) - 2.f * uz * ks_cc1 - ks_cc2;

	//Eq  Geiger 2015(95)
	f[25] = (ks_aa0 * (uz * uz - uz) + ks_aa1 * (2.f * uz - 1.f) + ks_aa2) * 0.5f;
	f[10] = (ks_ab0 * (uz * uz - uz) + ks_ab1 * (2.f * uz - 1.f) + ks_ab2) * 0.5f;
	f[19] = (ks_ac0 * (uz * uz - uz) + ks_ac1 * (2.f * uz - 1.f) + ks_ac2) * 0.5f;
	f[18] = (ks_ba0 * (uz * uz - uz) + ks_ba1 * (2.f * uz - 1.f) + ks_ba2) * 0.5f;
	f[3] = (ks_bb0 * (uz * uz - uz) + ks_bb1 * (2.f * uz - 1.f) + ks_bb2) * 0.5f;
	f[13] = (ks_bc0 * (uz * uz - uz) + ks_bc1 * (2.f * uz - 1.f) + ks_bc2) * 0.5f;
	f[23] = (ks_ca0 * (uz * uz - uz) + ks_ca1 * (2.f * uz - 1.f) + ks_ca2) * 0.5f;
	f[7] = (ks_cb0 * (uz * uz - uz) + ks_cb1 * (2.f * uz - 1.f) + ks_cb2) * 0.5f;
	f[22] = (ks_cc0 * (uz * uz - uz) + ks_cc1 * (2.f * uz - 1.f) + ks_cc2) * 0.5f;

	//Eq  Geiger 2015(96)
	f[21] = (ks_aa0 * (uz * uz + uz) + ks_aa1 * (2.f * uz + 1.f) + ks_aa2) * 0.5f;
	f[8] = (ks_ab0 * (uz * uz + uz) + ks_ab1 * (2.f * uz + 1.f) + ks_ab2) * 0.5f;
	f[24] = (ks_ac0 * (uz * uz + uz) + ks_ac1 * (2.f * uz + 1.f) + ks_ac2) * 0.5f;
	f[14] = (ks_ba0 * (uz * uz + uz) + ks_ba1 * (2.f * uz + 1.f) + ks_ba2) * 0.5f;
	f[4] = (ks_bb0 * (uz * uz + uz) + ks_bb1 * (2.f * uz + 1.f) + ks_bb2) * 0.5f;
	f[17] = (ks_bc0 * (uz * uz + uz) + ks_bc1 * (2.f * uz + 1.f) + ks_bc2) * 0.5f;
	f[20] = (ks_ca0 * (uz * uz + uz) + ks_ca1 * (2.f * uz + 1.f) + ks_ca2) * 0.5f;
	f[9] = (ks_cb0 * (uz * uz + uz) + ks_cb1 * (2.f * uz + 1.f) + ks_cb2) * 0.5f;
	f[26] = (ks_cc0 * (uz * uz + uz) + ks_cc1 * (2.f * uz + 1.f) + ks_cc2) * 0.5f;
}
*/

/*
// BACKUP
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
		
		int cellFineList[8];
		cellFineList[0] = cellFine0;
		cellFineList[1] = cellFine0 + 1; if ( cellFineList[1] >= InfoFine.cellCount ) cellFineList[1] = 0;
		cellFineList[2] = jPlusViewFine( cellFine0 );
		cellFineList[3] = cellFineList[2] + 1; if ( cellFineList[3] >= InfoFine.cellCount ) cellFineList[3] = 0;
		cellFineList[4] = kPlusViewFine( cellFine0 );
		cellFineList[5] = cellFineList[4] + 1; if ( cellFineList[5] >= InfoFine.cellCount ) cellFineList[5] = 0;
		cellFineList[6] = jkPlusViewFine( cellFine0 );
		cellFineList[7] = cellFineList[6] + 1; if ( cellFineList[7] >= InfoFine.cellCount ) cellFineList[7] = 0;
		
		float f[27] = {0.f};
		
		for ( int which = 0; which < 8; which++ )
		{
			const int cellFine = cellFineList[which];
			NBRStruct NBR;
			NBR.self = cellFine;
			NBR.jPlus = jPlusViewFine( cellFine );
			NBR.kPlus = kPlusViewFine( cellFine );
			NBR.jkPlus = jkPlusViewFine( cellFine );
			finishNBRPlus( NBR, InfoFine );
			
			float fFine[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipperFine, InfoFine );
			for ( int direction = 0; direction < 27; direction++ )	fFine[direction] = fViewFine(fReadIndex[direction], cellReadIndex[direction]);
			for (int direction = 0; direction < 27; direction++) f[direction] += fFine[direction];	
		}
		for (int direction = 0; direction < 27; direction++) f[direction] = f[direction] * 0.125f;
		
		const float scale = 2.0f;
		rescaleF( f, scale );
		
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jkPlusViewCoarse( cellCoarse );
		finishNBRPlus( NBR, InfoCoarse );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		for ( int direction = 0; direction < 27; direction++ ) fViewCoarse( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, InfoCoarse.fineToCoarseCount, cellLambda );
}
*/
