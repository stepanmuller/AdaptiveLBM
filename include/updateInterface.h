#pragma once

#include "./applyCollision.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"

// Helper table for second order moments
//  	id: { 0, 1, 2, 3, 4, 5, 6,		 7, 8, 9,10,11,12,13,14,15,16,17,18,		19,20,21,22,23,24,25,26 };
	
//  	cx: { 0, 1,-1, 0, 0, 0, 0,		 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,		-1, 1,-1, 1, 1,-1,-1, 1 };
//  	cy: { 0, 0, 0, 0, 0,-1, 1,		 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1,		 1,-1,-1, 1,-1, 1,-1, 1 };
//  	cz: { 0, 0, 0,-1, 1, 0, 0,		-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,		-1, 1, 1,-1,-1, 1,-1, 1 };

// cx * cx: { 0, 1, 1, 0, 0, 0, 0,		 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,		 1, 1, 1, 1, 1, 1, 1, 1 };
// cy * cy: { 0, 0, 0, 0, 0, 1, 1,		 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,		 1, 1, 1, 1, 1, 1, 1, 1 };
// cz * cz: { 0, 0, 0, 1, 1, 0, 0,		 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,		 1, 1, 1, 1, 1, 1, 1, 1 };

// cy * cz: { 0, 0, 0, 0, 0, 0, 0,		 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 1,		-1,-1,-1,-1, 1, 1, 1, 1 };
// cx * cz: { 0, 0, 0, 0, 0, 0, 0,		-1,-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,		 1, 1,-1,-1,-1,-1, 1, 1 };
// cx * cy: { 0, 0, 0, 0, 0, 0, 0,		 0, 0, 0, 0, 1, 1, 0, 0,-1,-1, 0, 0,		-1,-1, 1, 1,-1,-1, 1, 1 };

// cx2-cy2: { 0, 1, 1, 0, 0,-1,-1,		 1, 1, 1, 1, 0, 0,-1,-1, 0, 0,-1,-1,		 0, 0, 0, 0, 0, 0, 0, 0 };
// cx2-cz2: { 0, 1, 1,-1,-1, 0, 0,		 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,		 0, 0, 0, 0, 0, 0, 0, 0 };

__host__ __device__ void reconstructInterpolatedF( 	float (&f)[27], const float &rho, const float &ux, const float &uy, const float &uz, 
													const float &k_011, const float &k_101, const float &k_110, 
													const float &k_200, const float &k_020, const float &k_002 )
{
	const float dRho = rho - 1.f;
	const float ux2 = ux * ux;
	const float uy2 = uy * uy;
	const float uz2 = uz * uz;
	
	// D3Q27 weight moments needed by the well-conditioned transformation.
    const float K_aa0 = 1.f / 36.f;
    const float K_ab0 = 1.f / 9.f;
    const float K_ac0 = 1.f / 36.f;
    const float K_ba0 = 1.f / 9.f;
    const float K_bb0 = 4.f / 9.f;
    const float K_bc0 = 1.f / 9.f;
    const float K_ca0 = 1.f / 36.f;
    const float K_cb0 = 1.f / 9.f;
    const float K_cc0 = 1.f / 36.f;
    const float K_a00 = 1.f / 6.f;
    const float K_b00 = 2.f / 3.f;
    const float K_c00 = 1.f / 6.f;
    const float K_a02 = 1.f / 18.f;
    const float K_b02 = 2.f / 9.f;
    const float K_c02 = 1.f / 18.f;

    // -------------------------------------------------------------------------
    // Cumulants -> well-conditioned central moments, Geier 2017, Eqs. 53-56.
    // -------------------------------------------------------------------------

	const float k_000 = dRho;

    const float k_220 = dRho / 9.f;
    const float k_022 = k_220;
    const float k_202 = k_220;

    const float k_222 = dRho / 27.f;

    // -------------------------------------------------------------------------
    // Well-conditioned central moments -> shifted populations, Eqs. 57-65.
    // -------------------------------------------------------------------------

    const float k_b00 = k_000 * (1.f - ux2) - k_200 - ux2;
    const float k_b01 = - 2.f * ux * k_101;
    const float k_b02 = k_002 * (1.f - ux2) - k_202 - ux2 / 3.f;
    const float k_b10 = - 2.f * ux * k_110;
    const float k_b11 = k_011 * (1.f - ux2);
    const float k_b20 = k_020 * (1.f - ux2) - k_220 - ux2 / 3.f;
    const float k_b22 = k_022 * (1.f - ux2) - k_222 - ux2 / 9.f;

    const float k_a00 = ((k_000 + 1.f) * (ux2 - ux) + k_200) * 0.5f;
    const float k_a01 = (k_101 * (2.f * ux - 1.f)) * 0.5f;
    const float k_a02 = ((k_002 + 1.f / 3.f) * (ux2 - ux) + k_202) * 0.5f;
    const float k_a10 = (k_110 * (2.f * ux - 1.f)) * 0.5f;
    const float k_a11 = (k_011 * (ux2 - ux)) * 0.5f;
    const float k_a20 = ((k_020 + 1.f / 3.f) * (ux2 - ux) + k_220) * 0.5f;
    const float k_a22 = ((k_022 + 1.f / 9.f) * (ux2 - ux) + k_222) * 0.5f;

    const float k_c00 = ((k_000 + 1.f) * (ux2 + ux) + k_200) * 0.5f;
    const float k_c01 = (k_101 * (2.f * ux + 1.f)) * 0.5f;
    const float k_c02 = ((k_002 + 1.f / 3.f) * (ux2 + ux) + k_202) * 0.5f;
    const float k_c10 = (k_110 * (2.f * ux + 1.f)) * 0.5f;
    const float k_c11 = (k_011 * (ux2 + ux)) * 0.5f;
    const float k_c20 = ((k_020 + 1.f / 3.f) * (ux2 + ux) + k_220) * 0.5f;
    const float k_c22 = ((k_022 + 1.f / 9.f) * (ux2 + ux) + k_222) * 0.5f;

    const float k_ab0 = k_a00 * (1.f - uy2) - 2.f * uy * k_a10 - k_a20 - K_a00 * uy2;
    const float k_ab1 = k_a01 * (1.f - uy2) - 2.f * uy * k_a11 - 0.f;
    const float k_ab2 = k_a02 * (1.f - uy2) - 2.f * uy * 0.f - k_a22 - K_a02 * uy2;
    const float k_bb0 = k_b00 * (1.f - uy2) - 2.f * uy * k_b10 - k_b20 - K_b00 * uy2;
    const float k_bb1 = k_b01 * (1.f - uy2) - 2.f * uy * k_b11 - 0.f;
    const float k_bb2 = k_b02 * (1.f - uy2) - 2.f * uy * 0.f - k_b22 - K_b02 * uy2;
    const float k_cb0 = k_c00 * (1.f - uy2) - 2.f * uy * k_c10 - k_c20 - K_c00 * uy2;
    const float k_cb1 = k_c01 * (1.f - uy2) - 2.f * uy * k_c11 - 0.f;
    const float k_cb2 = k_c02 * (1.f - uy2) - 2.f * uy * 0.f - k_c22 - K_c02 * uy2;

    const float k_aa0 = ((k_a00 + K_a00) * (uy2 - uy) + k_a10 * (2.f * uy - 1.f) + k_a20) * 0.5f;
    const float k_aa1 = (k_a01 * (uy2 - uy) + k_a11 * (2.f * uy - 1.f) + 0.f) * 0.5f;
    const float k_aa2 = ((k_a02 + K_a02) * (uy2 - uy) + k_a22) * 0.5f;
    const float k_ba0 = ((k_b00 + K_b00) * (uy2 - uy) + k_b10 * (2.f * uy - 1.f) + k_b20) * 0.5f;
    const float k_ba1 = (k_b01 * (uy2 - uy) + k_b11 * (2.f * uy - 1.f)) * 0.5f;
    const float k_ba2 = ((k_b02 + K_b02) * (uy2 - uy) + k_b22) * 0.5f;
    const float k_ca0 = ((k_c00 + K_c00) * (uy2 - uy) + k_c10 * (2.f * uy - 1.f) + k_c20) * 0.5f;
    const float k_ca1 = (k_c01 * (uy2 - uy) + k_c11 * (2.f * uy - 1.f)) * 0.5f;
    const float k_ca2 = ((k_c02 + K_c02) * (uy2 - uy) + k_c22) * 0.5f;

    const float k_ac0 = ((k_a00 + K_a00) * (uy2 + uy) + k_a10 * (2.f * uy + 1.f) + k_a20) * 0.5f;
    const float k_ac1 = (k_a01 * (uy2 + uy) + k_a11 * (2.f * uy + 1.f)) * 0.5f;
    const float k_ac2 = ((k_a02 + K_a02) * (uy2 + uy) + k_a22) * 0.5f;
    const float k_bc0 = ((k_b00 + K_b00) * (uy2 + uy) + k_b10 * (2.f * uy + 1.f) + k_b20) * 0.5f;
    const float k_bc1 = (k_b01 * (uy2 + uy) + k_b11 * (2.f * uy + 1.f)) * 0.5f;
    const float k_bc2 = ((k_b02 + K_b02) * (uy2 + uy) + k_b22) * 0.5f;
    const float k_cc0 = ((k_c00 + K_c00) * (uy2 + uy) + k_c10 * (2.f * uy + 1.f) + k_c20) * 0.5f;
    const float k_cc1 = (k_c01 * (uy2 + uy) + k_c11 * (2.f * uy + 1.f)) * 0.5f;
    const float k_cc2 = ((k_c02 + K_c02) * (uy2 + uy) + k_c22) * 0.5f;

    f[MMO] = k_aa0 * (1.f - uz2) - 2.f * uz * k_aa1 - k_aa2 - K_aa0 * uz2;
    f[MOO]  = k_ab0 * (1.f - uz2) - 2.f * uz * k_ab1 - k_ab2 - K_ab0 * uz2;
    f[MPO] = k_ac0 * (1.f - uz2) - 2.f * uz * k_ac1 - k_ac2 - K_ac0 * uz2;
    f[OMO]  = k_ba0 * (1.f - uz2) - 2.f * uz * k_ba1 - k_ba2 - K_ba0 * uz2;
    f[OOO]  = k_bb0 * (1.f - uz2) - 2.f * uz * k_bb1 - k_bb2 - K_bb0 * uz2;
    f[OPO]  = k_bc0 * (1.f - uz2) - 2.f * uz * k_bc1 - k_bc2 - K_bc0 * uz2;
    f[PMO] = k_ca0 * (1.f - uz2) - 2.f * uz * k_ca1 - k_ca2 - K_ca0 * uz2;
    f[POO]  = k_cb0 * (1.f - uz2) - 2.f * uz * k_cb1 - k_cb2 - K_cb0 * uz2;
    f[PPO] = k_cc0 * (1.f - uz2) - 2.f * uz * k_cc1 - k_cc2 - K_cc0 * uz2;

    f[MMM] = ((k_aa0 + K_aa0) * (uz2 - uz) + k_aa1 * (2.f * uz - 1.f) + k_aa2) * 0.5f;
    f[MOM] = ((k_ab0 + K_ab0) * (uz2 - uz) + k_ab1 * (2.f * uz - 1.f) + k_ab2) * 0.5f;
    f[MPM] = ((k_ac0 + K_ac0) * (uz2 - uz) + k_ac1 * (2.f * uz - 1.f) + k_ac2) * 0.5f;
    f[OMM] = ((k_ba0 + K_ba0) * (uz2 - uz) + k_ba1 * (2.f * uz - 1.f) + k_ba2) * 0.5f;
    f[OOM]  = ((k_bb0 + K_bb0) * (uz2 - uz) + k_bb1 * (2.f * uz - 1.f) + k_bb2) * 0.5f;
    f[OPM] = ((k_bc0 + K_bc0) * (uz2 - uz) + k_bc1 * (2.f * uz - 1.f) + k_bc2) * 0.5f;
    f[PMM] = ((k_ca0 + K_ca0) * (uz2 - uz) + k_ca1 * (2.f * uz - 1.f) + k_ca2) * 0.5f;
    f[POM]  = ((k_cb0 + K_cb0) * (uz2 - uz) + k_cb1 * (2.f * uz - 1.f) + k_cb2) * 0.5f;
    f[PPM] = ((k_cc0 + K_cc0) * (uz2 - uz) + k_cc1 * (2.f * uz - 1.f) + k_cc2) * 0.5f;

    f[MMP] = ((k_aa0 + K_aa0) * (uz2 + uz) + k_aa1 * (2.f * uz + 1.f) + k_aa2) * 0.5f;
    f[MOP]  = ((k_ab0 + K_ab0) * (uz2 + uz) + k_ab1 * (2.f * uz + 1.f) + k_ab2) * 0.5f;
    f[MPP] = ((k_ac0 + K_ac0) * (uz2 + uz) + k_ac1 * (2.f * uz + 1.f) + k_ac2) * 0.5f;
    f[OMP] = ((k_ba0 + K_ba0) * (uz2 + uz) + k_ba1 * (2.f * uz + 1.f) + k_ba2) * 0.5f;
    f[OOP]  = ((k_bb0 + K_bb0) * (uz2 + uz) + k_bb1 * (2.f * uz + 1.f) + k_bb2) * 0.5f;
    f[OPP] = ((k_bc0 + K_bc0) * (uz2 + uz) + k_bc1 * (2.f * uz + 1.f) + k_bc2) * 0.5f;
    f[PMP] = ((k_ca0 + K_ca0) * (uz2 + uz) + k_ca1 * (2.f * uz + 1.f) + k_ca2) * 0.5f;
    f[POP]  = ((k_cb0 + K_cb0) * (uz2 + uz) + k_cb1 * (2.f * uz + 1.f) + k_cb2) * 0.5f;
    f[PPP] = ((k_cc0 + K_cc0) * (uz2 + uz) + k_cc1 * (2.f * uz + 1.f) + k_cc2) * 0.5f;
    /*
    const float weights[27] = { 8.f/27.f, 
		2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
		1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
		1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
    for ( int direction = 0; direction < 27; direction++ ) f[direction] += weights[direction];
    */
}

void updateFineToCoarseInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	// The interpolation and rescaling is based on Martin Schönherr's disertation 2015
	const InfoStruct &InfoCoarse = GridCoarse.Info;
	auto fViewCoarse = GridCoarse.fArray.getView();
	const bool &esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto jPlusViewCoarse = GridCoarse.NBR.jPlusArray.getConstView();
	auto kPlusViewCoarse = GridCoarse.NBR.kPlusArray.getConstView();
	const float tauCoarse = 3.f * InfoCoarse.nu + 0.5f;
	const float omega1Coarse =  1.f / tauCoarse;
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto jPlusViewFine = GridFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.NBR.kPlusArray.getConstView();
	const float tauFine = 3.f * InfoFine.nu + 0.5f;
	const float omega1Fine =  1.f / tauFine;
	
	auto fineToCoarseIndexView = GridCoarse.fineToCoarseIndexArray.getConstView();
	auto childMapView = GridCoarse.childMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cellCoarse = fineToCoarseIndexView( index );
		const int cellFine0 = childMapView( cellCoarse );
		
		int cellStencil[8];
		cellStencil[0] = cellFine0;
		cellStencil[1] = cellFine0 + 1; if ( cellStencil[1] >= InfoFine.cellCount ) cellStencil[1] = 0;
		cellStencil[2] = jPlusViewFine( cellFine0 );
		cellStencil[3] = cellStencil[2] + 1; if ( cellStencil[3] >= InfoFine.cellCount ) cellStencil[3] = 0;
		cellStencil[4] = kPlusViewFine( cellFine0 );
		cellStencil[5] = cellStencil[4] + 1; if ( cellStencil[5] >= InfoFine.cellCount ) cellStencil[5] = 0;
		cellStencil[6] = jPlusViewFine( cellStencil[4] );
		cellStencil[7] = cellStencil[6] + 1; if ( cellStencil[7] >= InfoFine.cellCount ) cellStencil[7] = 0;
		
		// Initialize stencil variables
		float rhoStencil[8]; float uxStencil[8]; float uyStencil[8]; float uzStencil[8];
		float kxyStencil[8]; float kyzStencil[8]; float kxzStencil[8]; float kxxMyyStencil[8]; float kxxMzzStencil[8];
		
		// Extract values from each stencil cell
		for ( int i = 0; i < 8; i++ )
		{
			const int nbr = cellStencil[i];
			NBRStruct NBRofNBR;
			NBRofNBR.self = nbr;
			NBRofNBR.jPlus = jPlusViewFine( nbr );
			NBRofNBR.kPlus = kPlusViewFine( nbr );
			NBRofNBR.jkPlus = jPlusViewFine( NBRofNBR.kPlus );
			finishNBRPlus( NBRofNBR, InfoFine );
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPreCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperFine, InfoFine );
			float fNbr[27];
			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewFine( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			
			getRhoUxUyUz( rhoStencil[i], uxStencil[i], uyStencil[i], uzStencil[i], fNbr );
			/*
			const float weights[27] = { 8.f/27.f, 
				2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
				1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
				1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] -= weights[direction];
			*/
			kxyStencil[i] = - 3.f * omega1Fine * ( ( 
					+ fNbr[11] + fNbr[12] - fNbr[15] - fNbr[16] 
					- fNbr[19] - fNbr[20] + fNbr[21] + fNbr[22] - fNbr[23] - fNbr[24] + fNbr[25] + fNbr[26]
													) / rhoStencil[i] - uxStencil[i] * uyStencil[i] );
			kyzStencil[i] = - 3.f * omega1Fine * ( (
					- fNbr[13] - fNbr[14] + fNbr[17] + fNbr[18] 
					- fNbr[19] - fNbr[20] - fNbr[21] - fNbr[22] + fNbr[23] + fNbr[24] + fNbr[25] + fNbr[26]
													) / rhoStencil[i] - uyStencil[i] * uzStencil[i] );
			kxzStencil[i] = - 3.f * omega1Fine * ( (
					- fNbr[7 ] - fNbr[8 ] + fNbr[9 ] + fNbr[10] 
					+ fNbr[19] + fNbr[20] - fNbr[21] - fNbr[22] - fNbr[23] - fNbr[24] + fNbr[25] + fNbr[26]
													) / rhoStencil[i] - uxStencil[i] * uzStencil[i] );
			kxxMyyStencil[i] = - 1.5f * omega1Fine * ( (
					+ fNbr[1 ] + fNbr[2 ] - fNbr[5 ] - fNbr[6 ] 
					+ fNbr[7 ] + fNbr[8 ] + fNbr[9 ] + fNbr[10] - fNbr[13] - fNbr[14] - fNbr[17] - fNbr[18]
													) / rhoStencil[i] - ( uxStencil[i] * uxStencil[i] - uyStencil[i] * uyStencil[i] ) );
			kxxMzzStencil[i] = - 1.5f * omega1Fine * ( (
					+ fNbr[1 ] + fNbr[2 ] - fNbr[3 ] - fNbr[4 ] 
					+ fNbr[11] + fNbr[12] - fNbr[13] - fNbr[14] + fNbr[15] + fNbr[16] - fNbr[17] - fNbr[18]
													) / rhoStencil[i] - ( uxStencil[i] * uxStencil[i] - uzStencil[i] * uzStencil[i] ) );
		}
		
		// get all required coefficients
		// eq Schönherr 2015 (7.10)
		float d0 = 0.f; for ( int i = 0; i < 8; i++ ) d0 += rhoStencil[i]; d0 *= 0.125f;
		
		// The following is directly taken from VirtualFluids (just renamed variables). https://github.com/irmb/virtualfluids 
		const float a0 = 0.015625f * (2.f * (((kxyStencil[0] - kxyStencil[7]) + (kxyStencil[4] - kxyStencil[3])) +
                                ((kxyStencil[1] - kxyStencil[6]) + (kxyStencil[5] - kxyStencil[2])) +
                                ((kxzStencil[0] - kxzStencil[7]) + (kxzStencil[3] - kxzStencil[4])) +
                                ((kxzStencil[1] - kxzStencil[6]) + (kxzStencil[2] - kxzStencil[5])) +
                                ((uyStencil[7] + uyStencil[0]) + (uyStencil[3] + uyStencil[4])) - ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[5] + uzStencil[2]) - (uzStencil[6] + uzStencil[1]))) +
                        8.f * (((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) + ((uxStencil[6] + uxStencil[1]) + (uxStencil[5] + uxStencil[2]))) +
                        ((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                        ((kxxMyyStencil[6] - kxxMyyStencil[1]) + (kxxMyyStencil[2] - kxxMyyStencil[5])) +
                        ((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                        ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[2] - kxxMzzStencil[5])));
        const float b0 = 0.015625f * (2.f * (((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                                ((kxxMyyStencil[6] - kxxMyyStencil[1]) + (kxxMyyStencil[2] - kxxMyyStencil[5])) +
                                ((kxyStencil[0] - kxyStencil[7]) + (kxyStencil[4] - kxyStencil[3])) +
                                ((kxyStencil[6] - kxyStencil[1]) + (kxyStencil[2] - kxyStencil[5])) +
                                ((kyzStencil[0] - kyzStencil[7]) + (kyzStencil[3] - kyzStencil[4])) +
                                ((kyzStencil[1] - kyzStencil[6]) + (kyzStencil[2] - kyzStencil[5])) +
                                ((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) - ((uxStencil[2] + uxStencil[6]) + (uxStencil[1] + uxStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[2] + uzStencil[5]))) +
                        8.f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[3] + uyStencil[4])) + ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5]))) +
                        ((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                        ((kxxMzzStencil[1] - kxxMzzStencil[6]) + (kxxMzzStencil[5] - kxxMzzStencil[2])));
        const float c0 = 0.015625f * (2.f * (((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                                ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[5] - kxxMzzStencil[2])) +
                                ((kxzStencil[0] - kxzStencil[7]) + (kxzStencil[4] - kxzStencil[3])) +
                                ((kxzStencil[6] - kxzStencil[1]) + (kxzStencil[2] - kxzStencil[5])) +
                                ((kyzStencil[0] - kyzStencil[7]) + (kyzStencil[4] - kyzStencil[3])) +
                                ((kyzStencil[1] - kyzStencil[6]) + (kyzStencil[5] - kyzStencil[2])) +
                                ((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[6] + uxStencil[1])) +
                                ((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[6] + uyStencil[1]) - (uyStencil[2] + uyStencil[5]))) +
                        8.f * (((uzStencil[7] + uzStencil[0]) + (uzStencil[3] + uzStencil[4])) + ((uzStencil[1] + uzStencil[6]) + (uzStencil[5] + uzStencil[2]))) +
                        ((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                        ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[2] - kxxMyyStencil[5])));

        const float ax = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[3] - uxStencil[4])) + ((uxStencil[1] - uxStencil[6]) + (uxStencil[5] - uxStencil[2])));
        const float bx = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[3] - uyStencil[4])) + ((uyStencil[1] - uyStencil[6]) + (uyStencil[5] - uyStencil[2])));
        const float cx = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[3] - uzStencil[4])) + ((uzStencil[1] - uzStencil[6]) + (uzStencil[5] - uzStencil[2])));

        const float ay = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[3] - uxStencil[4])) + ((uxStencil[6] - uxStencil[1]) + (uxStencil[2] - uxStencil[5])));
        const float by = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[3] - uyStencil[4])) + ((uyStencil[6] - uyStencil[1]) + (uyStencil[2] - uyStencil[5])));
        const float cy = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[3] - uzStencil[4])) + ((uzStencil[6] - uzStencil[1]) + (uzStencil[2] - uzStencil[5])));

        const float az = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[4] - uxStencil[3])) + ((uxStencil[6] - uxStencil[1]) + (uxStencil[5] - uxStencil[2])));
        const float bz = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[4] - uyStencil[3])) + ((uyStencil[6] - uyStencil[1]) + (uyStencil[5] - uyStencil[2])));
        const float cz = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[4] - uzStencil[3])) + ((uzStencil[6] - uzStencil[1]) + (uzStencil[5] - uzStencil[2])));
		
		// The rest of the coefficients is not needed for fineToCoarse interpolation, because the coarse cell has coords [0, 0, 0]
		/*
		a200 = 0.0625f * (2.f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[3] - uyStencil[6])) + ((uyStencil[4] - uyStencil[1]) - (uyStencil[2] + uyStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[6])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[4] + uzStencil[1]))) +
                        ((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                        ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[5] - kxxMyyStencil[2])) +
                        ((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                        ((kxxMzzStencil[1] - kxxMzzStencil[6]) + (kxxMzzStencil[5] - kxxMzzStencil[2])));
        b200 = 0.125f * (2.f * (-((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) + ((uxStencil[6] + uxStencil[1]) + (uxStencil[2] + uxStencil[5]))) +
                       ((kxyStencil[7] - kxyStencil[0]) + (kxyStencil[3] - kxyStencil[4])) +
                       ((kxyStencil[1] - kxyStencil[6]) + (kxyStencil[5] - kxyStencil[2])));
        c200 = 0.125f * (2.f * (((uxStencil[3] + uxStencil[4]) - (uxStencil[7] + uxStencil[0])) + ((uxStencil[6] + uxStencil[1]) - (uxStencil[2] + uxStencil[5]))) +
                       ((kxzStencil[7] - kxzStencil[0]) + (kxzStencil[3] - kxzStencil[4])) +
                       ((kxzStencil[1] - kxzStencil[6]) + (kxzStencil[5] - kxzStencil[2])));
        
        a020 = 0.125f * (2.f * (-((uyStencil[7] + uyStencil[0]) + (uyStencil[4] + uyStencil[3])) + ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5]))) +
                       ((kxyStencil[7] - kxyStencil[0]) + (kxyStencil[3] - kxyStencil[4])) +
                       ((kxyStencil[6] - kxyStencil[1]) + (kxyStencil[2] - kxyStencil[5])));
        b020 = 0.0625f * (2.f * (((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                                ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[5] - kxxMyyStencil[2])) +
                                ((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) - ((uxStencil[6] + uxStencil[1]) + (uxStencil[5] + uxStencil[2])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[2] + uzStencil[5]))) +
                        ((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                        ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[2] - kxxMzzStencil[5])));
        c020 = 0.125f * (2.f * (((uyStencil[4] + uyStencil[3]) - (uyStencil[7] + uyStencil[0])) + ((uyStencil[5] + uyStencil[2]) - (uyStencil[6] + uyStencil[1]))) +
                       ((kyzStencil[7] - kyzStencil[0]) + (kyzStencil[3] - kyzStencil[4])) +
                       ((kyzStencil[6] - kyzStencil[1]) + (kyzStencil[2] - kyzStencil[5])));
                 
        a002 = 0.125f * (2.f * (((uzStencil[3] + uzStencil[4]) - (uzStencil[7] + uzStencil[0])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[5] + uzStencil[2]))) +
                       ((kxzStencil[7] - kxzStencil[0]) + (kxzStencil[4] - kxzStencil[3])) +
                       ((kxzStencil[5] - kxzStencil[2]) + (kxzStencil[6] - kxzStencil[1])));
        b002 = 0.125f * (2.f * (((uzStencil[3] + uzStencil[4]) - (uzStencil[7] + uzStencil[0])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[1] + uzStencil[6]))) +
                       ((kyzStencil[7] - kyzStencil[0]) + (kyzStencil[4] - kyzStencil[3])) +
                       ((kyzStencil[5] - kyzStencil[2]) + (kyzStencil[6] - kyzStencil[1])));
        c002 = 0.0625f * (2.f * (((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                                ((kxxMzzStencil[2] - kxxMzzStencil[5]) + (kxxMzzStencil[1] - kxxMzzStencil[6])) +
                                ((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[1] + uxStencil[6])) +
                                ((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[1] + uyStencil[6]) - (uyStencil[2] + uyStencil[5]))) +
                        ((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                        ((kxxMyyStencil[5] - kxxMyyStencil[2]) + (kxxMyyStencil[6] - kxxMyyStencil[1])));
		
        a110 = 0.5f * (((uxStencil[7] + uxStencil[0]) + (uxStencil[4] + uxStencil[3])) - ((uxStencil[2] + uxStencil[5]) + (uxStencil[1] + uxStencil[6])));
        b110 = 0.5f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[4] + uyStencil[3])) - ((uyStencil[2] + uyStencil[5]) + (uyStencil[1] + uyStencil[6])));
        c110 = 0.5f * (((uzStencil[7] + uzStencil[0]) + (uzStencil[4] + uzStencil[3])) - ((uzStencil[2] + uzStencil[5]) + (uzStencil[1] + uzStencil[6])));

        a101 = 0.5f * (((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[1] + uxStencil[6])));
        b101 = 0.5f * (((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[2] + uyStencil[5]) - (uyStencil[1] + uyStencil[6])));
        c101 = 0.5f * (((uzStencil[7] + uzStencil[0]) - (uzStencil[4] + uzStencil[3])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[1] + uzStencil[6])));

        a011 = 0.5f * (((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[1] + uxStencil[6]) - (uxStencil[2] + uxStencil[5])));
        b011 = 0.5f * (((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[1] + uyStencil[6]) - (uyStencil[2] + uyStencil[5])));
        c011 = 0.5f * (((uzStencil[7] + uzStencil[0]) - (uzStencil[4] + uzStencil[3])) + ((uzStencil[1] + uzStencil[6]) - (uzStencil[2] + uzStencil[5])));

        a111 = ((uxStencil[7] - uxStencil[0]) + (uxStencil[4] - uxStencil[3])) + ((uxStencil[2] - uxStencil[5]) + (uxStencil[1] - uxStencil[6]));
        b111 = ((uyStencil[7] - uyStencil[0]) + (uyStencil[4] - uyStencil[3])) + ((uyStencil[2] - uyStencil[5]) + (uyStencil[1] - uyStencil[6]));
        c111 = ((uzStencil[7] - uzStencil[0]) + (uzStencil[4] - uzStencil[3])) + ((uzStencil[2] - uzStencil[5]) + (uzStencil[1] - uzStencil[6]));
		*/
		
		// get average second order moments
		// eq Schönherr 2015 (7.29 - 7.33)
		float kxyAvg = 0.f; for ( int i = 0; i < 8; i++ ) kxyAvg += kxyStencil[i]; kxyAvg *= 0.125f; kxyAvg -= ( ay + bx );
		float kyzAvg = 0.f; for ( int i = 0; i < 8; i++ ) kyzAvg += kyzStencil[i]; kyzAvg *= 0.125f; kyzAvg -= ( bz + cy );
		float kxzAvg = 0.f; for ( int i = 0; i < 8; i++ ) kxzAvg += kxzStencil[i]; kxzAvg *= 0.125f; kxzAvg -= ( az + cx );
		float kxxMyyAvg = 0.f; for ( int i = 0; i < 8; i++ ) kxxMyyAvg += kxxMyyStencil[i]; kxxMyyAvg *= 0.125f; kxxMyyAvg -= ( ax - by );
		float kxxMzzAvg = 0.f; for ( int i = 0; i < 8; i++ ) kxxMzzAvg += kxxMzzStencil[i]; kxxMzzAvg *= 0.125f; kxxMzzAvg -= ( ax - cz );
		
		// get interpolated variables for the coarse cell
		const float rho = d0; const float ux = a0; const float uy = b0; const float uz = c0;
		const float dRho = rho - 1.f;
		
		// calculate second order central moments
		// eq Schönherr 2015 (7.38 - 7.43)
		// note that A, B, C is all zeros because coarse cell is placed [0, 0, 0]
		const float sigma = 2.f; // fine to coarse
		const float k_011 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( (bz + cy) + kyzAvg );
		const float k_101 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( (az + cx) + kxzAvg );
		const float k_110 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( (ay + bx) + kxyAvg );
		
		const float mxxMyy = - (2.f / 3.f) * ((ax - by) + kxxMyyAvg) * sigma / omega1Coarse * rho;
		const float mxxMzz = - (2.f / 3.f) * ((ax - cz) + kxxMzzAvg) * sigma / omega1Coarse * rho;
		
		const float k_200 = (1.f / 3.f) * (       mxxMyy +       mxxMzz + dRho);
		const float k_020 = (1.f / 3.f) * (-2.f * mxxMyy +       mxxMzz + dRho);
		const float k_002 = (1.f / 3.f) * (       mxxMyy - 2.f * mxxMzz + dRho);
		
		// reconstruct f for the coarse cell
		float f[27];
		reconstructInterpolatedF( f, rho, ux, uy, uz, k_011, k_101, k_110, k_200, k_020, k_002 );
		
		// write reconstructed f into the coarse cell
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jPlusViewCoarse( NBR.kPlus );
		finishNBRPlus( NBR, InfoCoarse );
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
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
	auto jMinusViewCoarse = GridCoarse.NBR.jMinusArray.getConstView();
	auto kMinusViewCoarse = GridCoarse.NBR.kMinusArray.getConstView();
	const float tauCoarse = 3.f * InfoCoarse.nu + 0.5f;
	const float omega1Coarse =  1.f / tauCoarse;
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto jPlusViewFine = GridFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.NBR.kPlusArray.getConstView();
	const float tauFine = 3.f * InfoFine.nu + 0.5f;
	const float omega1Fine =  1.f / tauFine;
	
	auto coarseToFineIndexView = GridCoarse.coarseToFineIndexArray.getConstView();
	auto childMapView = GridCoarse.childMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cellCoarse = coarseToFineIndexView( index );
		// get base data = center cell
		NBRStruct NBR;
		NBR.self = cellCoarse;
		NBR.jPlus = jPlusViewCoarse( cellCoarse );
		NBR.kPlus = kPlusViewCoarse( cellCoarse );
		NBR.jkPlus = jPlusViewCoarse( NBR.kPlus );
		finishNBRPlus( NBR, InfoCoarse );
		int cellReadIndex[27], fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		
		float fBase[27];
		for ( int direction = 0; direction < 27; direction++ ) fBase[direction] = fViewCoarse(fReadIndex[direction], cellReadIndex[direction]);
		float rhoBase, uxBase, uyBase, uzBase;
		getRhoUxUyUz( rhoBase, uxBase, uyBase, uzBase, fBase );
		/*
		const float weights[27] = { 8.f/27.f, 
				2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
				1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
				1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
		for ( int direction = 0; direction < 27; direction++ ) fBase[direction] -= weights[direction];
		*/
		const float kxyBase = - 3.f * omega1Coarse * ( (
				+ fBase[11] + fBase[12] - fBase[15] - fBase[16] 
				- fBase[19] - fBase[20] + fBase[21] + fBase[22] - fBase[23] - fBase[24] + fBase[25] + fBase[26]
												) / rhoBase - uxBase * uyBase );
		const float kyzBase = - 3.f * omega1Coarse * ( (
				- fBase[13] - fBase[14] + fBase[17] + fBase[18] 
				- fBase[19] - fBase[20] - fBase[21] - fBase[22] + fBase[23] + fBase[24] + fBase[25] + fBase[26]
												) / rhoBase - uyBase * uzBase );
		const float kxzBase = - 3.f * omega1Coarse * ( (
				- fBase[7 ] - fBase[8 ] + fBase[9 ] + fBase[10] 
				+ fBase[19] + fBase[20] - fBase[21] - fBase[22] - fBase[23] - fBase[24] + fBase[25] + fBase[26]
												) / rhoBase - uxBase * uzBase );
		const float kxxMyyBase = - 1.5f * omega1Coarse * ( (
				+ fBase[1 ] + fBase[2 ] - fBase[5 ] - fBase[6 ] 
				+ fBase[7 ] + fBase[8 ] + fBase[9 ] + fBase[10] - fBase[13] - fBase[14] - fBase[17] - fBase[18]
												) / rhoBase - ( uxBase * uxBase - uyBase * uyBase ) );
		const float kxxMzzBase = - 1.5f * omega1Coarse * ( ( 
				+ fBase[1 ] + fBase[2 ] - fBase[3 ] - fBase[4 ] 
				+ fBase[11] + fBase[12] - fBase[13] - fBase[14] + fBase[15] + fBase[16] - fBase[17] - fBase[18]
												) / rhoBase - ( uxBase * uxBase - uzBase * uzBase ) );

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

		const int cellStencil[6] = { 
			cellCoarse+1, cellCoarse-1, 
			jPlusViewCoarse(cellCoarse), jMinusViewCoarse(cellCoarse), 
			kPlusViewCoarse(cellCoarse), kMinusViewCoarse(cellCoarse) 
		};
		
		// Accumulate contributions for each neighbour
		for ( int i = 0; i < 6; i++ )
		{
			const int nbr = cellStencil[i];
			NBRStruct NBRofNBR;
			NBRofNBR.self = nbr;
			NBRofNBR.jPlus = jPlusViewCoarse( nbr );
			NBRofNBR.kPlus = kPlusViewCoarse( nbr );
			NBRofNBR.jkPlus = jPlusViewCoarse( NBRofNBR.kPlus );
			finishNBRPlus( NBRofNBR, InfoCoarse );
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPreCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperCoarse, InfoCoarse );
			
			float fNbr[27];
			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewCoarse( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			float rhoNbr, uxNbr, uyNbr, uzNbr;
			getRhoUxUyUz( rhoNbr, uxNbr, uyNbr, uzNbr, fNbr );
			LocalDuStruct LocalDuNbr;
			getLocalDu( fNbr, InfoCoarse.nu, LocalDuNbr );
			
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
		
		// linear version
		// axy = 0.f; axz = 0.f; bxy = 0.f; byz = 0.f; cxz = 0.f; cyz = 0.f;
		
		const int cellFine0 = childMapView( cellCoarse );

		int cellFineList[8];
		cellFineList[0] = cellFine0;
		cellFineList[1] = cellFine0 + 1; if ( cellFineList[1] >= InfoFine.cellCount ) cellFineList[1] = 0;
		cellFineList[2] = jPlusViewFine( cellFine0 );
		cellFineList[3] = cellFineList[2] + 1; if ( cellFineList[3] >= InfoFine.cellCount ) cellFineList[3] = 0;
		cellFineList[4] = kPlusViewFine( cellFine0 );
		cellFineList[5] = cellFineList[4] + 1; if ( cellFineList[5] >= InfoFine.cellCount ) cellFineList[5] = 0;
		cellFineList[6] = jPlusViewFine( cellFineList[4] );
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
			NBR.jkPlus = jPlusViewFine( NBR.kPlus );
			finishNBRPlus( NBR, InfoFine );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperFine, InfoFine );
			
			const float dx = cellFineDx[which];
			const float dy = cellFineDy[which];
			const float dz = cellFineDz[which];
			const float rho = rhoBase + dRhodx * dx + dRhody * dy + dRhodz * dz;
			const float dRho = rho - 1.f;
			const float ux = uxBase + ax * dx + ay * dy + az * dz + axy * dx * dy + axz * dx * dz + ayz * dy * dz + axx * dx * dx + ayy * dy * dy + azz * dz * dz;
			const float uy = uyBase + bx * dx + by * dy + bz * dz + bxy * dx * dy + bxz * dx * dz + byz * dy * dz + bxx * dx * dx + byy * dy * dy + bzz * dz * dz;
			const float uz = uzBase + cx * dx + cy * dy + cz * dz + cxy * dx * dy + cxz * dx * dz + cyz * dy * dz + cxx * dx * dx + cyy * dy * dy + czz * dz * dz;
			
			// calculate second order central moments
			// eq Schönherr 2015 (7.38 - 7.43) - with base gradients mathematically cancelled
			const float sigma = 0.5f; // coarse to fine
			const float A011 = bxz * dx + cxy * dx + byz * dy + 2.f * cyy * dy + 2.f * bzz * dz + cyz * dz;
			const float A101 = axz * dx + 2.f * cxx * dx + ayz * dy + cxy * dy + 2.f * azz * dz + cxz * dz;
			const float A110 = axy * dx + 2.f * bxx * dx + 2.f * ayy * dy + bxy * dy + ayz * dz + bxz * dz;
			const float B = 2.f * axx * dx - bxy * dx + axy * dy - 2.f * byy * dy + axz * dz - byz * dz;
			const float C = 2.f * axx * dx - cxz * dx + axy * dy - cyz * dy + axz * dz - 2.f * czz * dz;
            
			const float k_011 = - ( sigma * rho ) / ( 3.f * omega1Fine ) * ( kyzBase + A011 );
			const float k_101 = - ( sigma * rho ) / ( 3.f * omega1Fine ) * ( kxzBase + A101 );
			const float k_110 = - ( sigma * rho ) / ( 3.f * omega1Fine ) * ( kxyBase + A110 );
			const float k_200 = dRho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Fine ) * ( kxxMyyBase + B + kxxMzzBase + C );
			const float k_020 = dRho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Fine ) * ( - 2.f * ( kxxMyyBase + B ) + kxxMzzBase + C );
			const float k_002 = dRho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Fine ) * ( kxxMyyBase + B - 2.f * ( kxxMzzBase + C ) );
			
			float f[27];
			reconstructInterpolatedF( f, rho, ux, uy, uz, k_011, k_101, k_110, k_200, k_020, k_002 );
			
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
