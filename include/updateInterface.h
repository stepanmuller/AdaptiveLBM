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
													const float &C_011, const float &C_101, const float &C_110, 
													const float &C_200, const float &C_020, const float &C_002 )
{
	//------------------------------------------------------------------------------------
	//------------------------------ CUMULANTS TO CENTRAL MOM. ---------------------------
	//------------------------------------------------------------------------------------

	const float ks_000 = rho;

	// Permutation again

	//Eq Geier 2015(47) backwards
	const float ks_110 = C_110;
	const float ks_101 = C_101;
	const float ks_011 = C_011;

	//Eq Geier 2015(48) backwards
	const float ks_200 = C_200;
	const float ks_020 = C_020;
	const float ks_002 = C_002;

	// Schönherr 2015, first order central moments are set to zero
	const float ks_100 = 0.f;
	const float ks_010 = 0.f;
	const float ks_001 = 0.f;

	//Eq. Geier 2015(81)
	const float ks_211 = (ks_200 * ks_011 + 2.f * ks_101 * ks_110) / rho;
	const float ks_121 = (ks_020 * ks_101 + 2.f * ks_110 * ks_011) / rho;
	const float ks_112 = (ks_002 * ks_110 + 2.f * ks_011 * ks_101) / rho;

	//Eq. Geier 2015(82)
	const float ks_220 = (ks_020 * ks_200 + 2.f * ks_110 * ks_110) / rho;
	const float ks_022 = (ks_002 * ks_020 + 2.f * ks_011 * ks_011) / rho;
	const float ks_202 = (ks_200 * ks_002 + 2.f * ks_101 * ks_101) / rho;

	// Eq. Geier 2015(84)
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

	//Eq Geier 2015(88)
	const float ks_b00 = ks_000 * (1.f - ux * ux) - 2.f * ux * ks_100 - ks_200;
	const float ks_b01 = ks_001 * (1.f - ux * ux) - 2.f * ux * ks_101;
	const float ks_b02 = ks_002 * (1.f - ux * ux) - ks_202;
	const float ks_b10 = ks_010 * (1.f - ux * ux) - 2.f * ux * ks_110;
	const float ks_b11 = ks_011 * (1.f - ux * ux) - ks_211;
	const float ks_b12 = - 2.f * ux * ks_112;
	const float ks_b20 = ks_020 * (1.f - ux * ux) - ks_220;
	const float ks_b21 = - 2.f * ux * ks_121;
	const float ks_b22 = ks_022 * (1.f - ux * ux) - ks_222;

	//Eq  Geier 2015(89)
	const float ks_a00 = (ks_000 * (ux * ux - ux) + ks_100 * (2.f * ux - 1.f) + ks_200) * 0.5f;
	const float ks_a01 = (ks_001 * (ux * ux - ux) + ks_101 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a02 = (ks_002 * (ux * ux - ux) + ks_202) * 0.5f;
	const float ks_a10 = (ks_010 * (ux * ux - ux) + ks_110 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a11 = (ks_011 * (ux * ux - ux) + ks_211) * 0.5f;
	const float ks_a12 = (ks_112 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a20 = (ks_020 * (ux * ux - ux) + ks_220) * 0.5f;
	const float ks_a21 = (ks_121 * (2.f * ux - 1.f)) * 0.5f;
	const float ks_a22 = (ks_022 * (ux * ux - ux) + ks_222) * 0.5f;

	//Eq  Geier 2015(90)
	const float ks_c00 = (ks_000 * (ux * ux + ux) + ks_100 * (2.f * ux + 1.f) + ks_200) * 0.5f;
	const float ks_c01 = (ks_001 * (ux * ux + ux) + ks_101 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c02 = (ks_002 * (ux * ux + ux) + ks_202) * 0.5f;
	const float ks_c10 = (ks_010 * (ux * ux + ux) + ks_110 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c11 = (ks_011 * (ux * ux + ux) + ks_211) * 0.5f;
	const float ks_c12 = (ks_112 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c20 = (ks_020 * (ux * ux + ux) + ks_220) * 0.5f;
	const float ks_c21 = (ks_121 * (2.f * ux + 1.f)) * 0.5f;
	const float ks_c22 = (ks_022 * (ux * ux + ux) + ks_222) * 0.5f;

	//Eq Geier 2015(91)
	const float ks_ab0 = ks_a00 * (1.f - uy * uy) - 2.f * uy * ks_a10 - ks_a20;
	const float ks_ab1 = ks_a01 * (1.f - uy * uy) - 2.f * uy * ks_a11 - ks_a21;
	const float ks_ab2 = ks_a02 * (1.f - uy * uy) - 2.f * uy * ks_a12 - ks_a22;
	const float ks_bb0 = ks_b00 * (1.f - uy * uy) - 2.f * uy * ks_b10 - ks_b20;
	const float ks_bb1 = ks_b01 * (1.f - uy * uy) - 2.f * uy * ks_b11 - ks_b21;
	const float ks_bb2 = ks_b02 * (1.f - uy * uy) - 2.f * uy * ks_b12 - ks_b22;
	const float ks_cb0 = ks_c00 * (1.f - uy * uy) - 2.f * uy * ks_c10 - ks_c20;
	const float ks_cb1 = ks_c01 * (1.f - uy * uy) - 2.f * uy * ks_c11 - ks_c21;
	const float ks_cb2 = ks_c02 * (1.f - uy * uy) - 2.f * uy * ks_c12 - ks_c22;

	//Eq  Geier 2015(92)
	const float ks_aa0 = (ks_a00 * (uy * uy - uy) + ks_a10 * (2.f * uy - 1.f) + ks_a20) * 0.5f;
	const float ks_aa1 = (ks_a01 * (uy * uy - uy) + ks_a11 * (2.f * uy - 1.f) + ks_a21) * 0.5f;
	const float ks_aa2 = (ks_a02 * (uy * uy - uy) + ks_a12 * (2.f * uy - 1.f) + ks_a22) * 0.5f;
	const float ks_ba0 = (ks_b00 * (uy * uy - uy) + ks_b10 * (2.f * uy - 1.f) + ks_b20) * 0.5f;
	const float ks_ba1 = (ks_b01 * (uy * uy - uy) + ks_b11 * (2.f * uy - 1.f) + ks_b21) * 0.5f;
	const float ks_ba2 = (ks_b02 * (uy * uy - uy) + ks_b12 * (2.f * uy - 1.f) + ks_b22) * 0.5f;
	const float ks_ca0 = (ks_c00 * (uy * uy - uy) + ks_c10 * (2.f * uy - 1.f) + ks_c20) * 0.5f;
	const float ks_ca1 = (ks_c01 * (uy * uy - uy) + ks_c11 * (2.f * uy - 1.f) + ks_c21) * 0.5f;
	const float ks_ca2 = (ks_c02 * (uy * uy - uy) + ks_c12 * (2.f * uy - 1.f) + ks_c22) * 0.5f;

	//Eq Geier 2015(93)
	const float ks_ac0 = (ks_a00 * (uy * uy + uy) + ks_a10 * (2.f * uy + 1.f) + ks_a20) * 0.5f;
	const float ks_ac1 = (ks_a01 * (uy * uy + uy) + ks_a11 * (2.f * uy + 1.f) + ks_a21) * 0.5f;
	const float ks_ac2 = (ks_a02 * (uy * uy + uy) + ks_a12 * (2.f * uy + 1.f) + ks_a22) * 0.5f;
	const float ks_bc0 = (ks_b00 * (uy * uy + uy) + ks_b10 * (2.f * uy + 1.f) + ks_b20) * 0.5f;
	const float ks_bc1 = (ks_b01 * (uy * uy + uy) + ks_b11 * (2.f * uy + 1.f) + ks_b21) * 0.5f;
	const float ks_bc2 = (ks_b02 * (uy * uy + uy) + ks_b12 * (2.f * uy + 1.f) + ks_b22) * 0.5f;
	const float ks_cc0 = (ks_c00 * (uy * uy + uy) + ks_c10 * (2.f * uy + 1.f) + ks_c20) * 0.5f;
	const float ks_cc1 = (ks_c01 * (uy * uy + uy) + ks_c11 * (2.f * uy + 1.f) + ks_c21) * 0.5f;
	const float ks_cc2 = (ks_c02 * (uy * uy + uy) + ks_c12 * (2.f * uy + 1.f) + ks_c22) * 0.5f;

	//Eq Geier 2015(94)
	f[11] = ks_aa0 * (1.f - uz * uz) - 2.f * uz * ks_aa1 - ks_aa2;
	f[2] = ks_ab0 * (1.f - uz * uz) - 2.f * uz * ks_ab1 - ks_ab2;
	f[15] = ks_ac0 * (1.f - uz * uz) - 2.f * uz * ks_ac1 - ks_ac2;
	f[5] = ks_ba0 * (1.f - uz * uz) - 2.f * uz * ks_ba1 - ks_ba2;
	f[0] = ks_bb0 * (1.f - uz * uz) - 2.f * uz * ks_bb1 - ks_bb2;
	f[6] = ks_bc0 * (1.f - uz * uz) - 2.f * uz * ks_bc1 - ks_bc2;
	f[16] = ks_ca0 * (1.f - uz * uz) - 2.f * uz * ks_ca1 - ks_ca2;
	f[1] = ks_cb0 * (1.f - uz * uz) - 2.f * uz * ks_cb1 - ks_cb2;
	f[12] = ks_cc0 * (1.f - uz * uz) - 2.f * uz * ks_cc1 - ks_cc2;

	//Eq  Geier 2015(95)
	f[25] = (ks_aa0 * (uz * uz - uz) + ks_aa1 * (2.f * uz - 1.f) + ks_aa2) * 0.5f;
	f[10] = (ks_ab0 * (uz * uz - uz) + ks_ab1 * (2.f * uz - 1.f) + ks_ab2) * 0.5f;
	f[19] = (ks_ac0 * (uz * uz - uz) + ks_ac1 * (2.f * uz - 1.f) + ks_ac2) * 0.5f;
	f[18] = (ks_ba0 * (uz * uz - uz) + ks_ba1 * (2.f * uz - 1.f) + ks_ba2) * 0.5f;
	f[3] = (ks_bb0 * (uz * uz - uz) + ks_bb1 * (2.f * uz - 1.f) + ks_bb2) * 0.5f;
	f[13] = (ks_bc0 * (uz * uz - uz) + ks_bc1 * (2.f * uz - 1.f) + ks_bc2) * 0.5f;
	f[23] = (ks_ca0 * (uz * uz - uz) + ks_ca1 * (2.f * uz - 1.f) + ks_ca2) * 0.5f;
	f[7] = (ks_cb0 * (uz * uz - uz) + ks_cb1 * (2.f * uz - 1.f) + ks_cb2) * 0.5f;
	f[22] = (ks_cc0 * (uz * uz - uz) + ks_cc1 * (2.f * uz - 1.f) + ks_cc2) * 0.5f;

	//Eq  Geier 2015(96)
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

void updateFineToCoarseInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	// The interpolation and rescaling is based on Martin Schönherr's disertation 2015
	const InfoStruct &InfoCoarse = GridCoarse.Info;
	auto fViewCoarse = GridCoarse.fArray.getView();
	const bool &esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto jPlusViewCoarse = GridCoarse.NBR.jPlusArray.getConstView();
	auto kPlusViewCoarse = GridCoarse.NBR.kPlusArray.getConstView();
	auto jkPlusViewCoarse = GridCoarse.NBR.jkPlusArray.getConstView();
	const float tauCoarse = 3.f * InfoCoarse.nu + 0.5f;
	const float omega1Coarse =  1.f / tauCoarse;
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto jPlusViewFine = GridFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.NBR.kPlusArray.getConstView();
	auto jkPlusViewFine = GridFine.NBR.jkPlusArray.getConstView();
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
		cellStencil[6] = jkPlusViewFine( cellFine0 );
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
			NBRofNBR.jkPlus = jkPlusViewFine( nbr );
			finishNBRPlus( NBRofNBR, InfoFine );
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPostCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperFine, InfoFine );
			float fNbr[27];
			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewFine( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			
			getRhoUxUyUz( rhoStencil[i], uxStencil[i], uyStencil[i], uzStencil[i], fNbr );
			
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
		
		// calculate second order cummulants
		// eq Schönherr 2015 (7.38 - 7.43)
		// note that A, B, C is all zeros because coarse cell is placed [0, 0, 0]
		const float sigma = 2.f; // fine to coarse
		const float C011 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( bz + cy + kyzAvg );
		const float C101 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( az + cx + kxzAvg );
		const float C110 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( ay + bx + kxyAvg );
		const float C200 = rho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Coarse ) * ( ax - by + kxxMyyAvg + ax - cz + kxxMzzAvg );
		const float C020 = rho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Coarse ) * ( - 2.f * ( ax - by + kxxMyyAvg ) + ax - cz + kxxMzzAvg );
		const float C002 = rho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Coarse ) * ( ax - by + kxxMyyAvg - 2.f * ( ax - cz + kxxMzzAvg ) );
		
		// reconstruct f for the coarse cell
		float f[27];
		reconstructInterpolatedF( f, rho, ux, uy, uz, C011, C101, C110, C200, C020, C002 );
		
		// write reconstructed f into the coarse cell
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


/*
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

		int cellStencil[8];
		cellStencil[0] = cellFine0;
		cellStencil[1] = cellFine0 + 1; if ( cellStencil[1] >= InfoFine.cellCount ) cellStencil[1] = 0;
		cellStencil[2] = jPlusViewFine( cellFine0 );
		cellStencil[3] = cellStencil[2] + 1; if ( cellStencil[3] >= InfoFine.cellCount ) cellStencil[3] = 0;
		cellStencil[4] = kPlusViewFine( cellFine0 );
		cellStencil[5] = cellStencil[4] + 1; if ( cellStencil[5] >= InfoFine.cellCount ) cellStencil[5] = 0;
		cellStencil[6] = jkPlusViewFine( cellFine0 );
		cellStencil[7] = cellStencil[6] + 1; if ( cellStencil[7] >= InfoFine.cellCount ) cellStencil[7] = 0;
		
		// Accumulate contributions for each neighbour
		for ( int i = 0; i < 8; i++ )
		{
			const int nbr = cellStencil[i];
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
*/

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
	const float tauCoarse = 3.f * InfoCoarse.nu + 0.5f;
	const float omega1Coarse =  1.f / tauCoarse;
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto jPlusViewFine = GridFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.NBR.kPlusArray.getConstView();
	auto jkPlusViewFine = GridFine.NBR.jkPlusArray.getConstView();
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
		NBR.jkPlus = jkPlusViewCoarse( cellCoarse );
		finishNBRPlus( NBR, InfoCoarse );
		int cellReadIndex[27], fReadIndex[27];
		getPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipperCoarse, InfoCoarse );
		
		float fBase[27];
		for ( int direction = 0; direction < 27; direction++ ) fBase[direction] = fViewCoarse(fReadIndex[direction], cellReadIndex[direction]);
		float rhoBase, uxBase, uyBase, uzBase;
		getRhoUxUyUz( rhoBase, uxBase, uyBase, uzBase, fBase );
	
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
			NBRofNBR.jkPlus = jkPlusViewCoarse( nbr );
			finishNBRPlus( NBRofNBR, InfoCoarse );
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPostCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperCoarse, InfoCoarse );
			
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
			
			// calculate second order cummulants
			// eq Schönherr 2015 (7.38 - 7.43) - with base gradients mathematically cancelled
			const float sigma = 0.5f; // coarse to fine
			const float A011 = bxz * dx + cxy * dx + byz * dy + 2.f * cyy * dy + 2.f * bzz * dz + cyz * dz;
			const float A101 = axz * dx + 2.f * cxx * dx + ayz * dy + cxy * dy + 2.f * azz * dz + cxz * dz;
			const float A110 = axy * dx + 2.f * bxx * dx + 2.f * ayy * dy + bxy * dy + ayz * dz + bxz * dz;
			const float B = 2.f * axx * dx - bxy * dx + axy * dy - 2.f * byy * dy + axz * dz - byz * dz;
			const float C = 2.f * axx * dx - cxz * dx + axy * dy - cyz * dy + axz * dz - 2.f * czz * dz;
            
			const float C011 = - ( sigma * rho ) / ( 3.f * omega1Fine ) * ( kyzBase + A011 );
			const float C101 = - ( sigma * rho ) / ( 3.f * omega1Fine ) * ( kxzBase + A101 );
			const float C110 = - ( sigma * rho ) / ( 3.f * omega1Fine ) * ( kxyBase + A110 );
			const float C200 = rho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Fine ) * ( kxxMyyBase + B + kxxMzzBase + C );
			const float C020 = rho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Fine ) * ( - 2.f * ( kxxMyyBase + B ) + kxxMzzBase + C );
			const float C002 = rho / 3.f - ( 2.f * sigma * rho ) / ( 9.f * omega1Fine ) * ( kxxMyyBase + B - 2.f * ( kxxMzzBase + C ) );
			
			float f[27];
			reconstructInterpolatedF( f, rho, ux, uy, uz, C011, C101, C110, C200, C020, C002 );
			
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
