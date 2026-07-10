// Velocity set. Small optimization - on the fine grid ghost cell layer, write only the distributions which will get streamed into the fine grid.
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

__host__ __device__ void rescaleF( float (&f)[27], const bool &coarseToFine )
{
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
	/*
		We are using cummulant collision, which involves transformation to central moments. Therefore to
		perform the transformation of the distribution functions when travelling between grids, we can perform
		the central moment transformation, scale the relevant moments, set the high order cummulants to zero,
		skip the relaxation part and perform the backwards transformation to receive the rescaled distribution
		functions on the target grid.
	*/
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

// DIAD VERSION //

void writeToCoarseGridInterface( DIADGridStruct &GridCoarse, DIADGridStruct &GridFine )
{
	auto fArrayViewCoarse = GridCoarse.fArray.getView();
	const InfoStruct InfoCoarse = GridCoarse.Info;
	auto fArrayViewFine  = GridFine.fArray.getView();
	const InfoStruct InfoFine = GridFine.Info;
	
	bool esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto iNbrViewCoarse = GridCoarse.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrViewCoarse = GridCoarse.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrViewCoarse = GridCoarse.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrViewCoarse = GridCoarse.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrViewCoarse = GridCoarse.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrViewCoarse = GridCoarse.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrViewCoarse = GridCoarse.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	bool esotwistFlipperFine = GridFine.esotwistFlipper;
	auto iNbrViewFine = GridFine.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrViewFine = GridFine.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrViewFine = GridFine.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrViewFine = GridFine.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrViewFine = GridFine.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrViewFine = GridFine.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrViewFine = GridFine.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	auto writeView = GridCoarse.fineToCoarseWriteArray.getConstView();
	auto readView = GridCoarse.fineToCoarseReadArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int writeCell = writeView( cell );
		const int readCell = readView( cell );
		
		int fineCellList[8] = {};
		fineCellList[0] = readCell;
		fineCellList[1] = iNbrViewFine( readCell );
		fineCellList[2] = jNbrViewFine( readCell );
		fineCellList[3] = kNbrViewFine( readCell );
		fineCellList[4] = ijNbrViewFine( readCell );
		fineCellList[5] = ikNbrViewFine( readCell );
		fineCellList[6] = jkNbrViewFine( readCell );
		fineCellList[7] = ijkNbrViewFine( readCell );
		
		float f[27] = {0};
		
		for ( int which = 0; which < 8; which++ )
		{
			const int fine = fineCellList[which];
			DIADEsotwistNbrStruct Nbr;
			Nbr.i = iNbrViewFine( fine );
			Nbr.j = jNbrViewFine( fine );
			Nbr.k = kNbrViewFine( fine );
			Nbr.ij = ijNbrViewFine( fine );
			Nbr.ik = ikNbrViewFine( fine );
			Nbr.jk = jkNbrViewFine( fine );
			Nbr.ijk = ijkNbrViewFine( fine );
			
			float fFine[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPostCollisionIndex( fine, cellReadIndex, fReadIndex, Nbr, esotwistFlipperFine, InfoFine );
			for ( int direction = 0; direction < 27; direction++ )	fFine[direction] = fArrayViewFine(fReadIndex[direction], cellReadIndex[direction]);
			for (int direction = 0; direction < 27; direction++) f[direction] += fFine[direction];	
		}
		for (int direction = 0; direction < 27; direction++) f[direction] = f[direction] * 0.125f;
		
		rescaleF( f, false );
		
		DIADEsotwistNbrStruct Nbr;
		Nbr.i = iNbrViewCoarse( writeCell );
		Nbr.j = jNbrViewCoarse( writeCell );
		Nbr.k = kNbrViewCoarse( writeCell );
		Nbr.ij = ijNbrViewCoarse( writeCell );
		Nbr.ik = ikNbrViewCoarse( writeCell );
		Nbr.jk = jkNbrViewCoarse( writeCell );
		Nbr.ijk = ijkNbrViewCoarse( writeCell );
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( writeCell, cellWriteIndex, fWriteIndex, Nbr, esotwistFlipperCoarse, InfoCoarse );
		for ( int direction = 0; direction < 27; direction++ ) fArrayViewCoarse( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.fineToCoarseWriteArray.getSize(), cellLambda );
}

void writeToFineGridInterface( DIADGridStruct &GridCoarse, DIADGridStruct &GridFine )
{
	auto fArrayViewCoarse = GridCoarse.fArray.getView();
	const InfoStruct InfoCoarse = GridCoarse.Info;
	auto fArrayViewFine  = GridFine.fArray.getView();
	const InfoStruct InfoFine = GridFine.Info;
	
	bool esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto iNbrViewCoarse = GridCoarse.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrViewCoarse = GridCoarse.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrViewCoarse = GridCoarse.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrViewCoarse = GridCoarse.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrViewCoarse = GridCoarse.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrViewCoarse = GridCoarse.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrViewCoarse = GridCoarse.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	bool esotwistFlipperFine = GridFine.esotwistFlipper;
	auto iNbrViewFine = GridFine.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrViewFine = GridFine.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrViewFine = GridFine.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrViewFine = GridFine.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrViewFine = GridFine.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrViewFine = GridFine.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrViewFine = GridFine.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	auto writeView = GridCoarse.coarseToFineWriteArray.getConstView();
	auto readView = GridCoarse.coarseToFineReadArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int writeCell = writeView( cell );
		const int readCell = readView( cell );
		
		int fineCellList[8] = {};
		fineCellList[0] = writeCell;
		fineCellList[1] = iNbrViewFine( writeCell );
		fineCellList[2] = jNbrViewFine( writeCell );
		fineCellList[3] = kNbrViewFine( writeCell );
		fineCellList[4] = ijNbrViewFine( writeCell );
		fineCellList[5] = ikNbrViewFine( writeCell );
		fineCellList[6] = jkNbrViewFine( writeCell );
		fineCellList[7] = ijkNbrViewFine( writeCell );
		
		float f[27] = {0};
		
		DIADEsotwistNbrStruct Nbr;
		Nbr.i = iNbrViewCoarse( readCell );
		Nbr.j = jNbrViewCoarse( readCell );
		Nbr.k = kNbrViewCoarse( readCell );
		Nbr.ij = ijNbrViewCoarse( readCell );
		Nbr.ik = ikNbrViewCoarse( readCell );
		Nbr.jk = jkNbrViewCoarse( readCell );
		Nbr.ijk = ijkNbrViewCoarse( readCell );
		int cellReadIndex[27];
		int fReadIndex[27];
		getPostCollisionIndex( readCell, cellReadIndex, fReadIndex, Nbr, esotwistFlipperCoarse, InfoCoarse );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayViewCoarse(fReadIndex[direction], cellReadIndex[direction]);
		
		rescaleF( f, true );
		
		for ( int which = 0; which < 8; which++ )
		{
			const int fine = fineCellList[which];
			Nbr.i = iNbrViewFine( fine );
			Nbr.j = jNbrViewFine( fine );
			Nbr.k = kNbrViewFine( fine );
			Nbr.ij = ijNbrViewFine( fine );
			Nbr.ik = ikNbrViewFine( fine );
			Nbr.jk = jkNbrViewFine( fine );
			Nbr.ijk = ijkNbrViewFine( fine );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPostCollisionIndex( fine, cellWriteIndex, fWriteIndex, Nbr, esotwistFlipperFine, InfoFine );
			for ( int direction = 0; direction < 27; direction++ ) fArrayViewFine( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		}
		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.coarseToFineWriteArray.getSize(), cellLambda );
}

void applyCoarseFineGridCommunication( DIADGridStruct &GridCoarse, DIADGridStruct &GridFine )
{
	writeToCoarseGridInterface( GridCoarse, GridFine );
	writeToFineGridInterface( GridCoarse, GridFine );
}
