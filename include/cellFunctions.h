#pragma once

// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cx * cx: { 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
// cy * cy: { 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
// cz * cz: { 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

// cy * cz: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1 };
// cx * cz: { 0, 0, 0, 0, 0, 0, 0,-1,-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1,-1,-1, 1, 1 };
// cx * cy: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,-1,-1, 0, 0,-1,-1, 1, 1,-1,-1, 1, 1 };

// w:  { 8/27, 2/27, 2/27, 2/27 , 2/27, 2/27, 2/27, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216 };

__cuda_callable__ void getIJKCellIndexFromXYZ( int& iCell, int& jCell, int& kCell, const float &x, const float &y, const float &z, const InfoStruct &Info)
{
    iCell = (int)(( x - Info.ox ) / Info.res + 0.5f);
    jCell = (int)(( y - Info.oy ) / Info.res + 0.5f);
    kCell = (int)(( z - Info.oz ) / Info.res + 0.5f);
}

__cuda_callable__ void getXYZFromIJKCellIndex( const int& iCell, const int& jCell, const int& kCell, float &x, float &y, float &z, const InfoStruct &Info)
{
    x = iCell * Info.res + Info.ox;
    y = jCell * Info.res + Info.oy;
    z = kCell * Info.res + Info.oz;
}

__cuda_callable__ void getOuterNormal( 	const int& iCell, const int& jCell, const int& kCell,
										int& outerNormalX, int& outerNormalY, int& outerNormalZ, const InfoStruct &Info )
{
    outerNormalX = 0;
    outerNormalY = 0;
    outerNormalZ = 0;
    if 			( iCell == 0 ) 						outerNormalX = -1;
    else if 	( iCell == Info.cellCountX - 1 ) 	outerNormalX = 1;
    if 			( jCell == 0 ) 						outerNormalY = -1;
    else if 	( jCell == Info.cellCountY - 1) 	outerNormalY = 1;
    if 			( kCell == 0 ) 						outerNormalZ = -1;
    else if 	( kCell == Info.cellCountZ - 1 ) 	outerNormalZ = 1;
}

__cuda_callable__ void getFeq(
	const float &rho, const float &ux, const float &uy, const float &uz, 
	float (&feq)[27]
	)
{

	const float u2 = ux*ux + uy*uy + uz*uz;

	const float cu0  = 0.f;
	const float cu1  = +ux;
	const float cu2  = -ux;
	const float cu3  = -uz;
	const float cu4  = +uz;
	const float cu5  = -uy;
	const float cu6  = +uy;
	const float cu7  = +ux -uz;
	const float cu8  = -ux +uz;
	const float cu9  = +ux +uz;
	const float cu10 = -ux -uz;
	const float cu11 = -ux -uy;
	const float cu12 = +ux +uy;
	const float cu13 = +uy -uz;
	const float cu14 = -uy +uz;
	const float cu15 = -ux +uy;
	const float cu16 = +ux -uy;
	const float cu17 = +uy +uz;
	const float cu18 = -uy -uz;
	const float cu19 = -ux +uy -uz;
	const float cu20 = +ux -uy +uz;
	const float cu21 = -ux -uy +uz;
	const float cu22 = +ux +uy -uz;
	const float cu23 = +ux -uy -uz;
	const float cu24 = -ux +uy +uz;
	const float cu25 = -ux -uy -uz;
	const float cu26 = +ux +uy +uz;

	constexpr float w0  = 8.f/27.f;
	constexpr float w1  = 2.f/27.f;
	constexpr float w2  = 1.f/54.f;
	constexpr float w3 = 1.f/216.f;

	feq[0]  = rho*w0 *(1.f + 3.f*cu0  + 4.5f*cu0 *cu0  - 1.5f*u2);
	feq[1]  = rho*w1 *(1.f + 3.f*cu1  + 4.5f*cu1 *cu1  - 1.5f*u2);
	feq[2]  = rho*w1 *(1.f + 3.f*cu2  + 4.5f*cu2 *cu2  - 1.5f*u2);
	feq[3]  = rho*w1 *(1.f + 3.f*cu3  + 4.5f*cu3 *cu3  - 1.5f*u2);
	feq[4]  = rho*w1 *(1.f + 3.f*cu4  + 4.5f*cu4 *cu4  - 1.5f*u2);
	feq[5]  = rho*w1 *(1.f + 3.f*cu5  + 4.5f*cu5 *cu5  - 1.5f*u2);
	feq[6]  = rho*w1 *(1.f + 3.f*cu6  + 4.5f*cu6 *cu6  - 1.5f*u2);
	feq[7]  = rho*w2 *(1.f + 3.f*cu7  + 4.5f*cu7 *cu7  - 1.5f*u2);
	feq[8]  = rho*w2 *(1.f + 3.f*cu8  + 4.5f*cu8 *cu8  - 1.5f*u2);
	feq[9]  = rho*w2 *(1.f + 3.f*cu9  + 4.5f*cu9 *cu9  - 1.5f*u2);
	feq[10] = rho*w2 *(1.f + 3.f*cu10 + 4.5f*cu10*cu10 - 1.5f*u2);
	feq[11] = rho*w2 *(1.f + 3.f*cu11 + 4.5f*cu11*cu11 - 1.5f*u2);
	feq[12] = rho*w2 *(1.f + 3.f*cu12 + 4.5f*cu12*cu12 - 1.5f*u2);
	feq[13] = rho*w2 *(1.f + 3.f*cu13 + 4.5f*cu13*cu13 - 1.5f*u2);
	feq[14] = rho*w2 *(1.f + 3.f*cu14 + 4.5f*cu14*cu14 - 1.5f*u2);
	feq[15] = rho*w2 *(1.f + 3.f*cu15 + 4.5f*cu15*cu15 - 1.5f*u2);
	feq[16] = rho*w2 *(1.f + 3.f*cu16 + 4.5f*cu16*cu16 - 1.5f*u2);
	feq[17] = rho*w2 *(1.f + 3.f*cu17 + 4.5f*cu17*cu17 - 1.5f*u2);
	feq[18] = rho*w2 *(1.f + 3.f*cu18 + 4.5f*cu18*cu18 - 1.5f*u2);
	feq[19] = rho*w3 *(1.f + 3.f*cu19 + 4.5f*cu19*cu19 - 1.5f*u2);
	feq[20] = rho*w3 *(1.f + 3.f*cu20 + 4.5f*cu20*cu20 - 1.5f*u2);
	feq[21] = rho*w3 *(1.f + 3.f*cu21 + 4.5f*cu21*cu21 - 1.5f*u2);
	feq[22] = rho*w3 *(1.f + 3.f*cu22 + 4.5f*cu22*cu22 - 1.5f*u2);
	feq[23] = rho*w3 *(1.f + 3.f*cu23 + 4.5f*cu23*cu23 - 1.5f*u2);
	feq[24] = rho*w3 *(1.f + 3.f*cu24 + 4.5f*cu24*cu24 - 1.5f*u2);
	feq[25] = rho*w3 *(1.f + 3.f*cu25 + 4.5f*cu25*cu25 - 1.5f*u2);
	feq[26] = rho*w3 *(1.f + 3.f*cu26 + 4.5f*cu26*cu26 - 1.5f*u2);	
}

__cuda_callable__ void getFneq(const float (&f)[27], const float (&feq)[27], float (&fneq)[27])
{
	for ( int i = 0; i < 27; i++ ) fneq[i] = f[i] - feq[i];
}

__cuda_callable__ void getRhoUxUyUz(
	float &rho, float &ux, float &uy, float &uz, 
	const float (&f)[27]
	)
{
	rho = 	f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8] + f[9] +
			f[10] + f[11] + f[12] + f[13] + f[14] + f[15] + f[16] + f[17] + f[18] + f[19] +
			f[20] + f[21] + f[22] + f[23] + f[24] + f[25] + f[26];     
	
	ux = 	(+f[1]  - f[2]  + f[7]  - f[8]  + f[9]  - f[10]
			-f[11] + f[12] - f[15] + f[16] - f[19] + f[20]
			-f[21] + f[22] + f[23] - f[24] - f[25] + f[26]) / rho;

	uy = 	(-f[5]  + f[6]  - f[11] + f[12] + f[13] - f[14]
			+f[15] - f[16] + f[17] - f[18] + f[19] - f[20]
			-f[21] + f[22] - f[23] + f[24] - f[25] + f[26]) / rho;

	uz = 	(-f[3]  + f[4]  - f[7]  + f[8]  + f[9]  - f[10]
			-f[13] + f[14] + f[17] - f[18] - f[19] + f[20]
			+f[21] - f[22] - f[23] + f[24] - f[25] + f[26]) / rho;
}

__cuda_callable__ void getOmegaLES( const float (&fneq)[27], const float &rho, const float &nu, float &omegaLES )
{
	const float tau = 3.f * nu + 0.5f;
	if (SMAGORINSKY_CONSTANT == 0)
	{
		omegaLES = 1 / tau;
		return;
	}
	
	float P = 0.f;

	const float cxcx = (fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] 
				+ fneq[16] + fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26]);
	P += (cxcx * cxcx);

	const float cycy = (fneq[5] + fneq[6] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[15] + fneq[16] + fneq[17]
				+ fneq[18] + fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26]);
	P += (cycy * cycy);

	const float czcz = (fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[17]
				+ fneq[18] + fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26]);
	P += (czcz * czcz);

	const float cycz = (-fneq[13] - fneq[14] + fneq[17] + fneq[18] - fneq[19] - fneq[20] - fneq[21] - fneq[22]
				+ fneq[23] + fneq[24] + fneq[25] + fneq[26]);
	P += 2.f*(cycz * cycz);

	const float cxcz = (-fneq[7] - fneq[8] + fneq[9] + fneq[10] + fneq[19] + fneq[20] - fneq[21] - fneq[22]
				- fneq[23] - fneq[24] + fneq[25] + fneq[26]);
	P += 2.f*(cxcz * cxcz);

	const float cxcy = (fneq[11] + fneq[12] - fneq[15] - fneq[16] - fneq[19] - fneq[20] + fneq[21] + fneq[22] 
				- fneq[23] - fneq[24] + fneq[25] + fneq[26]);
	P += 2.f*(cxcy * cxcy);

	P = sqrt(P);

	const float CLES_term = 18.f * SMAGORINSKY_CONSTANT * (1.f/rho);
	const float tauLES = 0.5 * tau + 0.5 * sqrt(tau * tau + CLES_term * P);

	omegaLES = 1 / tauLES;
}

__cuda_callable__ void convertToPhysicalVelocity( float &ux, float &uy, float &uz, const InfoStruct &Info )
{
	ux = ux * (Info.res/1000.f) / Info.dtPhys;
	uy = uy * (Info.res/1000.f) / Info.dtPhys;
	uz = uz * (Info.res/1000.f) / Info.dtPhys;
}

__cuda_callable__ void convertToPhysicalPressure( float &rho )
{
	// converts LBM rho to physical pressure, overwrites the variable (LBM rho -> physical p)
	const float p = (rho - 1.f) * rhoNominalPhys * soundspeedPhys * soundspeedPhys;
	rho = p;
}

__cuda_callable__ void convertToPhysicalPressure( float &rho, const InfoStruct &Info )
{
	// converts LBM rho to physical pressure, overwrites the variable (LBM rho -> physical p)
	const float p = (rho - 1.f) * rhoNominalPhys * soundspeedPhys * soundspeedPhys;
	rho = p;
}

__cuda_callable__ void convertToPhysicalForce( float &gx, float &gy, float &gz, const InfoStruct &Info )
{
	gx = gx * rhoNominalPhys * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
	gy = gy * rhoNominalPhys * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
	gz = gz * rhoNominalPhys * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
}

__host__ __device__ void getLocalDu( float (&f)[27], const float &nu, LocalDuStruct &localDu )
{
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
	//------------------------------------------------------------------------------------
	//--------------------------- TRANSFORM TO CENTRAL MOMENTS ---------------------------
	//------------------------------------------------------------------------------------

	//Eq Geier 2015(43)
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

	//Eq Geier 2015(44)
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

	//Eq Geier 2015(45)
	// third part of the central moments transformation
	const float k_000 = (k_c00 + k_a00) + k_b00;
	const float k_001 = (k_c01 + k_a01) + k_b01;
	const float k_002 = (k_c02 + k_a02) + k_b02;
	const float k_010 = (k_c10 + k_a10) + k_b10;
	const float k_011 = (k_c11 + k_a11) + k_b11;
	const float k_020 = (k_c20 + k_a20) + k_b20;

	// const float k_100 = (k_c00 - k_a00) - ux * k_000; // not needed
	const float k_101 = (k_c01 - k_a01) - ux * k_001;
	const float k_110 = (k_c10 - k_a10) - ux * k_010;

	const float k_200 = (k_c00 + k_a00) - 2.f * ux * (k_c00 - k_a00) + ux * ux * k_000;

	//------------------------------------------------------------------------------------
	//------------------------------ CENTRAL MOM. TO CUMULANTS ---------------------------
	//------------------------------------------------------------------------------------

	//Eq Geier 2015(47)
	const float C_110 = k_110;
	const float C_101 = k_101;
	const float C_011 = k_011;

	//Eq Geier 2015(48)
	const float C_200 = k_200;
	const float C_020 = k_020;
	const float C_002 = k_002;
	
	float feq[27];
	getFeq(rho, ux, uy, uz, feq);
	float fneq[27];
	getFneq(f, feq, fneq);
	float omegaLES;
	getOmegaLES(fneq, rho, nu, omegaLES);
	const float omega1 = omegaLES;
	//const float tau = 3.f * nu + 0.5f;
	//const float omega1 =  1 / tau;
	
	localDu.duxdx = (0.5f * omega1) * (-2.f * C_200 + C_020 + C_002) + 0.5f * (rho - C_200 - C_020 - C_002);
	localDu.duydy = localDu.duxdx + (1.5f * omega1) * (C_200 - C_020);
	localDu.duzdz = localDu.duxdx + (1.5f * omega1) * (C_200 - C_002);
	localDu.duxdyCross = - (3.f * omega1) * C_110;
	localDu.duydzCross = - (3.f * omega1) * C_011;
	localDu.duxdzCross = - (3.f * omega1) * C_101;
}

__host__ __device__ void getInterpolationVariables( float &rho, float &ux, float &uy, float &uz, 
													float &k_000,
													float &k_100, float &k_010, float &k_001, 
													float &k_200, float &k_020, float &k_002, 
													float &k_011, float &k_101, float &k_110,
													LocalDuStruct &localDu, 
													const float (&f)[27], const float &nu )
{
	getRhoUxUyUz( rho, ux, uy, uz, f );

	//-------------------------- CUMMULANT COLLISION EQUATIONS ---------------------------
	//------------------------------------------------------------------------------------
	//--------------------------- TRANSFORM TO CENTRAL MOMENTS ---------------------------
	//------------------------------------------------------------------------------------

	//Eq Geier 2015(43)
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

	//Eq Geier 2015(44)
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

	//Eq Geier 2015(45)
	// third part of the central moments transformation
	k_000 = (k_c00 + k_a00) + k_b00;
	k_001 = (k_c01 + k_a01) + k_b01;
	k_002 = (k_c02 + k_a02) + k_b02;
	k_010 = (k_c10 + k_a10) + k_b10;
	k_011 = (k_c11 + k_a11) + k_b11;
	k_020 = (k_c20 + k_a20) + k_b20;

	k_100 = (k_c00 - k_a00) - ux * k_000;
	k_101 = (k_c01 - k_a01) - ux * k_001;
	k_110 = (k_c10 - k_a10) - ux * k_010;

	k_200 = (k_c00 + k_a00) - 2.f * ux * (k_c00 - k_a00) + ux * ux * k_000;
	
	// Get local Du
	
	float feq[27];
	getFeq(rho, ux, uy, uz, feq);
	float fneq[27];
	getFneq(f, feq, fneq);
	float omegaLES;
	getOmegaLES(fneq, rho, nu, omegaLES);
	const float omega1 = omegaLES;
	
	localDu.duxdx = (0.5f * omega1) * (-2.f * k_200 + k_020 + k_002) + 0.5f * (rho - k_200 - k_020 - k_002);
	localDu.duydy = localDu.duxdx + (1.5f * omega1) * (k_200 - k_020);
	localDu.duzdz = localDu.duxdx + (1.5f * omega1) * (k_200 - k_002);
	localDu.duxdyCross = - (3.f * omega1) * k_110;
	localDu.duydzCross = - (3.f * omega1) * k_011;
	localDu.duxdzCross = - (3.f * omega1) * k_101;
}

__host__ __device__ void rescaleCentralMoments(	float &k_200, float &k_020, float &k_002, 
												float &k_011, float &k_101, float &k_110,
												const bool &coarseToFine )
{	
	//	We are using cummulant collision, which involves transformation to central moments. Therefore to
	//	perform the transformation of the distribution functions when travelling between grids, we can perform
	//	the central moment transformation, scale the relevant moments, set the high order cummulants to zero,
	//	skip the relaxation part and perform the backwards transformation to receive the rescaled distribution
	//	functions on the target grid.
	
	const float k_200prev = k_200;
	const float k_020prev = k_020;
	const float k_002prev = k_002;
	const float k_011prev = k_011;
	const float k_101prev = k_101;
	const float k_110prev = k_110;
	
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
}

__host__ __device__ void useInterpolatedVariables( 	const float &rho, const float &ux, const float &uy, const float &uz, 
													const float &k_000,
													const float &k_100, const float &k_010, const float &k_001, 
													const float &k_200, const float &k_020, const float &k_002, 
													const float &k_011, const float &k_101, const float &k_110, 
													float (&f)[27] )
{
	//------------------------------------------------------------------------------------
	//------------------------------ CENTRAL MOM. TO CUMULANTS ---------------------------
	//------------------------------------------------------------------------------------

	//Eq Geier 2015(47)
	const float C_110 = k_110;
	const float C_101 = k_101;
	const float C_011 = k_011;

	//Eq Geier 2015(48)
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

	//Eq Geier 2015(47) backwards
	const float ks_110 = C_110;
	const float ks_101 = C_101;
	const float ks_011 = C_011;

	//Eq Geier 2015(48) backwards
	const float ks_200 = C_200;
	const float ks_020 = C_020;
	const float ks_002 = C_002;

	//Eq. Geier 2015(85, 86, 87)
	const float ks_100 = -k_100;
	const float ks_010 = -k_010;
	const float ks_001 = -k_001;

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
