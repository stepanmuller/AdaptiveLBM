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

__host__ __device__ void getLocalDu( float (&f)[27], const float &nu, const float &SmagorinskyConstant, LocalDuStruct &localDu )
{
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
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

	//Eq Geiger 2015(47)
	const float C_110 = k_110;
	const float C_101 = k_101;
	const float C_011 = k_011;

	//Eq Geiger 2015(48)
	const float C_200 = k_200;
	const float C_020 = k_020;
	const float C_002 = k_002;
	
	//------------------------------------------------------------------------------------
	// -------------------------------CALCULATING LES OMEGA-------------------------------
	//------------------------------------------------------------------------------------

	float feq[27];
	getFeq(rho, ux, uy, uz, feq);
	float fneq[27];
	getFneq(f, feq, fneq);
	float omegaLES;
	getOmegaLES(fneq, rho, nu, omegaLES);

	const float omega1 = omegaLES;
	
	localDu.duxdx = (0.5f * omega1) * (-2.f * C_200 + C_020 + C_002) + 0.5f * (rho - C_200 - C_020 - C_002);
	localDu.duydy = localDu.duxdx + (1.5f * omega1) * (C_200 - C_020);
	localDu.duzdz = localDu.duxdx + (1.5f * omega1) * (C_200 - C_002);
	localDu.duXY = - (3.f * omega1) * C_110;
	localDu.duYZ = - (3.f * omega1) * C_011;
	localDu.duXZ = - (3.f * omega1) * C_101;
}
