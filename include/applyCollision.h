#pragma once

#include "./cellFunctions.h"

// Cummulant collision
//
// Martin Geier 2015, 2017

__host__ __device__ void applyCollision( float (&f)[27], const BCRhoUGStruct &BCRhoUG, const float &nu )
{
	const float gx = BCRhoUG.gx;
	const float gy = BCRhoUG.gy;
	const float gz = BCRhoUG.gz;
	
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
	
	//------------------------------------------------------------------------------------
	//---------------------------- APPLY FORCING - FIRST HALF ----------------------------
	//------------------------------------------------------------------------------------
	ux = ((ux * rho) + gx/2.f) / rho;
	uy = ((uy * rho) + gy/2.f) / rho;
	uz = ((uz * rho) + gz/2.f) / rho;
	
	float ux2 = ux * ux;
	float uy2 = uy * uy;
	// float uz2 = uz * uz;
	float rhoInv = 1.f / rho;
	
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
	
	const float k_a12 = (k_ac2 - k_aa2) - uy * k_a02;
	const float k_b12 = (k_bc2 - k_ba2) - uy * k_b02;
	const float k_c12 = (k_cc2 - k_ca2) - uy * k_c02;

	const float k_a20 = (k_ac0 + k_aa0) - 2.f * uy * (k_ac0 - k_aa0) + uy * uy * k_a00;
	const float k_b20 = (k_bc0 + k_ba0) - 2.f * uy * (k_bc0 - k_ba0) + uy * uy * k_b00;
	const float k_c20 = (k_cc0 + k_ca0) - 2.f * uy * (k_cc0 - k_ca0) + uy * uy * k_c00;
	
	const float k_a21 = (k_ac1 + k_aa1) - 2.f * uy * (k_ac1 - k_aa1) + uy2 * k_a01;
	const float k_b21 = (k_bc1 + k_ba1) - 2.f * uy * (k_bc1 - k_ba1) + uy2 * k_b01;
	const float k_c21 = (k_cc1 + k_ca1) - 2.f * uy * (k_cc1 - k_ca1) + uy2 * k_c01;
	const float k_a22 = (k_ac2 + k_aa2) - 2.f * uy * (k_ac2 - k_aa2) + uy2 * k_a02;
	const float k_b22 = (k_bc2 + k_ba2) - 2.f * uy * (k_bc2 - k_ba2) + uy2 * k_b02;
	const float k_c22 = (k_cc2 + k_ca2) - 2.f * uy * (k_cc2 - k_ca2) + uy2 * k_c02;

	//Eq Geier 2015(45)
	// third part of the central moments transformation
	const float k_000 = (k_c00 + k_a00) + k_b00;
	const float k_001 = (k_c01 + k_a01) + k_b01;
	const float k_002 = (k_c02 + k_a02) + k_b02;
	const float k_010 = (k_c10 + k_a10) + k_b10;
	const float k_011 = (k_c11 + k_a11) + k_b11;
	const float k_020 = (k_c20 + k_a20) + k_b20;
	
	const float k_012 = (k_c12 + k_a12) + k_b12;
	const float k_021 = (k_c21 + k_a21) + k_b21;
	const float k_022 = (k_c22 + k_a22) + k_b22;


	const float k_100 = (k_c00 - k_a00) - ux * k_000;
	const float k_101 = (k_c01 - k_a01) - ux * k_001;
	const float k_110 = (k_c10 - k_a10) - ux * k_010;
	
	const float k_102 = (k_c02 - k_a02) - ux * k_002;
	const float k_111 = (k_c11 - k_a11) - ux * k_011;
	const float k_112 = (k_c12 - k_a12) - ux * k_012;
	const float k_120 = (k_c20 - k_a20) - ux * k_020;
	const float k_121 = (k_c21 - k_a21) - ux * k_021;
	const float k_122 = (k_c22 - k_a22) - ux * k_022;

	const float k_200 = (k_c00 + k_a00) - 2.f * ux * (k_c00 - k_a00) + ux * ux * k_000;

	const float k_201 = (k_c01 + k_a01) - 2.f * ux * (k_c01 - k_a01) + ux2 * k_001;
	const float k_202 = (k_c02 + k_a02) - 2.f * ux * (k_c02 - k_a02) + ux2 * k_002;
	const float k_210 = (k_c10 + k_a10) - 2.f * ux * (k_c10 - k_a10) + ux2 * k_010;
	const float k_211 = (k_c11 + k_a11) - 2.f * ux * (k_c11 - k_a11) + ux2 * k_011;
	const float k_212 = (k_c12 + k_a12) - 2.f * ux * (k_c12 - k_a12) + ux2 * k_012;
	const float k_220 = (k_c20 + k_a20) - 2.f * ux * (k_c20 - k_a20) + ux2 * k_020;
	const float k_221 = (k_c21 + k_a21) - 2.f * ux * (k_c21 - k_a21) + ux2 * k_021;
	const float k_222 = (k_c22 + k_a22) - 2.f * ux * (k_c22 - k_a22) + ux2 * k_022;

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
	
	const float C_111 = k_111;
	const float C_120 = k_120;
	const float C_102 = k_102;
	const float C_210 = k_210;
	const float C_012 = k_012;
	const float C_201 = k_201;
	const float C_021 = k_021;
	
	// Eq 51 from Geier 2015
	const float C_211 = k_211 - (k_200 * k_011 + 2.f * k_101 * k_110) * rhoInv;
	const float C_121 = k_121 - (k_020 * k_101 + 2.f * k_110 * k_011) * rhoInv;
	const float C_112 = k_112 - (k_002 * k_110 + 2.f * k_011 * k_101) * rhoInv;

	// Eq 52 from Geier 2015
	const float C_220 = k_220 - (k_020 * k_200 + 2.f * k_110 * k_110) * rhoInv;
	const float C_022 = k_022 - (k_002 * k_020 + 2.f * k_011 * k_011) * rhoInv;
	const float C_202 = k_202 - (k_200 * k_002 + 2.f * k_101 * k_101) * rhoInv;

	// Eq 53 from Geier 2015
	const float C_122 = k_122 - (k_020 * k_102 + k_002 * k_120 + 4.f * k_011 * k_111 + 2.f * (k_110 * k_012 + k_101 * k_021)) * rhoInv;
	const float C_212 = k_212 - (k_002 * k_210 + k_200 * k_012 + 4.f * k_101 * k_111 + 2.f * (k_011 * k_201 + k_110 * k_102)) * rhoInv;
	const float C_221 = k_221 - (k_200 * k_021 + k_020 * k_201 + 4.f * k_110 * k_111 + 2.f * (k_101 * k_120 + k_011 * k_210)) * rhoInv;

	// Eq 54 from Geier 2015
	const float C_222 = k_222
					  - (4.f * k_111 * k_111 + k_200 * k_022 + k_020 * k_202 + k_002 * k_220
						 + 4.f * (k_011 * k_211 + k_101 * k_121 + k_110 * k_112) + 2.f * (k_120 * k_102 + k_210 * k_012 + k_201 * k_021))
							* rhoInv
					  + (16.f * k_110 * k_101 * k_011 + 4.f * (k_101 * k_101 * k_020 + k_011 * k_011 * k_200 + k_110 * k_110 * k_002)
						 + 2.f * k_200 * k_020 * k_002)
							* rhoInv * rhoInv;

	//------------------------------------------------------------------------------------
	// -------------------------------RELAXATION DEFINITIONS------------------------------
	//------------------------------------------------------------------------------------

	const float omega1 = 1.f / (3.f * nu + 0.5f);	// shear viscosity
	const float omega2 = 1.f;  //(3.f*KS.lbmViscosity*2.f + n1o2); // bulkViscosity > Viscosity ... test: bulkViscosity = 2/3 shearViscosity
	const float lambda3 = (float) (0.01);  // Section 7 @ Geier 2017 http://dx.doi.org/10.1016/j.jcp.2017.05.040
	const float lambda4 = (float) (0.01);
	const float lambda5 = (float) (0.01);
	const float omega3 = 8.f * (omega1 - 2.f) * (omega2 * (3.f * omega1 - 1.f) - 5.f * omega1)
					   / (8.f * (5.f - 2.f * omega1) * omega1 + omega2 * (8.f + omega1 * (9.f * omega1 - 26.f)));
	const float omega120p102 = omega3 + (1.f - omega3) * TNL::abs(C_120 + C_102) / (rho * lambda3 + TNL::abs(C_120 + C_102));  // limiter
	const float omega210p012 = omega3 + (1.f - omega3) * TNL::abs(C_210 + C_012) / (rho * lambda3 + TNL::abs(C_210 + C_012));  // limiter
	const float omega201p021 = omega3 + (1.f - omega3) * TNL::abs(C_201 + C_021) / (rho * lambda3 + TNL::abs(C_201 + C_021));  // limiter
	const float omega4 = 8.f * (omega1 - 2.f) * (omega1 + omega2 * (3.f * omega1 - 7.f))
					   / (omega2 * (56.f - 42.f * omega1 + 9.f * omega1 * omega1) - 8.f * omega1);
	const float omega120m102 = omega4 + (1.f - omega4) * TNL::abs(C_120 - C_102) / (rho * lambda4 + TNL::abs(C_120 - C_102));  // limiter
	const float omega210m012 = omega4 + (1.f - omega4) * TNL::abs(C_210 - C_012) / (rho * lambda4 + TNL::abs(C_210 - C_012));  // limiter
	const float omega201m021 = omega4 + (1.f - omega4) * TNL::abs(C_201 - C_021) / (rho * lambda4 + TNL::abs(C_201 - C_021));  // limiter
	const float omega5 =
		24.f * (omega1 - 2.f)
		* (4.f * omega1 * omega1 + omega1 * omega2 * (18.f - 13.f * omega1) + omega2 * omega2 * (2.f + omega1 * (6.f * omega1 - 11.f)))
		/ (16.f * omega1 * omega1 * (omega1 - 6.f) - 2.f * omega1 * omega2 * (216.f + 5.f * omega1 * (9.f * omega1 - 46.f))
		   + omega2 * omega2 * (omega1 * (3.f * omega1 - 10.f) * (15.f * omega1 - 28.f) - 48.f));
	const float omega111 = omega5 + (1.f - omega5) * TNL::abs(C_111) / (rho * lambda5 + TNL::abs(C_111));  // limiter
	const float omega6 = 1.f;
	const float omega7 = 1.f;
	const float omega8 = 1.f;
	const float omega9 = 1.f;
	const float omega10 = 1.f;
	// extra parameters
	const float A = (4.f * omega1 * omega1 + 2.f * omega1 * omega2 * (omega1 - 6.f) + omega2 * omega2 * (omega1 * (10.f - 3.f * omega1) - 4.f))
				  / (omega1 - omega2) / (omega2 * (2.f + 3.f * omega1) - 8.f * omega1);
	const float B =
		(4.f * omega1 * omega2 * (9.f * omega1 - 16.f) - 4.f * omega1 * omega1 - 2.f * omega2 * omega2 * (2.f + 9.f * omega1 * (omega1 - 2.f)))
		* (1.f/3.f) / (omega1 - omega2) / (omega2 * (2.f + 3.f * omega1) - 8.f * omega1);


	//------------------------------------------------------------------------------------
	// -------------------------------------COLLISION-------------------------------------
	//------------------------------------------------------------------------------------

	//  RELAX RATE Geier 2015(103) //2017 diff

	//Eq Geier 2015(58)
	const float Dxu = -omega1 * 0.5f / rho * (2.f * C_200 - C_020 - C_002) -	0.5f / rho * (C_200 + C_020 + C_002 - k_000); // -(-1-rho))

	//Eq Geier 2015(59)
	const float Dyv = Dxu + 3.0 * omega1 * 0.5f / rho * (C_200 - C_020);

	//Eq Geier 2015(60)
	const float Dzw = Dxu + 3.0 * omega1 * 0.5f / rho * (C_200 - C_002);
	const float DxvDyu = -3.f * omega1 * rhoInv * C_110;
	const float DxwDzu = -3.f * omega1 * rhoInv * C_101;
	const float DywDzv = -3.f * omega1 * rhoInv * C_011;

	//------------------------------------------------------------------------------------

	//Eq Geier 2015(55)
	const float Cs_110 = (1.f - omega1) * C_110;
	//Eq Geier 2015(56)
	const float Cs_101 = (1.f - omega1) * C_101;
	//Eq Geier 2015(57)
	const float Cs_011 = (1.f - omega1) * C_011;

	//---------------------------------------------------------------------------------

	//Eq Geier 2015(61, 62, 63)
	const float Eq61RHS = (1.f - omega1) * (C_200 - C_020) - 3.0 * rho * (1.f - omega1 * 0.5f) * (ux * ux * Dxu - uy * uy * Dyv);
	const float Eq64RHS = (1.f - omega1) * (C_200 - C_002) - 3.0 * rho * (1.f - omega1 * 0.5f) * (ux * ux * Dxu - uz * uz * Dzw);
	const float Eq65RHS = k_000 - 3.0 * rho * 0.5f * (ux * ux * Dxu + uy * uy * Dyv + uz * uz * Dzw);

	const float Cs_200 = 1.f / 3.0 * (Eq61RHS + Eq64RHS + Eq65RHS);
	const float Cs_020 = 1.f / 3.0 * (Eq64RHS - 2.f * Eq61RHS + Eq65RHS);
	const float Cs_002 = 1.f / 3.0 * (Eq61RHS - 2.f * Eq64RHS + Eq65RHS);
	
	// Limiter for omegas
	const float Eq117 = (1.f - omega120p102) * (C_120 + C_102);
	const float Eq118 = (1.f - omega210p012) * (C_210 + C_012);
	const float Eq119 = (1.f - omega201p021) * (C_201 + C_021);
	const float Eq120 = (1.f - omega120m102) * (C_120 - C_102);
	const float Eq121 = (1.f - omega210m012) * (C_210 - C_012);
	const float Eq122 = (1.f - omega201m021) * (C_201 - C_021);

	const float Cs_120 = 0.5f * (Eq120 + Eq117);
	const float Cs_102 = 0.5f * (-Eq120 + Eq117);
	const float Cs_210 = 0.5f * (Eq121 + Eq118);
	const float Cs_012 = 0.5f * (-Eq121 + Eq118);
	const float Cs_021 = 0.5f * (-Eq122 + Eq119);
	const float Cs_201 = 0.5f * (Eq122 + Eq119);
	// Eq 42
	const float Cs_111 = (1.f - omega111) * C_111;

	// Eqs 43-45
	const float Eq43RHS =
		(2.f/3.f) * (1.f / omega1 - 0.5f) * omega6 * A * rho * (Dxu - 2.f * Dyv + Dzw) + (1.f - omega6) * (C_220 - 2.f * C_202 + C_022);
	const float Eq44RHS =
		(2.f/3.f) * (1.f / omega1 - 0.5f) * omega6 * A * rho * (Dxu + Dyv - 2.f * Dzw) + (1.f - omega6) * (C_220 + C_202 - 2.f * C_022);
	const float Eq45RHS = -(4.f/3.f) * (1.f / omega1 - 0.5f) * omega7 * A * rho * (Dxu + Dyv + Dzw) + (1.f - omega7) * (C_220 + C_202 + C_022);

	const float Cs_220 = (1.f/3.f) * (Eq43RHS + Eq44RHS + Eq45RHS);
	const float Cs_202 = (1.f/3.f) * (-Eq43RHS + Eq45RHS);
	const float Cs_022 = (1.f/3.f) * (-Eq44RHS + Eq45RHS);
	// Eq 46-48
	const float Cs_211 = -(1.f/3.f) * (1.f / omega1 - 0.5f) * omega8 * B * rho * DywDzv + (1.f - omega8) * C_211;
	const float Cs_121 = -(1.f/3.f) * (1.f / omega1 - 0.5f) * omega8 * B * rho * DxwDzu + (1.f - omega8) * C_121;
	const float Cs_112 = -(1.f/3.f) * (1.f / omega1 - 0.5f) * omega8 * B * rho * DxvDyu + (1.f - omega8) * C_112;
	// Eqs 49-52
	const float Cs_221 = (1.f - omega9) * C_221;
	const float Cs_212 = (1.f - omega9) * C_212;
	const float Cs_122 = (1.f - omega9) * C_122;
	const float Cs_222 = (1.f - omega10) * C_222;

	// remark: collision step is the same as in well-CUM

	// 3.4 Backward cumulant transformation
	
	//const float ks_001 = Cs_001;
	//const float ks_010 = Cs_010;
	//const float ks_100 = Cs_100;
	const float ks_011 = Cs_011;
	const float ks_101 = Cs_101;
	const float ks_110 = Cs_110;
	const float ks_111 = Cs_111;
	const float ks_002 = Cs_002;
	const float ks_020 = Cs_020;
	const float ks_200 = Cs_200;
	const float ks_012 = Cs_012;
	const float ks_021 = Cs_021;
	const float ks_102 = Cs_102;
	const float ks_201 = Cs_201;
	const float ks_120 = Cs_120;
	const float ks_210 = Cs_210;
	
	// Eq. 81 from Geier 2015
	const float ks_211 = Cs_211 + (ks_200 * ks_011 + 2.f * ks_101 * ks_110) * rhoInv;
	const float ks_121 = Cs_121 + (ks_020 * ks_101 + 2.f * ks_110 * ks_011) * rhoInv;
	const float ks_112 = Cs_112 + (ks_002 * ks_110 + 2.f * ks_011 * ks_101) * rhoInv;

	// Eq. 82 from Geier 2015
	const float ks_220 = Cs_220 + (ks_020 * ks_200 + 2.f * ks_110 * ks_110) * rhoInv;
	const float ks_022 = Cs_022 + (ks_002 * ks_020 + 2.f * ks_011 * ks_011) * rhoInv;
	const float ks_202 = Cs_202 + (ks_200 * ks_002 + 2.f * ks_101 * ks_101) * rhoInv;

	// Eq. 83 from Geier 2015
	const float ks_122 =
		Cs_122 + (ks_020 * ks_102 + ks_002 * ks_120 + 4.f * ks_011 * ks_111 + 2.f * (ks_110 * ks_012 + ks_101 * ks_021)) * rhoInv;
	const float ks_212 =
		Cs_212 + (ks_002 * ks_210 + ks_200 * ks_012 + 4.f * ks_101 * ks_111 + 2.f * (ks_011 * ks_201 + ks_110 * ks_102)) * rhoInv;
	const float ks_221 =
		Cs_221 + (ks_200 * ks_021 + ks_020 * ks_201 + 4.f * ks_110 * ks_111 + 2.f * (ks_101 * ks_120 + ks_011 * ks_210)) * rhoInv;

	// Eq. 84 from Geier 2015
	const float ks_222 =
		Cs_222
		+ (4.f * ks_111 * ks_111 + ks_200 * ks_022 + ks_020 * ks_202 + ks_002 * ks_220
		   + 4.f * (ks_011 * ks_211 + ks_101 * ks_121 + ks_110 * ks_112) + 2.f * (ks_120 * ks_102 + ks_210 * ks_012 + ks_201 * ks_021))
			  * rhoInv
		- (16.f * ks_110 * ks_101 * ks_011 + 4.f * (ks_101 * ks_101 * ks_020 + ks_011 * ks_011 * ks_200 + ks_110 * ks_110 * ks_002)
		   + 2.f * ks_200 * ks_020 * ks_002)
			  * rhoInv * rhoInv;

	
	

	
	
	// backward central moment transformation
	const float ks_000 = k_000;
	// Geier 2017: forcing scheae
	const float ks_100 = -k_100;
	const float ks_010 = -k_010;
	const float ks_001 = -k_001;

	// Eq 88 froa Geier 2015
	const float ks_b00 = ks_000 * (1.f - ux2) - 2.f * ux * ks_100 - ks_200;
	const float ks_b01 = ks_001 * (1.f - ux2) - 2.f * ux * ks_101 - ks_201;
	const float ks_b02 = ks_002 * (1.f - ux2) - 2.f * ux * ks_102 - ks_202;
	const float ks_b10 = ks_010 * (1.f - ux2) - 2.f * ux * ks_110 - ks_210;
	const float ks_b11 = ks_011 * (1.f - ux2) - 2.f * ux * ks_111 - ks_211;
	const float ks_b12 = ks_012 * (1.f - ux2) - 2.f * ux * ks_112 - ks_212;
	const float ks_b20 = ks_020 * (1.f - ux2) - 2.f * ux * ks_120 - ks_220;
	const float ks_b21 = ks_021 * (1.f - ux2) - 2.f * ux * ks_121 - ks_221;
	const float ks_b22 = ks_022 * (1.f - ux2) - 2.f * ux * ks_122 - ks_222;

	// Eq 89 froa Geier 2015
	const float ks_a00 = (ks_000 * (ux2 - ux) + ks_100 * (2.f * ux - 1.f) + ks_200) * 0.5f;
	const float ks_a01 = (ks_001 * (ux2 - ux) + ks_101 * (2.f * ux - 1.f) + ks_201) * 0.5f;
	const float ks_a02 = (ks_002 * (ux2 - ux) + ks_102 * (2.f * ux - 1.f) + ks_202) * 0.5f;
	const float ks_a10 = (ks_010 * (ux2 - ux) + ks_110 * (2.f * ux - 1.f) + ks_210) * 0.5f;
	const float ks_a11 = (ks_011 * (ux2 - ux) + ks_111 * (2.f * ux - 1.f) + ks_211) * 0.5f;
	const float ks_a12 = (ks_012 * (ux2 - ux) + ks_112 * (2.f * ux - 1.f) + ks_212) * 0.5f;
	const float ks_a20 = (ks_020 * (ux2 - ux) + ks_120 * (2.f * ux - 1.f) + ks_220) * 0.5f;
	const float ks_a21 = (ks_021 * (ux2 - ux) + ks_121 * (2.f * ux - 1.f) + ks_221) * 0.5f;
	const float ks_a22 = (ks_022 * (ux2 - ux) + ks_122 * (2.f * ux - 1.f) + ks_222) * 0.5f;

	// Eq 90 froa Geier 2015
	const float ks_c00 = (ks_000 * (ux2 + ux) + ks_100 * (2.f * ux + 1.f) + ks_200) * 0.5f;
	const float ks_c01 = (ks_001 * (ux2 + ux) + ks_101 * (2.f * ux + 1.f) + ks_201) * 0.5f;
	const float ks_c02 = (ks_002 * (ux2 + ux) + ks_102 * (2.f * ux + 1.f) + ks_202) * 0.5f;
	const float ks_c10 = (ks_010 * (ux2 + ux) + ks_110 * (2.f * ux + 1.f) + ks_210) * 0.5f;
	const float ks_c11 = (ks_011 * (ux2 + ux) + ks_111 * (2.f * ux + 1.f) + ks_211) * 0.5f;
	const float ks_c12 = (ks_012 * (ux2 + ux) + ks_112 * (2.f * ux + 1.f) + ks_212) * 0.5f;
	const float ks_c20 = (ks_020 * (ux2 + ux) + ks_120 * (2.f * ux + 1.f) + ks_220) * 0.5f;
	const float ks_c21 = (ks_021 * (ux2 + ux) + ks_121 * (2.f * ux + 1.f) + ks_221) * 0.5f;
	const float ks_c22 = (ks_022 * (ux2 + ux) + ks_122 * (2.f * ux + 1.f) + ks_222) * 0.5f;

	// Eq 91 froa Geier 2015
	const float ks_ab0 = ks_a00 * (1.f - uy2) - 2.f * uy * ks_a10 - ks_a20;
	const float ks_ab1 = ks_a01 * (1.f - uy2) - 2.f * uy * ks_a11 - ks_a21;
	const float ks_ab2 = ks_a02 * (1.f - uy2) - 2.f * uy * ks_a12 - ks_a22;
	const float ks_bb0 = ks_b00 * (1.f - uy2) - 2.f * uy * ks_b10 - ks_b20;
	const float ks_bb1 = ks_b01 * (1.f - uy2) - 2.f * uy * ks_b11 - ks_b21;
	const float ks_bb2 = ks_b02 * (1.f - uy2) - 2.f * uy * ks_b12 - ks_b22;
	const float ks_cb0 = ks_c00 * (1.f - uy2) - 2.f * uy * ks_c10 - ks_c20;
	const float ks_cb1 = ks_c01 * (1.f - uy2) - 2.f * uy * ks_c11 - ks_c21;
	const float ks_cb2 = ks_c02 * (1.f - uy2) - 2.f * uy * ks_c12 - ks_c22;

	// Eq 92 froa Geier 2015
	const float ks_aa0 = (ks_a00 * (uy2 - uy) + ks_a10 * (2.f * uy - 1.f) + ks_a20) * 0.5f;
	const float ks_aa1 = (ks_a01 * (uy2 - uy) + ks_a11 * (2.f * uy - 1.f) + ks_a21) * 0.5f;
	const float ks_aa2 = (ks_a02 * (uy2 - uy) + ks_a12 * (2.f * uy - 1.f) + ks_a22) * 0.5f;
	const float ks_ba0 = (ks_b00 * (uy2 - uy) + ks_b10 * (2.f * uy - 1.f) + ks_b20) * 0.5f;
	const float ks_ba1 = (ks_b01 * (uy2 - uy) + ks_b11 * (2.f * uy - 1.f) + ks_b21) * 0.5f;
	const float ks_ba2 = (ks_b02 * (uy2 - uy) + ks_b12 * (2.f * uy - 1.f) + ks_b22) * 0.5f;
	const float ks_ca0 = (ks_c00 * (uy2 - uy) + ks_c10 * (2.f * uy - 1.f) + ks_c20) * 0.5f;
	const float ks_ca1 = (ks_c01 * (uy2 - uy) + ks_c11 * (2.f * uy - 1.f) + ks_c21) * 0.5f;
	const float ks_ca2 = (ks_c02 * (uy2 - uy) + ks_c12 * (2.f * uy - 1.f) + ks_c22) * 0.5f;

	// Eq 93 froa Geier 2015
	const float ks_ac0 = (ks_a00 * (uy2 + uy) + ks_a10 * (2.f * uy + 1.f) + ks_a20) * 0.5f;
	const float ks_ac1 = (ks_a01 * (uy2 + uy) + ks_a11 * (2.f * uy + 1.f) + ks_a21) * 0.5f;
	const float ks_ac2 = (ks_a02 * (uy2 + uy) + ks_a12 * (2.f * uy + 1.f) + ks_a22) * 0.5f;
	const float ks_bc0 = (ks_b00 * (uy2 + uy) + ks_b10 * (2.f * uy + 1.f) + ks_b20) * 0.5f;
	const float ks_bc1 = (ks_b01 * (uy2 + uy) + ks_b11 * (2.f * uy + 1.f) + ks_b21) * 0.5f;
	const float ks_bc2 = (ks_b02 * (uy2 + uy) + ks_b12 * (2.f * uy + 1.f) + ks_b22) * 0.5f;
	const float ks_cc0 = (ks_c00 * (uy2 + uy) + ks_c10 * (2.f * uy + 1.f) + ks_c20) * 0.5f;
	const float ks_cc1 = (ks_c01 * (uy2 + uy) + ks_c11 * (2.f * uy + 1.f) + ks_c21) * 0.5f;
	const float ks_cc2 = (ks_c02 * (uy2 + uy) + ks_c12 * (2.f * uy + 1.f) + ks_c22) * 0.5f;


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

/*
// BACKUP
// FORCING VERSION
__host__ __device__ void applyCollision( float (&f)[27], const BCRhoUGStruct &BCRhoUG, const float &nu )
{
	const float gx = BCRhoUG.gx;
	const float gy = BCRhoUG.gy;
	const float gz = BCRhoUG.gz;
	
	float rho, ux, uy, uz;
	getRhoUxUyUz( rho, ux, uy, uz, f );
	
	//------------------------------------------------------------------------------------
	//---------------------------- APPLY FORCING - FIRST HALF ----------------------------
	//------------------------------------------------------------------------------------
	ux = ((ux * rho) + gx/2.f) / rho;
	uy = ((uy * rho) + gy/2.f) / rho;
	uz = ((uz * rho) + gz/2.f) / rho;
	
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
	const float k_000 = (k_c00 + k_a00) + k_b00;
	const float k_001 = (k_c01 + k_a01) + k_b01;
	const float k_002 = (k_c02 + k_a02) + k_b02;
	const float k_010 = (k_c10 + k_a10) + k_b10;
	const float k_011 = (k_c11 + k_a11) + k_b11;
	const float k_020 = (k_c20 + k_a20) + k_b20;

	const float k_100 = (k_c00 - k_a00) - ux * k_000;
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

	//higher order cummulants all get relaxed to zero so they dont have to be calculated

	//------------------------------------------------------------------------------------
	// -------------------------------CALCULATING LES OMEGA-------------------------------
	//------------------------------------------------------------------------------------

	float feq[27];
	getFeq(rho, ux, uy, uz, feq);
	float fneq[27];
	getFneq(f, feq, fneq);
	float omegaLES;
	getOmegaLES(fneq, rho, nu, omegaLES);

	//------------------------------------------------------------------------------------
	// -------------------------------------COLLISION-------------------------------------
	//------------------------------------------------------------------------------------

	//  RELAX RATE Geier 2015(103) //2017 diff

	const float omega1 = omegaLES;

	//Eq Geier 2015(58)
	const float Dxu = -omega1 * 0.5f / rho * (2.f * C_200 - C_020 - C_002) -	0.5f / rho * (C_200 + C_020 + C_002 - k_000); // -(-1-rho))

	//Eq Geier 2015(59)
	const float Dyv = Dxu + 3.0 * omega1 * 0.5f / rho * (C_200 - C_020);

	//Eq Geier 2015(60)
	const float Dzw = Dxu + 3.0 * omega1 * 0.5f / rho * (C_200 - C_002);

	//------------------------------------------------------------------------------------

	//Eq Geier 2015(55)
	const float Cs_110 = (1.f - omega1) * C_110;
	//Eq Geier 2015(56)
	const float Cs_101 = (1.f - omega1) * C_101;
	//Eq Geier 2015(57)
	const float Cs_011 = (1.f - omega1) * C_011;

	//---------------------------------------------------------------------------------

	//Eq Geier 2015(61, 62, 63)
	const float Eq61RHS = (1.f - omega1) * (C_200 - C_020) - 3.0 * rho * (1.f - omega1 * 0.5f) * (ux * ux * Dxu - uy * uy * Dyv);
	const float Eq64RHS = (1.f - omega1) * (C_200 - C_002) - 3.0 * rho * (1.f - omega1 * 0.5f) * (ux * ux * Dxu - uz * uz * Dzw);
	const float Eq65RHS = k_000 - 3.0 * rho * 0.5f * (ux * ux * Dxu + uy * uy * Dyv + uz * uz * Dzw);

	const float Cs_200 = 1.f / 3.0 * (Eq61RHS + Eq64RHS + Eq65RHS);
	const float Cs_020 = 1.f / 3.0 * (Eq64RHS - 2.f * Eq61RHS + Eq65RHS);
	const float Cs_002 = 1.f / 3.0 * (Eq61RHS - 2.f * Eq64RHS + Eq65RHS);

	//------------------------------------------------------------------------------------
	//------------------------------ CUMULANTS TO CENTRAL MOM. ---------------------------
	//------------------------------------------------------------------------------------

	const float ks_000 = k_000;

	// Permutation again

	//Eq Geier 2015(47) backwards
	const float ks_110 = Cs_110;
	const float ks_101 = Cs_101;
	const float ks_011 = Cs_011;

	//Eq Geier 2015(48) backwards
	const float ks_200 = Cs_200;
	const float ks_020 = Cs_020;
	const float ks_002 = Cs_002;

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
*/
