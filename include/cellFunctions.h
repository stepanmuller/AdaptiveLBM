#pragma once

#include "./D3Q27Directions.h"

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

__host__ __device__ void getIJKCellIndexFromXYZ( int& iCell, int& jCell, int& kCell, const float &x, const float &y, const float &z, const InfoStruct &Info)
{
    iCell = (int)(( x - Info.ox ) / Info.res + 0.5f);
    jCell = (int)(( y - Info.oy ) / Info.res + 0.5f);
    kCell = (int)(( z - Info.oz ) / Info.res + 0.5f);
}

__host__ __device__ void getXYZFromIJKCellIndex( const int& iCell, const int& jCell, const int& kCell, float &x, float &y, float &z, const InfoStruct &Info)
{
    x = iCell * Info.res + Info.ox;
    y = jCell * Info.res + Info.oy;
    z = kCell * Info.res + Info.oz;
}

__host__ __device__ void getOuterNormal( 	const int& iCell, const int& jCell, const int& kCell,
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

__host__ __device__ void getFeq(
	const float &rho, const float &ux, const float &uy, const float &uz, 
	float (&feq)[27]
	)
{
	const float dRho = rho - 1.f;
	
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

	feq[0]  = w0 * (dRho + (3.f*cu0  + 4.5f*cu0 *cu0  - 1.5f*u2) * (dRho + 1.f));
	feq[1]  = w1 * (dRho + (3.f*cu1  + 4.5f*cu1 *cu1  - 1.5f*u2) * (dRho + 1.f));
	feq[2]  = w1 * (dRho + (3.f*cu2  + 4.5f*cu2 *cu2  - 1.5f*u2) * (dRho + 1.f));
	feq[3]  = w1 * (dRho + (3.f*cu3  + 4.5f*cu3 *cu3  - 1.5f*u2) * (dRho + 1.f));
	feq[4]  = w1 * (dRho + (3.f*cu4  + 4.5f*cu4 *cu4  - 1.5f*u2) * (dRho + 1.f));
	feq[5]  = w1 * (dRho + (3.f*cu5  + 4.5f*cu5 *cu5  - 1.5f*u2) * (dRho + 1.f));
	feq[6]  = w1 * (dRho + (3.f*cu6  + 4.5f*cu6 *cu6  - 1.5f*u2) * (dRho + 1.f));
	feq[7]  = w2 * (dRho + (3.f*cu7  + 4.5f*cu7 *cu7  - 1.5f*u2) * (dRho + 1.f));
	feq[8]  = w2 * (dRho + (3.f*cu8  + 4.5f*cu8 *cu8  - 1.5f*u2) * (dRho + 1.f));
	feq[9]  = w2 * (dRho + (3.f*cu9  + 4.5f*cu9 *cu9  - 1.5f*u2) * (dRho + 1.f));
	feq[10] = w2 * (dRho + (3.f*cu10 + 4.5f*cu10*cu10 - 1.5f*u2) * (dRho + 1.f));
	feq[11] = w2 * (dRho + (3.f*cu11 + 4.5f*cu11*cu11 - 1.5f*u2) * (dRho + 1.f));
	feq[12] = w2 * (dRho + (3.f*cu12 + 4.5f*cu12*cu12 - 1.5f*u2) * (dRho + 1.f));
	feq[13] = w2 * (dRho + (3.f*cu13 + 4.5f*cu13*cu13 - 1.5f*u2) * (dRho + 1.f));
	feq[14] = w2 * (dRho + (3.f*cu14 + 4.5f*cu14*cu14 - 1.5f*u2) * (dRho + 1.f));
	feq[15] = w2 * (dRho + (3.f*cu15 + 4.5f*cu15*cu15 - 1.5f*u2) * (dRho + 1.f));
	feq[16] = w2 * (dRho + (3.f*cu16 + 4.5f*cu16*cu16 - 1.5f*u2) * (dRho + 1.f));
	feq[17] = w2 * (dRho + (3.f*cu17 + 4.5f*cu17*cu17 - 1.5f*u2) * (dRho + 1.f));
	feq[18] = w2 * (dRho + (3.f*cu18 + 4.5f*cu18*cu18 - 1.5f*u2) * (dRho + 1.f));
	feq[19] = w3 * (dRho + (3.f*cu19 + 4.5f*cu19*cu19 - 1.5f*u2) * (dRho + 1.f));
	feq[20] = w3 * (dRho + (3.f*cu20 + 4.5f*cu20*cu20 - 1.5f*u2) * (dRho + 1.f));
	feq[21] = w3 * (dRho + (3.f*cu21 + 4.5f*cu21*cu21 - 1.5f*u2) * (dRho + 1.f));
	feq[22] = w3 * (dRho + (3.f*cu22 + 4.5f*cu22*cu22 - 1.5f*u2) * (dRho + 1.f));
	feq[23] = w3 * (dRho + (3.f*cu23 + 4.5f*cu23*cu23 - 1.5f*u2) * (dRho + 1.f));
	feq[24] = w3 * (dRho + (3.f*cu24 + 4.5f*cu24*cu24 - 1.5f*u2) * (dRho + 1.f));
	feq[25] = w3 * (dRho + (3.f*cu25 + 4.5f*cu25*cu25 - 1.5f*u2) * (dRho + 1.f));
	feq[26] = w3 * (dRho + (3.f*cu26 + 4.5f*cu26*cu26 - 1.5f*u2) * (dRho + 1.f));	
	/*
	const float weights[27] = { 8.f/27.f, 
		2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
		1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
		1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
	for ( int direction = 0; direction < 27; direction++ ) feq[direction] += weights[direction];
	*/
}

__host__ __device__ void getFneq(const float (&f)[27], const float (&feq)[27], float (&fneq)[27])
{
	for ( int i = 0; i < 27; i++ ) fneq[i] = f[i] - feq[i];
}

__host__ __device__ void getRhoUxUyUz(
	float &rho, float &ux, float &uy, float &uz, 
	float (&f)[27] // const float (&f)[27]
	)
{
	/*
	const float weights[27] = { 8.f/27.f, 
		2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
		1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
		1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
	for ( int direction = 0; direction < 27; direction++ ) f[direction] -= weights[direction];
	*/
	const float dRho = (((f[PPP]+f[MMM]) + (f[PMP]+f[MPM])) + ((f[PPM]+f[MMP]) + (f[PPM]+f[MMP])))
					  + (((f[OPP]+f[OMM]) + (f[OPM]+f[OMP])) + ((f[POP]+f[MOM]) + (f[POM]+f[MOP])) + ((f[PPO]+f[MMO]) + (f[PMO]+f[MPO])))
						+ ((f[POO]+f[MOO]) + (f[OPO]+f[OMO]) + (f[OOP]+f[OOM])) + f[OOO];			
    
    const float rhoInv = 1.f / (dRho + 1.f);
    
    const float momentumX = ((((f[PPP]-f[MMM]) + (f[PMP]-f[MPM])) + ((f[PPM]-f[MMP]) + (f[PMM]-f[MPP])))
                          + (((f[POP]-f[MOM]) + (f[POM]-f[MOP])) + ((f[PPO]-f[MMO]) + (f[PMO]-f[MPO])))
                            + (f[POO]-f[MOO]));

    const float momentumY = ((((f[PPP]-f[MMM]) - (f[PMP]-f[MPM])) + ((f[PPM]-f[MMP]) - (f[PMM]-f[MPP])))
                          + (((f[OPP]-f[OMM]) + (f[OPM]-f[OMP])) + ((f[PPO]-f[MMO]) - (f[PMO]-f[MPO])))
                            + (f[OPO]-f[OMO]));

    const float momentumZ = ((((f[PPP]-f[MMM]) + (f[PMP]-f[MPM])) - ((f[PPM]-f[MMP]) + (f[PMM]-f[MPP])))
                          + (((f[OPP]-f[OMM]) - (f[OPM]-f[OMP])) + ((f[POP]-f[MOM]) - (f[POM]-f[MOP])))
                            + (f[OOP]-f[OOM]));
	rho = dRho + 1.f;
    ux = momentumX * rhoInv;
    uy = momentumY * rhoInv;
    uz = momentumZ * rhoInv;
    
    //for ( int direction = 0; direction < 27; direction++ ) f[direction] += weights[direction];
}

__host__ __device__ void convertToPhysicalVelocity( float &ux, float &uy, float &uz, const InfoStruct &Info )
{
	ux = ux * (Info.res/1000.f) / Info.dtPhys;
	uy = uy * (Info.res/1000.f) / Info.dtPhys;
	uz = uz * (Info.res/1000.f) / Info.dtPhys;
}

__host__ __device__ void convertToPhysicalPressure( float &rho )
{
	// converts LBM rho to physical pressure, overwrites the variable (LBM rho -> physical p)
	const float p = (rho - 1.f) * rhoNominalPhys * soundspeedPhys * soundspeedPhys;
	rho = p;
}

__host__ __device__ void convertToPhysicalPressure( float &rho, const InfoStruct &Info )
{
	// converts LBM rho to physical pressure, overwrites the variable (LBM rho -> physical p)
	const float p = (rho - 1.f) * rhoNominalPhys * soundspeedPhys * soundspeedPhys;
	rho = p;
}

__host__ __device__ void convertToPhysicalForce( float &gx, float &gy, float &gz, const InfoStruct &Info )
{
	gx = gx * rhoNominalPhys * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
	gy = gy * rhoNominalPhys * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
	gz = gz * rhoNominalPhys * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
}

__host__ __device__ void getLocalDu( float (&f)[27], const float &nu, LocalDuStruct &localDu )
{
	/*
	const float weights[27] = { 8.f/27.f, 
		2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
		1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
		1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
	for ( int direction = 0; direction < 27; direction++ ) f[direction] -= weights[direction];
	*/
	
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

    // First part of the central-moment transformation, zeroth z moments.
    const float k_aa0 = (f[MMP] + f[MMM]) + f[MMO];
    const float k_ab0 = (f[MOP] + f[MOM]) + f[MOO];
    const float k_ac0 = (f[MPP] + f[MPM]) + f[MPO];
    const float k_ba0 = (f[OMP] + f[OMM]) + f[OMO];
    const float k_bb0 = (f[OOP] + f[OOM]) + f[OOO];
    const float k_bc0 = (f[OPP] + f[OPM]) + f[OPO];
    const float k_ca0 = (f[PMP] + f[PMM]) + f[PMO];
    const float k_cb0 = (f[POP] + f[POM]) + f[POO];
    const float k_cc0 = (f[PPP] + f[PPM]) + f[PPO];

    // The stored density is dRho. The physical density is 1 + dRho.
    const float dRho =
        ((k_aa0 + k_ab0) + k_ac0) +
        ((k_ba0 + k_bb0) + k_bc0) +
        ((k_ca0 + k_cb0) + k_cc0);
    const float rho = 1.f + dRho;
    const float rhoInv = 1.f / rho;

    // The weights carry no momentum, so momentum is obtained directly from f_bar.
    const float momentumX =
        ((k_ca0 + k_cb0) + k_cc0) - ((k_aa0 + k_ab0) + k_ac0);
    const float momentumY =
        ((k_ac0 + k_bc0) + k_cc0) - ((k_aa0 + k_ba0) + k_ca0);
    const float momentumZ =
        (((f[MMP] - f[MMM]) + (f[MOP] - f[MOM])) + ((f[MPP] - f[MPM]) + (f[OMP] - f[OMM]))) +
        (((f[OOP] - f[OOM]) + (f[OPP] - f[OPM])) + ((f[PMP] - f[PMM]) + (f[POP] - f[POM]))) +
        (f[PPP] - f[PPM]);

    // Apply the first half of the body force.
    const float ux = momentumX * rhoInv;
    const float uy = momentumY * rhoInv;
    const float uz = momentumZ * rhoInv;

    const float ux2 = ux * ux;
    const float uy2 = uy * uy;
    const float uz2 = uz * uz;

    // -------------------------------------------------------------------------
    // Well-conditioned populations -> central moments, Geier 2017, Eqs. 6-14.
    // -------------------------------------------------------------------------

    const float k_aa1 = (f[MMP] - f[MMM]) - uz * (k_aa0 + K_aa0);
    const float k_ab1 = (f[MOP] - f[MOM]) - uz * (k_ab0 + K_ab0);
    const float k_ac1 = (f[MPP] - f[MPM]) - uz * (k_ac0 + K_ac0);
    const float k_ba1 = (f[OMP] - f[OMM]) - uz * (k_ba0 + K_ba0);
    const float k_bb1 = (f[OOP] - f[OOM]) - uz * (k_bb0 + K_bb0);
    const float k_bc1 = (f[OPP] - f[OPM]) - uz * (k_bc0 + K_bc0);
    const float k_ca1 = (f[PMP] - f[PMM]) - uz * (k_ca0 + K_ca0);
    const float k_cb1 = (f[POP] - f[POM]) - uz * (k_cb0 + K_cb0);
    const float k_cc1 = (f[PPP] - f[PPM]) - uz * (k_cc0 + K_cc0);

    const float k_aa2 = (f[MMP] + f[MMM]) - 2.f * uz * (f[MMP] - f[MMM]) + uz2 * (k_aa0 + K_aa0);
    const float k_ab2 = (f[MOP] + f[MOM]) - 2.f * uz * (f[MOP] - f[MOM]) + uz2 * (k_ab0 + K_ab0);
    const float k_ac2 = (f[MPP] + f[MPM]) - 2.f * uz * (f[MPP] - f[MPM]) + uz2 * (k_ac0 + K_ac0);
    const float k_ba2 = (f[OMP] + f[OMM]) - 2.f * uz * (f[OMP] - f[OMM]) + uz2 * (k_ba0 + K_ba0);
    const float k_bb2 = (f[OOP] + f[OOM]) - 2.f * uz * (f[OOP] - f[OOM]) + uz2 * (k_bb0 + K_bb0);
    const float k_bc2 = (f[OPP] + f[OPM]) - 2.f * uz * (f[OPP] - f[OPM]) + uz2 * (k_bc0 + K_bc0);
    const float k_ca2 = (f[PMP] + f[PMM]) - 2.f * uz * (f[PMP] - f[PMM]) + uz2 * (k_ca0 + K_ca0);
    const float k_cb2 = (f[POP] + f[POM]) - 2.f * uz * (f[POP] - f[POM]) + uz2 * (k_cb0 + K_cb0);
    const float k_cc2 = (f[PPP] + f[PPM]) - 2.f * uz * (f[PPP] - f[PPM]) + uz2 * (k_cc0 + K_cc0);

    const float k_a00 = (k_ac0 + k_aa0) + k_ab0;
    const float k_b00 = (k_bc0 + k_ba0) + k_bb0;
    const float k_c00 = (k_cc0 + k_ca0) + k_cb0;
    const float k_a01 = (k_ac1 + k_aa1) + k_ab1;
    const float k_b01 = (k_bc1 + k_ba1) + k_bb1;
    const float k_c01 = (k_cc1 + k_ca1) + k_cb1;
    const float k_a02 = (k_ac2 + k_aa2) + k_ab2;
    const float k_b02 = (k_bc2 + k_ba2) + k_bb2;
    const float k_c02 = (k_cc2 + k_ca2) + k_cb2;

    const float k_a10 = (k_ac0 - k_aa0) - uy * (k_a00 + K_a00);
    const float k_b10 = (k_bc0 - k_ba0) - uy * (k_b00 + K_b00);
    const float k_c10 = (k_cc0 - k_ca0) - uy * (k_c00 + K_c00);
    const float k_a11 = (k_ac1 - k_aa1) - uy * k_a01;
    const float k_b11 = (k_bc1 - k_ba1) - uy * k_b01;
    const float k_c11 = (k_cc1 - k_ca1) - uy * k_c01;

    const float k_a20 = (k_ac0 + k_aa0) - 2.f * uy * (k_ac0 - k_aa0) + uy2 * (k_a00 + K_a00);
    const float k_b20 = (k_bc0 + k_ba0) - 2.f * uy * (k_bc0 - k_ba0) + uy2 * (k_b00 + K_b00);
    const float k_c20 = (k_cc0 + k_ca0) - 2.f * uy * (k_cc0 - k_ca0) + uy2 * (k_c00 + K_c00);

    // Use the same stored-density sum used for rho so collision and back-transform
    // conserve exactly the same zeroth-order quantity in finite precision.
    const float k_000 = dRho;
    const float k_001 = (k_c01 + k_a01) + k_b01;
    const float k_002 = (k_c02 + k_a02) + k_b02;
    const float k_010 = (k_c10 + k_a10) + k_b10;
    const float k_011 = (k_c11 + k_a11) + k_b11;
    const float k_020 = (k_c20 + k_a20) + k_b20;

    const float k_101 = (k_c01 - k_a01) - ux * k_001;
    const float k_110 = (k_c10 - k_a10) - ux * k_010;

    const float k_200 = (k_c00 + k_a00) - 2.f * ux * (k_c00 - k_a00) + ux2 * (k_000 + 1.f);

    // -------------------------------------------------------------------------
    // Central moments -> cumulants, Geier 2017, Eqs. 16-23.
    // -------------------------------------------------------------------------

    const float C_110 = k_110;
    const float C_101 = k_101;
    const float C_011 = k_011;
    const float C_200 = k_200;
    const float C_020 = k_020;
    const float C_002 = k_002;
        
    const float omega1 = 1.f / (3.f * nu + 0.5f); //= 1.f / (3.f * (nu * BC.nuMultiplier) + 0.5f);
	
	localDu.duxdx = ( (0.5f * omega1) * (-2.f * C_200 + C_020 + C_002) + 0.5f * (rho - C_200 - C_020 - C_002) ) / rho;
	localDu.duydy = localDu.duxdx + (1.5f * omega1) * (C_200 - C_020) / rho;
	localDu.duzdz = localDu.duxdx + (1.5f * omega1) * (C_200 - C_002) / rho;
	localDu.duxdyCross = - (3.f * omega1) * C_110 / rho;
	localDu.duydzCross = - (3.f * omega1) * C_011 / rho;
	localDu.duxdzCross = - (3.f * omega1) * C_101 / rho;
}
