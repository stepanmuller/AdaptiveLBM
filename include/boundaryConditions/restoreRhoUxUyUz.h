__host__ __device__ void restoreRhoUxUyUz(
	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,
	float &rho, float &ux, float &uy, float &uz,
	const float (&f)[27]
)
{
	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);
	if ( normalCode == 655 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (3.f/2.f) * f[0] + (3.f/2.f) * f[3] + (3.f/2.f) * f[4] + (3.f/2.f) * f[5] + (3.f/2.f) * f[6] + (3.f/2.f) * f[13] + (3.f/2.f) * f[14] + (3.f/2.f) * f[17] + (3.f/2.f) * f[18];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] + (-1.f/2.f) * f[3] + (-1.f/2.f) * f[4] + (-1.f/2.f) * f[5] + (-1.f/2.f) * f[6] + (2.f) * f[7] + (2.f) * f[9] + (2.f) * f[12] + (-1.f/2.f) * f[13] + (-1.f/2.f) * f[14] + (2.f) * f[16] + (-1.f/2.f) * f[17] + (-1.f/2.f) * f[18] + (2.f) * f[20] + (2.f) * f[22] + (2.f) * f[23] + (2.f) * f[26];
		const float q2 = - f[5] + f[6] + (2.f) * f[12] + f[13] - f[14] + (-2.f) * f[16] + f[17] - f[18] + (-2.f) * f[20] + (2.f) * f[22] + (-2.f) * f[23] + (2.f) * f[26];
		const float q3 = - f[3] + f[4] + (-2.f) * f[7] + (2.f) * f[9] - f[13] + f[14] + f[17] - f[18] + (2.f) * f[20] + (-2.f) * f[22] + (-2.f) * f[23] + (2.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 565 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (3.f/2.f) * f[0] + (3.f/2.f) * f[1] + (3.f/2.f) * f[2] + (3.f/2.f) * f[3] + (3.f/2.f) * f[4] + (3.f/2.f) * f[7] + (3.f/2.f) * f[8] + (3.f/2.f) * f[9] + (3.f/2.f) * f[10];
		const float q1 = + f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + (2.f) * f[12] + (-2.f) * f[15] + (-2.f) * f[19] + (2.f) * f[22] + (-2.f) * f[24] + (2.f) * f[26];
		const float q2 = + (-1.f/2.f) * f[0] + (-1.f/2.f) * f[1] + (-1.f/2.f) * f[2] + (-1.f/2.f) * f[3] + (-1.f/2.f) * f[4] + (2.f) * f[6] + (-1.f/2.f) * f[7] + (-1.f/2.f) * f[8] + (-1.f/2.f) * f[9] + (-1.f/2.f) * f[10] + (2.f) * f[12] + (2.f) * f[13] + (2.f) * f[15] + (2.f) * f[17] + (2.f) * f[19] + (2.f) * f[22] + (2.f) * f[24] + (2.f) * f[26];
		const float q3 = - f[3] + f[4] - f[7] + f[8] + f[9] - f[10] + (-2.f) * f[13] + (2.f) * f[17] + (-2.f) * f[19] + (-2.f) * f[22] + (2.f) * f[24] + (2.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 556 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (3.f/2.f) * f[0] + (3.f/2.f) * f[1] + (3.f/2.f) * f[2] + (3.f/2.f) * f[5] + (3.f/2.f) * f[6] + (3.f/2.f) * f[11] + (3.f/2.f) * f[12] + (3.f/2.f) * f[15] + (3.f/2.f) * f[16];
		const float q1 = + f[1] - f[2] + (-2.f) * f[8] + (2.f) * f[9] - f[11] + f[12] - f[15] + f[16] + (2.f) * f[20] + (-2.f) * f[21] + (-2.f) * f[24] + (2.f) * f[26];
		const float q2 = - f[5] + f[6] - f[11] + f[12] + (-2.f) * f[14] + f[15] - f[16] + (2.f) * f[17] + (-2.f) * f[20] + (-2.f) * f[21] + (2.f) * f[24] + (2.f) * f[26];
		const float q3 = + (-1.f/2.f) * f[0] + (-1.f/2.f) * f[1] + (-1.f/2.f) * f[2] + (2.f) * f[4] + (-1.f/2.f) * f[5] + (-1.f/2.f) * f[6] + (2.f) * f[8] + (2.f) * f[9] + (-1.f/2.f) * f[11] + (-1.f/2.f) * f[12] + (2.f) * f[14] + (-1.f/2.f) * f[15] + (-1.f/2.f) * f[16] + (2.f) * f[17] + (2.f) * f[20] + (2.f) * f[21] + (2.f) * f[24] + (2.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 455 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (3.f/2.f) * f[0] + (3.f/2.f) * f[3] + (3.f/2.f) * f[4] + (3.f/2.f) * f[5] + (3.f/2.f) * f[6] + (3.f/2.f) * f[13] + (3.f/2.f) * f[14] + (3.f/2.f) * f[17] + (3.f/2.f) * f[18];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + (1.f/2.f) * f[3] + (1.f/2.f) * f[4] + (1.f/2.f) * f[5] + (1.f/2.f) * f[6] + (-2.f) * f[8] + (-2.f) * f[10] + (-2.f) * f[11] + (1.f/2.f) * f[13] + (1.f/2.f) * f[14] + (-2.f) * f[15] + (1.f/2.f) * f[17] + (1.f/2.f) * f[18] + (-2.f) * f[19] + (-2.f) * f[21] + (-2.f) * f[24] + (-2.f) * f[25];
		const float q2 = - f[5] + f[6] + (-2.f) * f[11] + f[13] - f[14] + (2.f) * f[15] + f[17] - f[18] + (2.f) * f[19] + (-2.f) * f[21] + (2.f) * f[24] + (-2.f) * f[25];
		const float q3 = - f[3] + f[4] + (2.f) * f[8] + (-2.f) * f[10] - f[13] + f[14] + f[17] - f[18] + (-2.f) * f[19] + (2.f) * f[21] + (2.f) * f[24] + (-2.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 545 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (3.f/2.f) * f[0] + (3.f/2.f) * f[1] + (3.f/2.f) * f[2] + (3.f/2.f) * f[3] + (3.f/2.f) * f[4] + (3.f/2.f) * f[7] + (3.f/2.f) * f[8] + (3.f/2.f) * f[9] + (3.f/2.f) * f[10];
		const float q1 = + f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + (-2.f) * f[11] + (2.f) * f[16] + (2.f) * f[20] + (-2.f) * f[21] + (2.f) * f[23] + (-2.f) * f[25];
		const float q2 = + (1.f/2.f) * f[0] + (1.f/2.f) * f[1] + (1.f/2.f) * f[2] + (1.f/2.f) * f[3] + (1.f/2.f) * f[4] + (-2.f) * f[5] + (1.f/2.f) * f[7] + (1.f/2.f) * f[8] + (1.f/2.f) * f[9] + (1.f/2.f) * f[10] + (-2.f) * f[11] + (-2.f) * f[14] + (-2.f) * f[16] + (-2.f) * f[18] + (-2.f) * f[20] + (-2.f) * f[21] + (-2.f) * f[23] + (-2.f) * f[25];
		const float q3 = - f[3] + f[4] - f[7] + f[8] + f[9] - f[10] + (2.f) * f[14] + (-2.f) * f[18] + (2.f) * f[20] + (2.f) * f[21] + (-2.f) * f[23] + (-2.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 554 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (3.f/2.f) * f[0] + (3.f/2.f) * f[1] + (3.f/2.f) * f[2] + (3.f/2.f) * f[5] + (3.f/2.f) * f[6] + (3.f/2.f) * f[11] + (3.f/2.f) * f[12] + (3.f/2.f) * f[15] + (3.f/2.f) * f[16];
		const float q1 = + f[1] - f[2] + (2.f) * f[7] + (-2.f) * f[10] - f[11] + f[12] - f[15] + f[16] + (-2.f) * f[19] + (2.f) * f[22] + (2.f) * f[23] + (-2.f) * f[25];
		const float q2 = - f[5] + f[6] - f[11] + f[12] + (2.f) * f[13] + f[15] - f[16] + (-2.f) * f[18] + (2.f) * f[19] + (2.f) * f[22] + (-2.f) * f[23] + (-2.f) * f[25];
		const float q3 = + (1.f/2.f) * f[0] + (1.f/2.f) * f[1] + (1.f/2.f) * f[2] + (-2.f) * f[3] + (1.f/2.f) * f[5] + (1.f/2.f) * f[6] + (-2.f) * f[7] + (-2.f) * f[10] + (1.f/2.f) * f[11] + (1.f/2.f) * f[12] + (-2.f) * f[13] + (1.f/2.f) * f[15] + (1.f/2.f) * f[16] + (-2.f) * f[18] + (-2.f) * f[19] + (-2.f) * f[22] + (-2.f) * f[23] + (-2.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 566 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + (2.f) * f[1] + (2.f) * f[2] + f[4] + f[6] + f[8] + f[9] + f[12] + f[15] + (-4.f) * f[17] + (-4.f) * f[24] + (-4.f) * f[26];
		const float q1 = + f[1] - f[2] + (-2.f) * f[8] + (2.f) * f[9] + (2.f) * f[12] + (-2.f) * f[15] + (-4.f) * f[24] + (4.f) * f[26];
		const float q2 = + (-1.f/2.f) * f[0] + (-1.f/2.f) * f[1] + (-1.f/2.f) * f[2] - f[4] + (2.f) * f[6] - f[8] - f[9] + (2.f) * f[12] + (2.f) * f[15] + (4.f) * f[17] + (4.f) * f[24] + (4.f) * f[26];
		const float q3 = + (-1.f/2.f) * f[0] + (-1.f/2.f) * f[1] + (-1.f/2.f) * f[2] + (2.f) * f[4] - f[6] + (2.f) * f[8] + (2.f) * f[9] - f[12] - f[15] + (4.f) * f[17] + (4.f) * f[24] + (4.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 656 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[1] + f[4] + (2.f) * f[5] + (2.f) * f[6] + (-4.f) * f[9] + f[12] + f[14] + f[16] + f[17] + (-4.f) * f[20] + (-4.f) * f[26];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] - f[4] + (-1.f/2.f) * f[5] + (-1.f/2.f) * f[6] + (4.f) * f[9] + (2.f) * f[12] - f[14] + (2.f) * f[16] - f[17] + (4.f) * f[20] + (4.f) * f[26];
		const float q2 = - f[5] + f[6] + (2.f) * f[12] + (-2.f) * f[14] + (-2.f) * f[16] + (2.f) * f[17] + (-4.f) * f[20] + (4.f) * f[26];
		const float q3 = + (-1.f/2.f) * f[0] - f[1] + (2.f) * f[4] + (-1.f/2.f) * f[5] + (-1.f/2.f) * f[6] + (4.f) * f[9] - f[12] + (2.f) * f[14] - f[16] + (2.f) * f[17] + (4.f) * f[20] + (4.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 665 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[1] + (2.f) * f[3] + (2.f) * f[4] + f[6] + f[7] + f[9] + (-4.f) * f[12] + f[13] + f[17] + (-4.f) * f[22] + (-4.f) * f[26];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] + (-1.f/2.f) * f[3] + (-1.f/2.f) * f[4] - f[6] + (2.f) * f[7] + (2.f) * f[9] + (4.f) * f[12] - f[13] - f[17] + (4.f) * f[22] + (4.f) * f[26];
		const float q2 = + (-1.f/2.f) * f[0] - f[1] + (-1.f/2.f) * f[3] + (-1.f/2.f) * f[4] + (2.f) * f[6] - f[7] - f[9] + (4.f) * f[12] + (2.f) * f[13] + (2.f) * f[17] + (4.f) * f[22] + (4.f) * f[26];
		const float q3 = - f[3] + f[4] + (-2.f) * f[7] + (2.f) * f[9] + (-2.f) * f[13] + (2.f) * f[17] + (-4.f) * f[22] + (4.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 544 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + (2.f) * f[1] + (2.f) * f[2] + f[3] + f[5] + f[7] + f[10] + f[11] + f[16] + (-4.f) * f[18] + (-4.f) * f[23] + (-4.f) * f[25];
		const float q1 = + f[1] - f[2] + (2.f) * f[7] + (-2.f) * f[10] + (-2.f) * f[11] + (2.f) * f[16] + (4.f) * f[23] + (-4.f) * f[25];
		const float q2 = + (1.f/2.f) * f[0] + (1.f/2.f) * f[1] + (1.f/2.f) * f[2] + f[3] + (-2.f) * f[5] + f[7] + f[10] + (-2.f) * f[11] + (-2.f) * f[16] + (-4.f) * f[18] + (-4.f) * f[23] + (-4.f) * f[25];
		const float q3 = + (1.f/2.f) * f[0] + (1.f/2.f) * f[1] + (1.f/2.f) * f[2] + (-2.f) * f[3] + f[5] + (-2.f) * f[7] + (-2.f) * f[10] + f[11] + f[16] + (-4.f) * f[18] + (-4.f) * f[23] + (-4.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 454 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[2] + f[3] + (2.f) * f[5] + (2.f) * f[6] + (-4.f) * f[10] + f[11] + f[13] + f[15] + f[18] + (-4.f) * f[19] + (-4.f) * f[25];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + f[3] + (1.f/2.f) * f[5] + (1.f/2.f) * f[6] + (-4.f) * f[10] + (-2.f) * f[11] + f[13] + (-2.f) * f[15] + f[18] + (-4.f) * f[19] + (-4.f) * f[25];
		const float q2 = - f[5] + f[6] + (-2.f) * f[11] + (2.f) * f[13] + (2.f) * f[15] + (-2.f) * f[18] + (4.f) * f[19] + (-4.f) * f[25];
		const float q3 = + (1.f/2.f) * f[0] + f[2] + (-2.f) * f[3] + (1.f/2.f) * f[5] + (1.f/2.f) * f[6] + (-4.f) * f[10] + f[11] + (-2.f) * f[13] + f[15] + (-2.f) * f[18] + (-4.f) * f[19] + (-4.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 445 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[2] + (2.f) * f[3] + (2.f) * f[4] + f[5] + f[8] + f[10] + (-4.f) * f[11] + f[14] + f[18] + (-4.f) * f[21] + (-4.f) * f[25];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + (1.f/2.f) * f[3] + (1.f/2.f) * f[4] + f[5] + (-2.f) * f[8] + (-2.f) * f[10] + (-4.f) * f[11] + f[14] + f[18] + (-4.f) * f[21] + (-4.f) * f[25];
		const float q2 = + (1.f/2.f) * f[0] + f[2] + (1.f/2.f) * f[3] + (1.f/2.f) * f[4] + (-2.f) * f[5] + f[8] + f[10] + (-4.f) * f[11] + (-2.f) * f[14] + (-2.f) * f[18] + (-4.f) * f[21] + (-4.f) * f[25];
		const float q3 = - f[3] + f[4] + (2.f) * f[8] + (-2.f) * f[10] + (2.f) * f[14] + (-2.f) * f[18] + (4.f) * f[21] + (-4.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 564 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + (2.f) * f[1] + (2.f) * f[2] + f[3] + f[6] + f[7] + f[10] + f[12] + (-4.f) * f[13] + f[15] + (-4.f) * f[19] + (-4.f) * f[22];
		const float q1 = + f[1] - f[2] + (2.f) * f[7] + (-2.f) * f[10] + (2.f) * f[12] + (-2.f) * f[15] + (-4.f) * f[19] + (4.f) * f[22];
		const float q2 = + (-1.f/2.f) * f[0] + (-1.f/2.f) * f[1] + (-1.f/2.f) * f[2] - f[3] + (2.f) * f[6] - f[7] - f[10] + (2.f) * f[12] + (4.f) * f[13] + (2.f) * f[15] + (4.f) * f[19] + (4.f) * f[22];
		const float q3 = + (1.f/2.f) * f[0] + (1.f/2.f) * f[1] + (1.f/2.f) * f[2] + (-2.f) * f[3] + f[6] + (-2.f) * f[7] + (-2.f) * f[10] + f[12] + (-4.f) * f[13] + f[15] + (-4.f) * f[19] + (-4.f) * f[22];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 654 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[1] + f[3] + (2.f) * f[5] + (2.f) * f[6] + (-4.f) * f[7] + f[12] + f[13] + f[16] + f[18] + (-4.f) * f[22] + (-4.f) * f[23];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] - f[3] + (-1.f/2.f) * f[5] + (-1.f/2.f) * f[6] + (4.f) * f[7] + (2.f) * f[12] - f[13] + (2.f) * f[16] - f[18] + (4.f) * f[22] + (4.f) * f[23];
		const float q2 = - f[5] + f[6] + (2.f) * f[12] + (2.f) * f[13] + (-2.f) * f[16] + (-2.f) * f[18] + (4.f) * f[22] + (-4.f) * f[23];
		const float q3 = + (1.f/2.f) * f[0] + f[1] + (-2.f) * f[3] + (1.f/2.f) * f[5] + (1.f/2.f) * f[6] + (-4.f) * f[7] + f[12] + (-2.f) * f[13] + f[16] + (-2.f) * f[18] + (-4.f) * f[22] + (-4.f) * f[23];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 645 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[1] + (2.f) * f[3] + (2.f) * f[4] + f[5] + f[7] + f[9] + f[14] + (-4.f) * f[16] + f[18] + (-4.f) * f[20] + (-4.f) * f[23];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] + (-1.f/2.f) * f[3] + (-1.f/2.f) * f[4] - f[5] + (2.f) * f[7] + (2.f) * f[9] - f[14] + (4.f) * f[16] - f[18] + (4.f) * f[20] + (4.f) * f[23];
		const float q2 = + (1.f/2.f) * f[0] + f[1] + (1.f/2.f) * f[3] + (1.f/2.f) * f[4] + (-2.f) * f[5] + f[7] + f[9] + (-2.f) * f[14] + (-4.f) * f[16] + (-2.f) * f[18] + (-4.f) * f[20] + (-4.f) * f[23];
		const float q3 = - f[3] + f[4] + (-2.f) * f[7] + (2.f) * f[9] + (2.f) * f[14] + (-2.f) * f[18] + (4.f) * f[20] + (-4.f) * f[23];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 546 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + (2.f) * f[1] + (2.f) * f[2] + f[4] + f[5] + f[8] + f[9] + f[11] + (-4.f) * f[14] + f[16] + (-4.f) * f[20] + (-4.f) * f[21];
		const float q1 = + f[1] - f[2] + (-2.f) * f[8] + (2.f) * f[9] + (-2.f) * f[11] + (2.f) * f[16] + (4.f) * f[20] + (-4.f) * f[21];
		const float q2 = + (1.f/2.f) * f[0] + (1.f/2.f) * f[1] + (1.f/2.f) * f[2] + f[4] + (-2.f) * f[5] + f[8] + f[9] + (-2.f) * f[11] + (-4.f) * f[14] + (-2.f) * f[16] + (-4.f) * f[20] + (-4.f) * f[21];
		const float q3 = + (-1.f/2.f) * f[0] + (-1.f/2.f) * f[1] + (-1.f/2.f) * f[2] + (2.f) * f[4] - f[5] + (2.f) * f[8] + (2.f) * f[9] - f[11] + (4.f) * f[14] - f[16] + (4.f) * f[20] + (4.f) * f[21];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 456 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[2] + f[4] + (2.f) * f[5] + (2.f) * f[6] + (-4.f) * f[8] + f[11] + f[14] + f[15] + f[17] + (-4.f) * f[21] + (-4.f) * f[24];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + f[4] + (1.f/2.f) * f[5] + (1.f/2.f) * f[6] + (-4.f) * f[8] + (-2.f) * f[11] + f[14] + (-2.f) * f[15] + f[17] + (-4.f) * f[21] + (-4.f) * f[24];
		const float q2 = - f[5] + f[6] + (-2.f) * f[11] + (-2.f) * f[14] + (2.f) * f[15] + (2.f) * f[17] + (-4.f) * f[21] + (4.f) * f[24];
		const float q3 = + (-1.f/2.f) * f[0] - f[2] + (2.f) * f[4] + (-1.f/2.f) * f[5] + (-1.f/2.f) * f[6] + (4.f) * f[8] - f[11] + (2.f) * f[14] - f[15] + (2.f) * f[17] + (4.f) * f[21] + (4.f) * f[24];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 465 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (2.f) * f[0] + f[2] + (2.f) * f[3] + (2.f) * f[4] + f[6] + f[8] + f[10] + f[13] + (-4.f) * f[15] + f[17] + (-4.f) * f[19] + (-4.f) * f[24];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + (1.f/2.f) * f[3] + (1.f/2.f) * f[4] + f[6] + (-2.f) * f[8] + (-2.f) * f[10] + f[13] + (-4.f) * f[15] + f[17] + (-4.f) * f[19] + (-4.f) * f[24];
		const float q2 = + (-1.f/2.f) * f[0] - f[2] + (-1.f/2.f) * f[3] + (-1.f/2.f) * f[4] + (2.f) * f[6] - f[8] - f[10] + (2.f) * f[13] + (4.f) * f[15] + (2.f) * f[17] + (4.f) * f[19] + (4.f) * f[24];
		const float q3 = - f[3] + f[4] + (2.f) * f[8] + (-2.f) * f[10] + (-2.f) * f[13] + (2.f) * f[17] + (-4.f) * f[19] + (4.f) * f[24];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 666 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[1] + (2.f) * f[4] + (2.f) * f[6] + (-2.f) * f[9] + (-2.f) * f[12] + (-2.f) * f[17] + (-16.f) * f[26];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] - f[4] - f[6] + (4.f) * f[9] + (4.f) * f[12] + (-2.f) * f[17] + (8.f) * f[26];
		const float q2 = + (-1.f/2.f) * f[0] - f[1] - f[4] + (2.f) * f[6] + (-2.f) * f[9] + (4.f) * f[12] + (4.f) * f[17] + (8.f) * f[26];
		const float q3 = + (-1.f/2.f) * f[0] - f[1] + (2.f) * f[4] - f[6] + (4.f) * f[9] + (-2.f) * f[12] + (4.f) * f[17] + (8.f) * f[26];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 444 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[2] + (2.f) * f[3] + (2.f) * f[5] + (-2.f) * f[10] + (-2.f) * f[11] + (-2.f) * f[18] + (-16.f) * f[25];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + f[3] + f[5] + (-4.f) * f[10] + (-4.f) * f[11] + (2.f) * f[18] + (-8.f) * f[25];
		const float q2 = + (1.f/2.f) * f[0] + f[2] + f[3] + (-2.f) * f[5] + (2.f) * f[10] + (-4.f) * f[11] + (-4.f) * f[18] + (-8.f) * f[25];
		const float q3 = + (1.f/2.f) * f[0] + f[2] + (-2.f) * f[3] + f[5] + (-4.f) * f[10] + (2.f) * f[11] + (-4.f) * f[18] + (-8.f) * f[25];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 466 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[2] + (2.f) * f[4] + (2.f) * f[6] + (-2.f) * f[8] + (-2.f) * f[15] + (-2.f) * f[17] + (-16.f) * f[24];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + f[4] + f[6] + (-4.f) * f[8] + (-4.f) * f[15] + (2.f) * f[17] + (-8.f) * f[24];
		const float q2 = + (-1.f/2.f) * f[0] - f[2] - f[4] + (2.f) * f[6] + (-2.f) * f[8] + (4.f) * f[15] + (4.f) * f[17] + (8.f) * f[24];
		const float q3 = + (-1.f/2.f) * f[0] - f[2] + (2.f) * f[4] - f[6] + (4.f) * f[8] + (-2.f) * f[15] + (4.f) * f[17] + (8.f) * f[24];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 646 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[1] + (2.f) * f[4] + (2.f) * f[5] + (-2.f) * f[9] + (-2.f) * f[14] + (-2.f) * f[16] + (-16.f) * f[20];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] - f[4] - f[5] + (4.f) * f[9] + (-2.f) * f[14] + (4.f) * f[16] + (8.f) * f[20];
		const float q2 = + (1.f/2.f) * f[0] + f[1] + f[4] + (-2.f) * f[5] + (2.f) * f[9] + (-4.f) * f[14] + (-4.f) * f[16] + (-8.f) * f[20];
		const float q3 = + (-1.f/2.f) * f[0] - f[1] + (2.f) * f[4] - f[5] + (4.f) * f[9] + (4.f) * f[14] + (-2.f) * f[16] + (8.f) * f[20];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 664 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[1] + (2.f) * f[3] + (2.f) * f[6] + (-2.f) * f[7] + (-2.f) * f[12] + (-2.f) * f[13] + (-16.f) * f[22];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] - f[3] - f[6] + (4.f) * f[7] + (4.f) * f[12] + (-2.f) * f[13] + (8.f) * f[22];
		const float q2 = + (-1.f/2.f) * f[0] - f[1] - f[3] + (2.f) * f[6] + (-2.f) * f[7] + (4.f) * f[12] + (4.f) * f[13] + (8.f) * f[22];
		const float q3 = + (1.f/2.f) * f[0] + f[1] + (-2.f) * f[3] + f[6] + (-4.f) * f[7] + (2.f) * f[12] + (-4.f) * f[13] + (-8.f) * f[22];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 644 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[1] + (2.f) * f[3] + (2.f) * f[5] + (-2.f) * f[7] + (-2.f) * f[16] + (-2.f) * f[18] + (-16.f) * f[23];
		const float q1 = + (-1.f/2.f) * f[0] + (2.f) * f[1] - f[3] - f[5] + (4.f) * f[7] + (4.f) * f[16] + (-2.f) * f[18] + (8.f) * f[23];
		const float q2 = + (1.f/2.f) * f[0] + f[1] + f[3] + (-2.f) * f[5] + (2.f) * f[7] + (-4.f) * f[16] + (-4.f) * f[18] + (-8.f) * f[23];
		const float q3 = + (1.f/2.f) * f[0] + f[1] + (-2.f) * f[3] + f[5] + (-4.f) * f[7] + (2.f) * f[16] + (-4.f) * f[18] + (-8.f) * f[23];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 464 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[2] + (2.f) * f[3] + (2.f) * f[6] + (-2.f) * f[10] + (-2.f) * f[13] + (-2.f) * f[15] + (-16.f) * f[19];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + f[3] + f[6] + (-4.f) * f[10] + (2.f) * f[13] + (-4.f) * f[15] + (-8.f) * f[19];
		const float q2 = + (-1.f/2.f) * f[0] - f[2] - f[3] + (2.f) * f[6] + (-2.f) * f[10] + (4.f) * f[13] + (4.f) * f[15] + (8.f) * f[19];
		const float q3 = + (1.f/2.f) * f[0] + f[2] + (-2.f) * f[3] + f[6] + (-4.f) * f[10] + (-4.f) * f[13] + (2.f) * f[15] + (-8.f) * f[19];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
	else if ( normalCode == 446 )
	{
		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk
		const float q0 = + (5.f/2.f) * f[0] + (2.f) * f[2] + (2.f) * f[4] + (2.f) * f[5] + (-2.f) * f[8] + (-2.f) * f[11] + (-2.f) * f[14] + (-16.f) * f[21];
		const float q1 = + (1.f/2.f) * f[0] + (-2.f) * f[2] + f[4] + f[5] + (-4.f) * f[8] + (-4.f) * f[11] + (2.f) * f[14] + (-8.f) * f[21];
		const float q2 = + (1.f/2.f) * f[0] + f[2] + f[4] + (-2.f) * f[5] + (2.f) * f[8] + (-4.f) * f[11] + (-4.f) * f[14] + (-8.f) * f[21];
		const float q3 = + (-1.f/2.f) * f[0] - f[2] + (2.f) * f[4] - f[5] + (4.f) * f[8] + (-2.f) * f[11] + (4.f) * f[14] + (8.f) * f[21];
		rho = q0;
		ux = q1 / rho;
		uy = q2 / rho;
		uz = q3 / rho;
	}
}
