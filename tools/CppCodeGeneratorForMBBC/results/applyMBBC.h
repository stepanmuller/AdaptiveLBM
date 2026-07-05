__host__ __device__ void applyMBBC(
	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,
	const float &rho, const float &ux, const float &uy, const float &uz,
	float (&f)[27]
)
{
	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);
	if ( normalCode == 655 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[3] + f[4] + f[5] + f[6] + f[7] + f[9] + f[12] + f[13] + f[14] + f[16] + f[17] + f[18] + f[20] + f[22] + f[23] + f[26];
		const float smf1 = - f[5] + f[6] + f[12] + f[13] - f[14] - f[16] + f[17] - f[18] - f[20] + f[22] - f[23] + f[26];
		const float smf2 = - f[3] + f[4] - f[7] + f[9] - f[13] + f[14] + f[17] - f[18] + f[20] - f[22] - f[23] + f[26];
		const float smf3 = + f[5] + f[6] + f[12] + f[13] + f[14] + f[16] + f[17] + f[18] + f[20] + f[22] + f[23] + f[26];
		const float smf4 = + f[3] + f[4] + f[7] + f[9] + f[13] + f[14] + f[17] + f[18] + f[20] + f[22] + f[23] + f[26];
		const float smf5 = - f[13] - f[14] + f[17] + f[18] - f[20] - f[22] + f[23] + f[26];
		const float smf6 = - f[13] + f[14] + f[17] - f[18] + f[20] - f[22] - f[23] + f[26];
		const float smf7 = + f[13] - f[14] + f[17] - f[18] - f[20] + f[22] - f[23] + f[26];
		const float smf8 = + f[13] + f[14] + f[17] + f[18] + f[20] + f[22] + f[23] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * uy;
		const float m2 = rho * uz;
		const float m3 = (1.f/3.f) * rho + rho * uy * uy;
		const float m4 = (1.f/3.f) * rho + rho * uz * uz;
		const float m5 = rho * uy * uz;
		const float m6 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m7 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m8 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s3 - s4 + s8;
		f[8] = + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (-1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[10] = + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[15] = + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (-1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[19] = + (-1.f/4.f) * s5 + (-1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[21] = + (-1.f/4.f) * s5 + (1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[24] = + (1.f/4.f) * s5 + (1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[25] = + (1.f/4.f) * s5 + (-1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
	}
	if ( normalCode == 565 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[3] + f[4] + f[6] + f[7] + f[8] + f[9] + f[10] + f[12] + f[13] + f[15] + f[17] + f[19] + f[22] + f[24] + f[26];
		const float smf1 = + f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[12] - f[15] - f[19] + f[22] - f[24] + f[26];
		const float smf2 = - f[3] + f[4] - f[7] + f[8] + f[9] - f[10] - f[13] + f[17] - f[19] - f[22] + f[24] + f[26];
		const float smf3 = + f[1] + f[2] + f[7] + f[8] + f[9] + f[10] + f[12] + f[15] + f[19] + f[22] + f[24] + f[26];
		const float smf4 = + f[3] + f[4] + f[7] + f[8] + f[9] + f[10] + f[13] + f[17] + f[19] + f[22] + f[24] + f[26];
		const float smf5 = - f[7] - f[8] + f[9] + f[10] + f[19] - f[22] - f[24] + f[26];
		const float smf6 = - f[7] + f[8] + f[9] - f[10] - f[19] - f[22] + f[24] + f[26];
		const float smf7 = + f[7] - f[8] + f[9] - f[10] - f[19] + f[22] - f[24] + f[26];
		const float smf8 = + f[7] + f[8] + f[9] + f[10] + f[19] + f[22] + f[24] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uz;
		const float m3 = (1.f/3.f) * rho + rho * ux * ux;
		const float m4 = (1.f/3.f) * rho + rho * uz * uz;
		const float m5 = rho * ux * uz;
		const float m6 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m7 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m8 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[5] = + s0 - s3 - s4 + s8;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[14] = + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (-1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[16] = + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (-1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[20] = + (1.f/4.f) * s5 + (1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[21] = + (-1.f/4.f) * s5 + (1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[23] = + (-1.f/4.f) * s5 + (-1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[25] = + (1.f/4.f) * s5 + (-1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
	}
	if ( normalCode == 556 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[4] + f[5] + f[6] + f[8] + f[9] + f[11] + f[12] + f[14] + f[15] + f[16] + f[17] + f[20] + f[21] + f[24] + f[26];
		const float smf1 = + f[1] - f[2] - f[8] + f[9] - f[11] + f[12] - f[15] + f[16] + f[20] - f[21] - f[24] + f[26];
		const float smf2 = - f[5] + f[6] - f[11] + f[12] - f[14] + f[15] - f[16] + f[17] - f[20] - f[21] + f[24] + f[26];
		const float smf3 = + f[1] + f[2] + f[8] + f[9] + f[11] + f[12] + f[15] + f[16] + f[20] + f[21] + f[24] + f[26];
		const float smf4 = + f[5] + f[6] + f[11] + f[12] + f[14] + f[15] + f[16] + f[17] + f[20] + f[21] + f[24] + f[26];
		const float smf5 = + f[11] + f[12] - f[15] - f[16] - f[20] + f[21] - f[24] + f[26];
		const float smf6 = - f[11] + f[12] + f[15] - f[16] - f[20] - f[21] + f[24] + f[26];
		const float smf7 = - f[11] + f[12] - f[15] + f[16] + f[20] - f[21] - f[24] + f[26];
		const float smf8 = + f[11] + f[12] + f[15] + f[16] + f[20] + f[21] + f[24] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = (1.f/3.f) * rho + rho * ux * ux;
		const float m4 = (1.f/3.f) * rho + rho * uy * uy;
		const float m5 = rho * ux * uy;
		const float m6 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m7 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m8 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[3] = + s0 - s3 - s4 + s8;
		f[7] = + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (-1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[10] = + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[13] = + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (-1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[19] = + (-1.f/4.f) * s5 + (1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[22] = + (1.f/4.f) * s5 + (1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[23] = + (-1.f/4.f) * s5 + (-1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[25] = + (1.f/4.f) * s5 + (-1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
	}
	if ( normalCode == 455 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[3] + f[4] + f[5] + f[6] + f[8] + f[10] + f[11] + f[13] + f[14] + f[15] + f[17] + f[18] + f[19] + f[21] + f[24] + f[25];
		const float smf1 = - f[5] + f[6] - f[11] + f[13] - f[14] + f[15] + f[17] - f[18] + f[19] - f[21] + f[24] - f[25];
		const float smf2 = - f[3] + f[4] + f[8] - f[10] - f[13] + f[14] + f[17] - f[18] - f[19] + f[21] + f[24] - f[25];
		const float smf3 = + f[5] + f[6] + f[11] + f[13] + f[14] + f[15] + f[17] + f[18] + f[19] + f[21] + f[24] + f[25];
		const float smf4 = + f[3] + f[4] + f[8] + f[10] + f[13] + f[14] + f[17] + f[18] + f[19] + f[21] + f[24] + f[25];
		const float smf5 = - f[13] - f[14] + f[17] + f[18] - f[19] - f[21] + f[24] + f[25];
		const float smf6 = - f[13] + f[14] + f[17] - f[18] - f[19] + f[21] + f[24] - f[25];
		const float smf7 = + f[13] - f[14] + f[17] - f[18] + f[19] - f[21] + f[24] - f[25];
		const float smf8 = + f[13] + f[14] + f[17] + f[18] + f[19] + f[21] + f[24] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * uy;
		const float m2 = rho * uz;
		const float m3 = (1.f/3.f) * rho + rho * uy * uy;
		const float m4 = (1.f/3.f) * rho + rho * uz * uz;
		const float m5 = rho * uy * uz;
		const float m6 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m7 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m8 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s3 - s4 + s8;
		f[7] = + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[9] = + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (-1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (-1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[16] = + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[20] = + (-1.f/4.f) * s5 + (1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[22] = + (-1.f/4.f) * s5 + (-1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[23] = + (1.f/4.f) * s5 + (-1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[26] = + (1.f/4.f) * s5 + (1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
	}
	if ( normalCode == 545 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[7] + f[8] + f[9] + f[10] + f[11] + f[14] + f[16] + f[18] + f[20] + f[21] + f[23] + f[25];
		const float smf1 = + f[1] - f[2] + f[7] - f[8] + f[9] - f[10] - f[11] + f[16] + f[20] - f[21] + f[23] - f[25];
		const float smf2 = - f[3] + f[4] - f[7] + f[8] + f[9] - f[10] + f[14] - f[18] + f[20] + f[21] - f[23] - f[25];
		const float smf3 = + f[1] + f[2] + f[7] + f[8] + f[9] + f[10] + f[11] + f[16] + f[20] + f[21] + f[23] + f[25];
		const float smf4 = + f[3] + f[4] + f[7] + f[8] + f[9] + f[10] + f[14] + f[18] + f[20] + f[21] + f[23] + f[25];
		const float smf5 = - f[7] - f[8] + f[9] + f[10] + f[20] - f[21] - f[23] + f[25];
		const float smf6 = - f[7] + f[8] + f[9] - f[10] + f[20] + f[21] - f[23] - f[25];
		const float smf7 = + f[7] - f[8] + f[9] - f[10] + f[20] - f[21] + f[23] - f[25];
		const float smf8 = + f[7] + f[8] + f[9] + f[10] + f[20] + f[21] + f[23] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uz;
		const float m3 = (1.f/3.f) * rho + rho * ux * ux;
		const float m4 = (1.f/3.f) * rho + rho * uz * uz;
		const float m5 = rho * ux * uz;
		const float m6 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m7 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m8 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[6] = + s0 - s3 - s4 + s8;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (-1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[13] = + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[15] = + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (-1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[19] = + (1.f/4.f) * s5 + (-1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[22] = + (-1.f/4.f) * s5 + (-1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[24] = + (-1.f/4.f) * s5 + (1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[26] = + (1.f/4.f) * s5 + (1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
	}
	if ( normalCode == 554 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[3] + f[5] + f[6] + f[7] + f[10] + f[11] + f[12] + f[13] + f[15] + f[16] + f[18] + f[19] + f[22] + f[23] + f[25];
		const float smf1 = + f[1] - f[2] + f[7] - f[10] - f[11] + f[12] - f[15] + f[16] - f[19] + f[22] + f[23] - f[25];
		const float smf2 = - f[5] + f[6] - f[11] + f[12] + f[13] + f[15] - f[16] - f[18] + f[19] + f[22] - f[23] - f[25];
		const float smf3 = + f[1] + f[2] + f[7] + f[10] + f[11] + f[12] + f[15] + f[16] + f[19] + f[22] + f[23] + f[25];
		const float smf4 = + f[5] + f[6] + f[11] + f[12] + f[13] + f[15] + f[16] + f[18] + f[19] + f[22] + f[23] + f[25];
		const float smf5 = + f[11] + f[12] - f[15] - f[16] - f[19] + f[22] - f[23] + f[25];
		const float smf6 = - f[11] + f[12] + f[15] - f[16] + f[19] + f[22] - f[23] - f[25];
		const float smf7 = - f[11] + f[12] - f[15] + f[16] - f[19] + f[22] + f[23] - f[25];
		const float smf8 = + f[11] + f[12] + f[15] + f[16] + f[19] + f[22] + f[23] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = (1.f/3.f) * rho + rho * ux * ux;
		const float m4 = (1.f/3.f) * rho + rho * uy * uy;
		const float m5 = rho * ux * uy;
		const float m6 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m7 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m8 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[4] = + s0 - s3 - s4 + s8;
		f[8] = + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[9] = + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (-1.f/2.f) * s7 + (-1.f/2.f) * s8;
		f[14] = + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (-1.f/2.f) * s6 + (-1.f/2.f) * s8;
		f[20] = + (-1.f/4.f) * s5 + (-1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[21] = + (1.f/4.f) * s5 + (-1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[24] = + (-1.f/4.f) * s5 + (1.f/4.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8;
		f[26] = + (1.f/4.f) * s5 + (1.f/4.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8;
	}
	if ( normalCode == 566 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[4] + f[6] + f[8] + f[9] + f[12] + f[15] + f[17] + f[24] + f[26];
		const float smf1 = + f[1] - f[2] - f[8] + f[9] + f[12] - f[15] - f[24] + f[26];
		const float smf2 = + f[6] + f[12] + f[15] + f[17] + f[24] + f[26];
		const float smf3 = + f[4] + f[8] + f[9] + f[17] + f[24] + f[26];
		const float smf4 = + f[1] + f[2] + f[8] + f[9] + f[12] + f[15] + f[24] + f[26];
		const float smf5 = + f[6] + f[12] + f[15] + f[17] + f[24] + f[26];
		const float smf6 = + f[4] + f[8] + f[9] + f[17] + f[24] + f[26];
		const float smf7 = - f[8] + f[9] - f[24] + f[26];
		const float smf8 = + f[12] - f[15] - f[24] + f[26];
		const float smf9 = + f[12] + f[15] + f[24] + f[26];
		const float smf10 = + f[8] + f[9] + f[24] + f[26];
		const float smf11 = + f[12] - f[15] - f[24] + f[26];
		const float smf12 = - f[8] + f[9] - f[24] + f[26];
		const float smf13 = + f[8] + f[9] + f[24] + f[26];
		const float smf14 = + f[12] + f[15] + f[24] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * ux * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[3] = + s0 - s4 - s5 + s14;
		f[5] = + s0 - s4 - s6 + s13;
		f[7] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[10] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[13] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[14] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[16] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[18] = - s0 + (-1.f/2.f) * s2 + (-1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (1.f/2.f) * s9 + (1.f/2.f) * s10 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[19] = + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[20] = + (1.f/4.f) * s7 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[21] = + (-1.f/4.f) * s7 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[22] = + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[23] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[25] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
	}
	if ( normalCode == 656 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[4] + f[5] + f[6] + f[9] + f[12] + f[14] + f[16] + f[17] + f[20] + f[26];
		const float smf1 = + f[1] + f[9] + f[12] + f[16] + f[20] + f[26];
		const float smf2 = - f[5] + f[6] + f[12] - f[14] - f[16] + f[17] - f[20] + f[26];
		const float smf3 = + f[4] + f[9] + f[14] + f[17] + f[20] + f[26];
		const float smf4 = + f[1] + f[9] + f[12] + f[16] + f[20] + f[26];
		const float smf5 = + f[5] + f[6] + f[12] + f[14] + f[16] + f[17] + f[20] + f[26];
		const float smf6 = + f[4] + f[9] + f[14] + f[17] + f[20] + f[26];
		const float smf7 = - f[14] + f[17] - f[20] + f[26];
		const float smf8 = + f[12] - f[16] - f[20] + f[26];
		const float smf9 = + f[12] - f[16] - f[20] + f[26];
		const float smf10 = + f[12] + f[16] + f[20] + f[26];
		const float smf11 = + f[14] + f[17] + f[20] + f[26];
		const float smf12 = - f[14] + f[17] - f[20] + f[26];
		const float smf13 = + f[14] + f[17] + f[20] + f[26];
		const float smf14 = + f[12] + f[16] + f[20] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s13;
		f[3] = + s0 - s4 - s5 + s14;
		f[7] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s10 + (-1.f/2.f) * s14;
		f[8] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s13;
		f[10] = - s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (1.f/2.f) * s11 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[11] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[13] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[15] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[21] = + (-1.f/4.f) * s7 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[22] = + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[23] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[24] = + (1.f/4.f) * s7 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[25] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
	}
	if ( normalCode == 665 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[3] + f[4] + f[6] + f[7] + f[9] + f[12] + f[13] + f[17] + f[22] + f[26];
		const float smf1 = + f[1] + f[7] + f[9] + f[12] + f[22] + f[26];
		const float smf2 = + f[6] + f[12] + f[13] + f[17] + f[22] + f[26];
		const float smf3 = - f[3] + f[4] - f[7] + f[9] - f[13] + f[17] - f[22] + f[26];
		const float smf4 = + f[1] + f[7] + f[9] + f[12] + f[22] + f[26];
		const float smf5 = + f[6] + f[12] + f[13] + f[17] + f[22] + f[26];
		const float smf6 = + f[3] + f[4] + f[7] + f[9] + f[13] + f[17] + f[22] + f[26];
		const float smf7 = - f[13] + f[17] - f[22] + f[26];
		const float smf8 = - f[7] + f[9] - f[22] + f[26];
		const float smf9 = - f[7] + f[9] - f[22] + f[26];
		const float smf10 = - f[13] + f[17] - f[22] + f[26];
		const float smf11 = + f[7] + f[9] + f[22] + f[26];
		const float smf12 = + f[13] + f[17] + f[22] + f[26];
		const float smf13 = + f[13] + f[17] + f[22] + f[26];
		const float smf14 = + f[7] + f[9] + f[22] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m10 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s13;
		f[5] = + s0 - s4 - s6 + s14;
		f[8] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[10] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[11] = - s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (1.f/2.f) * s11 + (1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[14] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[15] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[16] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[18] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[20] = + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[21] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[23] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[24] = + (1.f/4.f) * s7 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[25] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
	}
	if ( normalCode == 544 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[3] + f[5] + f[7] + f[10] + f[11] + f[16] + f[18] + f[23] + f[25];
		const float smf1 = + f[1] - f[2] + f[7] - f[10] - f[11] + f[16] + f[23] - f[25];
		const float smf2 = - f[5] - f[11] - f[16] - f[18] - f[23] - f[25];
		const float smf3 = - f[3] - f[7] - f[10] - f[18] - f[23] - f[25];
		const float smf4 = + f[1] + f[2] + f[7] + f[10] + f[11] + f[16] + f[23] + f[25];
		const float smf5 = + f[5] + f[11] + f[16] + f[18] + f[23] + f[25];
		const float smf6 = + f[3] + f[7] + f[10] + f[18] + f[23] + f[25];
		const float smf7 = - f[7] + f[10] - f[23] + f[25];
		const float smf8 = + f[11] - f[16] - f[23] + f[25];
		const float smf9 = - f[11] - f[16] - f[23] - f[25];
		const float smf10 = - f[7] - f[10] - f[23] - f[25];
		const float smf11 = - f[11] + f[16] + f[23] - f[25];
		const float smf12 = + f[7] - f[10] + f[23] - f[25];
		const float smf13 = + f[7] + f[10] + f[23] + f[25];
		const float smf14 = + f[11] + f[16] + f[23] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * ux * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[4] = + s0 - s4 - s5 + s14;
		f[6] = + s0 - s4 - s6 + s13;
		f[8] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[9] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[13] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[14] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[15] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[17] = - s0 + (1.f/2.f) * s2 + (1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s9 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[19] = + (1.f/4.f) * s7 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[20] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[21] = + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[22] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[24] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[26] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
	}
	if ( normalCode == 454 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[3] + f[5] + f[6] + f[10] + f[11] + f[13] + f[15] + f[18] + f[19] + f[25];
		const float smf1 = - f[2] - f[10] - f[11] - f[15] - f[19] - f[25];
		const float smf2 = - f[5] + f[6] - f[11] + f[13] + f[15] - f[18] + f[19] - f[25];
		const float smf3 = - f[3] - f[10] - f[13] - f[18] - f[19] - f[25];
		const float smf4 = + f[2] + f[10] + f[11] + f[15] + f[19] + f[25];
		const float smf5 = + f[5] + f[6] + f[11] + f[13] + f[15] + f[18] + f[19] + f[25];
		const float smf6 = + f[3] + f[10] + f[13] + f[18] + f[19] + f[25];
		const float smf7 = - f[13] + f[18] - f[19] + f[25];
		const float smf8 = + f[11] - f[15] - f[19] + f[25];
		const float smf9 = - f[11] + f[15] + f[19] - f[25];
		const float smf10 = - f[11] - f[15] - f[19] - f[25];
		const float smf11 = - f[13] - f[18] - f[19] - f[25];
		const float smf12 = + f[13] - f[18] + f[19] - f[25];
		const float smf13 = + f[13] + f[18] + f[19] + f[25];
		const float smf14 = + f[11] + f[15] + f[19] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s13;
		f[4] = + s0 - s4 - s5 + s14;
		f[7] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s13;
		f[8] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s10 + (-1.f/2.f) * s14;
		f[9] = - s0 + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s11 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[12] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[14] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[16] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[20] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[21] = + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[22] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[23] = + (1.f/4.f) * s7 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[24] = + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[26] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
	}
	if ( normalCode == 445 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[3] + f[4] + f[5] + f[8] + f[10] + f[11] + f[14] + f[18] + f[21] + f[25];
		const float smf1 = - f[2] - f[8] - f[10] - f[11] - f[21] - f[25];
		const float smf2 = - f[5] - f[11] - f[14] - f[18] - f[21] - f[25];
		const float smf3 = - f[3] + f[4] + f[8] - f[10] + f[14] - f[18] + f[21] - f[25];
		const float smf4 = + f[2] + f[8] + f[10] + f[11] + f[21] + f[25];
		const float smf5 = + f[5] + f[11] + f[14] + f[18] + f[21] + f[25];
		const float smf6 = + f[3] + f[4] + f[8] + f[10] + f[14] + f[18] + f[21] + f[25];
		const float smf7 = - f[14] + f[18] - f[21] + f[25];
		const float smf8 = - f[8] + f[10] - f[21] + f[25];
		const float smf9 = + f[8] - f[10] + f[21] - f[25];
		const float smf10 = + f[14] - f[18] + f[21] - f[25];
		const float smf11 = - f[8] - f[10] - f[21] - f[25];
		const float smf12 = - f[14] - f[18] - f[21] - f[25];
		const float smf13 = + f[14] + f[18] + f[21] + f[25];
		const float smf14 = + f[8] + f[10] + f[21] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m10 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s13;
		f[6] = + s0 - s4 - s6 + s14;
		f[7] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[9] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[12] = - s0 + (1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[13] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[15] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[16] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[17] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[20] = + (-1.f/4.f) * s7 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[22] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[23] = + (1.f/4.f) * s7 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[24] = + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[26] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
	}
	if ( normalCode == 564 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[3] + f[6] + f[7] + f[10] + f[12] + f[13] + f[15] + f[19] + f[22];
		const float smf1 = + f[1] - f[2] + f[7] - f[10] + f[12] - f[15] - f[19] + f[22];
		const float smf2 = + f[6] + f[12] + f[13] + f[15] + f[19] + f[22];
		const float smf3 = - f[3] - f[7] - f[10] - f[13] - f[19] - f[22];
		const float smf4 = + f[1] + f[2] + f[7] + f[10] + f[12] + f[15] + f[19] + f[22];
		const float smf5 = + f[6] + f[12] + f[13] + f[15] + f[19] + f[22];
		const float smf6 = + f[3] + f[7] + f[10] + f[13] + f[19] + f[22];
		const float smf7 = - f[7] + f[10] + f[19] - f[22];
		const float smf8 = + f[12] - f[15] - f[19] + f[22];
		const float smf9 = + f[12] + f[15] + f[19] + f[22];
		const float smf10 = - f[7] - f[10] - f[19] - f[22];
		const float smf11 = + f[12] - f[15] - f[19] + f[22];
		const float smf12 = + f[7] - f[10] - f[19] + f[22];
		const float smf13 = + f[7] + f[10] + f[19] + f[22];
		const float smf14 = + f[12] + f[15] + f[19] + f[22];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * ux * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[4] = + s0 - s4 - s5 + s14;
		f[5] = + s0 - s4 - s6 + s13;
		f[8] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[9] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[14] = - s0 + (-1.f/2.f) * s2 + (1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (1.f/2.f) * s9 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[16] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[18] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[20] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[21] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[23] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[24] = + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[25] = + (1.f/4.f) * s7 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[26] = + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
	}
	if ( normalCode == 654 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[3] + f[5] + f[6] + f[7] + f[12] + f[13] + f[16] + f[18] + f[22] + f[23];
		const float smf1 = + f[1] + f[7] + f[12] + f[16] + f[22] + f[23];
		const float smf2 = - f[5] + f[6] + f[12] + f[13] - f[16] - f[18] + f[22] - f[23];
		const float smf3 = - f[3] - f[7] - f[13] - f[18] - f[22] - f[23];
		const float smf4 = + f[1] + f[7] + f[12] + f[16] + f[22] + f[23];
		const float smf5 = + f[5] + f[6] + f[12] + f[13] + f[16] + f[18] + f[22] + f[23];
		const float smf6 = + f[3] + f[7] + f[13] + f[18] + f[22] + f[23];
		const float smf7 = - f[13] + f[18] - f[22] + f[23];
		const float smf8 = + f[12] - f[16] + f[22] - f[23];
		const float smf9 = + f[12] - f[16] + f[22] - f[23];
		const float smf10 = + f[12] + f[16] + f[22] + f[23];
		const float smf11 = - f[13] - f[18] - f[22] - f[23];
		const float smf12 = + f[13] - f[18] + f[22] - f[23];
		const float smf13 = + f[13] + f[18] + f[22] + f[23];
		const float smf14 = + f[12] + f[16] + f[22] + f[23];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s13;
		f[4] = + s0 - s4 - s5 + s14;
		f[8] = - s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s11 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[9] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s10 + (-1.f/2.f) * s14;
		f[10] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s13;
		f[11] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[14] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[15] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[20] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[21] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[24] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[25] = + (1.f/4.f) * s7 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[26] = + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s14;
	}
	if ( normalCode == 645 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[3] + f[4] + f[5] + f[7] + f[9] + f[14] + f[16] + f[18] + f[20] + f[23];
		const float smf1 = + f[1] + f[7] + f[9] + f[16] + f[20] + f[23];
		const float smf2 = - f[5] - f[14] - f[16] - f[18] - f[20] - f[23];
		const float smf3 = - f[3] + f[4] - f[7] + f[9] + f[14] - f[18] + f[20] - f[23];
		const float smf4 = + f[1] + f[7] + f[9] + f[16] + f[20] + f[23];
		const float smf5 = + f[5] + f[14] + f[16] + f[18] + f[20] + f[23];
		const float smf6 = + f[3] + f[4] + f[7] + f[9] + f[14] + f[18] + f[20] + f[23];
		const float smf7 = - f[14] + f[18] - f[20] + f[23];
		const float smf8 = - f[7] + f[9] + f[20] - f[23];
		const float smf9 = - f[7] + f[9] + f[20] - f[23];
		const float smf10 = + f[14] - f[18] + f[20] - f[23];
		const float smf11 = + f[7] + f[9] + f[20] + f[23];
		const float smf12 = - f[14] - f[18] - f[20] - f[23];
		const float smf13 = + f[14] + f[18] + f[20] + f[23];
		const float smf14 = + f[7] + f[9] + f[20] + f[23];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m10 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s13;
		f[6] = + s0 - s4 - s6 + s14;
		f[8] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[10] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[11] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[13] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[15] = - s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[17] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[21] = + (-1.f/4.f) * s7 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[22] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[24] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[25] = + (1.f/4.f) * s7 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[26] = + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
	}
	if ( normalCode == 546 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[2] + f[4] + f[5] + f[8] + f[9] + f[11] + f[14] + f[16] + f[20] + f[21];
		const float smf1 = + f[1] - f[2] - f[8] + f[9] - f[11] + f[16] + f[20] - f[21];
		const float smf2 = - f[5] - f[11] - f[14] - f[16] - f[20] - f[21];
		const float smf3 = + f[4] + f[8] + f[9] + f[14] + f[20] + f[21];
		const float smf4 = + f[1] + f[2] + f[8] + f[9] + f[11] + f[16] + f[20] + f[21];
		const float smf5 = + f[5] + f[11] + f[14] + f[16] + f[20] + f[21];
		const float smf6 = + f[4] + f[8] + f[9] + f[14] + f[20] + f[21];
		const float smf7 = - f[8] + f[9] + f[20] - f[21];
		const float smf8 = + f[11] - f[16] - f[20] + f[21];
		const float smf9 = - f[11] - f[16] - f[20] - f[21];
		const float smf10 = + f[8] + f[9] + f[20] + f[21];
		const float smf11 = - f[11] + f[16] + f[20] - f[21];
		const float smf12 = - f[8] + f[9] + f[20] - f[21];
		const float smf13 = + f[8] + f[9] + f[20] + f[21];
		const float smf14 = + f[11] + f[16] + f[20] + f[21];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * ux * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[3] = + s0 - s4 - s5 + s14;
		f[6] = + s0 - s4 - s6 + s13;
		f[7] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[10] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[13] = - s0 + (1.f/2.f) * s2 + (-1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s9 + (1.f/2.f) * s10 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[15] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[17] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[22] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[23] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[24] = + (-1.f/4.f) * s7 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[25] = + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[26] = + (1.f/4.f) * s7 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
	}
	if ( normalCode == 456 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[4] + f[5] + f[6] + f[8] + f[11] + f[14] + f[15] + f[17] + f[21] + f[24];
		const float smf1 = - f[2] - f[8] - f[11] - f[15] - f[21] - f[24];
		const float smf2 = - f[5] + f[6] - f[11] - f[14] + f[15] + f[17] - f[21] + f[24];
		const float smf3 = + f[4] + f[8] + f[14] + f[17] + f[21] + f[24];
		const float smf4 = + f[2] + f[8] + f[11] + f[15] + f[21] + f[24];
		const float smf5 = + f[5] + f[6] + f[11] + f[14] + f[15] + f[17] + f[21] + f[24];
		const float smf6 = + f[4] + f[8] + f[14] + f[17] + f[21] + f[24];
		const float smf7 = - f[14] + f[17] - f[21] + f[24];
		const float smf8 = + f[11] - f[15] + f[21] - f[24];
		const float smf9 = - f[11] + f[15] - f[21] + f[24];
		const float smf10 = - f[11] - f[15] - f[21] - f[24];
		const float smf11 = + f[14] + f[17] + f[21] + f[24];
		const float smf12 = - f[14] + f[17] - f[21] + f[24];
		const float smf13 = + f[14] + f[17] + f[21] + f[24];
		const float smf14 = + f[11] + f[15] + f[21] + f[24];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uy;
		const float m9 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m10 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s13;
		f[3] = + s0 - s4 - s5 + s14;
		f[7] = - s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (1.f/2.f) * s11 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[9] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s13;
		f[10] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s10 + (-1.f/2.f) * s14;
		f[12] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[13] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[16] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[19] = + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[20] = + (-1.f/4.f) * s7 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[22] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[23] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[25] = + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s14;
		f[26] = + (1.f/4.f) * s7 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
	}
	if ( normalCode == 465 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[3] + f[4] + f[6] + f[8] + f[10] + f[13] + f[15] + f[17] + f[19] + f[24];
		const float smf1 = - f[2] - f[8] - f[10] - f[15] - f[19] - f[24];
		const float smf2 = + f[6] + f[13] + f[15] + f[17] + f[19] + f[24];
		const float smf3 = - f[3] + f[4] + f[8] - f[10] - f[13] + f[17] - f[19] + f[24];
		const float smf4 = + f[2] + f[8] + f[10] + f[15] + f[19] + f[24];
		const float smf5 = + f[6] + f[13] + f[15] + f[17] + f[19] + f[24];
		const float smf6 = + f[3] + f[4] + f[8] + f[10] + f[13] + f[17] + f[19] + f[24];
		const float smf7 = - f[13] + f[17] - f[19] + f[24];
		const float smf8 = - f[8] + f[10] + f[19] - f[24];
		const float smf9 = + f[8] - f[10] - f[19] + f[24];
		const float smf10 = - f[13] + f[17] - f[19] + f[24];
		const float smf11 = - f[8] - f[10] - f[19] - f[24];
		const float smf12 = + f[13] + f[17] + f[19] + f[24];
		const float smf13 = + f[13] + f[17] + f[19] + f[24];
		const float smf14 = + f[8] + f[10] + f[19] + f[24];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m10 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m11 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m12 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m13 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m14 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s13;
		f[5] = + s0 - s4 - s6 + s14;
		f[7] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[9] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s13;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s11 + (-1.f/2.f) * s14;
		f[12] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13;
		f[14] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[16] = - s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (-1.f/2.f) * s11 + (1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s14;
		f[18] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s9 + (-1.f/2.f) * s14;
		f[20] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[21] = + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[22] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
		f[23] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14;
		f[25] = + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s11 + (1.f/4.f) * s14;
		f[26] = + (1.f/4.f) * s7 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13;
	}
	if ( normalCode == 666 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[4] + f[6] + f[9] + f[12] + f[17] + f[26];
		const float smf1 = + f[1] + f[9] + f[12] + f[26];
		const float smf2 = + f[6] + f[12] + f[17] + f[26];
		const float smf3 = + f[4] + f[9] + f[17] + f[26];
		const float smf4 = + f[1] + f[9] + f[12] + f[26];
		const float smf5 = + f[6] + f[12] + f[17] + f[26];
		const float smf6 = + f[4] + f[9] + f[17] + f[26];
		const float smf7 = + f[17] + f[26];
		const float smf8 = + f[9] + f[26];
		const float smf9 = + f[12] + f[26];
		const float smf10 = + f[12] + f[26];
		const float smf11 = + f[9] + f[26];
		const float smf12 = + f[12] + f[26];
		const float smf13 = + f[17] + f[26];
		const float smf14 = + f[9] + f[26];
		const float smf15 = + f[17] + f[26];
		const float smf16 = + f[17] + f[26];
		const float smf17 = + f[9] + f[26];
		const float smf18 = + f[12] + f[26];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s16;
		f[3] = + s0 - s4 - s5 + s18;
		f[5] = + s0 - s4 - s6 + s17;
		f[7] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[8] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[10] = - s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (1.f/2.f) * s12 + (1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[11] = - s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (1.f/2.f) * s14 + (1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[13] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[14] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[15] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[16] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[18] = - s0 + (-1.f/2.f) * s2 + (-1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[19] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (-1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[20] = + (1.f/4.f) * s8 + (1.f/4.f) * s11 + (1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[21] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s11 + (1.f/4.f) * s13 + (-1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[22] = + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[23] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[24] = + (1.f/4.f) * s7 + (1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[25] = + s0 + (1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (-1.f/4.f) * s13 + (-1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
	}
	if ( normalCode == 444 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[3] + f[5] + f[10] + f[11] + f[18] + f[25];
		const float smf1 = - f[2] - f[10] - f[11] - f[25];
		const float smf2 = - f[5] - f[11] - f[18] - f[25];
		const float smf3 = - f[3] - f[10] - f[18] - f[25];
		const float smf4 = + f[2] + f[10] + f[11] + f[25];
		const float smf5 = + f[5] + f[11] + f[18] + f[25];
		const float smf6 = + f[3] + f[10] + f[18] + f[25];
		const float smf7 = + f[18] + f[25];
		const float smf8 = + f[10] + f[25];
		const float smf9 = + f[11] + f[25];
		const float smf10 = - f[11] - f[25];
		const float smf11 = - f[10] - f[25];
		const float smf12 = - f[11] - f[25];
		const float smf13 = - f[18] - f[25];
		const float smf14 = - f[10] - f[25];
		const float smf15 = - f[18] - f[25];
		const float smf16 = + f[18] + f[25];
		const float smf17 = + f[10] + f[25];
		const float smf18 = + f[11] + f[25];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s16;
		f[4] = + s0 - s4 - s5 + s18;
		f[6] = + s0 - s4 - s6 + s17;
		f[7] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[8] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[9] = - s0 + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[12] = - s0 + (1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (-1.f/2.f) * s14 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[13] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[14] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[15] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[16] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[17] = - s0 + (1.f/2.f) * s2 + (1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[19] = + (1.f/4.f) * s8 + (-1.f/4.f) * s11 + (-1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[20] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[21] = + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[22] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s11 + (-1.f/4.f) * s13 + (1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[23] = + (1.f/4.f) * s7 + (-1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[24] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (-1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[26] = + s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s2 + (-1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
	}
	if ( normalCode == 466 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[4] + f[6] + f[8] + f[15] + f[17] + f[24];
		const float smf1 = - f[2] - f[8] - f[15] - f[24];
		const float smf2 = + f[6] + f[15] + f[17] + f[24];
		const float smf3 = + f[4] + f[8] + f[17] + f[24];
		const float smf4 = + f[2] + f[8] + f[15] + f[24];
		const float smf5 = + f[6] + f[15] + f[17] + f[24];
		const float smf6 = + f[4] + f[8] + f[17] + f[24];
		const float smf7 = + f[17] + f[24];
		const float smf8 = - f[8] - f[24];
		const float smf9 = - f[15] - f[24];
		const float smf10 = + f[15] + f[24];
		const float smf11 = + f[8] + f[24];
		const float smf12 = - f[15] - f[24];
		const float smf13 = + f[17] + f[24];
		const float smf14 = - f[8] - f[24];
		const float smf15 = + f[17] + f[24];
		const float smf16 = + f[17] + f[24];
		const float smf17 = + f[8] + f[24];
		const float smf18 = + f[15] + f[24];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s16;
		f[3] = + s0 - s4 - s5 + s18;
		f[5] = + s0 - s4 - s6 + s17;
		f[7] = - s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s12 + (1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[9] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[10] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[12] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[13] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[14] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[16] = - s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (-1.f/2.f) * s14 + (1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[18] = - s0 + (-1.f/2.f) * s2 + (-1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[19] = + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[20] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s11 + (1.f/4.f) * s13 + (1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[21] = + (-1.f/4.f) * s8 + (1.f/4.f) * s11 + (-1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[22] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (-1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[23] = + s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (-1.f/4.f) * s13 + (1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[25] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (-1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[26] = + (1.f/4.f) * s7 + (1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16;
	}
	if ( normalCode == 646 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[4] + f[5] + f[9] + f[14] + f[16] + f[20];
		const float smf1 = + f[1] + f[9] + f[16] + f[20];
		const float smf2 = - f[5] - f[14] - f[16] - f[20];
		const float smf3 = + f[4] + f[9] + f[14] + f[20];
		const float smf4 = + f[1] + f[9] + f[16] + f[20];
		const float smf5 = + f[5] + f[14] + f[16] + f[20];
		const float smf6 = + f[4] + f[9] + f[14] + f[20];
		const float smf7 = - f[14] - f[20];
		const float smf8 = + f[9] + f[20];
		const float smf9 = - f[16] - f[20];
		const float smf10 = - f[16] - f[20];
		const float smf11 = + f[9] + f[20];
		const float smf12 = + f[16] + f[20];
		const float smf13 = + f[14] + f[20];
		const float smf14 = + f[9] + f[20];
		const float smf15 = - f[14] - f[20];
		const float smf16 = + f[14] + f[20];
		const float smf17 = + f[9] + f[20];
		const float smf18 = + f[16] + f[20];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s16;
		f[3] = + s0 - s4 - s5 + s18;
		f[6] = + s0 - s4 - s6 + s17;
		f[7] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[8] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[10] = - s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (1.f/2.f) * s12 + (1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[11] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[13] = - s0 + (1.f/2.f) * s2 + (-1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[15] = - s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (1.f/2.f) * s14 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[17] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[19] = + s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (-1.f/4.f) * s13 + (-1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[21] = + (-1.f/4.f) * s7 + (1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[22] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[23] = + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[24] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s11 + (1.f/4.f) * s13 + (-1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[25] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (-1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[26] = + (1.f/4.f) * s8 + (1.f/4.f) * s11 + (1.f/4.f) * s14 + (1.f/4.f) * s17;
	}
	if ( normalCode == 664 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[3] + f[6] + f[7] + f[12] + f[13] + f[22];
		const float smf1 = + f[1] + f[7] + f[12] + f[22];
		const float smf2 = + f[6] + f[12] + f[13] + f[22];
		const float smf3 = - f[3] - f[7] - f[13] - f[22];
		const float smf4 = + f[1] + f[7] + f[12] + f[22];
		const float smf5 = + f[6] + f[12] + f[13] + f[22];
		const float smf6 = + f[3] + f[7] + f[13] + f[22];
		const float smf7 = - f[13] - f[22];
		const float smf8 = - f[7] - f[22];
		const float smf9 = + f[12] + f[22];
		const float smf10 = + f[12] + f[22];
		const float smf11 = - f[7] - f[22];
		const float smf12 = + f[12] + f[22];
		const float smf13 = - f[13] - f[22];
		const float smf14 = + f[7] + f[22];
		const float smf15 = + f[13] + f[22];
		const float smf16 = + f[13] + f[22];
		const float smf17 = + f[7] + f[22];
		const float smf18 = + f[12] + f[22];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s16;
		f[4] = + s0 - s4 - s5 + s18;
		f[5] = + s0 - s4 - s6 + s17;
		f[8] = - s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[9] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[10] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[11] = - s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (1.f/2.f) * s14 + (1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[14] = - s0 + (-1.f/2.f) * s2 + (1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[15] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[16] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[18] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[19] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[20] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[21] = + s0 + (1.f/2.f) * s1 + (1.f/2.f) * s2 + (-1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (-1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[23] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s11 + (1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[24] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[25] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s11 + (-1.f/4.f) * s13 + (-1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[26] = + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s18;
	}
	if ( normalCode == 644 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[1] + f[3] + f[5] + f[7] + f[16] + f[18] + f[23];
		const float smf1 = + f[1] + f[7] + f[16] + f[23];
		const float smf2 = - f[5] - f[16] - f[18] - f[23];
		const float smf3 = - f[3] - f[7] - f[18] - f[23];
		const float smf4 = + f[1] + f[7] + f[16] + f[23];
		const float smf5 = + f[5] + f[16] + f[18] + f[23];
		const float smf6 = + f[3] + f[7] + f[18] + f[23];
		const float smf7 = + f[18] + f[23];
		const float smf8 = - f[7] - f[23];
		const float smf9 = - f[16] - f[23];
		const float smf10 = - f[16] - f[23];
		const float smf11 = - f[7] - f[23];
		const float smf12 = + f[16] + f[23];
		const float smf13 = - f[18] - f[23];
		const float smf14 = + f[7] + f[23];
		const float smf15 = - f[18] - f[23];
		const float smf16 = + f[18] + f[23];
		const float smf17 = + f[7] + f[23];
		const float smf18 = + f[16] + f[23];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[2] = + s0 - s5 - s6 + s16;
		f[4] = + s0 - s4 - s5 + s18;
		f[6] = + s0 - s4 - s6 + s17;
		f[8] = - s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[9] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[10] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[11] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[12] = + (1.f/2.f) * s1 + (1.f/2.f) * s4 + (-1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[13] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[14] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[15] = - s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (1.f/2.f) * s14 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[17] = - s0 + (1.f/2.f) * s2 + (1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[19] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s11 + (-1.f/4.f) * s13 + (-1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[20] = + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[21] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (-1.f/4.f) * s7 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[22] = + (-1.f/4.f) * s8 + (-1.f/4.f) * s11 + (1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[24] = + s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s2 + (-1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (1.f/4.f) * s13 + (-1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[25] = + (1.f/4.f) * s7 + (-1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[26] = + (-1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
	}
	if ( normalCode == 464 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[3] + f[6] + f[10] + f[13] + f[15] + f[19];
		const float smf1 = - f[2] - f[10] - f[15] - f[19];
		const float smf2 = + f[6] + f[13] + f[15] + f[19];
		const float smf3 = - f[3] - f[10] - f[13] - f[19];
		const float smf4 = + f[2] + f[10] + f[15] + f[19];
		const float smf5 = + f[6] + f[13] + f[15] + f[19];
		const float smf6 = + f[3] + f[10] + f[13] + f[19];
		const float smf7 = - f[13] - f[19];
		const float smf8 = + f[10] + f[19];
		const float smf9 = - f[15] - f[19];
		const float smf10 = + f[15] + f[19];
		const float smf11 = - f[10] - f[19];
		const float smf12 = - f[15] - f[19];
		const float smf13 = - f[13] - f[19];
		const float smf14 = - f[10] - f[19];
		const float smf15 = + f[13] + f[19];
		const float smf16 = + f[13] + f[19];
		const float smf17 = + f[10] + f[19];
		const float smf18 = + f[15] + f[19];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s16;
		f[4] = + s0 - s4 - s5 + s18;
		f[5] = + s0 - s4 - s6 + s17;
		f[7] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[8] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[9] = - s0 + (1.f/2.f) * s1 + (1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s12 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[11] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[12] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[14] = - s0 + (-1.f/2.f) * s2 + (1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (1.f/2.f) * s10 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[16] = - s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (-1.f/2.f) * s14 + (1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[17] = + (1.f/2.f) * s2 + (1.f/2.f) * s5 + (-1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[18] = + (-1.f/2.f) * s3 + (1.f/2.f) * s6 + (1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[20] = + s0 + (-1.f/2.f) * s1 + (1.f/2.f) * s2 + (-1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[21] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s11 + (-1.f/4.f) * s12 + (-1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[22] = + (-1.f/4.f) * s7 + (-1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[23] = + (1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (-1.f/4.f) * s8 + (-1.f/4.f) * s11 + (-1.f/4.f) * s13 + (1.f/4.f) * s14 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
		f[24] = + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[25] = + (1.f/4.f) * s8 + (-1.f/4.f) * s11 + (-1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[26] = + (-1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (1.f/4.f) * s12 + (1.f/4.f) * s13 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
	}
	if ( normalCode == 446 )
	{
		// Multiply S Mfk fk
		const float smf0 = + f[0] + f[2] + f[4] + f[5] + f[8] + f[11] + f[14] + f[21];
		const float smf1 = - f[2] - f[8] - f[11] - f[21];
		const float smf2 = - f[5] - f[11] - f[14] - f[21];
		const float smf3 = + f[4] + f[8] + f[14] + f[21];
		const float smf4 = + f[2] + f[8] + f[11] + f[21];
		const float smf5 = + f[5] + f[11] + f[14] + f[21];
		const float smf6 = + f[4] + f[8] + f[14] + f[21];
		const float smf7 = - f[14] - f[21];
		const float smf8 = - f[8] - f[21];
		const float smf9 = + f[11] + f[21];
		const float smf10 = - f[11] - f[21];
		const float smf11 = + f[8] + f[21];
		const float smf12 = - f[11] - f[21];
		const float smf13 = + f[14] + f[21];
		const float smf14 = - f[8] - f[21];
		const float smf15 = - f[14] - f[21];
		const float smf16 = + f[14] + f[21];
		const float smf17 = + f[8] + f[21];
		const float smf18 = + f[11] + f[21];
		// Calculate equilibrium moments
		const float m0 = rho;
		const float m1 = rho * ux;
		const float m2 = rho * uy;
		const float m3 = rho * uz;
		const float m4 = (1.f/3.f) * rho + rho * ux * ux;
		const float m5 = (1.f/3.f) * rho + rho * uy * uy;
		const float m6 = (1.f/3.f) * rho + rho * uz * uz;
		const float m7 = rho * uy * uz;
		const float m8 = rho * ux * uz;
		const float m9 = rho * ux * uy;
		const float m10 = (1.f/3.f) * rho * uy + rho * ux * ux * uy;
		const float m11 = (1.f/3.f) * rho * uz + rho * ux * ux * uz;
		const float m12 = (1.f/3.f) * rho * ux + rho * ux * uy * uy;
		const float m13 = (1.f/3.f) * rho * uz + rho * uy * uy * uz;
		const float m14 = (1.f/3.f) * rho * ux + rho * ux * uz * uz;
		const float m15 = (1.f/3.f) * rho * uy + rho * uy * uz * uz;
		const float m16 = (1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz;
		const float m17 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz;
		const float m18 = (1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy;
		// Subtract Sm - S Mfk fk
		const float s0 = m0 - smf0;
		const float s1 = m1 - smf1;
		const float s2 = m2 - smf2;
		const float s3 = m3 - smf3;
		const float s4 = m4 - smf4;
		const float s5 = m5 - smf5;
		const float s6 = m6 - smf6;
		const float s7 = m7 - smf7;
		const float s8 = m8 - smf8;
		const float s9 = m9 - smf9;
		const float s10 = m10 - smf10;
		const float s11 = m11 - smf11;
		const float s12 = m12 - smf12;
		const float s13 = m13 - smf13;
		const float s14 = m14 - smf14;
		const float s15 = m15 - smf15;
		const float s16 = m16 - smf16;
		const float s17 = m17 - smf17;
		const float s18 = m18 - smf18;
		// Multiply (S Mfu)^-1 * Smf to get unknown distributions
		f[1] = + s0 - s5 - s6 + s16;
		f[3] = + s0 - s4 - s5 + s18;
		f[6] = + s0 - s4 - s6 + s17;
		f[7] = - s0 + (1.f/2.f) * s1 + (-1.f/2.f) * s3 + (1.f/2.f) * s4 + s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s12 + (1.f/2.f) * s13 + (-1.f/2.f) * s16 + (-1.f/2.f) * s18;
		f[9] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s13 + (-1.f/2.f) * s16;
		f[10] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s12 + (-1.f/2.f) * s18;
		f[12] = - s0 + (1.f/2.f) * s1 + (1.f/2.f) * s2 + (1.f/2.f) * s4 + (1.f/2.f) * s5 + s6 + (-1.f/2.f) * s14 + (-1.f/2.f) * s15 + (-1.f/2.f) * s16 + (-1.f/2.f) * s17;
		f[13] = - s0 + (1.f/2.f) * s2 + (-1.f/2.f) * s3 + s4 + (1.f/2.f) * s5 + (1.f/2.f) * s6 + (-1.f/2.f) * s10 + (1.f/2.f) * s11 + (-1.f/2.f) * s17 + (-1.f/2.f) * s18;
		f[15] = + (-1.f/2.f) * s1 + (1.f/2.f) * s4 + (1.f/2.f) * s14 + (-1.f/2.f) * s17;
		f[16] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s15 + (-1.f/2.f) * s16;
		f[17] = + (1.f/2.f) * s3 + (1.f/2.f) * s6 + (-1.f/2.f) * s11 + (-1.f/2.f) * s17;
		f[18] = + (-1.f/2.f) * s2 + (1.f/2.f) * s5 + (1.f/2.f) * s10 + (-1.f/2.f) * s18;
		f[19] = + (1.f/2.f) * s1 + (-1.f/2.f) * s4 + (1.f/4.f) * s8 + (-1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (-1.f/4.f) * s12 + (-1.f/4.f) * s14 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[20] = + (-1.f/4.f) * s7 + (1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16;
		f[22] = + s0 + (-1.f/2.f) * s1 + (-1.f/2.f) * s2 + (1.f/2.f) * s3 + (-1.f/2.f) * s4 + (-1.f/2.f) * s5 + (-1.f/2.f) * s6 + (-1.f/4.f) * s7 + (-1.f/4.f) * s8 + (1.f/4.f) * s9 + (1.f/4.f) * s10 + (-1.f/4.f) * s11 + (1.f/4.f) * s12 + (-1.f/4.f) * s13 + (1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17 + (1.f/4.f) * s18;
		f[23] = + (1.f/2.f) * s2 + (-1.f/2.f) * s5 + (1.f/4.f) * s7 + (-1.f/4.f) * s9 + (-1.f/4.f) * s10 + (1.f/4.f) * s12 + (-1.f/4.f) * s13 + (-1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s18;
		f[24] = + (-1.f/4.f) * s8 + (1.f/4.f) * s11 + (-1.f/4.f) * s14 + (1.f/4.f) * s17;
		f[25] = + (1.f/4.f) * s9 + (-1.f/4.f) * s10 + (-1.f/4.f) * s12 + (1.f/4.f) * s18;
		f[26] = + (-1.f/2.f) * s3 + (-1.f/2.f) * s6 + (1.f/4.f) * s7 + (1.f/4.f) * s8 + (1.f/4.f) * s11 + (1.f/4.f) * s13 + (1.f/4.f) * s14 + (1.f/4.f) * s15 + (1.f/4.f) * s16 + (1.f/4.f) * s17;
	}
}