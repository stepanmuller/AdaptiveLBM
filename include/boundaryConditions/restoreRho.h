__host__ __device__ void restoreRho(
	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,
	float &rho, const float &ux, const float &uy, const float &uz,
	const float (&f)[27]
)
{
	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);
	if ( normalCode == 655 )
	{
		const float scqu = + (1.f) * ux + (0.f) * uy + (0.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[1] + f[3] + f[4] + f[5] + f[6] + (2.f) * f[7] + (2.f) * f[9] + (2.f) * f[12] + f[13] + f[14] + (2.f) * f[16] + f[17] + f[18] + (2.f) * f[20] + (2.f) * f[22] + (2.f) * f[23] + (2.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 565 )
	{
		const float scqu = + (0.f) * ux + (1.f) * uy + (0.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + f[1] + f[2] + f[3] + f[4] + (2.f) * f[6] + f[7] + f[8] + f[9] + f[10] + (2.f) * f[12] + (2.f) * f[13] + (2.f) * f[15] + (2.f) * f[17] + (2.f) * f[19] + (2.f) * f[22] + (2.f) * f[24] + (2.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 556 )
	{
		const float scqu = + (0.f) * ux + (0.f) * uy + (1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + f[1] + f[2] + (2.f) * f[4] + f[5] + f[6] + (2.f) * f[8] + (2.f) * f[9] + f[11] + f[12] + (2.f) * f[14] + f[15] + f[16] + (2.f) * f[17] + (2.f) * f[20] + (2.f) * f[21] + (2.f) * f[24] + (2.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 455 )
	{
		const float scqu = + (1.f) * ux + (0.f) * uy + (0.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[2] - f[3] - f[4] - f[5] - f[6] + (-2.f) * f[8] + (-2.f) * f[10] + (-2.f) * f[11] - f[13] - f[14] + (-2.f) * f[15] - f[17] - f[18] + (-2.f) * f[19] + (-2.f) * f[21] + (-2.f) * f[24] + (-2.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 545 )
	{
		const float scqu = + (0.f) * ux + (1.f) * uy + (0.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] - f[1] - f[2] - f[3] - f[4] + (-2.f) * f[5] - f[7] - f[8] - f[9] - f[10] + (-2.f) * f[11] + (-2.f) * f[14] + (-2.f) * f[16] + (-2.f) * f[18] + (-2.f) * f[20] + (-2.f) * f[21] + (-2.f) * f[23] + (-2.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 554 )
	{
		const float scqu = + (0.f) * ux + (0.f) * uy + (1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] - f[1] - f[2] + (-2.f) * f[3] - f[5] - f[6] + (-2.f) * f[7] + (-2.f) * f[10] - f[11] - f[12] + (-2.f) * f[13] - f[15] - f[16] + (-2.f) * f[18] + (-2.f) * f[19] + (-2.f) * f[22] + (-2.f) * f[23] + (-2.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 566 )
	{
		const float scqu = + (0.f) * ux + (1.f) * uy + (1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + f[1] + f[2] + (2.f) * f[4] + (2.f) * f[6] + (2.f) * f[8] + (2.f) * f[9] + (2.f) * f[12] + (2.f) * f[15] + (4.f) * f[17] + (4.f) * f[24] + (4.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 656 )
	{
		const float scqu = + (1.f) * ux + (0.f) * uy + (1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[1] + (2.f) * f[4] + f[5] + f[6] + (4.f) * f[9] + (2.f) * f[12] + (2.f) * f[14] + (2.f) * f[16] + (2.f) * f[17] + (4.f) * f[20] + (4.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 665 )
	{
		const float scqu = + (1.f) * ux + (1.f) * uy + (0.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[1] + f[3] + f[4] + (2.f) * f[6] + (2.f) * f[7] + (2.f) * f[9] + (4.f) * f[12] + (2.f) * f[13] + (2.f) * f[17] + (4.f) * f[22] + (4.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 544 )
	{
		const float scqu = + (0.f) * ux + (-1.f) * uy + (-1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + f[1] + f[2] + (2.f) * f[3] + (2.f) * f[5] + (2.f) * f[7] + (2.f) * f[10] + (2.f) * f[11] + (2.f) * f[16] + (4.f) * f[18] + (4.f) * f[23] + (4.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 454 )
	{
		const float scqu = + (-1.f) * ux + (0.f) * uy + (-1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[2] + (2.f) * f[3] + f[5] + f[6] + (4.f) * f[10] + (2.f) * f[11] + (2.f) * f[13] + (2.f) * f[15] + (2.f) * f[18] + (4.f) * f[19] + (4.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 445 )
	{
		const float scqu = + (-1.f) * ux + (-1.f) * uy + (0.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[2] + f[3] + f[4] + (2.f) * f[5] + (2.f) * f[8] + (2.f) * f[10] + (4.f) * f[11] + (2.f) * f[14] + (2.f) * f[18] + (4.f) * f[21] + (4.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 564 )
	{
		const float scqu = + (0.f) * ux + (-1.f) * uy + (1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] - f[1] - f[2] + (-2.f) * f[3] + (-2.f) * f[6] + (-2.f) * f[7] + (-2.f) * f[10] + (-2.f) * f[12] + (-4.f) * f[13] + (-2.f) * f[15] + (-4.f) * f[19] + (-4.f) * f[22];
		rho = scmf / s;
	}
	else if ( normalCode == 654 )
	{
		const float scqu = + (-1.f) * ux + (0.f) * uy + (1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[1] + (-2.f) * f[3] - f[5] - f[6] + (-4.f) * f[7] + (-2.f) * f[12] + (-2.f) * f[13] + (-2.f) * f[16] + (-2.f) * f[18] + (-4.f) * f[22] + (-4.f) * f[23];
		rho = scmf / s;
	}
	else if ( normalCode == 645 )
	{
		const float scqu = + (-1.f) * ux + (1.f) * uy + (0.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[1] - f[3] - f[4] + (-2.f) * f[5] + (-2.f) * f[7] + (-2.f) * f[9] + (-2.f) * f[14] + (-4.f) * f[16] + (-2.f) * f[18] + (-4.f) * f[20] + (-4.f) * f[23];
		rho = scmf / s;
	}
	else if ( normalCode == 546 )
	{
		const float scqu = + (0.f) * ux + (1.f) * uy + (-1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] - f[1] - f[2] + (-2.f) * f[4] + (-2.f) * f[5] + (-2.f) * f[8] + (-2.f) * f[9] + (-2.f) * f[11] + (-4.f) * f[14] + (-2.f) * f[16] + (-4.f) * f[20] + (-4.f) * f[21];
		rho = scmf / s;
	}
	else if ( normalCode == 456 )
	{
		const float scqu = + (1.f) * ux + (0.f) * uy + (-1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[2] + (-2.f) * f[4] - f[5] - f[6] + (-4.f) * f[8] + (-2.f) * f[11] + (-2.f) * f[14] + (-2.f) * f[15] + (-2.f) * f[17] + (-4.f) * f[21] + (-4.f) * f[24];
		rho = scmf / s;
	}
	else if ( normalCode == 465 )
	{
		const float scqu = + (1.f) * ux + (-1.f) * uy + (0.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[2] - f[3] - f[4] + (-2.f) * f[6] + (-2.f) * f[8] + (-2.f) * f[10] + (-2.f) * f[13] + (-4.f) * f[15] + (-2.f) * f[17] + (-4.f) * f[19] + (-4.f) * f[24];
		rho = scmf / s;
	}
	else if ( normalCode == 666 )
	{
		const float scqu = + (1.f) * ux + (1.f) * uy + (1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[1] + (2.f) * f[4] + (2.f) * f[6] + (4.f) * f[9] + (4.f) * f[12] + (4.f) * f[17] + (8.f) * f[26];
		rho = scmf / s;
	}
	else if ( normalCode == 444 )
	{
		const float scqu = + (1.f) * ux + (1.f) * uy + (1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[2] + (-2.f) * f[3] + (-2.f) * f[5] + (-4.f) * f[10] + (-4.f) * f[11] + (-4.f) * f[18] + (-8.f) * f[25];
		rho = scmf / s;
	}
	else if ( normalCode == 466 )
	{
		const float scqu = + (1.f) * ux + (-1.f) * uy + (-1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[2] + (-2.f) * f[4] + (-2.f) * f[6] + (-4.f) * f[8] + (-4.f) * f[15] + (-4.f) * f[17] + (-8.f) * f[24];
		rho = scmf / s;
	}
	else if ( normalCode == 646 )
	{
		const float scqu = + (-1.f) * ux + (1.f) * uy + (-1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[1] + (-2.f) * f[4] + (-2.f) * f[5] + (-4.f) * f[9] + (-4.f) * f[14] + (-4.f) * f[16] + (-8.f) * f[20];
		rho = scmf / s;
	}
	else if ( normalCode == 664 )
	{
		const float scqu = + (-1.f) * ux + (-1.f) * uy + (1.f) * uz;
		const float s = -1 + scqu;
		const float scmf = - f[0] + (-2.f) * f[1] + (-2.f) * f[3] + (-2.f) * f[6] + (-4.f) * f[7] + (-4.f) * f[12] + (-4.f) * f[13] + (-8.f) * f[22];
		rho = scmf / s;
	}
	else if ( normalCode == 644 )
	{
		const float scqu = + (1.f) * ux + (-1.f) * uy + (-1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[1] + (2.f) * f[3] + (2.f) * f[5] + (4.f) * f[7] + (4.f) * f[16] + (4.f) * f[18] + (8.f) * f[23];
		rho = scmf / s;
	}
	else if ( normalCode == 464 )
	{
		const float scqu = + (-1.f) * ux + (1.f) * uy + (-1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[2] + (2.f) * f[3] + (2.f) * f[6] + (4.f) * f[10] + (4.f) * f[13] + (4.f) * f[15] + (8.f) * f[19];
		rho = scmf / s;
	}
	else if ( normalCode == 446 )
	{
		const float scqu = + (-1.f) * ux + (-1.f) * uy + (1.f) * uz;
		const float s = 1 + scqu;
		const float scmf = + f[0] + (2.f) * f[2] + (2.f) * f[4] + (2.f) * f[5] + (4.f) * f[8] + (4.f) * f[11] + (4.f) * f[14] + (8.f) * f[21];
		rho = scmf / s;
	}
}
