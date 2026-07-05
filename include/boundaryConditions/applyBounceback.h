#pragma once

// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

__host__ __device__ void applyBounceback( float (&f)[27] )
{
	float fTemp;

	fTemp = f[1];  f[1]  = f[2];  f[2]  = fTemp;
	fTemp = f[3];  f[3]  = f[4];  f[4]  = fTemp;
	fTemp = f[5];  f[5]  = f[6];  f[6]  = fTemp;

	fTemp = f[7];  f[7]  = f[8];  f[8]  = fTemp;
	fTemp = f[9];  f[9]  = f[10]; f[10] = fTemp;
	fTemp = f[11]; f[11] = f[12]; f[12] = fTemp;
	fTemp = f[13]; f[13] = f[14]; f[14] = fTemp;
	fTemp = f[15]; f[15] = f[16]; f[16] = fTemp;
	fTemp = f[17]; f[17] = f[18]; f[18] = fTemp;

	fTemp = f[19]; f[19] = f[20]; f[20] = fTemp;
	fTemp = f[21]; f[21] = f[22]; f[22] = fTemp;
	fTemp = f[23]; f[23] = f[24]; f[24] = fTemp;
	fTemp = f[25]; f[25] = f[26]; f[26] = fTemp;
}
