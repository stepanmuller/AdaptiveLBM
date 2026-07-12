#pragma once

// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

__host__ __device__ void applyMovingBounceback( float (&f)[27], const BCRhoUGStruct &BCRhoUG )
{
	
	const float ux = BCRhoUG.ux;
	const float uy = BCRhoUG.uy;
	const float uz = BCRhoUG.uz;
	
	float fTemp, cu;

	// Precalculated multipliers: 2 * w_i / c_s^2 (where c_s^2 = 1/3) -> 6 * w_i
	const float w1 = 4.0f / 9.0f;  // Face centers (w = 2/27)
	const float w2 = 1.0f / 9.0f;  // Edge centers (w = 1/54)
	const float w3 = 1.0f / 36.0f; // Corners      (w = 1/216)

	// --- Group 1: Face centers ---
	cu = ux;
	fTemp = f[1]; f[1] = f[2] + w1 * cu; f[2] = fTemp - w1 * cu;

	cu = -uz;
	fTemp = f[3]; f[3] = f[4] + w1 * cu; f[4] = fTemp - w1 * cu;

	cu = -uy;
	fTemp = f[5]; f[5] = f[6] + w1 * cu; f[6] = fTemp - w1 * cu;


	// --- Group 2: Edge centers ---
	cu = ux - uz;
	fTemp = f[7]; f[7] = f[8] + w2 * cu; f[8] = fTemp - w2 * cu;

	cu = ux + uz;
	fTemp = f[9]; f[9] = f[10] + w2 * cu; f[10] = fTemp - w2 * cu;

	cu = -ux - uy;
	fTemp = f[11]; f[11] = f[12] + w2 * cu; f[12] = fTemp - w2 * cu;

	cu = uy - uz;
	fTemp = f[13]; f[13] = f[14] + w2 * cu; f[14] = fTemp - w2 * cu;

	cu = -ux + uy;
	fTemp = f[15]; f[15] = f[16] + w2 * cu; f[16] = fTemp - w2 * cu;

	cu = uy + uz;
	fTemp = f[17]; f[17] = f[18] + w2 * cu; f[18] = fTemp - w2 * cu;


	// --- Group 3: Corners ---
	cu = -ux + uy - uz;
	fTemp = f[19]; f[19] = f[20] + w3 * cu; f[20] = fTemp - w3 * cu;

	cu = -ux - uy + uz;
	fTemp = f[21]; f[21] = f[22] + w3 * cu; f[22] = fTemp - w3 * cu;

	cu = ux - uy - uz;
	fTemp = f[23]; f[23] = f[24] + w3 * cu; f[24] = fTemp - w3 * cu;

	cu = -ux - uy - uz;
	fTemp = f[25]; f[25] = f[26] + w3 * cu; f[26] = fTemp - w3 * cu;
}
