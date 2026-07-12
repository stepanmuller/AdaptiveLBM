#pragma once

// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// Esotwist streaming step: Just flip the EsotwistFlipper to alter between odd / even iterations
void applyStreaming( GridStruct& Grid )
{
	Grid.esotwistFlipper = !Grid.esotwistFlipper;
}

__cuda_callable__ void getPreCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NBRStruct &NBR, const bool &esotwistFlipper, const InfoStruct &Info )
{
    cellIndex[0]  = NBR.self;
    cellIndex[1]  = NBR.self;
    cellIndex[2]  = NBR.iPlus;
    cellIndex[3]  = NBR.kPlus;
    cellIndex[4]  = NBR.self;
    cellIndex[5]  = NBR.jPlus;
    cellIndex[6]  = NBR.self;
    cellIndex[7]  = NBR.kPlus;
    cellIndex[8]  = NBR.iPlus;
    cellIndex[9]  = NBR.self;
    cellIndex[10] = NBR.ikPlus;
    cellIndex[11] = NBR.ijPlus;
    cellIndex[12] = NBR.self;
    cellIndex[13] = NBR.kPlus;
    cellIndex[14] = NBR.jPlus;
    cellIndex[15] = NBR.iPlus;
    cellIndex[16] = NBR.jPlus;
    cellIndex[17] = NBR.self;
    cellIndex[18] = NBR.jkPlus;
    cellIndex[19] = NBR.ikPlus;
    cellIndex[20] = NBR.jPlus;
    cellIndex[21] = NBR.ijPlus;
    cellIndex[22] = NBR.kPlus;
    cellIndex[23] = NBR.jkPlus;
    cellIndex[24] = NBR.iPlus;
    cellIndex[25] = NBR.ijkPlus;
    cellIndex[26] = NBR.self;

    if ( esotwistFlipper )
    {
        fIndex[0]  = 0;   fIndex[1]  = 1;   fIndex[2]  = 2;
        fIndex[3]  = 3;   fIndex[4]  = 4;   fIndex[5]  = 5;
        fIndex[6]  = 6;   fIndex[7]  = 7;   fIndex[8]  = 8;
        fIndex[9]  = 9;   fIndex[10] = 10;  fIndex[11] = 11;
        fIndex[12] = 12;  fIndex[13] = 13;  fIndex[14] = 14;
        fIndex[15] = 15;  fIndex[16] = 16;  fIndex[17] = 17;
        fIndex[18] = 18;  fIndex[19] = 19;  fIndex[20] = 20;
        fIndex[21] = 21;  fIndex[22] = 22;  fIndex[23] = 23;
        fIndex[24] = 24;  fIndex[25] = 25;  fIndex[26] = 26;
    }
    else
    {
        fIndex[0]  = 0;   fIndex[1]  = 2;   fIndex[2]  = 1;
        fIndex[3]  = 4;   fIndex[4]  = 3;   fIndex[5]  = 6;
        fIndex[6]  = 5;   fIndex[7]  = 8;   fIndex[8]  = 7;
        fIndex[9]  = 10;  fIndex[10] = 9;   fIndex[11] = 12;
        fIndex[12] = 11;  fIndex[13] = 14;  fIndex[14] = 13;
        fIndex[15] = 16;  fIndex[16] = 15;  fIndex[17] = 18;
        fIndex[18] = 17;  fIndex[19] = 20;  fIndex[20] = 19;
        fIndex[21] = 22;  fIndex[22] = 21;  fIndex[23] = 24;
        fIndex[24] = 23;  fIndex[25] = 26;  fIndex[26] = 25;
    }
}

__cuda_callable__ void getPostCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NBRStruct &NBR, const bool &esotwistFlipper, const InfoStruct &Info )
{
    cellIndex[0]  = NBR.self;
    cellIndex[1]  = NBR.iPlus;
    cellIndex[2]  = NBR.self;
    cellIndex[3]  = NBR.self;
    cellIndex[4]  = NBR.kPlus;
    cellIndex[5]  = NBR.self;
    cellIndex[6]  = NBR.jPlus;
    cellIndex[7]  = NBR.iPlus;
    cellIndex[8]  = NBR.kPlus;
    cellIndex[9]  = NBR.ikPlus;
    cellIndex[10] = NBR.self;
    cellIndex[11] = NBR.self;
    cellIndex[12] = NBR.ijPlus;
    cellIndex[13] = NBR.jPlus;
    cellIndex[14] = NBR.kPlus;
    cellIndex[15] = NBR.jPlus;
    cellIndex[16] = NBR.iPlus;
    cellIndex[17] = NBR.jkPlus;
    cellIndex[18] = NBR.self;
    cellIndex[19] = NBR.jPlus;
    cellIndex[20] = NBR.ikPlus;
    cellIndex[21] = NBR.kPlus;
    cellIndex[22] = NBR.ijPlus;
    cellIndex[23] = NBR.iPlus;
    cellIndex[24] = NBR.jkPlus;
    cellIndex[25] = NBR.self;
    cellIndex[26] = NBR.ijkPlus;

    if ( esotwistFlipper )
    {
        fIndex[0]  = 0;   fIndex[1]  = 2;   fIndex[2]  = 1;
        fIndex[3]  = 4;   fIndex[4]  = 3;   fIndex[5]  = 6;
        fIndex[6]  = 5;   fIndex[7]  = 8;   fIndex[8]  = 7;
        fIndex[9]  = 10;  fIndex[10] = 9;   fIndex[11] = 12;
        fIndex[12] = 11;  fIndex[13] = 14;  fIndex[14] = 13;
        fIndex[15] = 16;  fIndex[16] = 15;  fIndex[17] = 18;
        fIndex[18] = 17;  fIndex[19] = 20;  fIndex[20] = 19;
        fIndex[21] = 22;  fIndex[22] = 21;  fIndex[23] = 24;
        fIndex[24] = 23;  fIndex[25] = 26;  fIndex[26] = 25;
    }
    else
    {
        fIndex[0]  = 0;   fIndex[1]  = 1;   fIndex[2]  = 2;
        fIndex[3]  = 3;   fIndex[4]  = 4;   fIndex[5]  = 5;
        fIndex[6]  = 6;   fIndex[7]  = 7;   fIndex[8]  = 8;
        fIndex[9]  = 9;   fIndex[10] = 10;  fIndex[11] = 11;
        fIndex[12] = 12;  fIndex[13] = 13;  fIndex[14] = 14;
        fIndex[15] = 15;  fIndex[16] = 16;  fIndex[17] = 17;
        fIndex[18] = 18;  fIndex[19] = 19;  fIndex[20] = 20;
        fIndex[21] = 21;  fIndex[22] = 22;  fIndex[23] = 23;
        fIndex[24] = 24;  fIndex[25] = 25;  fIndex[26] = 26;
    }
}
