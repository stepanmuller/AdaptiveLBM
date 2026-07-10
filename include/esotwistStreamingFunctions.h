// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// Esotwist streaming step: Just flip the EsotwistFlipper to alter between odd / even iterations
void applyStreaming( GridStruct& Grid )
{
	Grid.esotwistFlipper = !Grid.esotwistFlipper;
}

__cuda_callable__ void getPreCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NbrStruct &Nbr, const bool &esotwistFlipper, const InfoStruct &Info )
{
    cellIndex[0]  = Nbr.self;
    cellIndex[1]  = Nbr.self;
    cellIndex[2]  = Nbr.iPlus;
    cellIndex[3]  = Nbr.kPlus;
    cellIndex[4]  = Nbr.self;
    cellIndex[5]  = Nbr.jPlus;
    cellIndex[6]  = Nbr.self;
    cellIndex[7]  = Nbr.kPlus;
    cellIndex[8]  = Nbr.iPlus;
    cellIndex[9]  = Nbr.self;
    cellIndex[10] = Nbr.ikPlus;
    cellIndex[11] = Nbr.ijPlus;
    cellIndex[12] = Nbr.self;
    cellIndex[13] = Nbr.kPlus;
    cellIndex[14] = Nbr.jPlus;
    cellIndex[15] = Nbr.iPlus;
    cellIndex[16] = Nbr.jPlus;
    cellIndex[17] = Nbr.self;
    cellIndex[18] = Nbr.jkPlus;
    cellIndex[19] = Nbr.ikPlus;
    cellIndex[20] = Nbr.jPlus;
    cellIndex[21] = Nbr.ijPlus;
    cellIndex[22] = Nbr.kPlus;
    cellIndex[23] = Nbr.jkPlus;
    cellIndex[24] = Nbr.iPlus;
    cellIndex[25] = Nbr.ijkPlus;
    cellIndex[26] = Nbr.self;

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

__cuda_callable__ void getPostCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NbrStruct &Nbr, const bool &esotwistFlipper, const InfoStruct &Info )
{
    cellIndex[0]  = Nbr.self;
    cellIndex[1]  = Nbr.iPlus;
    cellIndex[2]  = Nbr.self;
    cellIndex[3]  = Nbr.self;
    cellIndex[4]  = Nbr.kPlus;
    cellIndex[5]  = Nbr.self;
    cellIndex[6]  = Nbr.jPlus;
    cellIndex[7]  = Nbr.iPlus;
    cellIndex[8]  = Nbr.kPlus;
    cellIndex[9]  = Nbr.ikPlus;
    cellIndex[10] = Nbr.self;
    cellIndex[11] = Nbr.self;
    cellIndex[12] = Nbr.ijPlus;
    cellIndex[13] = Nbr.jPlus;
    cellIndex[14] = Nbr.kPlus;
    cellIndex[15] = Nbr.jPlus;
    cellIndex[16] = Nbr.iPlus;
    cellIndex[17] = Nbr.jkPlus;
    cellIndex[18] = Nbr.self;
    cellIndex[19] = Nbr.jPlus;
    cellIndex[20] = Nbr.ikPlus;
    cellIndex[21] = Nbr.kPlus;
    cellIndex[22] = Nbr.ijPlus;
    cellIndex[23] = Nbr.iPlus;
    cellIndex[24] = Nbr.jkPlus;
    cellIndex[25] = Nbr.self;
    cellIndex[26] = Nbr.ijkPlus;

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
