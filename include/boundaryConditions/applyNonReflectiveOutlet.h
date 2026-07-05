// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cz is negative for: { 3, 7, 10, 13, 18, 19, 22, 23, 25 };

// Works for positive Z direction outlet only!
void applyNonReflectiveOutletZ( DIADGridStruct &Grid )
{
	InfoStruct Info = Grid.Info;
	auto outletCellView = Grid.outletCellArray.getConstView();
	
	auto fArrayView  = Grid.fArray.getView();
	
	bool esotwistFlipper = Grid.esotwistFlipper;
	auto iNbrView = Grid.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrView = Grid.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrView = Grid.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrView = Grid.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrView = Grid.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrView = Grid.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrView = Grid.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	auto iMinNbrView = Grid.EsotwistNbrArray.iMinNbrArray.getConstView();
	auto jMinNbrView = Grid.EsotwistNbrArray.jMinNbrArray.getConstView();
	auto kMinNbrView = Grid.EsotwistNbrArray.kMinNbrArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = outletCellView( index );
		
		DIADEsotwistNbrStruct Nbr;
		Nbr.i = iNbrView( cell );
		Nbr.j = jNbrView( cell );
		Nbr.k = kNbrView( cell );
		Nbr.ij = ijNbrView( cell );
		Nbr.ik = ikNbrView( cell );
		Nbr.jk = jkNbrView( cell );
		Nbr.ijk = ijkNbrView( cell ); 
		
		const int kMin = kMinNbrView( cell );
		
		DIADEsotwistNbrStruct kMinNbr;
		kMinNbr.i = iNbrView( kMin );
		kMinNbr.j = jNbrView( kMin );
		kMinNbr.k = kNbrView( kMin );
		kMinNbr.ij = ijNbrView( kMin );
		kMinNbr.ik = ikNbrView( kMin );
		kMinNbr.jk = jkNbrView( kMin );
		kMinNbr.ijk = ijkNbrView( kMin ); 
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getEsotwistWriteIndex( cell, cellWriteIndex, fWriteIndex, Nbr, esotwistFlipper, Info );
		int kMinCellWriteIndex[27];
		int kMinfWriteIndex[27];
		getEsotwistWriteIndex( kMin, kMinCellWriteIndex, kMinfWriteIndex, kMinNbr, esotwistFlipper, Info );
		
		float f[27];
		for (int direction : { 3, 7, 10, 13, 18, 19, 22, 23, 25 })
		{
			f[direction] = invSqrt3 * fArrayView(kMinfWriteIndex[direction], kMinCellWriteIndex[direction]) + (1.f - invSqrt3) * fArrayView(fWriteIndex[direction], cellWriteIndex[direction]);
		}
		
		const bool esotwistFlipperInverted = !esotwistFlipper;
		int cellReadIndex[27];
		int fReadIndex[27];
		getEsotwistReadIndex( cell, cellReadIndex, fReadIndex, Nbr, esotwistFlipperInverted, Info );
		for (int direction : { 3, 7, 10, 13, 18, 19, 22, 23, 25 })
		{
			fArrayView(fReadIndex[direction], cellReadIndex[direction]) = f[direction];
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.outletCellArray.getSize(), cellLambda );
}
