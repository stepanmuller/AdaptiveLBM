// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cz is negative for: { 3, 7, 10, 13, 18, 19, 22, 23, 25 };

// So far works for positive Z direction outlet only!

void applyNonReflectiveOutletZ( GridStruct &Grid )
{
	InfoStruct Info = Grid.Info;
	if ( Info.nonReflectiveOutletCount == 0 ) return;
	
	auto nonReflectiveOutletIndexView = Grid.nonReflectiveOutletIndexArray.getConstView();
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	//auto iView = Grid.IJK.iArray.getConstView();
	//auto jView = Grid.IJK.jArray.getConstView();
	//auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.NBR.jkPlusArray.getConstView();
	// auto jMinusView = Grid.NBR.jMinusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = nonReflectiveOutletIndexView( index );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jkPlusView( cell );
		finishNBRPlus( NBR, Info );
		
		const int kMin = kMinusView( cell );
		
		NBRStruct kMinNBR;
		kMinNBR.self = kMin;
		kMinNBR.jPlus = jPlusView( kMin );
		kMinNBR.kPlus = kPlusView( kMin );
		kMinNBR.jkPlus = jkPlusView( kMin );
		finishNBRPlus( kMinNBR, Info );
		
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		int kMinCellReadIndex[27];
		int kMinfReadIndex[27];
		getPreviousPostCollisionIndex( kMinCellReadIndex, kMinfReadIndex, kMinNBR, esotwistFlipper, Info );
		
		float f[27];
		for (int direction : { 3, 7, 10, 13, 18, 19, 22, 23, 25 })
		{
			f[direction] = 0.577350269f * fView(kMinfReadIndex[direction], kMinCellReadIndex[direction]) + (1.f - 0.577350269f) * fView(fReadIndex[direction], cellReadIndex[direction]);
		}
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for (int direction : { 3, 7, 10, 13, 18, 19, 22, 23, 25 })
		{
			fView(fWriteIndex[direction], cellWriteIndex[direction]) = f[direction];
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.nonReflectiveOutletCount, cellLambda );
}
