// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cz is negative for: { 3, 7, 10, 13, 18, 19, 22, 23, 25 };

void applyNonReflectiveOutlet( GridStruct &Grid )
{
	InfoStruct Info = Grid.Info;
	if ( Info.nonReflectiveOutletCount == 0 ) return;
	
	auto nonReflectiveOutletIndexView = Grid.nonReflectiveOutletIndexArray.getConstView();
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jMinusView = Grid.NBR.jMinusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = nonReflectiveOutletIndexView( index );
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		int outerNormalX, outerNormalY, outerNormalZ;
		getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		int upstreamCell;
		int directionList[9];
		if ( outerNormalX > 0 ) { 
			upstreamCell = 
		}
		
		const int upstreamCell = kMinusView( cell );
		
		NBRStruct upstreamCellNBR;
		upstreamCellNBR.self = upstreamCell;
		upstreamCellNBR.jPlus = jPlusView( upstreamCell );
		upstreamCellNBR.kPlus = kPlusView( upstreamCell );
		upstreamCellNBR.jkPlus = jPlusView( upstreamCellNBR.kPlus );
		finishNBRPlus( upstreamCellNBR, Info );
		
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		int upstreamCellCellReadIndex[27];
		int upstreamCellfReadIndex[27];
		getPreviousPostCollisionIndex( upstreamCellCellReadIndex, upstreamCellfReadIndex, upstreamCellNBR, esotwistFlipper, Info );
		
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
