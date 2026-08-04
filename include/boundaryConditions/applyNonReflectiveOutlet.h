#pragma once

// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cx is negative for: { 2, 8, 10, 11, 15, 19, 21, 24, 25 };
// cx is positive for: { 1, 7, 9, 12, 16, 20, 22, 23, 26 };
// cy is negative for: { 5, 11, 14, 16, 18, 20, 21, 23, 25 };
// cy is positive for: { 6, 12, 13, 15, 17, 19, 22, 24, 26 };
// cz is negative for: { 3, 7, 10, 13, 18, 19, 22, 23, 25 };
// cz is positive for: { 4, 8, 9, 14, 17, 20, 21, 24, 26};

// Better idea, to implement later:
// The issue with this outlet is that as soon as backflow happens anywhere in the area,
// it crashes. I tried to fix this by smoothly switching to hard pressure MBBC, but then
// I have to manually choose the velocity threshold and even then there is still a risk
// of creating artifacts in places where backflow happens.
// Now, what I can do is this:
// Run a reduction on the non reflective outlet cells and calculate their average pressure
// There is no need to write the resulting f, just save the pressure to Info
// Then apply a completely normal and reliable MBBC but use some interpolated
// value between the non reflective average pressure and the target pressure

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
		const int* directionList = nullptr;
		static const int dir_xp[] = { 2, 8, 10, 11, 15, 19, 21, 24, 25 };
		static const int dir_xm[] = { 1, 7, 9, 12, 16, 20, 22, 23, 26 };
		static const int dir_yp[] = { 5, 11, 14, 16, 18, 20, 21, 23, 25 };
		static const int dir_ym[] = { 6, 12, 13, 15, 17, 19, 22, 24, 26 };
		static const int dir_zp[] = { 3, 7, 10, 13, 18, 19, 22, 23, 25 };
		static const int dir_zm[] = { 4, 8, 9, 14, 17, 20, 21, 24, 26 };
		if ( outerNormalX > 0 ) { 
			upstreamCell = cell-1; if ( upstreamCell < 0 ) upstreamCell = Info.cellCount-1;
			directionList = dir_xp;
		}
		else if ( outerNormalX < 0 ) { 
			upstreamCell = cell+1; if ( upstreamCell >= Info.cellCount ) upstreamCell = 0;
			directionList = dir_xm;
		}
		else if ( outerNormalY > 0 ) { 
			upstreamCell = jMinusView( cell );
			directionList = dir_yp;
		}
		else if ( outerNormalY < 0 ) { 
			upstreamCell = jPlusView( cell );
			directionList = dir_ym;
		}
		else if ( outerNormalZ > 0 ) { 
			upstreamCell = kMinusView( cell );
			directionList = dir_zp;
		}
		else { 
			upstreamCell = kPlusView( cell );
			directionList = dir_zm;
		}
		
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
		for (int i = 0; i < 9; i++)
		{
			const int direction = directionList[i];
			f[direction] = 0.577350269f * fView(upstreamCellfReadIndex[direction], upstreamCellCellReadIndex[direction]) + (1.f - 0.577350269f) * fView(fReadIndex[direction], cellReadIndex[direction]);
		}
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for (int i = 0; i < 9; i++)
		{
			const int direction = directionList[i];
			fView(fWriteIndex[direction], cellWriteIndex[direction]) = f[direction];
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.nonReflectiveOutletCount, cellLambda );
}
