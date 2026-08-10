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

// Schlaffer disertation 2013 - non reflective impedance based inlet with fixed reference point

void applyNonReflectiveInlet( GridStruct &Grid )
{
	InfoStruct &Info = Grid.Info;
	if ( Info.nonReflectiveInletCount == 0 ) return;
	
	auto nonReflectiveInletIndexView = Grid.nonReflectiveInletIndexArray.getConstView();
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	
	auto fetch = [=] __cuda_callable__ ( const int index ) -> MultiResultHolder
	{
		const int cell = nonReflectiveInletIndexView( index );
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
		
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		
		const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
		const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
		const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
		
		MarkerStruct Marker;
		Marker.nonReflectiveInlet = 1;
		BCStruct BC;
		// pass the current state into the boundary condition function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, U
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		
		float rhoZ = 1.f;
		for (int direction = 0; direction < 27; direction++)
		{
			const int cx = cxArray[direction]; const int cy = cyArray[direction]; const int cz = czArray[direction];
			if ( outerNormalX != 0 )
			{
				if ( cx == 0 ) rhoZ += fView(fReadIndex[direction], cellReadIndex[direction]);
				else if ( cx * outerNormalX > 0 ) rhoZ += 2.f * fView(fReadIndex[direction], cellReadIndex[direction]);
			}
			else if ( outerNormalY != 0 )
			{
				if ( cy == 0 ) rhoZ += fView(fReadIndex[direction], cellReadIndex[direction]);
				else if ( cy * outerNormalY > 0 ) rhoZ += 2.f * fView(fReadIndex[direction], cellReadIndex[direction]);
			}
			else
			{
				if ( cz == 0 ) rhoZ += fView(fReadIndex[direction], cellReadIndex[direction]);
				else if ( cz * outerNormalZ > 0 ) rhoZ += 2.f * fView(fReadIndex[direction], cellReadIndex[direction]);
			}
		}
		
		const float rhoZInv = 1.f / rhoZ; // rhoOld / rhoZ;
		const float temp = 0.577350269f + 1.f/3.f * rhoZInv;
		
		float uImpAbs;
		if ( outerNormalX != 0 )
		{
			uImpAbs = TNL::abs( BC.ux ) - temp + TNL::sqrt( temp*temp - 2.f/3.f * ( rhoZInv * ( TNL::abs( BC.ux ) - 1.f ) + 1.f ) );
		}
		else if ( outerNormalY != 0 )
		{
			uImpAbs = TNL::abs( BC.uy ) - temp + TNL::sqrt( temp*temp - 2.f/3.f * ( rhoZInv * ( TNL::abs( BC.uy ) - 1.f ) + 1.f ) );
		}
		else
		{
			uImpAbs = TNL::abs( BC.uz ) - temp + TNL::sqrt( temp*temp - 2.f/3.f * ( rhoZInv * ( TNL::abs( BC.uz ) - 1.f ) + 1.f ) );
		}	
		float rhoImp = rhoZ / ( 1.f - uImpAbs );
		return { rhoZ - 1.f, rhoImp - 1.f };
		
	};
	auto reduction = [] __cuda_callable__( const MultiResultHolder& a, const MultiResultHolder& b ) -> MultiResultHolder {
		return { a.rhoZSum + b.rhoZSum, a.rhoImpSum + b.rhoImpSum }; };
	
	MultiResultHolder zeros{ 0.f, 0.f };
	
	MultiResultHolder MultiResult = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Info.nonReflectiveInletCount, fetch, reduction, zeros );
	
	const float rhoZAvg = MultiResult.rhoZSum / (float)Info.nonReflectiveInletCount + 1.f;
	const float rhoImpAvg = MultiResult.rhoImpSum / (float)Info.nonReflectiveInletCount + 1.f;
	
	Info.nonReflectiveInletRhoZ = rhoZAvg;
	Info.nonReflectiveInletRhoImp = rhoImpAvg;
}
