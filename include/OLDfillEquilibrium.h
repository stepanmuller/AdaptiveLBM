// DIAD VERSION //
void fillEquilibriumFromFunction( DIADGridStruct &Grid )
{
	InfoStruct Info = Grid.Info;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto fArrayView  = Grid.fArray.getView();
	bool useBouncebackArray = ( Grid.bouncebackMarkerArray.getSize() > 0 );
	auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
	bool esotwistFlipper = Grid.esotwistFlipper;
	auto iNbrView = Grid.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrView = Grid.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrView = Grid.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrView = Grid.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrView = Grid.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrView = Grid.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrView = Grid.EsotwistNbrArray.ijkNbrArray.getConstView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		DIADEsotwistNbrStruct Nbr;
		Nbr.i = iNbrView( cell );
		Nbr.j = jNbrView( cell );
		Nbr.k = kNbrView( cell );
		Nbr.ij = ijNbrView( cell );
		Nbr.ik = ikNbrView( cell );
		Nbr.jk = jkNbrView( cell );
		Nbr.ijk = ijkNbrView( cell );
		float feq[27];
		float rho = 1.f;
		float ux, uy, uz = 0.f;
		MarkerStruct Marker;
		if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		getInitialRhoUxUyUz( iCell, jCell, kCell, rho, ux, uy, uz, Marker, Info );
		getFeq(rho, ux, uy, uz, feq);
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cell, cellWriteIndex, fWriteIndex, Nbr, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = feq[direction];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Info.cellCount, cellLambda );
}
