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
	auto iNBRView = Grid.EsotwistNBRArray.iNBRArray.getConstView();
	auto jNBRView = Grid.EsotwistNBRArray.jNBRArray.getConstView();
	auto kNBRView = Grid.EsotwistNBRArray.kNBRArray.getConstView();
	auto ijNBRView = Grid.EsotwistNBRArray.ijNBRArray.getConstView();
	auto ikNBRView = Grid.EsotwistNBRArray.ikNBRArray.getConstView();
	auto jkNBRView = Grid.EsotwistNBRArray.jkNBRArray.getConstView();
	auto ijkNBRView = Grid.EsotwistNBRArray.ijkNBRArray.getConstView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		DIADEsotwistNBRStruct NBR;
		NBR.i = iNBRView( cell );
		NBR.j = jNBRView( cell );
		NBR.k = kNBRView( cell );
		NBR.ij = ijNBRView( cell );
		NBR.ik = ikNBRView( cell );
		NBR.jk = jkNBRView( cell );
		NBR.ijk = ijkNBRView( cell );
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
		getPostCollisionIndex( cell, cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = feq[direction];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Info.cellCount, cellLambda );
}
