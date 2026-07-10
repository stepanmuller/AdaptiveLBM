// DIAD VERSION //
void applyLocalCellUpdate( DIADGridStruct &Grid )
{
	auto fArrayView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	
	bool esotwistFlipper = Grid.esotwistFlipper;
	auto iNBRView = Grid.EsotwistNBRArray.iNBRArray.getConstView();
	auto jNBRView = Grid.EsotwistNBRArray.jNBRArray.getConstView();
	auto kNBRView = Grid.EsotwistNBRArray.kNBRArray.getConstView();
	auto ijNBRView = Grid.EsotwistNBRArray.ijNBRArray.getConstView();
	auto ikNBRView = Grid.EsotwistNBRArray.ikNBRArray.getConstView();
	auto jkNBRView = Grid.EsotwistNBRArray.jkNBRArray.getConstView();
	auto ijkNBRView = Grid.EsotwistNBRArray.ijkNBRArray.getConstView();
	
	bool useBouncebackArray = false;
	auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
	if ( Grid.bouncebackMarkerArray.getSize() > 0 )
	{
		useBouncebackArray = true;
	}
	InfoStruct Info = Grid.Info;
	
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
		
		MarkerStruct Marker;
		if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		
		if ( Marker.bounceback )
		{
			return; // bounceback gets implicitly applied by Esotwist
		}
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cell, cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
		
		float rho, ux, uy, uz;
		
		if ( Marker.fluid )
		{
			// do nothing, just skip the else block below
		}
		else
		{
			int outerNormalX, outerNormalY, outerNormalZ;
			getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
			if ( Marker.periodicX ) outerNormalX = 0;
			if ( Marker.periodicY ) outerNormalY = 0;
			if ( Marker.periodicZ ) outerNormalZ = 0;
			getGivenRhoUxUyUz( iCell, jCell, kCell, rho, ux, uy, uz, Info );
			if ( Marker.givenRho && !Marker.givenUxUyUz )
			{
				restoreUxUyUz( outerNormalX, outerNormalY, outerNormalZ, rho, ux, uy, uz, f );
			}
			else if ( !Marker.givenRho && Marker.givenUxUyUz )
			{
				restoreRho( outerNormalX, outerNormalY, outerNormalZ, rho, ux, uy, uz, f );
			}
			else if ( !Marker.givenRho && !Marker.givenUxUyUz )
			{
				restoreRhoUxUyUz( outerNormalX, outerNormalY, outerNormalZ, rho, ux, uy, uz, f );
			}
			applyMBBC( outerNormalX, outerNormalY, outerNormalZ, rho, ux, uy, uz, f );
		}
		const float SmagorinskyConstant = getSmagorinskyConstant( iCell, jCell, kCell, Info );
		applyCollision( f, Info.nu, SmagorinskyConstant );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cell, cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
}
