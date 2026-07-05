// DIAD VERSION //
void applyLocalCellUpdate( DIADGridStruct &Grid )
{
	auto fArrayView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	
	bool esotwistFlipper = Grid.esotwistFlipper;
	auto iNbrView = Grid.EsotwistNbrArray.iNbrArray.getConstView();
	auto jNbrView = Grid.EsotwistNbrArray.jNbrArray.getConstView();
	auto kNbrView = Grid.EsotwistNbrArray.kNbrArray.getConstView();
	auto ijNbrView = Grid.EsotwistNbrArray.ijNbrArray.getConstView();
	auto ikNbrView = Grid.EsotwistNbrArray.ikNbrArray.getConstView();
	auto jkNbrView = Grid.EsotwistNbrArray.jkNbrArray.getConstView();
	auto ijkNbrView = Grid.EsotwistNbrArray.ijkNbrArray.getConstView();
	
	bool useBouncebackArray = false;
	auto bouncebackMarkerArrayView = Grid.bouncebackMarkerArray.getConstView();
	if ( Grid.bouncebackMarkerArray.getSize() > 0 )
	{
		useBouncebackArray = true;
	}
	auto forcedVelocityMarkerArrayView = Grid.forcedVelocityMarkerArray.getConstView();
	bool useForcedVelocityArray = false;
	if ( Grid.forcedVelocityMarkerArray.getSize() > 0 )
	{
		useForcedVelocityArray = true;
	}
	InfoStruct Info = Grid.Info;
	
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
		
		MarkerStruct Marker;
		if ( useBouncebackArray ) Marker.bounceback = bouncebackMarkerArrayView( cell );
		if ( useForcedVelocityArray ) Marker.forcedVelocity = forcedVelocityMarkerArrayView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		
		if ( Marker.bounceback )
		{
			return; // bounceback gets implicitly applied by Esotwist
		}
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getEsotwistReadIndex( cell, cellReadIndex, fReadIndex, Nbr, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fArrayView(fReadIndex[direction], cellReadIndex[direction]);
		
		float rho, ux, uy, uz;
		float gx = 0.f;
		float gy = 0.f;
		float gz = 0.f;
		
		if ( Marker.movingBounceback )
		{
			getGivenRhoUxUyUz( iCell, jCell, kCell, rho, ux, uy, uz, Info, Marker );
			applyMovingBounceback( f, ux, uy, uz );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getEsotwistWriteIndex( cell, cellWriteIndex, fWriteIndex, Nbr, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
			return;
		}
		
		if ( Marker.fluid )
		{
			// do nothing, just skip the else block below
		}
		else if ( Marker.forcedVelocity )
		{
			getForcing( iCell, jCell, kCell, f, gx, gy, gz, Info );
		}
		else if ( Marker.nonReflectiveOutlet )
		{
			// also do nothing, just skip the else block below
		}
		else
		{
			int outerNormalX, outerNormalY, outerNormalZ;
			getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
			if ( Marker.periodicX ) outerNormalX = 0;
			if ( Marker.periodicY ) outerNormalY = 0;
			if ( Marker.periodicZ ) outerNormalZ = 0;
			getGivenRhoUxUyUz( iCell, jCell, kCell, rho, ux, uy, uz, Info, Marker );
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
		
		applyCollision( f, Info.nu, SmagorinskyConstant, gx, gy, gz );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getEsotwistWriteIndex( cell, cellWriteIndex, fWriteIndex, Nbr, esotwistFlipper, Info );
		
		for ( int direction = 0; direction < 27; direction++ ) fArrayView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
}
