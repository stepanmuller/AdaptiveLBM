AdaptiveLBM is a LBM solver based on the TNL library: https://tnl-project.org/

Compile using tnlcxx: https://gitlab.com/tnl-project/tnlcxx

tnlcxx --release main.cu

Features
- D3Q27 set
- cummulant collision taken from TNL-LBM: https://gitlab.com/tnl-project/tnl-lbm
- LES Smagorinsky model taken from: https://github.com/stloufra/LB/tree/thesis
- fully local moment-based boundary conditions
- esotwist streaming
- rest of this readme hasn't been written yet
