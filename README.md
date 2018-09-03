# FoE_STDC
Field-of-Experts Filters Guided Tensor Completion  
%% The Code is created based on the method described in the following papers:   
% [1] Jiaojiao Xiong, Sanqian Li, Qiegen Liu*, Xiaoling Xu. Analysis-operator guided simultaneous tensor decomposition and completion,2017 IEEE International Conference on Image Processing (ICIP), 17-20 Sept. 2017 ,2017 ,2677-2681.
% [2] B. Xiong#, Q. Liu#, J. Xiong, S. Li, S. Wang, D. Liang. Field-of-Experts Filters Guided Tensor Completion, IEEE Transactions on Multimedia (2018).   
% Author: B. Xiong#, Q. Liu# (Co-first author), J. Xiong, S. Li, D. Liang*   
% Date : 02/14/2018   
% Version : 1.0   
% The code and the algorithm are for non-comercial use only.   
% Copyright 2018, Department of Electronic Information Engineering, Nanchang University.  
% The current version is not optimized.  
% FoE-STDC - FoE filters guided simultaneous tensor decomposition and completion  
% 
% Paras:  
% @m_rate : Missing rates  
% @balance_lambda : Balances the tradeoff between naive STDC and STDC based on multi-view features domain  
% @lambda_FoE : Augmented lagrange multiplier  
% @nfilters : Number of filters  
