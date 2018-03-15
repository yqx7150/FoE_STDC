%% The Code is created based on the method described in the following papers: 
% [1] Jiaojiao Xiong, Sanqian Li, Qiegen Liu*, Xiaoling Xu. Analysis-operator guided simultaneous tensor decomposition and completion,
%     2017 IEEE International Conference on Image Processing (ICIP), 17-20 Sept. 2017 ,2017 ,2677-2681.
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
%
% Example 
% ========== 
clear;clc;
path(path,'utilities\');
path(path,'utilities\STDCfun\');
path(path,'utilities\tensorfun\');

FOEkernels = load('csf_3x3.mat');
m_rate =  0.7 ; 
mcell.nfilters = 3;  
balance_lambda = 0.08;
lambda_FoE= 0.0015;
iter =146;

%% load image
X = double(imread('airplane.bmp'))/255;
tsize = size(X);
% construction of missing data
rand('seed',0);%rng(0);
idx = randperm(numel(X));
mark = zeros(tsize);
mark(idx(1:floor(m_rate*numel(X)))) = 1;
mark = boolean(mark);
Xm = X ;
Xm(mark) = 0;
figure(1);imshow(Xm);
% parameters: kappa/omega/tau/gnns/pnns/maxitr/mode_dim/mode_PoM/mode_noise/VSet/Rate/Affinity
para = initial_para(10^0.2,1,0.1,2,25,iter,true,false,{[1],[2]},[1,1],{[],[]},tsize);
para_ST = para;
Xg = X;
X = Xm;

%% Initialization
tic;
tsize = size(X); 
% initialize manifold graphs
para_ST.H = construct_graphL(tsize,para_ST.VSet,para_ST.Rate,para_ST.gnns,para_ST.Affinity);
for i = 1 : size(para_ST.H,1)
    for j = 1 : numel(tsize)
        para_ST.Ds{i,j} = eye(tsize(j)); 
    end
end
% initilize factor matrices & permutation matrices (w.r.t the modes of tensor dimension & PoM)
vsize = tsize;
N = numel(tsize);
if para_ST.mode_dim, N = N-1; end;
for i = 1 : N
    V{i} = eye(tsize(i));
    vsize(i) = size(V{i},1);
    P{i} = (1:tsize(i))';
end
% initialize core tensor and augmented multiplier
Z = X;
Y = zeros(tsize);
% initialize algorithm parameters
norm_gt = norm(Xg(:));
norm_x = norm(X(:));
para_ST.alpha = ones(N,1);
para_ST.gamma = (para_ST.omega/para_ST.tau)/(norm_x^2);
xxt = reshape(X,tsize(1),[]);
xxt = norm(xxt*xxt');
ita = 1/(para_ST.tau*xxt);
ct = zeros(1,N);
for i = 1 : size(para_ST.H,1)
    ct = ct+double(para_ST.VSet{i});
end
for i = 1 : size(para_ST.H,1)
    list = 1 : N;
    list = list(para_ST.VSet{i});
    for j = 1 : size(para_ST.H,2)
        U{j} = para_ST.Ds{i,list(j)};
    end
    for j = 1 : size(para_ST.H,2)
        llt = reshape(TensorChainProduct(para_ST.H{i,j},U,[1:j-1 j+1:size(para_ST.H,2)]),tsize(list(j)),[]);
        llt = norm(llt*llt');
        para_ST.H{i,j} = para_ST.H{i,j}*para_ST.kappa*sqrt(ita*xxt/(2*llt*ct(list(j))));
    end
end
% message
disp(['Finish the initialization of all parameters within ',num2str(toc),' seconds...']);
disp('------------------------------------------------------------------------------');
disp('--                          Start FoE_STDC algorithm..                          --');
disp('------------------------------------------------------------------------------');
%% Main algorithm
lambda = ita*(1.1^(para_ST.maxitr))/2;

%% feature domain
for jjj = 1:mcell.nfilters      
    subFOEfilter{jjj} = FOEkernels.model.f{jjj,1};
    subFOEfilter_tr{jjj} = flipud(fliplr(FOEkernels.model.f{jjj,1}));
end
mcell.f = subFOEfilter;  
mcell.f_tr = subFOEfilter_tr;
bndry = [ 25 , 25 , 0];  
pad   = @(x) padarray(x,bndry,'replicate','both');
crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2),:);
[transfer,retain] = deal(@(d)d);
X1 = transfer(X); X1  = pad(X1);
mcell.imdims = [size(X1,1),size(X1,2)];
mcell.zeroimg = transfer(zeros(mcell.imdims));
mcell = circpadidx(mcell,transfer);
dim = size(X1);
nchannels = dim(3);
for iii = 1:nchannels  
    for jjj = 1:mcell.nfilters 
        X_FoE(:,:,mcell.nfilters*(iii-1)+jjj) = filter_circ_conv(X1(:,:,iii),mcell,jjj);
    end
end
rand('seed',0);%
tsize_FoE = size(X_FoE);
para_FoE = initial_para(10^0.2,1,0.1,2,25,iter,true,false,{[1],[2]},[1,1],{[],[]},tsize_FoE);
para_ST_FoE = para_FoE;

%% feature domain Initialization
tic;
tsize_FoE = size(X_FoE); 
% initialize manifold graphs
para_ST_FoE.H = construct_graphL(tsize_FoE,para_ST_FoE.VSet,para_ST_FoE.Rate,para_ST_FoE.gnns,para_ST_FoE.Affinity);
for i = 1 : size(para_ST_FoE.H,1)
    for j = 1 : numel(tsize_FoE)
        para_ST_FoE.Ds{i,j} = eye(tsize_FoE(j)); 
    end
end
vsize_FoE = tsize_FoE;
N_FoE = numel(tsize_FoE);
if para_ST_FoE.mode_dim, N_FoE = N_FoE-1; end;
for i = 1 : N_FoE
    V_FoE{i} = eye(tsize_FoE(i));
    vsize_FoE(i) = size(V_FoE{i},1);
    P_FoE{i} = (1:tsize_FoE(i))';
end
% initialize core tensor and augmented multiplier
Z_FoE = X_FoE;
Y_FoE = zeros(tsize_FoE);
W_FoE = Y_FoE ;
% initialize algorithm parameters
norm_x_FoE = norm(X_FoE(:));
para_ST_FoE.alpha = ones(N_FoE,1);
para_ST_FoE.gamma = (para_ST_FoE.omega/para_ST_FoE.tau)/(norm_x_FoE^2);
xxt_FoE = reshape(X_FoE,tsize_FoE(1),[]);
xxt_FoE = norm(xxt_FoE*xxt_FoE');
ita_FoE = 1/(para_ST_FoE.tau*xxt_FoE);
ct_FoE = zeros(1,N_FoE);
for i = 1 : size(para_ST_FoE.H,1)
    ct_FoE = ct_FoE+double(para_ST_FoE.VSet{i});
end
for i = 1 : size(para_ST_FoE.H,1)
    list_FoE = 1 : N_FoE;
    list_FoE = list_FoE(para_ST_FoE.VSet{i});
    for j = 1 : size(para_ST_FoE.H,2)
        U_FoE{j} = para_ST_FoE.Ds{i,list(j)};
    end
    for j = 1 : size(para_ST_FoE.H,2)
        llt_FoE = reshape(TensorChainProduct(para_ST_FoE.H{i,j},U_FoE,[1:j-1 j+1:size(para_ST_FoE.H,2)]),tsize_FoE(list(j)),[]);
        llt_FoE = norm(llt_FoE*llt_FoE');
        para_ST_FoE.H{i,j} = para_ST_FoE.H{i,j}*para_ST_FoE.kappa*sqrt(ita_FoE*xxt_FoE/(2*llt_FoE*ct_FoE(list_FoE(j))));
    end
end
RSE=zeros(para_ST.maxitr,1);PSNR=zeros(para_ST.maxitr,1);  SSIM=zeros(para_ST.maxitr,1);  
for itr = 1 : para_ST.maxitr
   
 for iii = 1:nchannels  
    for jjj = 1:mcell.nfilters  
        X_FoE(:,:,mcell.nfilters*(iii-1)+jjj) = (filter_circ_conv(pad(X(:,:,iii)),mcell,jjj));
    end
end
  %% step1 image domain   
    % update V1,...,Vn
    [V,P,rank_vi,pstr] = optimize_V(X,Y,Z,V,P,ita,tsize,vsize,para_ST);
    % update Z
    Z = optimize_Z(V,X,Y,ita,para_ST.gamma);
    % update X
    Xt = TensorChainProductT(Z,V,1:numel(V));
    residual = (norm(X(:)-Xt(:)))/norm(Xt(:));
    % update Y
    Y = Y+ita*(X-Xt);

  %% step2 feature domain   
    %  update V1,...,Vn_FoE
     [V_FoE,P_FoE,rank_vi_FoE,pstr_FoE] = optimize_V(X_FoE,Y_FoE,Z_FoE,V_FoE,P_FoE,ita_FoE,tsize_FoE,vsize_FoE,para_ST_FoE);                              
    % update Z_FoE     
     Z_FoE = optimize_Z(V_FoE,X_FoE,Y_FoE,ita_FoE,para_ST_FoE.gamma);       
    % update X_FoE
     Xt_FoE = TensorChainProductT(Z_FoE,V_FoE,1:numel(V_FoE));     
     X2_FoE = (ita_FoE*Xt_FoE+lambda_FoE*X_FoE - Y_FoE - W_FoE)/(ita_FoE + lambda_FoE);  
 
  %% step3 image domain and feature domain  
  for iii = 1:3  %        
        top  = 0;   bottom = 0;
        for jjj = 1:mcell.nfilters  %s.nfilters
            bottom = bottom + abs(mypsf2otf(jjj,mcell)).^2;
            for kkk=1:mcell.nfilters
              top = top + filter_circ_corr( (X2_FoE(:,:,mcell.nfilters*(iii-1)+jjj)+W_FoE(:,:,mcell.nfilters*(iii-1)+jjj)/lambda_FoE) ,mcell,kkk);
            end   
        end
        X111(:,:,iii) = crop(real(ifft2(   fft2(balance_lambda*lambda_FoE*top + ita*pad(Xt(:,:,iii) - Y(:,:,iii)/ita)) ./ (ita+balance_lambda*lambda_FoE*bottom ))));
  end      
     X(mark) = X111(mark);  
    % update W_FoE
    W_FoE = W_FoE + lambda_FoE*(X2_FoE-X_FoE);       
    % update Y_FoE
    Y_FoE = Y_FoE+ita_FoE*(X2_FoE-Xt_FoE); 
    % assessment
    info.rse(itr) = norm(X(mark)-Xg(mark))/norm_gt;
    RSE(itr)  = info.rse(itr);
    PSNR(itr) = psnr1(X,Xg);  
    fprintf(1, 'Iter=%d, PSNR=%f, RSE=%f\n', itr, PSNR(itr),RSE(itr)); 
    info.rank_vi(:,itr) = rank_vi;
    info.residual(:,itr) = residual;
    % display
    figure(2);imshow(X);
    pause(0.1);
    ita = ita*1.1;
    ita_FoE=ita_FoE*1.1  ;
    lambda_FoE=lambda_FoE*1.1;
    lambda=lambda*1.1;
end


   