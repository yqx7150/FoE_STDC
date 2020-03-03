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

  
## Recover the Baboon, Barbara, House, Lena, Peppers image with 90% of missing entries using different methods
 ![repeat-MDAEP](https://github.com/yqx7150/FoE_STDC/blob/master/Recovery%20results.PNG)
    
(a) Original image. (b) Observed data. (c) M2SA. (d) M2SA-G. (e) LRTC. (f) HaLRTC. (g) STDC. (h) SPCTV. (i) SPCQV. (j) FoE-LRTC. (k) FoE-STDC.

## The average recovery performance (PSNR/RSE) on eight images with missing rates of 70%, 80% and 90%
  
![repeat-MDAEP](https://github.com/yqx7150/FoE_STDC/blob/master/RECOVERY%20PERFORMANCE.PNG)

## Other Related Projects
  * Predual dictionary learning (PDL) / augmented Lagrangian multi-scale dictionary learning(ALM-DL) [<font size=5>**[Paper]**</font>](http://www.escience.cn/people/liuqiegen/index.html;jsessionid=5E20FEE3694E8BB3249B64202A8E25C8-n1)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PDL_ALM_DL_code) 

  * Adaptive dictionary learning in sparse gradient domain for image recovery [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/6578193/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GradDL) 
 
  * Highly undersampled magnetic resonance image reconstruction using two-level Bregman method with dictionary updating [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/6492252)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/TBMDU) 
   
  * Convolutional Sparse Coding in Gradient Domain for MRI Reconstruction [<font size=5>**[Paper]**</font>](http://html.rhhz.net/ZDHXBZWB/html/2017-10-1841.htm)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/GradCSC)
  
  * Synthesis-analysis deconvolutional network for compressed sensing [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/8296620)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SADN)
 
  * Multi-filters guided low-rank tensor coding for image inpainting [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0923596518308877?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MFLRTC_inpainting)
    
  * Sparse and dense hybrid representation via subspace modeling for dynamic MRI [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S089561111730006X)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/SDR)

