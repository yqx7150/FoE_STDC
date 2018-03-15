% GPU-friendly 2D convolution, results equivalent to imfilter(x,s.f{i},'same','circular',{'conv','corr'});
function x = filter_circ_corr(x,s,j), x = conv2(x(s.circpadidx{j}),s.f_tr{j},'valid'); end
