function X = TensorProduct(Tensor,U,d)

ndim0 = size(Tensor);
ndim0(d) = size(U,1);

X = shiftdim(Tensor,1);
ndim = size(X);
X = reshape(X,size(X,1),prod(size(X))/size(X,1));
X = U*X;
X = reshape(X,[size(X,1) ndim(2:end)]);
X = shiftdim(X,ndims(Tensor)-(1));

X = reshape(X,ndim0);