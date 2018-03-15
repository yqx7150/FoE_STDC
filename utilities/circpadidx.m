% compute circular padding indices for all filters (assume same sizes for all stages)
% used for 2D convolution and mypsf2otf
function s = circpadidx(s,transfer)
  idx = reshape(1:prod(s.imdims),s.imdims);
  s.circpadidx = cell(1,s.nfilters);
  fdims = size(s.f{1});
  if fdims(1) == fdims(2)
    % learned filters (assume that all have same size)
    s.circpadidx(:) = {transfer(padarray(idx,(fdims-1)/2,'circular','both'))};
  else
    % pairwise filters (different sizes)
    for i = 1:s.nfilters
      s.circpadidx{i} = transfer(padarray(idx,(size(s.f{i})-1)/2,'circular','both'));
    end
  end
end