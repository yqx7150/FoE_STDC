% optimized and GPU-friendly psf2otf, basically equivalent to psf2otf(s.f{j},s.imdims)
function otf = mypsf2otf(j,s)
  fdims = size(s.f{j});
  psf = s.zeroimg;
  idx_circ = s.circpadidx{j}(1:fdims(1),1:fdims(2));
  psf(idx_circ) = s.f{j};
  otf = fft2(psf);
end