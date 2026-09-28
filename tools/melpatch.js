// The learned model's input, computed in the page exactly as the pipeline does:
// 16 kHz mono, n_fft 512, hop 256, periodic Hann, centred with reflect padding, power spectrum,
// 96 Slaney mel bands (0 to 8 kHz, Slaney-normalised), log1p(1000 x mel); then eight 188-frame slices evenly spaced.
function melPatches16k(y){
  var N=512, H=256, NM=96, SR=16000, W=188, nb=N/2+1;
  var win=new Float64Array(N); for(var i=0;i<N;i++) win[i]=0.5-0.5*Math.cos(2*Math.PI*i/N);
  var hz2mel=function(f){ var f_sp=200/3, logs=1000/f_sp, ls=Math.log(6.4)/27; return f<1000? f/f_sp : logs+Math.log(f/1000)/ls; };
  var mel2hz=function(m){ var f_sp=200/3, logs=1000/f_sp, ls=Math.log(6.4)/27; return m<logs? m*f_sp : 1000*Math.exp(ls*(m-logs)); };
  var mpts=[], lo=hz2mel(0), hi=hz2mel(SR/2); for(i=0;i<NM+2;i++) mpts.push(mel2hz(lo+(hi-lo)*i/(NM+1)));
  var fft_f=[]; for(i=0;i<nb;i++) fft_f.push(i*SR/N);
  var fb=[]; for(var m=0;m<NM;m++){ var row=new Float64Array(nb), l=mpts[m], c=mpts[m+1], r=mpts[m+2], enorm=2/(r-l);
    for(i=0;i<nb;i++){ var f=fft_f[i], up=(f-l)/(c-l), dn=(r-f)/(r-c); row[i]=Math.max(0,Math.min(up,dn))*enorm; } fb.push(row); }
  var pad=N/2, L=y.length, yp=new Float64Array(L+2*pad);
  for(i=0;i<L;i++) yp[i+pad]=y[i];
  // centred with silence at each end, as the pipeline's audio library does by default
  var nf=1+Math.floor((yp.length-N)/H), M=[]; for(m=0;m<NM;m++) M.push(new Float32Array(nf));
  var re=new Float64Array(N), im=new Float64Array(N);
  for(var t=0;t<nf;t++){ for(i=0;i<N;i++){ re[i]=yp[t*H+i]*win[i]; im[i]=0; } fftRadix2(re,im);
    var p=new Float64Array(nb); for(i=0;i<nb;i++) p[i]=re[i]*re[i]+im[i]*im[i];
    for(m=0;m<NM;m++){ var s=0, row=fb[m]; for(i=0;i<nb;i++) s+=row[i]*p[i]; M[m][t]=Math.log1p(1000*s); } }
  if(nf<W) return null;
  var out=[]; for(var k=0;k<8;k++){ var st=Math.floor(k*(nf-W)/7); var sl=[]; for(m=0;m<NM;m++) sl.push(M[m].slice(st,st+W)); out.push(sl); }
  return out; }
function fftRadix2(re, im){ var n=re.length; for(var i=1,j=0;i<n;i++){ var bit=n>>1; for(;j&bit;bit>>=1) j^=bit; j^=bit; if(i<j){ var t=re[i]; re[i]=re[j]; re[j]=t; t=im[i]; im[i]=im[j]; im[j]=t; } }
  for(var len=2;len<=n;len<<=1){ var ang=-2*Math.PI/len, wr=Math.cos(ang), wi=Math.sin(ang); for(i=0;i<n;i+=len){ var cr=1, ci=0; for(var k=0;k<len/2;k++){ var ar=re[i+k], ai=im[i+k], br=re[i+k+len/2]*cr-im[i+k+len/2]*ci, bi=re[i+k+len/2]*ci+im[i+k+len/2]*cr; re[i+k]=ar+br; im[i+k]=ai+bi; re[i+k+len/2]=ar-br; im[i+k+len/2]=ai-bi; var nr=cr*wr-ci*wi; ci=cr*wi+ci*wr; cr=nr; } } } }
if(typeof module!=='undefined') module.exports={melPatches16k:melPatches16k};
