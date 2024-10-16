function gabor = drawGabor(ori,cycles,gaborWidth,gridRadius)

if ~exist('gridRadius','var'), gridRadius = 25; end
if ~exist('gaborWidth','var'), gaborWidth = 25; end
if ~exist('cycles','var'), cycles = 2; end

gridLength = 2*gridRadius+1;

F = zeros(gridLength);
sqrCycles = round(sqrt(cycles));
switch ori
    case 1
        F(1+sqrCycles,1+sqrCycles) = 1;
        F(end-sqrCycles+1,end-sqrCycles+1) = 1;
    case 2
        F(1,1+cycles) = 1;
        F(1,end-cycles+1) = 1;
    case 3
        F(1+sqrCycles,end-sqrCycles+1) = 1;
        F(end-sqrCycles+1,1+sqrCycles) = 1;
    case 4
        F(1+cycles,1) = 1;
        F(end-cycles+1,1) = 1;
end

gridIdx = -gridRadius:gridRadius;
[xmesh,ymesh] = meshgrid(gridIdx);
expDecay = exp(-(xmesh.^2 + ymesh.^2)/gaborWidth^2);

gabor = expDecay .* real(ifft2(F));
gabor = gabor./max(abs(gabor(:)));
