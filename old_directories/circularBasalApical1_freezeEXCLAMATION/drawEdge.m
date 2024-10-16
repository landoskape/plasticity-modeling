function edge = drawEdge(orientation,L)

if ~exist('L','var'), L = 51; end
if rem(L,2)~=1, error('L has to be odd'); end

expandFactor = 3;
LL = (L-1)*expandFactor + 1;
horzEdge = zeros(LL);
horzEdge((LL-1)/2,:) = 1;
edge = imrotate(horzEdge, orientation*180/pi, 'crop');
idxCenter = (LL-1)/2 + (-(L-1):(L-1));
edge = edge(idxCenter, idxCenter);
    
    