

numAngles = 8; % between 0 and 180 (because it's orientation and not direction)
L = 3;
L2 = L^2;
dangle = pi/numAngles; 
angles = (1:numAngles)/numAngles;
x = randi(numAngles,L)/numAngles*180;
if rand()<0.5
    cedge = randi(4);
    edgeIdx = getEdgeIdx(cedge,L,L);
    lookUpAngle = [0 pi/4 pi/2 3*pi/4];
    x(edgeIdx) = lookUpAngle(cedge);
end

im = cell2mat(cellfun(@(c) drawEdge(c,51), num2cell(x), 'uni', 0));
figure(1);clf;
imagesc(im);


%%

% -- to do list --
% 1. Test relation between tuning sharpness and 4 angles
% 2. Check whether double angle formulation is the correct one. (Should the
% tuning angles spread from 0-180 or 0-360?)

numAngles = 4; % between 0 and 180 (because it's orientation and not direction)
angles = (1:numAngles)/numAngles*pi; % angles to choose tuning from (0, pi]

tc = 0;
tuningSharpness = 1;
getRate = @(stim,tc) 5 + 45*exp(tuningSharpness*cos(2*(stim-tc)))./(2*pi*besseli(0,tuningSharpness));

stim = linspace(-pi,pi,100);
figure(1); clf;
rate = getRate(stim,tc);
plot(stim, rate); hold on;
for na = 1:numAngles
    line(angles(na)*[1,1],[0 max(rate)],'color','k');
end




