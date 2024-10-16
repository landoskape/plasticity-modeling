
T = 2;

tau_c = 20; % time steps (should be ms!!!!)
options.numOrientation = 4; % vert, horz, diagLeft, diagRight
options.gridLength = 5; 

NO = options.numOrientation;
L = options.gridLength;
eL = 3; % edge length

numEdge = 1; % num edges per image

stimMap = randi(NO,L,L,T);
edgeOri = zeros(1,T);
startPos = zeros(1,T);
for t = 1:T
    edgeOri(t) = randi(NO,numEdge,1);
    if rand<0.99, edgeOri(t)=1; end
    [edgeIdx,startPos(t)] = getEdgeIdx(edgeOri(t),L,eL,[T,t]);
    stimMap(edgeIdx) = edgeOri(t);
end
vecMap = reshape(stimMap,L*L,T) + NO*(0:L*L-1)';
inputRate = zeros(L*L*NO,T);
for t = 1:T
    inputRate(ismember(1:L*L*NO,vecMap(:,t)),t) = 1;
end

corrCor = corrcoef(inputRate');
corrCor = corrCor - diag(diag(corrCor));
figure(1); clf; imagesc(corrCor);

%%
figure(1); clf;
im = cell2mat(cellfun(@(c) drawEdge(c,51), num2cell(stimMap(:,:,randi(T))), 'uni', 0));
imagesc(im);
title(sprintf('Orientation: %d, startPos: %d', edgeOri(t), startPos(t)));

            
