


[iaf, spikes, y, aweights, sweights, vm, homRateEstimate,numInputs] = runo2_icaEdges(1, 3, 4);


%%
tCheck = length(y)/24;
NI = iaf.numInputs;
act = zeros(NI,tCheck);
for tc = 1:tCheck
    act(ismember(1:NI,y(:,tc)),tc) = 1;
end
figure(1); clf; 
imagesc(cov(act'));


%%
NO = iaf.numOrientation;
NP = iaf.numPosition;
L = sqrt(iaf.numPosition);
aSumWeight = zeros(1,NO*NP); 
sSumWeight = zeros(1,NO*NP); 
idxActive = [true(1, iaf.numActive), false(1,iaf.numSilent)];
idxSilent = [false(1, iaf.numActive), true(1,iaf.numSilent)];
for i = 1:NO*NP
    aSumWeight(i) = sum(iaf.ampaWeights(iaf.inputConnection==i & idxActive));
    sSumWeight(i) = sum(iaf.ampaWeights(iaf.inputConnection==i & idxSilent));
end

% convert to spatial grid
gridASumWeights = cell2mat(squeeze(cellfun(@(c) padarray(c,[1,1],-1),num2cell(reshape(permute(reshape(aSumWeight,NO,NP),[3 2 1]),L,L,NO),[1 2]),'uni',0)));
gridSSumWeights = cell2mat(squeeze(cellfun(@(c) padarray(c,[1,1],-1),num2cell(reshape(permute(reshape(sSumWeight,NO,NP),[3 2 1]),L,L,NO),[1 2]),'uni',0)));

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.33 0.49 0.41 0.29]);

subplot(1,6,1); 
imagesc(gridASumWeights);

subplot(1,6,2); 
imagesc(gridSSumWeights);

subplot(1,6,3); 
imagesc(aweights);

subplot(1,6,4); 
imagesc(sweights);

% subplot(1,6,5);
% imagesc(numInputs);
% 
% subplot(1,6,6);
% imagesc(aweights./numInputs);

dpsth = 1/iaf.dt;
subplot(1,6,5); hold on;
psth = sum(reshape(spikes,dpsth,length(spikes)/dpsth));
plot(1:length(spikes)/dpsth, psth);

subplot(1,6,6); 
plot((1:length(spikes))*iaf.dt, homRateEstimate);


%%

TT = 500;
T = size(spikes,2); 
xtime = randi(T/2-TT,1) + T/2;
dpsth = 10; 

figure(2); clf; 
subplot(3,1,1); 
plot(dpsth:dpsth:TT, sum(reshape(spikes(xtime:xtime+TT-1),dpsth,TT/dpsth))*(1000/dpsth));

subplot(3,1,2);
plot(1:TT, vm(xtime:xtime+TT-1));

subplot(3,1,3); 
yFocus = y(:,xtime:xtime+TT-1);
yFocus = mod(yFocus,3); 
yFocus(yFocus==0)=3;
imagesc(yFocus);


%% --

NO = 3; 
NP = 7; 


iOn = zeros(T,NO*NP);
for t = 1:T
    iOn(t,y(:,t)) = 1;
end

idxOrder = [1:3:NO*NP, 2:3:NO*NP, 3:3:NO*NP];

figure(1); clf;
subplot(1,2,1); 
imagesc(iOn(end-10000:end,idxOrder));

subplot(1,2,2);
imagesc(cov(iOn(:,idxOrder)))



























