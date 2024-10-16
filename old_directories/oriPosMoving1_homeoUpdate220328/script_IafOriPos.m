


[iaf, spikes, y, aweights, sweights, vm, homRateEstimate, numAInputs] = runo2_iafOriPos(1, 3, 3);

%%
NO = iaf.numOrientation;
NP = iaf.numPosition;
aSumWeight = zeros(1,NO*NP); 
sSumWeight = zeros(1,NO*NP); 
idxActive = [true(1, iaf.numActive), false(1,iaf.numSilent)];
idxSilent = [false(1, iaf.numActive), true(1,iaf.numSilent)];
for i = 1:NO*NP
    aSumWeight(i) = sum(iaf.ampaWeights(iaf.inputConnection==i & idxActive));
    sSumWeight(i) = sum(iaf.ampaWeights(iaf.inputConnection==i & idxSilent));
end

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.33 0.49 0.41 0.29]);

subplot(1,6,1); 
bar(reshape(aSumWeight,NO,NP))

subplot(1,6,2); 
bar(reshape(sSumWeight,NO,NP))

subplot(1,6,3); 
imagesc(aweights);

subplot(1,6,4); 
imagesc(sweights);

subplot(1,6,5);
imagesc(numAInputs);
subplot(1,6,6);
imagesc(aweights./numAInputs);

%%
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



























