
% -- to do list --
% 1. Test relation between tuning sharpness and 4 angles
% 2. Check whether double angle formulation is the correct one. (Should the
% tuning angles spread from 0-180 or 0-360?)

clc
apDepIdx = 3;
[iaf,spkTimes,vm,y,smallBasalWeight,smallApicalWeight,bweight,aweight] = runo2_corrD(1,apDepIdx);
spikes = zeros(1,iaf.T);
spikes(spkTimes) = 1;

psthWindow = 1; % in seconds
dpsth = round(psthWindow/iaf.dt); % num samples for psth
psth = sum(reshape(spikes,dpsth,iaf.T/dpsth),1)/psthWindow; % 

tvec = iaf.dt:iaf.dt:iaf.T*iaf.dt;

%% --
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.17 0.63 0.22 0.34]);

subplot(3,1,1); 
plot(1:iaf.T/dpsth,psth);

subplot(3,1,2);
imagesc(smallBasalWeight);

subplot(3,1,3); 
imagesc(smallApicalWeight);

figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.63 0.22 0.34]);

subplot(1,2,1); 
imagesc(iaf.sourceLoading');
subplot(1,2,2); hold on;
smoothFactor = round(iaf.numInputs/iaf.numSignals/10);
sbw = smoothsmooth(smallBasalWeight(:,end),smoothFactor);
saw = smoothsmooth(smallApicalWeight(:,end),smoothFactor);
plot(sbw/max(sbw),iaf.numInputs:-1:1,'color','k','linewidth',1.5);
plot(saw/max(saw),iaf.numInputs:-1:1,'color','b','linewidth',1.5);

%% -- analyze data -- 

basalWeights = arrayfun(@(inputIdx) sum(iaf.basalWeight(iaf.basalTuneIdx==inputIdx)), 1:iaf.numInputs, 'uni', 1);
apicalWeights = arrayfun(@(inputIdx) sum(iaf.apicalWeight(iaf.apicalTuneIdx==inputIdx)), 1:iaf.numInputs, 'uni', 1);

bw = zeros(1,100);
for i = 1:100, bw(i) = sum(iaf.basalWeight(iaf.basalTuneIdx==i)); end 

figure(1); clf; 
imagesc(cat(1, bw, basalWeights, apicalWeights))

%% -- analyze data --

angles = iaf.angles;
repAngles = [angles; angles+pi; angles(1)]; % Full circle
dangles = mean(diff(angles));
numIdx = max(iaf.apicalTuneIdx);
stimHistBins = angles(1)-dangles/2 : dangles : angles(end) + dangles/2;
[~,~,shBasalBin] = histcounts(iaf.basalTuneCenter,stimHistBins);
[~,~,shApicalBin] = histcounts(iaf.apicalTuneCenter,stimHistBins);
xBasal = cos(repAngles); % double angle because orientation not direction
yBasal = sin(repAngles);
bAngleWeight = zeros(length(repAngles),1);
xApical = repmat(xBasal,1,numIdx);
yApical = repmat(yBasal,1,numIdx);
aAngleWeight = repmat(bAngleWeight,1,numIdx);
for a = 1:length(angles)
    cidx = [a,a+length(angles)];
    if a==1, cidx(end+1)=length(repAngles); end %#ok
    bAngleWeight(cidx) = sum(iaf.basalWeight(shBasalBin==a));
    xBasal(cidx) = xBasal(cidx)*sum(iaf.basalWeight(shBasalBin==a));
    yBasal(cidx) = yBasal(cidx)*sum(iaf.basalWeight(shBasalBin==a));
    for ix = 1:numIdx
        idxIdx = (iaf.apicalTuneIdx(:)==ix); 
        aAngleWeight(cidx,ix) = sum(iaf.apicalWeight(shApicalBin==a & idxIdx));
        xApical(cidx,ix) = xApical(cidx,ix) * sum(iaf.apicalWeight(shApicalBin==a & idxIdx));
        yApical(cidx,ix) = yApical(cidx,ix) * sum(iaf.apicalWeight(shApicalBin==a & idxIdx));
    end
end

rLim = [0 max(aAngleWeight(:))];
figure(1);clf;
set(gcf,'units','normalized','outerposition',[0.16 0.5 0.72 0.48]);

subplot(3,4,5);
polarplot(repAngles,bAngleWeight,'Color','k','linewidth',1.5)
set(gca,'ThetaTick',linspace(0,360,5));
set(gca,'RTick',[]);
    
spidx = [2,6,10,3,7,11,4,8,12];
for ix = 1:numIdx
    subplot(3,4,spidx(ix));
    p=polarplot(repAngles,aAngleWeight(:,ix),'Color','k','linewidth',1.5);
    set(gca,'ThetaTick',linspace(0,360,5));
    set(gca,'RTick',[]);
    rlim(rLim);
end


figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.16 0 0.72 0.49]);

subplot(3,4,5);
imagesc(smallBasalWeight);
    
spidx = [2,6,10,3,7,11,4,8,12];
for ix = 1:numIdx
    subplot(3,4,spidx(ix));
    imagesc(smallApicalWeight(:,:,ix));
end





