
% -- things I want to vary --
% apDepIdx
% eProbEdx
% numAngles (4/8)

clc
apDepIdx = 1;
eProbIdx = 4;
[iaf,spkTimes,y,smallBasalWeight,smallApicalWeight] = runo2_cba(1,apDepIdx,eProbIdx);
spikes = zeros(1,iaf.T);
spikes(spkTimes) = 1;

psthWindow = 1; % in seconds
dpsth = round(psthWindow/iaf.dt); % num samples for psth
psth = sum(reshape(spikes,dpsth,iaf.T/dpsth),1)/psthWindow; % 

tvec = iaf.dt:iaf.dt:iaf.T*iaf.dt;

%% --
figure(1); clf;
plot(1:iaf.T/dpsth,psth);


%% -- analyze data --

angles = iaf.angles;
repAngles = [angles; angles+pi; angles(1)]; % Full circle 
dangles = mean(diff(angles));
numIdx = max(iaf.apicalIndices);
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
% rLim = [0 iaf.maxApicalWeight * iaf.numApical / 9];
figure(1);clf;
set(gcf,'units','normalized','outerposition',[0.16 0.5 0.72 0.48]);

subplot(3,4,5);
polarplot(repAngles,bAngleWeight,'Color','k','linewidth',1.5)
set(gca,'ThetaTick',linspace(0,360,5));
set(gca,'RTick',[]);
    
numIdx = 9;
spidx = [2,6,10,3,7,11,4,8,12];
for ix = 1:numIdx
    subplot(3,4,spidx(ix));
    p=polarplot(repAngles,aAngleWeight(:,ix),'Color','k','linewidth',1.5);
    set(gca,'ThetaTick',linspace(0,360,5));
    set(gca,'RTick',[]);
    rlim(rLim);
end


cLim = [0 max(smallApicalWeight(:))];
% cLim = [0 iaf.maxApicalWeight * iaf.numApical / 9];
figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.16 0 0.72 0.49]);

subplot(3,4,5);
imagesc(smallBasalWeight);
    
spidx = [2,6,10,3,7,11,4,8,12];
for ix = 1:numIdx
    subplot(3,4,spidx(ix));
    imagesc(smallApicalWeight(:,:,ix));
    caxis(cLim);
end





