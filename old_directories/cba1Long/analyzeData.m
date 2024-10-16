%% -- project notes --
%{

% I just reran everything on the server with some changes:
## - longer 
## - apical weights ranging from 1:0.5:0.9
## - edge probability 0.33:0.33:1

%}
%% -- load data --

hpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/cba1Long';
dpath = [hpath,'_data'];

nameConvention = @(ridx,ad,ep) sprintf('cba1Long_AD%d_EP%d_Run%d.mat',ad,ep,ridx);

numRuns = 50;
numAD = 5;
numEP = 3;

% load one to get params
sampleData = load(fullfile(dpath,nameConvention(1,1,1)));
numAngles = sampleData.iaf.numAngles;
numPosition = length(sampleData.iaf.apicalIndices);
T = sampleData.iaf.T;
lengthTraj = T/10000;

% load final weights
keepIdx = lengthTraj-100:lengthTraj;
scaleApicalWeight = 3e-9;
scaleBasalWeight = 35e-9;
dsFactor = 10;
apicalWeights = nan(numAngles, numPosition, numAD, numEP, numRuns);
basalWeights = nan(numAngles, numAD, numEP, numRuns);
basalTrajectory = nan(numAngles, lengthTraj/dsFactor, numAD,numEP,numRuns);
apicalTrajectory = nan(numAngles, lengthTraj/dsFactor, numPosition, numAD,numEP,numRuns);
for ridx = 1:numRuns
    for ad = 1:numAD
        for ep = 1:numEP
            cdata = load(fullfile(dpath,nameConvention(ridx,ad,ep)),'apicalWeightTrajectory','basalWeightTrajectory');
            apicalWeights(:,:,ad,ep,ridx) = mean(cdata.apicalWeightTrajectory(:,keepIdx,:),2);
            basalWeights(:,ad,ep,ridx) = mean(cdata.basalWeightTrajectory(:,keepIdx),2);
            apicalTrajectory(:,:,:,ad,ep,ridx) = permute(mean(reshape(permute(cdata.apicalWeightTrajectory,[4,2,1,3]),dsFactor,lengthTraj/dsFactor,numAngles,numPosition),1),[3,2,4,1]);
            basalTrajectory(:,:,ad,ep,ridx) = permute(mean(reshape(permute(cdata.basalWeightTrajectory,[3,2,1]),dsFactor,lengthTraj/dsFactor,numAngles),1),[3,2,1]);
        end
    end
end

% Analyze Edge Tuning 
centerIdx = 5;
edgeIndices = [3,7; 4,6; 1,6; 2,8];
basalPidx = nan(numAngles,numAD,numEP,numRuns);
basalPref = nan(numAngles,numAD,numEP,numRuns);
apicalPref = nan(numAngles,5,numAD,numEP,numRuns); % (1:5)={leftNonpref,leftPref,centerPref,rightPref,rightNonPref}
basalPTraj = nan(numAngles, lengthTraj/dsFactor, numAD, numEP, numRuns);
apicalPTraj = nan(numAngles,lengthTraj/dsFactor,3, numAD, numEP, numRuns);
for ridx = 1:numRuns
    for ad = 1:numAD
        for ep = 1:numEP
            [basalPref(:,ad,ep,ridx),basalPidx(:,ad,ep,ridx)] = sort(basalWeights(:,ad,ep,ridx),'descend');
            cidx = basalPidx(:,ad,ep,ridx);
            basalPTraj(:,:,ad,ep,ridx) = basalTrajectory(cidx,:,ad,ep,ridx); % just sort it...
            for ori = 1:numAngles
                apicalPref(ori,3,ad,ep,ridx) = apicalWeights(cidx(ori),5,ad,ep,ridx); % Center
                apicalPref(ori,2,ad,ep,ridx) = apicalWeights(cidx(ori),edgeIndices(cidx(ori),1),ad,ep,ridx); % left - on
                apicalPref(ori,4,ad,ep,ridx) = apicalWeights(cidx(ori),edgeIndices(cidx(ori),2),ad,ep,ridx); % right - on
                
                %leftOff = mean(apicalWeights(:,edgeIndices(cidx(ori),1),ad,ep,ridx)) - apicalWeights(cidx(ori),edgeIndices(cidx(ori),1),ad,ep,ridx);
                %rightOff = mean(apicalWeights(:,edgeIndices(cidx(ori),2),ad,ep,ridx)) - apicalWeights(cidx(ori),edgeIndices(cidx(ori),2),ad,ep,ridx);
                %apicalPref(ori,1,ad,ep,ridx) = leftOff;
                %apicalPref(ori,5,ad,ep,ridx) = rightOff;
                apicalPref(ori,1,ad,ep,ridx) = nanmean(apicalWeights(~ismember(1:numAngles,cidx(ori)),edgeIndices(cidx(ori),1),ad,ep,ridx),1);
                apicalPref(ori,5,ad,ep,ridx) = nanmean(apicalWeights(~ismember(1:numAngles,cidx(ori)),edgeIndices(cidx(ori),2),ad,ep,ridx),1);
                
                apicalPTraj(ori,:,1,ad,ep,ridx) = apicalTrajectory(cidx(ori),:,5,ad,ep,ridx);
                apicalPTraj(ori,:,2,ad,ep,ridx) = nanmean(apicalTrajectory(cidx(ori),:,edgeIndices(cidx(ori),:),ad,ep,ridx),3);
                apicalPTraj(ori,:,3,ad,ep,ridx) = nanmean(nanmean(apicalTrajectory(~ismember(1:4,cidx(ori)),:,edgeIndices(cidx(ori),:),ad,ep,ridx),1),3);
            end
        end
    end
end


%% -- look at examples --
ad = 5;
ep = 3;
figure(1); clf;
subplot(1,2,1); plot(1:4,nanmean(basalPref(:,ad,ep,:),4));
subplot(1,2,2); imagesc(nanmean(apicalPref(:,:,ad,ep,:),5));

%% -- look differently --

cc = cat(1, linspace(0,1,5), zeros(2,5));
AD = 5;
EP = 3;
figure(1); clf; 
for i = 1:4
    subplot(4,2,2*(i-1)+1); hold on;
    for ep = 1:3
        mnPlot = nanmean(apicalPref(i,:,AD,ep,:),5);
        sdPlot = nanstd(apicalPref(i,:,AD,ep,:),1,5);
        patch([1:5 fliplr(1:5)],[mnPlot+sdPlot fliplr(mnPlot-sdPlot)],'k','EdgeColor','none','FaceColor',cc(:,ep),'FaceAlpha',0.5);
        plot(1:5, mnPlot, 'color',cc(:,ep),'linewidth',1.5);
    end
    ylim([0 max(apicalPref(:,:,AD,:,:),[],'all')]);
    
    subplot(4,2,2*i); hold on;
    for ad = 1:5
        mnPlot = nanmean(apicalPref(i,:,ad,EP,:),5);
        sdPlot = nanstd(apicalPref(i,:,ad,EP,:),1,5);
        patch([1:5 fliplr(1:5)],[mnPlot+sdPlot fliplr(mnPlot-sdPlot)],'k','EdgeColor','none','FaceColor',cc(:,ad),'FaceAlpha',0.5);
        plot(1:5, mnPlot, 'color',cc(:,ad),'linewidth',1.5);
    end
    ylim([0 max(apicalPref(:,:,:,EP,:),[],'all')]);
end
subplot(4,2,1); title('Varying Edge Probability'); 
subplot(4,2,2); title('Varying Apical Depression'); 

%% -- look at examples --
r2plot = 3;
ad = 5;
ep = 3;
ridx = randperm(numRuns,r2plot);
figure(1); clf; 
for r = 1:r2plot
    subplot(2,r2plot,r);
    imagesc(basalWeights(:,ad,ep,ridx(r))');
    
    subplot(2,r2plot,r+r2plot);
    imagesc(apicalWeights(:,:,ad,ep,ridx(r))');
end


%% -- summary plot (populations) --
apicalSummary = zeros(3,numAD,numEP,numRuns);
for ad = 1:numAD
    for ep = 1:numEP
        for ridx = 1:numRuns
            apicalSummary(1,ad,ep,ridx) = apicalPref(1,3,ad,ep,ridx);
            apicalSummary(2,ad,ep,ridx) = nanmean(apicalPref(1,[2,4],ad,ep,ridx));
            apicalSummary(3,ad,ep,ridx) = (nansum(apicalPref(:,:,ad,ep,ridx),'all')-nansum(apicalPref(1,2:4,ad,ep,ridx)))/17;
        end
    end
end

apicalSumMean = nanmean(apicalSummary,4);
apicalSumStd = nanstd(apicalSummary,1,4);

figure(1);clf; 
set(gcf,'units','normalized','outerposition',[0 0.3 0.94 0.5]);

epLabel = [0.33 0.66 1];
adLabel = 110:-5:90;
cmap = [0,0,1; 0,185/255,1; 0,0,0];
%cmap = cat(2, linspace(0,1,numAD)',zeros(numAD,2));
for ep = 1:numEP
    subplot(1,numEP,ep); hold on;
    for pop = 1:3
        plot(1:5, squeeze(apicalSummary(pop,:,ep,:))*1e9, 'color',mean([cmap(pop,:);ones(2,3)],1),'linewidth',0.5,'marker','.','markersize',16);
    end
    for pop = 1:3
        %shadedErrorBar(1:3, apicalSumMean(:,ad,ep), apicalSumStd(:,ad,ep), {'color',cmap(ad,:),'linewidth',2.5,'marker','.','markersize',32},1);
        plot(1:5, apicalSumMean(pop,:,ep)*1e9,'color',cmap(pop,:),'linewidth',2.5,'marker','.','markersize',32);
    end
    xlim([0.8 5.2]);
    ylim([0 max(apicalSummary,[],'all')*1e9])
    set(gca,'xtick',1:5,'xticklabel',cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));%,'xticklabelrotation',45);
    set(gca,'ytick',0:2:6);
    xlabel('Apical Depression Ratio');
    ylabel('Input Weight (ns)');
    title(sprintf('P(edge)=%.2f',epLabel(ep)));
    set(gca,'fontsize',24);
end


%% -- look at trajectory --

cc = cat(1, linspace(1,0,5), zeros(2,5));
AD = 4; 
EP = 3;
figure(1); clf; 
subplot(2,4,1); imagesc(nanmean(basalPTraj(:,:,AD,EP,:),5));
for l = 1:3
    subplot(2,4,l+1); imagesc(nanmean(apicalPTraj(:,:,l,AD,EP,:),6));
end
subplot(2,4,1+4); hold on;
for ang = 1:4
    mnplot = nanmean(basalPTraj(ang,:,AD,EP,:),5);
    sdplot = nanstd(basalPTraj(ang,:,AD,EP,:),1,5);
    patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(:,ang),'EdgeColor','none','FaceAlpha',0.1);
    plot(1:lengthTraj/dsFactor,mnplot,'color',cc(:,ang),'linewidth',1.5);
end
for l = 1:3
    subplot(2,4,l+1+4); hold on;
    for ang = 1:4
        mnplot = nanmean(apicalPTraj(ang,:,l,AD,EP,:),6);
        sdplot = nanstd(apicalPTraj(ang,:,l,AD,EP,:),1,6);
        patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(:,ang),'EdgeColor','none','FaceAlpha',0.1);
        plot(1:lengthTraj/dsFactor,mnplot,'color',cc(:,ang),'linewidth',1.5);
    end
    ylim([0 max(apicalPref(:,:,AD,EP,:),[],'all')]);
end


%% -- practice plotting gabor images -- 

gaborCycles = 2;
gaborWidth = 18;
imFiltWidth = 3;
angles = (1:4)/4*pi;

ridx = randi(numRuns);
AD = 1;
EP = 3;
redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');
sdata = load(fullfile(dpath,nameConvention(ridx,AD,EP)));
sampleInput = reshape(sdata.y(:,randi(size(sdata.y,2))),3,3);
[~,sampleOri] = ismember(sampleInput,angles);
gaborGrid = cell2mat(cellfun(@(c) drawGabor(c,gaborCycles,gaborWidth), num2cell(sampleOri), 'uni', 0));

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.133 0.3733 0.665 0.547]);

% plot example gabor image
subplot(1,2,1); 
imagesc(imgaussfilt(gaborGrid,imFiltWidth))
colormap(redblue)
cAxis = caxis;
caxis(max(abs(cAxis))*[-1,1])
title(' A random stimulus');

gaborWeight = zeros(size(gaborGrid,1),size(gaborGrid,2),numAngles);
for ang = 1:numAngles
    gaborWeight(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(reshape(apicalWeights(ang,:,AD,EP,ridx),3,3)), 'uni', 0));
    %gaborWeight(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(reshape(apicalTrajectory(ang,1,:,AD,EP,ridx),3,3)), 'uni', 0));
end
subplot(1,2,2); 
imagesc(imgaussfilt(sum(gaborWeight,3),imFiltWidth));
colormap(redblue)
caxis(max(abs(sum(gaborWeight,3)),[],'all')*[-1 1]);
title(' When Depression is High');


%% -- get covariance of data --

ridx = randi(numRuns);
AD = 1;
EP = 4;
sdata = load(fullfile(dpath,nameConvention(ridx,AD,EP)));
[~,yangles] = ismember(sdata.y, angles);
yunique = (0:numPosition-1)'*numAngles + yangles;

activity = 5 * ones(numPosition * numAngles, size(yunique,2));
for ui = 1:numPosition*numAngles
    activity(ui,yunique(ceil(ui/4),:)==ui) = 45;
end

figure(1); clf;
imagesc(cov(activity'))
% dendrogram(Z)









