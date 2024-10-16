%% -- project notes --
%{

It looks good! The 1-2-3-4 method is the way to go for summary data.
We'll have: (sorted to basal preference)
1: Center Pos, Preferred Orientation
2: Edge Positions, Parallel orientations
3: Edge Positions, Non-preferred orientations
4: All other input tuning

Then, I can plot each weight population as a function of AD & EP

To introduce those groups, I want to start with the polar plots
Where the basal polar weight plot is on the left (larger)
And the apical polar weight plots are on the right, in the grid. 
 -- Show a couple examples
 -- Use colors to introduce 1-2-3-4 groups
 -- switch to summary data

% I just reran everything on the server with three changes:
## - the angle loaded into y changes 0 to pi (same but cleaner)
## - running for 3200*1000 instead of 2400
## - using 100 apical weights with 100e-12 maxweight instead of 180/25e-12

%}
%% -- load data --

hpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/circularBasalApical1';
dpath = [hpath,'_data'];

nameConvention = @(ridx,ad,ep) sprintf('cba1_AD%d_EP%d_Run%d.mat',ad,ep,ridx);

numRuns = 50;
numAD = 4;
numEP = 4;

% load one to get params
sampleData = load(fullfile(dpath,nameConvention(1,1,1)));
numAngles = sampleData.iaf.numAngles;
numPosition = length(sampleData.iaf.apicalIndices);
T = sampleData.iaf.T;
lengthTraj = T/1000;

% load final weights
keepIdx = lengthTraj-100:lengthTraj;
scaleApicalWeight = 3e-9;
scaleBasalWeight = 35e-9;
dsFactor = 10;
apicalWeights = zeros(numAngles, numPosition, numAD, numEP, numRuns);
basalWeights = zeros(numAngles, numAD, numEP, numRuns);
basalTrajectory = zeros(numAngles, lengthTraj/dsFactor, numAD,numEP,numRuns);
apicalTrajectory = zeros(numAngles, lengthTraj/dsFactor, numPosition, numAD,numEP,numRuns);
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
basalPidx = zeros(numAngles,numAD,numEP,numRuns);
basalPref = zeros(numAngles,numAD,numEP,numRuns);
apicalPref = zeros(numAngles,5,numAD,numEP,numRuns); % (1:5)={leftNonpref,leftPref,centerPref,rightPref,rightNonPref}
basalPTraj = zeros(numAngles, lengthTraj/dsFactor, numAD, numEP, numRuns);
apicalPTraj = zeros(numAngles,lengthTraj/dsFactor,3, numAD, numEP, numRuns);
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
                apicalPref(ori,1,ad,ep,ridx) = mean(apicalWeights(~ismember(1:numAngles,cidx(ori)),edgeIndices(cidx(ori),1),ad,ep,ridx),1);
                apicalPref(ori,5,ad,ep,ridx) = mean(apicalWeights(~ismember(1:numAngles,cidx(ori)),edgeIndices(cidx(ori),2),ad,ep,ridx),1);
                
                apicalPTraj(ori,:,1,ad,ep,ridx) = apicalTrajectory(cidx(ori),:,5,ad,ep,ridx);
                apicalPTraj(ori,:,2,ad,ep,ridx) = mean(apicalTrajectory(cidx(ori),:,edgeIndices(cidx(ori),:),ad,ep,ridx),3);
                apicalPTraj(ori,:,3,ad,ep,ridx) = mean(mean(apicalTrajectory(~ismember(1:4,cidx(ori)),:,edgeIndices(cidx(ori),:),ad,ep,ridx),1),3);
            end
        end
    end
end


%% -- look at examples --
ad = 1;
ep = 3;
figure(1); clf;
subplot(1,2,1); plot(1:4,mean(basalPref(:,ad,ep,:),4));
subplot(1,2,2); imagesc(mean(apicalPref(:,:,ad,ep,:),5));

%% -- look differently --

cc = cat(1, linspace(0,1,4), zeros(2,4));
AD = 4;
EP = 4;
figure(1); clf; 
for i = 1:4
    subplot(4,2,2*(i-1)+1); hold on;
    for ep = 1:4
        mnPlot = mean(apicalPref(i,:,AD,ep,:),5);
        sdPlot = std(apicalPref(i,:,AD,ep,:),1,5);
        patch([1:5 fliplr(1:5)],[mnPlot+sdPlot fliplr(mnPlot-sdPlot)],'k','EdgeColor','none','FaceColor',cc(:,ep),'FaceAlpha',0.5);
        plot(1:5, mnPlot, 'color',cc(:,ep),'linewidth',1.5);
    end
    ylim([0 max(apicalPref(:,:,AD,:,:),[],'all')]);
    
    subplot(4,2,2*i); hold on;
    for ad = 1:4
        mnPlot = mean(apicalPref(i,:,ad,EP,:),5);
        sdPlot = std(apicalPref(i,:,ad,EP,:),1,5);
        patch([1:5 fliplr(1:5)],[mnPlot+sdPlot fliplr(mnPlot-sdPlot)],'k','EdgeColor','none','FaceColor',cc(:,ad),'FaceAlpha',0.5);
        plot(1:5, mnPlot, 'color',cc(:,ad),'linewidth',1.5);
    end
    ylim([0 max(apicalPref(:,:,:,EP,:),[],'all')]);
end
subplot(4,2,1); title('Varying Edge Probability'); 
subplot(4,2,2); title('Varying Apical Depression'); 

%% -- look at examples --
r2plot = 3;
ad = 4;
ep = 2;
ridx = randperm(numRuns,r2plot);
figure(1); clf; 
for r = 1:r2plot
    subplot(2,r2plot,r);
    imagesc(basalWeights(:,ad,ep,ridx(r))');
    
    subplot(2,r2plot,r+r2plot);
    imagesc(apicalWeights(:,:,ad,ep,ridx(r))');
end

%% -- look at trajectory --

cc = cat(1, linspace(1,0,4), zeros(2,4));
AD = 3; 
EP = 4;
figure(1); clf; 
subplot(2,4,1); imagesc(mean(basalPTraj(:,:,AD,EP,:),5));
for l = 1:3
    subplot(2,4,l+1); imagesc(mean(apicalPTraj(:,:,l,AD,EP,:),6));
end
subplot(2,4,1+4); hold on;
for ang = 1:4
    mnplot = mean(basalPTraj(ang,:,AD,EP,:),5);
    sdplot = std(basalPTraj(ang,:,AD,EP,:),1,5);
    patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(:,ang),'EdgeColor','none','FaceAlpha',0.1);
    plot(1:lengthTraj/dsFactor,mnplot,'color',cc(:,ang),'linewidth',1.5);
end
for l = 1:3
    subplot(2,4,l+1+4); hold on;
    for ang = 1:4
        mnplot = mean(apicalPTraj(ang,:,l,AD,EP,:),6);
        sdplot = std(apicalPTraj(ang,:,l,AD,EP,:),1,6);
        patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(:,ang),'EdgeColor','none','FaceAlpha',0.1);
        plot(1:lengthTraj/dsFactor,mnplot,'color',cc(:,ang),'linewidth',1.5);
    end
    ylim([0 max(apicalPref(:,:,:,EP,:),[],'all')]);
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









