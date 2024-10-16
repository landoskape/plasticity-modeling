
%{
Making figures here for my written thesis. 

# Figure 1 - description of model (we already have IAF described earlier)
- show stimulus and basal/apical organization (using schematic I made for
carandini & harris)
- starting weights
%}

%% -- load data --

hpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/circularBasalApical1';
dpath = [hpath,'_data'];
fpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/thesisFigures';

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
apicalPidx = zeros(numAngles,numAD,numEP,numRuns); % (just for central position to notate alignment between B/A)
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
            % Only for alignment plot, always sort to basal!!!!
            [~,apicalPidx(:,ad,ep,ridx)] = sort(apicalWeights(:,5,ad,ep,ridx),'descend');
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

%% -- create gabor to demonstrate stimulus --

gaborCycles = 2;
gaborWidth = 17;
padWidth = 0;
gaborRadius = 25;
gaborLength = 2*gaborRadius + 1;
imFiltWidth = 3;
boundaryWidth = 0.5;
angles = (1:4)/4*pi;

rng(1);

ridx = randi(numRuns);
AD = 1;
EP = 4;
redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');
sdata = load(fullfile(dpath,nameConvention(ridx,AD,EP)));
sampleInput = reshape(sdata.y(:,randi(size(sdata.y,2))),3,3);
[~,sampleOri] = ismember(sampleInput,angles);
gaborGrid = cell2mat(cellfun(@(c) drawGabor(c,gaborCycles,gaborWidth,gaborRadius), num2cell(sampleOri), 'uni', 0));

EP = 1;
sdata = load(fullfile(dpath,nameConvention(ridx,AD,EP)));
sampleInput = reshape(sdata.y(:,randi(size(sdata.y,2))),3,3);
[~,sampleOri] = ismember(sampleInput,angles);
gaborGridNoEdge = cell2mat(cellfun(@(c) drawGabor(c,gaborCycles,gaborWidth,gaborRadius), num2cell(sampleOri), 'uni', 0));

rng('shuffle');

close all

% ### Figure 1 is input to basal/apical across grid positions ###
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.47 0.37 0.33 0.29]);

% plot example gabor image for basal inputs
subplot(1,2,1); hold on;
imagesc(imgaussfilt(gaborGridNoEdge,imFiltWidth)); %imgaussfilt(gaborGrid,imFiltWidth)
for i = 1:9
    xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
    xend = xstart + gaborLength+2*padWidth;
    ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
    yend = ystart + gaborLength+2*padWidth;
    line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','-');
    line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    if i==5
        line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
        line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
        line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
    end
end
colormap(redblue)
cAxis = caxis;
caxis(max(abs(cAxis))*[-1,1]);
xlim([0.5 length(basalGabors)+0.5]);
ylim([0.5 length(basalGabors)+0.5]);
set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
% title('Input to Apical Synapses');'
set(gca,'visible','off');
set(gca,'fontsize',24);

subplot(1,2,2); hold on;
imagesc(imgaussfilt(gaborGrid,imFiltWidth)); %imgaussfilt(gaborGrid,imFiltWidth)
for i = 1:9
    xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
    xend = xstart + gaborLength+2*padWidth;
    ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
    yend = ystart + gaborLength+2*padWidth;
    line([xstart xstart],[ystart yend],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
    line([xend xend],[ystart yend],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
    line([xstart xend],[ystart ystart],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
    line([xstart xend],[yend yend],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
end
colormap(redblue)
cAxis = caxis;
caxis(max(abs(cAxis))*[-1,1]);
xlim([0.5 length(basalGabors)+0.5]);
ylim([0.5 length(basalGabors)+0.5]);
set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
% title('Input to Apical Synapses');
set(gca,'visible','off');
set(gca,'fontsize',24);
% print(gcf,'-painters',fullfile(fpath,'inputExampleWithBoundaries'),'-depsc');

% print(gcf,'-painters',fullfile(fpath,'inputExampleBasalApicalNoEdgeEdge'),'-depsc');


%% -- create gabor patches and demonstrate net weight representation --

gaborCycles = 2;
gaborWidth = 17;
padWidth = 0;
gaborRadius = 25;
gaborLength = 2*gaborRadius + 1;
imFiltWidth = 3;
boundaryWidth = 0.5;

redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');

figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.13 0.58 0.72 0.33]);
for ang = 1:4
    rfMap = cell2mat(arrayfun(@(c) drawGabor(c,gaborCycles,gaborWidth,gaborRadius),ang*ones(1),'uni',0));
    subplot(1,4,ang); hold on;
    imagesc(imgaussfilt(rfMap,imFiltWidth));
    for i = 1
        xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
        xend = xstart + gaborLength+2*padWidth;
        ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
        yend = ystart + gaborLength+2*padWidth;
        line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
        line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    end
    cAxis = caxis;
    colormap(redblue);
    xlim([0.5 size(rfMap,2)+0.5]);
    ylim([0.5 size(rfMap,1)+0.5]);
    daspect([1,1,1]);
    caxis(max(abs(cAxis))*[-1,1]);
    set(gca,'visible','off');
end
tightfig;
% print(gcf,'-painters',fullfile(fpath,'fourAngles'),'-depsc');

oriTuning = [1,1,1,1; 1,0,0,0];
oriTuning = oriTuning .* [0.175;1];
numTuneExample = size(oriTuning,1);
figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.34 0.49 0.27 0.27]);
for nte = 1:numTuneExample
    subplot(1,numTuneExample,nte);
    rfMap = zeros(gaborLength);
    for ang = 1:4
        rfMap = rfMap + cell2mat(arrayfun(@(c) oriTuning(nte,ang)*drawGabor(c,gaborCycles,gaborWidth,gaborRadius),ang*ones(1),'uni',0));
    end
    imagesc(imgaussfilt(rfMap,imFiltWidth));
    set(gca,'ydir','normal');
    colormap(redblue);
    for i = 1:9
        xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
        xend = xstart + gaborLength+2*padWidth;
        ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
        yend = ystart + gaborLength+2*padWidth;
        line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
        line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    end
    cAxis = caxis;
    xlim([0.5 size(rfMap,2)+0.5]);
    ylim([0.5 size(rfMap,1)+0.5]);
    daspect([1,1,1]);
    caxis([-1,1]);
    set(gca,'visible','off');
end
tightfig;
% print(gcf,'-painters',fullfile(fpath,'netWeightRepresentation'),'-depsc');


%%

% ### Figure 1 is input to basal/apical across grid positions ###
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.47 0.37 0.33 0.29]);

% plot example gabor image for basal inputs
subplot(1,2,1); hold on;
imagesc(imgaussfilt(gaborGridNoEdge,imFiltWidth)); %imgaussfilt(gaborGrid,imFiltWidth)
for i = 1:9
    xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
    xend = xstart + gaborLength+2*padWidth;
    ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
    yend = ystart + gaborLength+2*padWidth;
    line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','-');
    line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
    if i==5
        line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
        line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
        line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth*6,'linestyle','-');
    end
end
colormap(redblue)
caxis(max(abs(cAxis))*[-1,1]);
xlim([0.5 length(basalGabors)+0.5]);
ylim([0.5 length(basalGabors)+0.5]);
set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
% title('Input to Apical Synapses');'
set(gca,'visible','off');
set(gca,'fontsize',24);

subplot(1,2,2); hold on;
imagesc(imgaussfilt(gaborGrid,imFiltWidth)); %imgaussfilt(gaborGrid,imFiltWidth)
for i = 1:9
    xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
    xend = xstart + gaborLength+2*padWidth;
    ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
    yend = ystart + gaborLength+2*padWidth;
    line([xstart xstart],[ystart yend],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
    line([xend xend],[ystart yend],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
    line([xstart xend],[ystart ystart],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
    line([xstart xend],[yend yend],'color','b','linewidth',boundaryWidth*6,'linestyle','-');
end
colormap(redblue)
caxis(max(abs(cAxis))*[-1,1]);
xlim([0.5 length(basalGabors)+0.5]);
ylim([0.5 length(basalGabors)+0.5]);
set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
set(gca,'visible','off');
set(gca,'fontsize',24);


padTuneId = [0,5];
gaborGrid = cell2mat(cellfun(@(c) padarray(drawGabor(c,gaborCycles,gaborWidth,gaborRadius),padTuneId,0), num2cell([3,2,1,4]), 'uni', 0));
szGabor = size(gaborGrid);
tuneSharpness = 5;%sdata.iaf.basalSharpness;
rate = @(input,btc) sdata.iaf.baseBasal + sdata.iaf.driveBasal * ...
    exp(tuneSharpness * cos(2*(input - btc))) / (2 * pi * besseli(0, tuneSharpness));

eachWidth = 250;
xWid = 4*eachWidth + 8*padTuneId(2)/gaborLength;
yWid = eachWidth;

% ### Figure 2 is side-by-side receptive fields ###
figure(2); clf;
set(gcf,'units','pixels','outerposition',[1050 450 xWid*1.01 yWid*1.5]);
set(gca,'units','pixels','position',[75 50 xWid*0.9 yWid/xWid * xWid*0.9]);
hold on;
imagesc(gaborGrid);
for i = 1:4
    xstart = (i-1)*(gaborLength + 2*padTuneId(2)) + padTuneId(2);
    xend = xstart + gaborLength;
    ystart = 1;
    yend = gaborLength;
    patch([xstart xstart xend xend],[ystart yend yend ystart],'k','FaceColor','none','EdgeColor','k','linewidth',1.5);
    %text(mean([xstart xend]),ystart+5,'Receptive Field','Fontsize',24,'HorizontalAlignment','Center');
end
colormap('hot');
colormap(redblue)
caxis(max(abs(cAxis))*[-1,1]);
xlim([1 4*gaborLength + 8*padTuneId(2)]);
ylim([1 gaborLength]);
set(gca,'visible','off');
% print(gcf,'-painters',fullfile(fpath,'inputImagesNoText'),'-depsc');
% print(gcf,'-painters',fullfile(fpath,'inputImages'),'-depsc');

% ### Figure 3 is firing rates of presynaptic neurons ###
figure(3); clf; 
set(gcf,'units','pixels','outerposition',[1050 50 xWid*1.01 400]);
set(gca,'units','pixels','position',[75 40 xWid*0.9 400*0.6]);
rate2plot = rate(pi/4,(1:4)/4*pi);
bar(1:4, rate2plot,'FaceColor','k');
xlim([0.5 4.5]);
ylim([0 max(rate2plot)+3]);
set(gca,'xtick',1:4,'xticklabel',cellfun(@(c) sprintf('%s',c),{'\pi/4','\pi/2','3\pi/4','\pi'},'uni',0));
set(gca,'ytick',0:10:max(rate2plot));
ylabel('Firing Rate (hz)')
title('Presynaptic Firing Rates');
set(gca,'fontsize',24)
% print(gcf,'-painters',fullfile(fpath,'inputRates'),'-depsc');



%% -- Second round of figures is about the development of tuning --

boundaryWidth = 6;
runPrms = [1,1,2; 4,4,1];
for ex = 1:2
    gaborCycles = 2;
    gaborWidth = 18;
    imFiltWidth = 3;
    angles = (1:4)/4*pi;
    numAngles = length(angles);
    basalXScale = 8;
    basalXWidth = 3;

    ridx = runPrms(ex,3);
    AD = runPrms(ex,1);
    EP = runPrms(ex,2);
    redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');
    sdata = load(fullfile(dpath,nameConvention(ridx,AD,EP)));
    sampleInput = reshape(sdata.y(:,randi(size(sdata.y,2))),3,3);
    [~,sampleOri] = ismember(sampleInput,angles);
    gaborGrid = cell2mat(cellfun(@(c) drawGabor(c,gaborCycles,gaborWidth), num2cell(sampleOri), 'uni', 0));

    weightIdx = 1;
    gaborWeightBasal = zeros(size(gaborGrid,1),size(gaborGrid,2),numAngles);
    gaborWeightApical = zeros(size(gaborGrid,1),size(gaborGrid,2),numAngles);
    for ang = 1:numAngles
        %gaborWeightApical(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(reshape(apicalWeights(ang,:,AD,EP,ridx),3,3)), 'uni', 0));
        gaborWeightBasal(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(repmat(basalTrajectory(ang,weightIdx,AD,EP,ridx),3,3)), 'uni', 0));
        gaborWeightApical(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(reshape(apicalTrajectory(ang,weightIdx,:,AD,EP,ridx),3,3)), 'uni', 0));
    end

    for i = 1:9
        if i==5, continue, end
        xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
        xend = xstart + gaborLength+2*padWidth + 1;
        ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
        yend = ystart + gaborLength+2*padWidth + 1;
        %patch([xstart xstart xend xend],[ystart yend yend ystart],'w','EdgeColor','none','FaceAlpha',0.9);
        gaborWeightBasal(xstart+1:xend-1,ystart+1:yend-1,:) = 0;
    end

    % ### Figure 1 - Plot initial and final weights ###
    figure(ex); clf;
    %set(gcf,'units','normalized','outerposition',[0.27 0.1 0.53 0.86]);
    set(gcf,'units','normalized','outerposition',[0.39 0.1 0.3 0.57]);
    
    subplot(2,2,1); hold on;
    imagesc(imgaussfilt(sum(gaborWeightBasal,3),imFiltWidth))
    for i = 1:9
        xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
        xend = xstart + gaborLength+2*padWidth;
        ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
        yend = ystart + gaborLength+2*padWidth;
        if i==5
            line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
            line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
            line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','-');
            line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
        end
        line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        if i~=5
            xcenter = mean([xstart xend]);
            ycenter = mean([ystart yend]);
            line(xcenter + gaborLength/basalXScale*[-1 1],ycenter + gaborLength/basalXScale*[-1 1],'color','k','linewidth',basalXWidth);
            line(xcenter + gaborLength/basalXScale*[1 -1],ycenter + gaborLength/basalXScale*[-1 1],'color','k','linewidth',basalXWidth);
        end
    end
    colormap(redblue)
    cAxis = caxis;
    caxis(max(abs(sum(gaborWeightBasal,3)),[],'all')*[-1 1]);
    xlim([0.5 gaborLength*3+0.5]);
    ylim([0.5 gaborLength*3+0.5]);
    set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
    set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
    set(gca,'visible','off');
    set(gca,'fontsize',24);
    daspect([1,1,1])
    
    subplot(2,2,2); hold on;
    imagesc(imgaussfilt(sum(gaborWeightApical,3),imFiltWidth));
    for i = 1:9
        xstart = rem(i-1,3)*gaborLength + 0.5;
        xend = xstart + gaborLength;
        ystart = rem(floor((i-1)/3),3)*gaborLength + 0.5;
        yend = ystart + gaborLength;
        line([xstart xstart],[ystart yend],'color','b','linewidth',boundaryWidth,'linestyle','-');
        line([xend xend],[ystart yend],'color','b','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','b','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[yend yend],'color','b','linewidth',boundaryWidth,'linestyle','-');
    end
    colormap(redblue)
    caxis(max(abs(sum(gaborWeightBasal,3)),[],'all')/9*[-1 1]);
    xlim([0.5 gaborLength*3+0.5]);
    ylim([0.5 gaborLength*3+0.5]);
    set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
    set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
    set(gca,'visible','off');
    set(gca,'fontsize',24);
    daspect([1,1,1])

    weightIdx = size(basalTrajectory,2);
    gaborWeightBasal = zeros(size(gaborGrid,1),size(gaborGrid,2),numAngles);
    gaborWeightApical = zeros(size(gaborGrid,1),size(gaborGrid,2),numAngles);
    for ang = 1:numAngles
        %gaborWeightApical(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(reshape(apicalWeights(ang,:,AD,EP,ridx),3,3)), 'uni', 0));
        gaborWeightBasal(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(repmat(basalTrajectory(ang,weightIdx,AD,EP,ridx),3,3)), 'uni', 0));
        gaborWeightApical(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth), num2cell(reshape(apicalTrajectory(ang,weightIdx,:,AD,EP,ridx),3,3)), 'uni', 0));
    end

    for i = 1:9
        if i==5, continue, end
        xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
        xend = xstart + gaborLength+2*padWidth + 1;
        ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
        yend = ystart + gaborLength+2*padWidth + 1;
        %patch([xstart xstart xend xend],[ystart yend yend ystart],'w','EdgeColor','none','FaceAlpha',0.9);
        gaborWeightBasal(xstart+1:xend-1,ystart+1:yend-1,:) = 0;
    end

    subplot(2,2,3); hold on;
    imagesc(imgaussfilt(sum(gaborWeightBasal,3),imFiltWidth))
    for i = 1:9
        xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
        xend = xstart + gaborLength+2*padWidth;
        ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
        yend = ystart + gaborLength+2*padWidth;
        if i==5
            line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
            line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
            line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','-');
            line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','-');
        end
        line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth/6,'linestyle','-');
        if i~=5
            xcenter = mean([xstart xend]);
            ycenter = mean([ystart yend]);
            line(xcenter + gaborLength/basalXScale*[-1 1],ycenter + gaborLength/basalXScale*[-1 1],'color','k','linewidth',basalXWidth);
            line(xcenter + gaborLength/basalXScale*[1 -1],ycenter + gaborLength/basalXScale*[-1 1],'color','k','linewidth',basalXWidth);
        end
    end
    colormap(redblue)
    cAxis = caxis;
    caxis(max(abs(sum(gaborWeightBasal,3)),[],'all')*[-1 1]);
    xlim([0.5 gaborLength*3+0.5]);
    ylim([0.5 gaborLength*3+0.5]);
    set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
    set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
    set(gca,'visible','off');
    set(gca,'fontsize',24);
    daspect([1,1,1])

    subplot(2,2,4); hold on;
    imagesc(imgaussfilt(sum(gaborWeightApical,3),imFiltWidth));
    for i = 1:9
        xstart = rem(i-1,3)*gaborLength + 0.5;
        xend = xstart + gaborLength;
        ystart = rem(floor((i-1)/3),3)*gaborLength + 0.5;
        yend = ystart + gaborLength;
        line([xstart xstart],[ystart yend],'color','b','linewidth',boundaryWidth,'linestyle','-');
        line([xend xend],[ystart yend],'color','b','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[ystart ystart],'color','b','linewidth',boundaryWidth,'linestyle','-');
        line([xstart xend],[yend yend],'color','b','linewidth',boundaryWidth,'linestyle','-');
    end
    colormap(redblue)
    caxis(max(abs(sum(gaborWeightBasal,3)),[],'all')/9*[-1 1]);
    xlim([0.5 gaborLength*3+0.5]);
    ylim([0.5 gaborLength*3+0.5]);
    set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
    set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
    set(gca,'visible','off');
    set(gca,'fontsize',24);
    daspect([1,1,1])
    
    tightfig;
%     print(gcf,'-painters',fullfile(fpath,sprintf('WeightsInitFinal_%d',ex)),'-depsc');
end

%% Plot D/P Ratio Schematic

dpXPos = [69 8 3];
dpYPos = [4 4.6];
dpXTitle = dpXPos(1) + dpXPos(2) + dpXPos(3)/2;
dpYTitle = 5.3;
dpMrkSize = 32;
dpYPosIdx = [2,2; 1,1; 2,1];
legName = {'High/High','Low/Low','High/Low'};

combo = 1; 
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.43 0.62 0.11 0.22]);
hold on;
line(dpXPos(1)+[0 dpXPos(2)],dpYPos(1)*[1,1],'color','k','linewidth',2);
line(sum(dpXPos(1:3))+[0 dpXPos(2)],dpYPos(1)*[1,1],'color','k','linewidth',2);
plot(dpXPos(1)+dpXPos(2)/2,dpYPos(dpYPosIdx(combo,1)),'color','k','marker','.','markersize',dpMrkSize*2);
plot(sum(dpXPos(1:3))+dpXPos(2)/2,dpYPos(dpYPosIdx(combo,2)),'color','b','marker','.','markersize',dpMrkSize*2);
text(sum(dpXPos([1:3,2]))+1,dpYPos(1),'100%','Fontsize',24);
text(sum(dpXPos([1:3,2]))+1,dpYPos(2),'110%','Fontsize',24);
text(dpXTitle,dpYTitle,'D/P Ratio','HorizontalAlignment','Center','Fontsize',24);
xlim([69 99]);
ylim([3 6]);
set(gca,'visible','off');
% print(gcf,'-painters',fullfile(fpath,'DPSchematicHighHigh'),'-depsc');

combo = 3; 
figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.43 0.62 0.11 0.22]);
hold on;
line(dpXPos(1)+[0 dpXPos(2)],dpYPos(1)*[1,1],'color','k','linewidth',2);
line(sum(dpXPos(1:3))+[0 dpXPos(2)],dpYPos(1)*[1,1],'color','k','linewidth',2);
plot(dpXPos(1)+dpXPos(2)/2,dpYPos(dpYPosIdx(combo,1)),'color','k','marker','.','markersize',dpMrkSize*2);
plot(sum(dpXPos(1:3))+dpXPos(2)/2,dpYPos(dpYPosIdx(combo,2)),'color','b','marker','.','markersize',dpMrkSize*2);
text(sum(dpXPos([1:3,2]))+1,dpYPos(1),'100%','Fontsize',24);
text(sum(dpXPos([1:3,2]))+1,dpYPos(2),'110%','Fontsize',24);
text(dpXTitle,dpYTitle,'D/P Ratio','HorizontalAlignment','Center','Fontsize',24);
xlim([69 99]);
ylim([3 6]);
set(gca,'visible','off');
% print(gcf,'-painters',fullfile(fpath,'DPSchematicHighLow'),'-depsc');


%% -- orientation preference for apical / basal --  & alignment between apical & basal --

confusionMatrix = zeros(4,4,numAD,numEP);
for ad = 1:numAD
    for ep = 1:numEP
        idx = sub2ind([4,4,numAD,numEP],squeeze(basalPidx(1,ad,ep,:)),squeeze(apicalPidx(1,ad,ep,:)),ad*ones(numRuns,1),ep*ones(numRuns,1));
        for ii = idx(:)'
            confusionMatrix(ii) = confusionMatrix(ii)+1;
        end
    end
end

prefMagnitude = zeros(numRuns,2,numAD,numEP);
prefMagSoftmax = zeros(numRuns,2,numAD,numEP);
basalSoftmax = basalPref./sum(basalPref,1);
apicalSoftmax = apicalPref./sum(apicalPref,1);
for ad = 1:numAD
    for ep = 1:numEP
        prefMagnitude(:,1,ad,ep) = basalPref(1,ad,ep,:) / sampleData.iaf.maxBasalWeight / sampleData.iaf.numBasal;
        prefMagnitude(:,2,ad,ep) = apicalPref(1,3,ad,ep,:) / sampleData.iaf.maxApicalWeight / sampleData.iaf.numApical;
        
        prefMagSoftmax(:,1,ad,ep) = basalSoftmax(1,ad,ep,:);
        prefMagSoftmax(:,2,ad,ep) = apicalSoftmax(1,3,ad,ep,:);
    end
end
        
        


smCMat = sum(sum(confusionMatrix,3),4) / sum(confusionMatrix,'all');
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.51 0.55 0.29 0.4]);
imagesc(1-smCMat);
for ii = 0.5:4.5
    for jj = 0.5:4.5
        line([0.5 4.5],[jj jj],'color','k','linewidth',0.5,'linestyle','-');
        line([ii ii],[0.5 4.5],'color','k','linewidth',0.5,'linestyle','-');
    end
end
xlim([0.5 4.5]);
ylim([0.5 4.5]);
for ii = 1:4
    for jj = 1:4
        text(ii,jj,sprintf('%s',num2str(round(100*smCMat(ii,jj))/100)),'horizontalalignment','center','verticalalignment','middle','fontsize',24);
    end
end
colormap('gray');
caxis([0 1]);
label = {'\pi/4','\pi/2','3\pi/4','\pi'};
set(gca,'xtick',1:4,'xticklabel',label);
set(gca,'ytick',1:4,'yticklabel',label);
xlabel('Apical Preference');
ylabel('Basal Preference');
daspect([1,1,1]);
set(gca,'fontsize',24);

print(gcf,'-painters',fullfile(fpath,'confusionMatrix'),'-depsc');


figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.11 0.05 0.29 0.4]);
for ad = 1:numAD
    for ep = 1:numEP
        subplot(numAD,numEP,(ad-1)*numEP+ep); hold on;
        scatter(prefMagnitude(:,1,ad,ep),prefMagnitude(:,2,ad,ep),'k');
        %scatter(prefMagSoftmax(:,1,ad,ep),prefMagSoftmax(:,2,ad,ep),'r');
        xlim([0 1]);
        ylim([0 1]);
        refline(1,0);
    end
end

figure(3); clf; 
set(gcf,'units','normalized','outerposition',[0.51 0.05 0.29 0.4]);
for ad = 1:numAD
    for ep = 1:numEP
        subplot(numAD,numEP,(ad-1)*numEP+ep); hold on;
        %scatter(prefMagnitude(:,1,ad,ep),prefMagnitude(:,2,ad,ep),'k');
        scatter(prefMagSoftmax(:,1,ad,ep),prefMagSoftmax(:,2,ad,ep),'r');
        xlim([0 1]);
        ylim([0 1]);
        refline(1,0);
    end
end

%% -- then plot grid of AD and EP, just for apical -- 

gaborCycles = 2;
gaborWidth = 18;
gaborRadius = 25;
gaborLength = 2*gaborRadius + 1;
imFiltWidth = 3;
angles = (1:4)/4*pi;
numAngles = length(angles);
boundaryWidth = 0.5;
redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');

adLabel = [110,105,102.5,100];
epLabel = 0.5:0.25:1;
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.33 0.26 0.65]);
for ad = 1:numAD
    for ep = 2:numEP
        ridx = randi(numRuns);

        sdata = load(fullfile(dpath,nameConvention(ridx,ad,ep)));
        sampleInput = reshape(sdata.y(:,randi(size(sdata.y,2))),3,3);
        [~,sampleOri] = ismember(sampleInput,angles);
        gaborGrid = cell2mat(cellfun(@(c) drawGabor(c,gaborCycles,gaborWidth,gaborRadius), num2cell(sampleOri), 'uni', 0));

        weightIdx = size(basalTrajectory,2);
        gaborWeightApical = zeros(size(gaborGrid,1),size(gaborGrid,2),numAngles);
        for ang = 1:numAngles
            gaborWeightApical(:,:,ang) = cell2mat(cellfun(@(weight) weight * drawGabor(ang,gaborCycles,gaborWidth,gaborRadius), num2cell(reshape(apicalTrajectory(ang,weightIdx,:,ad,ep,ridx),3,3)), 'uni', 0));
        end
        
        subplot(numAD,numEP-1,(ad-1)*(numEP-1)+(ep-1));
        imagesc(imgaussfilt(sum(gaborWeightApical,3),imFiltWidth));
        colormap(redblue);
        caxis(max(abs(apicalWeights(:)))*[-1 1]);
        %caxis(max(abs(sum(gaborWeightApical,3)),[],'all')*[-1 1]);
        
        for i = 1:9
            xstart = rem(i-1,3)*gaborLength + 0.5;
            xend = xstart + gaborLength;
            ystart = rem(floor((i-1)/3),3)*gaborLength + 0.5;
            yend = ystart + gaborLength;
            line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','--');
            line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','--');
            line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','--');
            line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','--');
        end
        xlim([0.5 gaborLength*3+0.5]);
        ylim([0.5 gaborLength*3+0.5]);
        set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',[]);
        set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',[]);
        
        if ad==1
            title(sprintf('%.2f',round(100*epLabel(ep-1))/100));
        end
        if ep==2
            ylabel(sprintf('%.1f%%',adLabel(ad)),'fontweight','bold');
        end
        set(gca,'fontsize',24);
    end
end
% print(gcf,'-painters',fullfile(fpath,'apicalTuningGrid3'),'-djpeg');



%% -- Summary Data --

apicalSummary = zeros(3,numAD,numEP,numRuns);
for ad = 1:numAD
    for ep = 1:numEP
        for ridx = 1:numRuns
            apicalSummary(1,ad,ep,ridx) = apicalPref(1,3,ad,ep,ridx);
            apicalSummary(2,ad,ep,ridx) = sum(apicalPref(1,[2,4],ad,ep,ridx));
            apicalSummary(3,ad,ep,ridx) = (sum(apicalPref(:,:,ad,ep,ridx),'all')-sum(apicalPref(1,2:4,ad,ep,ridx)));
        end
    end
end
apicalSummary = 100 * apicalSummary / sampleData.iaf.maxApicalWeight / sampleData.iaf.numApical;

apicalSumMean = mean(apicalSummary,4);
apicalSumStd = std(apicalSummary,1,4);

figure(1);clf; 
set(gcf,'units','normalized','outerposition',[0.19 0.55 0.75 0.34]);

epLabel = 0.50:0.25:1;
adLabel = [110,105,102.5,100];
cmap = [0,0,1; 0,185/255,1; 0,0,0];
%cmap = cat(2, linspace(0,1,numAD)',zeros(numAD,2));
for ep = 2:numEP
    subplot(1,numEP-1,ep-1); hold on;
    for pop = 1:3
        plot(1:4, squeeze(apicalSummary(pop,:,ep,:)), 'color',mean([cmap(pop,:);ones(2,3)],1),'linewidth',0.5,'marker','.','markersize',16);
    end
    for pop = 1:3
        %shadedErrorBar(1:3, apicalSumMean(:,ad,ep), apicalSumStd(:,ad,ep), {'color',cmap(ad,:),'linewidth',2.5,'marker','.','markersize',32},1);
        plot(1:4, apicalSumMean(pop,:,ep),'color',cmap(pop,:),'linewidth',2.5,'marker','.','markersize',32);
    end
    xlim([0.8 4.2]);
    ylim([0 100])
    set(gca,'xtick',1:4,'xticklabel',cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));%,'xticklabelrotation',45);
    set(gca,'ytick',0:50:100);
    xlabel('D/P Ratio');
    ylabel('Net Synaptic Weight (%)');
    %title(sprintf('P(edge)=%.2f',epLabel(ep-1)));
    text(1.0,90,sprintf('P(edge)=%.2f',epLabel(ep-1)),'Fontsize',24);
    set(gca,'fontsize',24);
end
%print(gcf,'-painters',fullfile(fpath,'EmergenceEdgeTuningPopulations'),'-depsc');

figure(2); clf; 
set(gcf,'units','inches','outerposition',[12 2 9 8]);
set(gca,'units','inches','position',[2 2 5 4.5]);
fractionEdgeMap = squeeze(apicalSumMean(2,:,:));
imagesc(1:4,1:4,fractionEdgeMap');
colormap('pink');
cb = colorbar();
ylabel(cb,'Net synaptic weight (%)','Fontsize',24);
caxis([0 30]);
set(gca,'xtick',1:4,'xticklabel',cellfun(@(c) sprintf('%s%%',num2str(c)),num2cell(adLabel),'uni', 0));
set(gca,'ytick',1:4,'yticklabel',0.25:0.25:1);
set(cb,'ticks',0:10:30);
xlabel('D/P Ratio');
ylabel('Edge Probability');
set(gca,'fontsize',22);
daspect([1,1,1])
print(gcf,'-painters',fullfile(fpath,'EdgeProbabilityMap'),'-depsc');





%% -- create gabor to demonstrate stimulus --

gaborCycles = 2;
gaborWidth = 17;
padWidth = 0;
gaborRadius = 25;
gaborLength = 2*gaborRadius + 1;
imFiltWidth = 3;
boundaryWidth = 0.5;
angles = (1:4)/4*pi;

ridx = randi(numRuns);
AD = 1;
EP = 4;
redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');


% ### Figure 1 is input to basal/apical across grid positions ###
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.09 0.39 0.25 0.4]);
for pop = 1:3
    for ang = 1:numAngles
        gaborGrid = cell2mat(cellfun(@(c) drawGabor(c,gaborCycles,gaborWidth,gaborRadius), num2cell(repmat(ang,3,3)), 'uni', 0));
        subplot(3,numAngles,(pop-1)*4+ang);
        if pop==1
            for i = 1:9
                if ang==2 && i==5, continue, end
                xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
                xend = xstart + gaborLength+2*padWidth + 1;
                ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
                yend = ystart + gaborLength+2*padWidth + 1;
                gaborGrid(xstart+1:xend-1,ystart+1:yend-1) = 0;
            end
        elseif pop==2
            for i = 1:9
                if ang==2 && (i == 4 || i == 6), continue, end
                xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
                xend = xstart + gaborLength+2*padWidth + 1;
                ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
                yend = ystart + gaborLength+2*padWidth + 1;
                gaborGrid(xstart+1:xend-1,ystart+1:yend-1) = 0;
            end
        else
            for i = 1:9
                if ~(ang==2 && (i == 4 || i==5 || i == 6)), continue, end
                xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
                xend = xstart + gaborLength+2*padWidth + 1;
                ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
                yend = ystart + gaborLength+2*padWidth + 1;
                gaborGrid(xstart+1:xend-1,ystart+1:yend-1) = 0;
            end
        end
        imagesc(imgaussfilt(gaborGrid,imFiltWidth));
        for i = 1:9
            xstart = rem(i-1,3)*(gaborLength + 2*padWidth) + 0.5;
            xend = xstart + gaborLength+2*padWidth;
            ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth) + 0.5;
            yend = ystart + gaborLength+2*padWidth;
            line([xstart xstart],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','--');
            line([xend xend],[ystart yend],'color','k','linewidth',boundaryWidth,'linestyle','--');
            line([xstart xend],[ystart ystart],'color','k','linewidth',boundaryWidth,'linestyle','--');
            line([xstart xend],[yend yend],'color','k','linewidth',boundaryWidth,'linestyle','--');
        end
        cAxis = caxis;
        colormap(redblue)
        caxis(max(abs(cAxis))*[-1,1]);
        xlim([0.5 length(basalGabors)+0.5]);
        ylim([0.5 length(basalGabors)+0.5]);
        %set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
        %set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
        set(gca,'visible','off');
        set(gca,'fontsize',24);
    end
end
tightfig
% print(gcf,'-painters',fullfile(fpath,'gridTuningPopulations'),'-djpeg');


%% ### Figure 1 is input to basal/apical across grid positions ###

gaborCycles = 2;
gaborWidth = 17;
padWidth = 0;
gaborRadius = 25;
gaborLength = 2*gaborRadius + 1;
imFiltWidth = 3;
boundaryWidth = 4;
angles = (1:4)/4*pi;
redblue = cat(2, [linspace(0,1,100),ones(1,100)]', [linspace(0,1,100),linspace(1,0,100)]', [ones(1,100), linspace(1,0,100)]');

prefAngle = 2;
padWidth = 3;

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.3 0.59 0.52 0.25]);
for ang = 1:numAngles
    gaborGrid = cell2mat(cellfun(@(c) padarray(drawGabor(c,gaborCycles,gaborWidth,gaborRadius),padWidth*[1,1]), num2cell(repmat(ang,3,3)), 'uni', 0));
    subplot(1,numAngles,ang);
    
    if ang==prefAngle 
        for i = [4 5 6]
            xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
            xend = xstart + gaborLength+2*padWidth + 1;
            ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
            yend = ystart + gaborLength+2*padWidth + 1;
            gaborGrid(xstart+1:xend-1,ystart+1:yend-1) = 0;
        end
    end
    
    imagesc(imgaussfilt(gaborGrid,imFiltWidth));
    
    for i = 1:9
        if ang==prefAngle && (i==2 || i==5 || i==8)
            lineStyle = '--';
            lineWidth = 0.5;
        else
            lineStyle = '-';
            lineWidth = boundaryWidth;
        end
        xstart = rem(i-1,3)*(gaborLength+padWidth*2) + padWidth + 0.5;
        xend = xstart + gaborLength;
        ystart = rem(floor((i-1)/3),3)*(gaborLength+padWidth*2) + 0.5 + padWidth;
        yend = ystart + gaborLength;
        line([xstart xstart],[ystart yend],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
        line([xend xend],[ystart yend],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
        line([xstart xend],[ystart ystart],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
        line([xstart xend],[yend yend],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
    end    

    cAxis = caxis;
    colormap(redblue)
    caxis(max(abs(cAxis))*[-1,1]);
    xlim([0.5 length(gaborGrid)+0.5]);
    ylim([0.5 length(gaborGrid)+0.5]);
    %set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
    %set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
    set(gca,'visible','off');
    set(gca,'fontsize',24);
    daspect([1,1,1]);
end
tightfig
% print(gcf,'-painters',fullfile(fpath,'gridTuningNonpreferred'),'-depsc');

% -- again for center/edge/preferred --
figure(2); clf;
set(gcf,'units','normalized','outerposition',[0.3 0.2 0.13 0.25]);
gaborGrid = cell2mat(cellfun(@(c) padarray(drawGabor(c,gaborCycles,gaborWidth,gaborRadius),padWidth*[1,1]), num2cell(repmat(prefAngle,3,3)), 'uni', 0));
for i = [1:3 7:9]
    xstart = rem(i-1,3)*(gaborLength + 2*padWidth);
    xend = xstart + gaborLength+2*padWidth + 1;
    ystart = rem(floor((i-1)/3),3)*(gaborLength + 2*padWidth);
    yend = ystart + gaborLength+2*padWidth + 1;
    gaborGrid(xstart+1:xend-1,ystart+1:yend-1) = 0;
end

imagesc(imgaussfilt(gaborGrid,imFiltWidth));

for i = 1:9
    if (i==2 || i==8)
        colorBox = 'c';
        lineStyle = '-';
        lineWidth = boundaryWidth;
    elseif i==5
        colorBox = 'b';
        lineStyle = '-';
        lineWidth = boundaryWidth;
    else
        colorBox = 'k';
        lineStyle = '--';
        lineWidth = 0.5;
    end
    xstart = rem(i-1,3)*(gaborLength+padWidth*2) + padWidth + 0.5;
    xend = xstart + gaborLength;
    ystart = rem(floor((i-1)/3),3)*(gaborLength+padWidth*2) + 0.5 + padWidth;
    yend = ystart + gaborLength;
    line([xstart xstart],[ystart yend],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
    line([xend xend],[ystart yend],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
    line([xstart xend],[ystart ystart],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
    line([xstart xend],[yend yend],'color',colorBox,'linewidth',lineWidth,'linestyle',lineStyle);
end    
cAxis = caxis;
colormap(redblue)
caxis(max(abs(cAxis))*[-1,1]);
xlim([0.5 length(gaborGrid)+0.5]);
ylim([0.5 length(gaborGrid)+0.5]);
%set(gca,'xtick',(0:2)*gaborLength + gaborRadius+1,'xticklabel',1:3);
%set(gca,'ytick',(0:2)*gaborLength + gaborRadius+1,'yticklabel',1:3);
set(gca,'visible','off');
set(gca,'fontsize',24);
daspect([1,1,1]);
tightfig
% print(gcf,'-painters',fullfile(fpath,'gridTuningPreferred'),'-depsc');

%% -- plot trajectory --

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



















