%% -- project notes --
%{

Looks good as well! Couldn't have asked for cleaner data...

It might be nice to run some more of these so that the distribution is
cleaner (noise creates banding in the average tuning)

I like the plots where I have the source loading on the left and the basal
& apical weights on the right, shifted to the source preference and sorted
by within-source idx preference. This is good. Then I can just spend my
time explaining the model...

## I changed the model slightly -
## - first I doubled the length because it was very much still in flux.
## - second I increased the number of runs as requested above
## - third I corrected the organization of the sourceLoading so each source
##   is the same distance from each other circularly.
## - fourth, I added a setting that makes basal depression "follow" apical
##   depression, such that all inputs have reduced depression sometimes. 

Doing the basalFollow worked great because it looks like they have the same
distribution. This rocks. So the plan for presentation is to present the
results with uniform depression across basal and apical inputs, show that
the depression parameter scales how much of a relay vs. integrator.

Then show that if some have high and some have low, the properties are
preserved. 

Then I can take it to the more interesting Iacaruso model, which is here:


%}
%% -- load data --

hpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/correlation1';
dpath = [hpath,'_data'];
fpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/thesisFigures';

nameConvention = @(ridx,ad,ns,bf) sprintf('correlation1_AD%d_NS%d_BF%d_Run%d.mat',ad,ns,bf,ridx);

numRuns = 50;
numAD = 4;
numNS = 1;
numBF = 2;

% load one to get params
sampleData = load(fullfile(dpath,nameConvention(1,1,1,1)));
T = sampleData.iaf.T;
lengthTraj = T/10000;

numInputs = sampleData.iaf.numInputs;
numSignals = sampleData.iaf.numSignals;
sourceLoading = {1,numNS};
sourcePeakIdx = {1,numNS};
firstSignalIdx = zeros(numNS,1);
shiftSignalIdx = zeros(numNS,1);

for ns = 1:numNS
    sampleData = load(fullfile(dpath,nameConvention(1,1,ns,1)));
    [~,sourcePeakIdx{ns}] = max(sampleData.iaf.sourceLoading');
    firstSignalIdx(ns) = sourcePeakIdx{ns}(1);
    shiftSignalIdx(ns) = diff(sourcePeakIdx{ns}(1:2));
    sourceLoading{ns} = sampleData.iaf.sourceLoading';
end

% load final weights
keepIdx = lengthTraj-1500:lengthTraj-1400;
scaleApicalWeight = 3e-9;
scaleBasalWeight = 35e-9;
dsFactor = 10;
apicalWeights = zeros(numInputs, numAD, numNS, numBF, numRuns);
basalWeights = zeros(numInputs, numAD, numNS, numBF, numRuns);
basalTrajectory = zeros(numInputs, lengthTraj/dsFactor, numAD,numNS,numBF,numRuns);
apicalTrajectory = zeros(numInputs, lengthTraj/dsFactor, numAD,numNS,numBF,numRuns);
for ridx = 1:numRuns
    for ad = 1:numAD
        for ns = 1:numNS
            for bf = 1:numBF
                cdata = load(fullfile(dpath,nameConvention(ridx,ad,ns,bf-1)),'smallBasalWeight','smallApicalWeight');
                apicalWeights(:,ad,ns,bf,ridx) = mean(cdata.smallApicalWeight(:,keepIdx),2);
                basalWeights(:,ad,ns,bf,ridx) = mean(cdata.smallBasalWeight(:,keepIdx),2);
                apicalTrajectory(:,:,ad,ns,bf,ridx) = permute(mean(reshape(permute(cdata.smallApicalWeight,[3,2,1]),dsFactor,lengthTraj/dsFactor,numInputs),1),[3,2,1]);
                basalTrajectory(:,:,ad,ns,bf,ridx) = permute(mean(reshape(permute(cdata.smallBasalWeight,[3,2,1]),dsFactor,lengthTraj/dsFactor,numInputs),1),[3,2,1]);
            end
        end
    end
end



% Analyze Edge Tuning 
sourcePref = zeros(numAD,numNS,numBF,numRuns);
basalPidx = zeros(numInputs,numAD,numNS,numBF,numRuns);
basalPref = zeros(numInputs,numAD,numNS,numBF,numRuns);
apicalPref = zeros(numInputs,numAD,numNS,numBF,numRuns);
basalPTraj = zeros(numInputs, lengthTraj/dsFactor, numAD, numNS, numBF,numRuns);
apicalPTraj = zeros(numInputs,lengthTraj/dsFactor, numAD, numNS, numBF,numRuns);
for ridx = 1:numRuns
    for ad = 1:numAD
        for ns = 1:numNS
            for bf = 1:numBF
                [~,peakWeightIdx] = max(basalWeights(:,ad,ns,bf,ridx));
                peakWeightIdx = peakWeightIdx(1); % in case there's multiple
                [~,closestIdx] = min(abs(peakWeightIdx - sourcePeakIdx{ns}));
                shiftSamples = -shiftSignalIdx(ns) * (closestIdx-1);
                sourcePref(ad,ns,bf,ridx) = closestIdx;

                basalPref(:,ad,ns,bf,ridx) = circshift(basalWeights(:,ad,ns,bf,ridx),shiftSamples);
                apicalPref(:,ad,ns,bf,ridx) = circshift(apicalWeights(:,ad,ns,bf,ridx),shiftSamples);
                basalPTraj(:,:,ad,ns,bf,ridx) = circshift(basalTrajectory(:,:,ad,ns,bf,ridx),shiftSamples,1);
                apicalPTraj(:,:,ad,ns,bf,ridx) = circshift(apicalTrajectory(:,:,ad,ns,bf,ridx),shiftSamples,1);
            end
        end
    end
end


%% -- look at examples --
ad = 4;
ns = 1;
bf = 1;

%b2plot = squeeze(basalPref(:,ad,ns,bf,:));
b2plot = squeeze(basalWeights(:,ad,ns,bf,:));
% b2plot = b2plot ./ max(b2plot,[],1);
%a2plot = squeeze(apicalPref(:,ad,ns,bf,:));
a2plot = squeeze(apicalWeights(:,ad,ns,bf,:));
% a2plot = a2plot ./ max(a2plot,[],1);
[~,pkidx] = max(b2plot,[],1);
[~,idx] = sort(pkidx);

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.16 0.43 0.73 0.44]);
subplot(1,3,1); 
imagesc(sourceLoading{ns});
xlabel('Source');
ylabel('Input');
title('Loadings');
set(gca,'fontsize',24);
colormap('hot');

subplot(1,3,2); 
imagesc(b2plot(:,idx));
xlabel('Run #');
ylabel('Input');
title('Basal Weights');
set(gca,'fontsize',24);

% subplot(1,5,3); hold on;
% plot(normalize(mean(sourceLoading{ns},2),'range',[0,1]),numInputs:-1:1,'color','k','linewidth',1.5);
% plot(normalize(mean(b2plot(:,idx),2),'range',[0,1]),numInputs:-1:1,'color','r','linewidth',1.5);
% xlabel({'Loading &';'Mean Basal Weight'});
% ylabel('Input');
% title('Basal Weights');
% set(gca,'fontsize',24);

subplot(1,3,3); 
imagesc(a2plot(:,idx));
xlabel('Run #');
ylabel('Input');
title('Apical Weights');
set(gca,'fontsize',24);

% subplot(1,5,5); hold on;
% plot(normalize(mean(sourceLoading{ns},2),'range',[0,1]),numInputs:-1:1,'color','k','linewidth',1.5);
% plot(normalize(mean(a2plot(:,idx),2),'range',[0,1]),numInputs:-1:1,'color','r','linewidth',1.5);
% xlabel({'Loading &';'Mean Apical Weight'});
% ylabel('Input');
% title('Apical Weights');
% set(gca,'fontsize',24);



%% -- look at examples --
r2plot = 3;
ad = 4;
ns = 2;
bf = 1;
ridx = randperm(numRuns,r2plot);
figure(1); clf; 
for r = 1:r2plot
    subplot(2,r2plot,r);
    imagesc(basalWeights(:,ad,ns,bf,ridx(r))');
    
    subplot(2,r2plot,r+r2plot);
    imagesc(apicalWeights(:,:,ad,ns,bf,ridx(r))');
end

%% -- look at trajectory --

cc = zeros(numInputs,3);%hsv(numInputs);
AD = 4; 
NS = 1;
BF = 1;
figure(1); clf; 
subplot(2,2,1); imagesc(mean(basalPTraj(:,:,AD,NS,BF,:),6));
subplot(2,2,2); imagesc(mean(apicalPTraj(:,:,AD,NS,BF,:),6));

smWidth = 10;
subplot(2,2,3); hold on;
for in = 1:numInputs
    mnplot = mean(filter(ones(1,smWidth),1,basalPTraj(in,:,AD,NS,BF,:),[],2),6);
    sdplot = std(filter(ones(1,smWidth),1,basalPTraj(in,:,AD,NS,BF,:),[],2),1,6);
    patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(in,:),'EdgeColor','none','FaceAlpha',0.1);
    plot(1:lengthTraj/dsFactor,mnplot,'color',cc(in,:),'linewidth',1.5);
end
subplot(2,2,4); hold on;
for in = 1:numInputs
    mnplot = mean(filter(ones(1,smWidth),1,apicalPTraj(in,:,AD,NS,BF,:),[],2),6);
    sdplot = std(filter(ones(1,smWidth),1,apicalPTraj(in,:,AD,NS,BF,:),[],2),1,6);
    patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(in,:),'EdgeColor','none','FaceAlpha',0.1);
    plot(1:lengthTraj/dsFactor,mnplot,'color',cc(in,:),'linewidth',1.5);
end










