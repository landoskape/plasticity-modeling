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


## Figure making notes ##
- when I measure pairwise correlation between pre and post, I have to do it
of the (smoothed) spike counts, rather than the internal signal. I think
this is why I'm getting weird data from the Low/Low condition
- that being said, I should also compare the correlation between
postsynaptic activity and source activity. Since each individual
presynaptic cell is noisy, I'd expect wider tuning to have better
correlation with the source, which can explain the above result and
provides further computational motivation for the mismatched D/P ratio.

- also, I kinda want to run it with a proper exponential drop-off rather
than a circular gaussian smooth peak... hmmmmm

%}


%% -- load data --

hpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/correlation1';
dpath = [hpath,'_data/old'];
fpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/thesisFigures';

nameConvention = @(ridx,ad,ns,bf) sprintf('correlation1_AD%d_NS%d_BF%d_Run%d.mat',ad,ns,bf,ridx);

numRuns = 100;
numAD = 4;
numNS = 1;
numBF = 2;

% load one to get params
sampleData = load(fullfile(dpath,nameConvention(1,1,1,1)));
T = sampleData.iaf.T;
lengthTraj = T/10000;
maxBasalWeight = sampleData.iaf.maxBasalWeight;
maxApicalWeight = sampleData.iaf.maxApicalWeight;

numInputs = sampleData.iaf.numInputs;
numSignals = sampleData.iaf.numSignals;
numApical = sampleData.iaf.numApical;
numBasal = sampleData.iaf.numBasal;
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

% -- compute average pairwise correlation score --
cvTheory = (sampleData.iaf.rateStd.^2 ./ (sampleData.iaf.varAdjustment' * sampleData.iaf.varAdjustment)) .* (eye(100) + sourceLoading{1} * sourceLoading{1}');
crTheory = cvTheory ./ (sqrt(diag(cvTheory))*sqrt(diag(cvTheory))'); % correlation theory

% load final weights
keepIdx = lengthTraj-100:lengthTraj;
scaleApicalWeight = 3e-9;
scaleBasalWeight = 35e-9;
dsFactor = 10;
apicalWeights = zeros(numInputs, numAD, numNS, numBF, numRuns);
basalWeights = zeros(numInputs, numAD, numNS, numBF, numRuns);
apicalTrajectory = zeros(numInputs, lengthTraj/dsFactor, numAD,numNS,numBF,numRuns);
basalTrajectory = zeros(numInputs, lengthTraj/dsFactor, numAD,numNS,numBF,numRuns);
sourcePref = zeros(numAD,numNS,numBF,numRuns);
basalPidx = zeros(numInputs,numAD,numNS,numBF,numRuns);
basalPref = zeros(numInputs,numAD,numNS,numBF,numRuns);
apicalPref = zeros(numInputs,numAD,numNS,numBF,numRuns);
basalCounts = nan(numInputs,numAD,numNS,numBF,numRuns);
apicalCounts = nan(numInputs,numAD,numNS,numBF,numRuns);
basalPTraj = zeros(numInputs, lengthTraj/dsFactor, numAD, numNS, numBF,numRuns);
apicalPTraj = zeros(numInputs,lengthTraj/dsFactor, numAD, numNS, numBF,numRuns);
basalIndex = zeros(numBasal, numAD,numNS,numBF,numRuns);
apicalIndex = zeros(numApical, numAD,numNS,numBF,numRuns);
basalSynWeight = zeros(numBasal, numAD,numNS,numBF,numRuns);
apicalSynWeight = zeros(numApical, numAD,numNS,numBF,numRuns);
% preallocate for average pairwise correlation score
useCondThresh = true; 
basalCondThresh = sampleData.iaf.basalCondThresh;
apicalCondThresh = sampleData.iaf.apicalCondThresh;
pwCorrBasal = zeros(numAD,numNS,numBF,numRuns);
pwCorrApical = zeros(numAD,numNS,numBF,numRuns);
ppCorrBasalSource = zeros(numAD,numNS,numBF,numRuns);
ppCorrApicalSource = zeros(numAD,numNS,numBF,numRuns);
ppCorrBasalSpikes = zeros(numAD,numNS,numBF,numRuns);
ppCorrApicalSpikes = zeros(numAD,numNS,numBF,numRuns);
pwCorrSources = zeros(3,numAD,numNS,numBF,numRuns);
ppCorrBasalSource1 = zeros(numAD,numNS,numBF,numRuns);
ppCorrApicalSource1 = zeros(numAD,numNS,numBF,numRuns);
ppCorrBasalSpikes1 = zeros(numAD,numNS,numBF,numRuns);
ppCorrApicalSpikes1 = zeros(numAD,numNS,numBF,numRuns);
pwCorrSources1 = zeros(3,numAD,numNS,numBF,numRuns);

numKeepSpikes = 1200;
numBootSpikes = 5;
% Run model
T = 75*1000;
spikeFilter = 20; % in ms
verbose = false;

msg = '';
for ridx = 1:numRuns
    for ad = 1:numAD
        for ns = 1:numNS
            for bf = 1:numBF
                fprintf(repmat('\b',1,length(msg)));
                msg = sprintf('%d/%d, %d/%d, %d/%d, %d/%d\n',ridx,numRuns,ad,numAD,ns,numNS,bf,numBF);
                fprintf(msg);
                
                cdata = load(fullfile(dpath,nameConvention(ridx,ad,ns,bf-1)),'iaf','smallBasalWeight','smallApicalWeight');
                apicalWeights(:,ad,ns,bf,ridx) = mean(cdata.smallApicalWeight(:,keepIdx),2);
                basalWeights(:,ad,ns,bf,ridx) = mean(cdata.smallBasalWeight(:,keepIdx),2);
                apicalTrajectory(:,:,ad,ns,bf,ridx) = permute(mean(reshape(permute(cdata.smallApicalWeight,[3,2,1]),dsFactor,lengthTraj/dsFactor,numInputs),1),[3,2,1]);
                basalTrajectory(:,:,ad,ns,bf,ridx) = permute(mean(reshape(permute(cdata.smallBasalWeight,[3,2,1]),dsFactor,lengthTraj/dsFactor,numInputs),1),[3,2,1]);
                
                [~,peakWeightIdx] = max(basalWeights(:,ad,ns,bf,ridx));
                peakWeightIdx = peakWeightIdx(1); % in case there's multiple
                [~,closestIdx] = min(abs(peakWeightIdx - sourcePeakIdx{ns}));
                shiftSamples = -shiftSignalIdx(ns) * (closestIdx-2);
                sourcePref(ad,ns,bf,ridx) = closestIdx;
                
                centerBasalIdx = mod(cdata.iaf.basalTuneIdx + shiftSamples - 1, 100)+1;
                centerApicalIdx = mod(cdata.iaf.apicalTuneIdx + shiftSamples - 1, 100)+1;
                basalPref(:,ad,ns,bf,ridx) = circshift(basalWeights(:,ad,ns,bf,ridx),shiftSamples);
                apicalPref(:,ad,ns,bf,ridx) = circshift(apicalWeights(:,ad,ns,bf,ridx),shiftSamples);
                basalCounts(:,ad,ns,bf,ridx) = arrayfun(@(inputIdx) sum(centerBasalIdx==inputIdx), (1:100)', 'uni', 1);
                apicalCounts(:,ad,ns,bf,ridx) = arrayfun(@(inputIdx) sum(centerApicalIdx==inputIdx), (1:100)', 'uni', 1);
                basalPTraj(:,:,ad,ns,bf,ridx) = circshift(basalTrajectory(:,:,ad,ns,bf,ridx),shiftSamples,1);
                apicalPTraj(:,:,ad,ns,bf,ridx) = circshift(apicalTrajectory(:,:,ad,ns,bf,ridx),shiftSamples,1);
                
                basalIndex(:,ad,ns,bf,ridx) = cdata.iaf.basalTuneIdx;
                apicalIndex(:,ad,ns,bf,ridx) = cdata.iaf.apicalTuneIdx;
                basalSynWeight(:,ad,ns,bf,ridx) = cdata.iaf.basalWeight;
                apicalSynWeight(:,ad,ns,bf,ridx) = cdata.iaf.apicalWeight;
                
                % Weighted average of pairwise correlations-basal
                cbWeights = basalSynWeight(:,ad,ns,bf,ridx);
                if useCondThresh, cbWeights(cbWeights<basalCondThresh)=0; end
                caWeights = apicalSynWeight(:,ad,ns,bf,ridx);
                if useCondThresh, caWeights(caWeights<apicalCondThresh)=0; end
                % Using the above later
                [iIdx,jIdx] = meshgrid(basalIndex(:,ad,ns,bf,ridx));
                idx = sub2ind(size(crTheory),iIdx,jIdx);
                cCorr = crTheory(idx);
                cWeightMat = cbWeights * cbWeights';
                adjustedWeightMat = cWeightMat - diag(diag(cWeightMat));
                pwCorrBasal(ad,ns,bf,ridx) = sum(cCorr .* adjustedWeightMat, 'all') / sum(adjustedWeightMat,'all');
                % Weighted average of pairwise correlations-apical
                [iIdx,jIdx] = meshgrid(apicalIndex(:,ad,ns,bf,ridx));
                idx = sub2ind(size(crTheory),iIdx,jIdx);
                cCorr = crTheory(idx);
                cWeightMat = caWeights * caWeights';
                adjustedWeightMat = cWeightMat - diag(diag(cWeightMat));
                pwCorrApical(ad,ns,bf,ridx) = sum(cCorr .* adjustedWeightMat, 'all') / sum(adjustedWeightMat,'all');
                
                % Compute weighted correlation between pre & post
                [~,spkTimes,y,source,bPreSpikes,aPreSpikes] = runFitModels(cdata.iaf,T,verbose);
                basalInput = y(basalIndex(:,ad,ns,bf,ridx),:);
                apicalInput = y(apicalIndex(:,ad,ns,bf,ridx),:);
                spikes = zeros(T,1);
                spikes(spkTimes) = 1;
                piBSource1 = (basalInput - mean(basalInput,2)) * (spikes - mean(spikes)) / T;
                piASource1 = (apicalInput - mean(apicalInput,2)) * (spikes - mean(spikes)) / T;
                ppCorrBasalSource1(ad,ns,bf,ridx) = (piBSource1' * cbWeights) / sum(cbWeights);
                ppCorrApicalSource1(ad,ns,bf,ridx) = (piASource1' * caWeights) / sum(caWeights);
                piBasal = (bPreSpikes - mean(bPreSpikes,2)) * (spikes - mean(spikes)) / T;
                piApical = (aPreSpikes - mean(aPreSpikes,2)) * (spikes - mean(spikes)) / T;
                ppCorrBasalSpikes1(ad,ns,bf,ridx) = (mean(piBasal,2)' * cbWeights) / sum(cbWeights);
                ppCorrApicalSpikes1(ad,ns,bf,ridx) = (mean(piApical,2)' * caWeights) / sum(caWeights);
                pwCorrSources1(:,ad,ns,bf,ridx) = (source - mean(source,2)) * (spikes - mean(spikes)) / T;                
                % Now bootstrap over spikes to keep spike count consistent
                if length(spkTimes)<numKeepSpikes
                    error('shit');
                end
                piBSource = zeros(numBasal,numBootSpikes);
                piASource = zeros(numApical,numBootSpikes);
                piBasal = zeros(numBasal,numBootSpikes);
                piApical = zeros(numApical,numBootSpikes);
                crSource = zeros(numSignals,numBootSpikes);
                % Compute pi from underlying membrane potential (once)
                for sbs = 1:numBootSpikes
                    cspkTimes = spkTimes(randperm(length(spkTimes),numKeepSpikes));
                    cspikes = zeros(T,1);
                    cspikes(cspkTimes) = 1;
                    cspikes = smoothsmooth(cspikes, spikeFilter); 
                    
                    % Compute pi from membrane potential
                    piBSource(:,sbs) = (basalInput - mean(basalInput,2)) * (cspikes - mean(cspikes)) / T;
                    piASource(:,sbs) = (apicalInput - mean(apicalInput,2)) * (cspikes - mean(cspikes)) / T;
                    
                    % Compute pi from smoothed spike rates
                    bPreSpikes = smoothsmooth(bPreSpikes',spikeFilter)';
                    aPreSpikes = smoothsmooth(aPreSpikes',spikeFilter)';
                    piBasal(:,sbs) = (bPreSpikes - mean(bPreSpikes,2)) * (cspikes - mean(cspikes)) / T;
                    piApical(:,sbs) = (aPreSpikes - mean(aPreSpikes,2)) * (cspikes - mean(cspikes)) / T;
                    crSource(:,sbs) = (source - mean(source,2)) * (cspikes - mean(cspikes)) / T;
                end
                %ppCorrBasalSource(ad,ns,bf,ridx) = (piBasal' * cbWeights) / sum(cbWeights);
                %ppCorrApicalSource(ad,ns,bf,ridx) = (piApical' * caWeights) / sum(caWeights);
                ppCorrBasalSource(ad,ns,bf,ridx) = (mean(piBSource,2)' * cbWeights) / sum(cbWeights);
                ppCorrApicalSource(ad,ns,bf,ridx) = (mean(piASource,2)' * caWeights) / sum(caWeights);
                %ppCorrBasalSpikes(ad,ns,bf,ridx) = (piBasal' * cbWeights) / sum(cbWeights);
                %ppCorrApicalSpikes(ad,ns,bf,ridx) = (piApical' * caWeights) / sum(caWeights);
                ppCorrBasalSpikes(ad,ns,bf,ridx) = (mean(piBasal,2)' * cbWeights) / sum(cbWeights);
                ppCorrApicalSpikes(ad,ns,bf,ridx) = (mean(piApical,2)' * caWeights) / sum(caWeights);
                % Compute pi with sources
                pwCorrSources(:,ad,ns,bf,ridx) = mean(crSource,2);
            end
        end
    end
end

% Just in case I want this --- It's weird because it's biphasic do to vast
% overrepresentation of central weight sometimes...
basalAverageWeight = basalPref ./ basalCounts / maxBasalWeight;
apicalAverageWeight = apicalPref ./ apicalCounts / maxApicalWeight;
basalAverageWeight(basalCounts==0)=0;
apicalAverageWeight(apicalCounts==0)=0;

% save(fullfile(hpath,'compiledDataThesisVersion'),'-v7.3');

%%
msg = '';
spkRate = zeros(numAD,numNS,numBF,numRuns);
for ridx = 1:numRuns
    for ad = 1:numAD
        for ns = 1:numNS
            for bf = 1:numBF
                fprintf(repmat('\b',1,length(msg)));
                msg = sprintf('%d/%d, %d/%d, %d/%d, %d/%d\n',ridx,numRuns,ad,numAD,ns,numNS,bf,numBF);
                fprintf(msg);
                cdata = load(fullfile(dpath,nameConvention(ridx,ad,ns,bf-1)),'spkTimes','iaf');
                spkRate(ad,ns,bf,ridx) = length(cdata.spkTimes(cdata.spkTimes>cdata.iaf.T/2))/(cdata.iaf.T/2);
            end
        end
    end
end

%% 
numSpikes = spkRate * 75*1000;
figure(1); clf; 
adbf = [2,1,1; 2,4,1; 1,4,1]; % adbf combinations for each figure (adding state in case I want it later)
hold on;
for combo = 1:size(adbf,1)
    cad = adbf(combo,2);
    cns = adbf(combo,3);
    cbf = adbf(combo,1);
    d2plot = squeeze(spkRate(cad,cns,cbf,:))*1000;
    scatter(combo + xscat(d2plot,xRange), d2plot ,48,'k','marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor','k','markerfacealpha',0.3);
end

%% -- Plot Source Data and Example Rates --

T = 5000;
filterWidth = 200;

sdata = load(fullfile(dpath,nameConvention(1,1,ns,1)));
sources = randn(T,size(sourceLoading{1},2));
sources = filter(ones(1,filterWidth)/sqrt(filterWidth),1,sources,[],1);

input = (randn(T,sdata.iaf.numInputs) + sources*sourceLoading{1}')./sdata.iaf.varAdjustment;
rate = sdata.iaf.rateStd * input + sdata.iaf.rateMean;
rate(rate<0) = 0;

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.16 0.09 0.21 0.78]);

subplot(4,1,1); 
imagesc(sourceLoading{ns}');
set(gca,'xtick',[]);
%xlabel('Presynaptic Cell #');
ylabel('Sources');
title('Loadings');
set(gca,'fontsize',24);
colormap('hot');

subplot(4,1,2:4); 
imagesc(rate)
set(gca,'ytick',[]);
xlabel('Presynaptic Cell #');
ylabel('Time');
title('Firing Rate');
set(gca,'fontsize',24);

% print(gcf,'-painters',fullfile(fpath,'loadingAndRateSample'),'-djpeg');


%% -- Plot Summary Data --
% in this figure I want the D/P ratio schematic on the left (will do that
% in illustrator)
% then I want a histogram showing how often each source was preferred
% then I want a histogram showing the number of B/A synapses connected each
% loading strength (where the x axis is loading strength)
% also a net synaptic weight for loading
% -- for the above two graphs, let's do it like an exponential (where it
% starts high and goes low, representing both sides of the peak), and then
% have a similarly structured inset for the non-preferred sources. 


sampleData = load(fullfile(dpath,nameConvention(1,1,1,1)));
maxBasalWeight = sampleData.iaf.maxBasalWeight;
maxApicalWeight = sampleData.iaf.maxApicalWeight;
adbf = [2,1,1; 2,3,1; 1,3,1]; % adbf combinations for each figure (adding state in case I want it later)
numCombo = size(adbf,1);

normBasalPref = 100 * basalPref / maxBasalWeight / numBasal;
normApicalPref = 100 * apicalPref / maxApicalWeight / numApical;
normBasalCounts = basalCounts / numBasal;
normApicalCounts = apicalCounts / numApical;
normBasalCounts(normBasalCounts==0)=nan;
normApicalCounts(normApicalCounts==0)=nan;

sourceColors = 'rkr';
sourceAlpha = [0.1 0.5 0.1];
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.21 0.18 0.42 0.78]);

useBWeight = basalPref;
useAWeight = apicalPref;
yMaxWeight = max([max(mean(useBWeight,5),[],'all'), max(mean(useAWeight,5),[],'all')]);

dpXPos = [69 8 3];
dpYPos = [4 4.6];
dpXTitle = dpXPos(1) + dpXPos(2) + dpXPos(3)/2;
dpYTitle = 5.3;
dpMrkSize = 32;
dpYPosIdx = [2,2; 1,1; 2,1];
legName = {'High/High','Low/Low','High/Low'};

fwhm = zeros(numCombo,numRuns,2);

for combo = 1:numCombo
    cad = adbf(combo,2);
    cbf = adbf(combo,1);
    cns = adbf(combo,3);
    
    subplot(4,2,combo*2); hold on;
    bar(1:3, arrayfun(@(source) sum(sourcePref(cad,cns,cbf,:)==source), 1:3, 'uni', 1),0.75, 'FaceColor','k','FaceAlpha',1);    
    set(gca,'xtick',1:3);
    set(gca,'ytick',0:20:40);
    xlim([0.5 3.5]);
    ylim([0 50]);
    if combo==numCombo
        xlabel('Source');
    end
    ylabel('# Preferred');
    if combo==1
        title('Source Preference');
    end
    set(gca,'fontsize',20);

    mnBasalPref = mean(useBWeight(:,cad,cns,cbf,:),5);
    mnApicalPref = mean(useAWeight(:,cad,cns,cbf,:),5);
    subplot(4,2,(combo-1)*2+1); hold on;
    shadedErrorBar(1:100, mnBasalPref, nanstd(useBWeight(:,cad,cns,cbf,:),1,5)/sqrt(numRuns),{'color','k','linewidth',1.5},1);
    shadedErrorBar(1:100, mnApicalPref, nanstd(useAWeight(:,cad,cns,cbf,:),1,5)/sqrt(numRuns),{'color','b','linewidth',1.5},1);
    set(gca,'ytick',0:2:yMaxWeight);
    xlim([0.5 100.5]);
    ylim([0 yMaxWeight]);
    ylabel('Weight (%)');
    if combo==1
        title('Weight Distribution');
    end
    line(dpXPos(1)+[0 dpXPos(2)],dpYPos(1)*[1,1],'color','k','linewidth',1);
    line(sum(dpXPos(1:3))+[0 dpXPos(2)],dpYPos(1)*[1,1],'color','k','linewidth',1);
    plot(dpXPos(1)+dpXPos(2)/2,dpYPos(dpYPosIdx(combo,1)),'color','k','marker','.','markersize',dpMrkSize);
    plot(sum(dpXPos(1:3))+dpXPos(2)/2,dpYPos(dpYPosIdx(combo,2)),'color','b','marker','.','markersize',dpMrkSize);
    text(sum(dpXPos([1:3,2]))+1,dpYPos(1),'100%','Fontsize',12);
    text(sum(dpXPos([1:3,2]))+1,dpYPos(2),'110%','Fontsize',12);
    text(dpXTitle,dpYTitle,'D/P Ratio','HorizontalAlignment','Center','Fontsize',12);
    text(5,5.25,legName{combo},'Fontsize',16,'Fontweight','Bold');
    set(gca,'fontsize',20);
    
    % And compute fwhm
    for ridx = 1:numRuns
        mxb = max(normBasalPref(:,cad,cns,cbf,ridx));
        mxa = max(normApicalPref(:,cad,cns,cbf,ridx));
        fwhm(combo,ridx,1) = find(normBasalPref(:,cad,cns,cbf,ridx)>=mxb/2,1,'last')-find(normBasalPref(:,cad,cns,cbf,ridx)>=mxb/2,1,'first')+1;
        fwhm(combo,ridx,2) = find(normApicalPref(:,cad,cns,cbf,ridx)>=mxa/2,1,'last')-find(normApicalPref(:,cad,cns,cbf,ridx)>=mxa/2,1,'first')+1;
    end
end

subplot(4,2,7);
for source = [2,1,3]
    patch([1:numInputs numInputs:-1:1],[sourceLoading{cns}(:,source)' zeros(1,numInputs)],...
        sourceColors(source),'FaceColor',sourceColors(source),'FaceAlpha',sourceAlpha(source),'EdgeColor','none');
end
set(gca,'ytick',[0 1]);
xlim([0.5 100.5]);
ylim([0 1.6]);
xlabel('Presynaptic Cell #');
ylabel('Loading');
set(gca,'fontsize',20);
legend('Preferred','Nonpreferred','location','northwest','Fontsize',12);

subplot(4,2,8); hold on;
shadedErrorBar(1:3, mean(fwhm(:,:,1),2),std(fwhm(:,:,1),1,2),{'color','k','linewidth',1.5,'marker','.','markersize',24},1);
shadedErrorBar(1:3, mean(fwhm(:,:,2),2),std(fwhm(:,:,2),1,2),{'color','b','linewidth',1.5,'marker','.','markersize',24},1);
set(gca,'xtick',1:3,'xticklabel',legName,'xticklabelrotation',45);
xlim([0.8 3.2]);
ylim([0 25]);
ylabel('FWHM');
title('Tuning Width');
set(gca,'fontsize',20);

% print(gcf,'-painters',fullfile(fpath,'sourceSummaryData'),'-depsc');


%% -- plot complexity data --

adbf = [2,1,1; 2,4,1; 1,4,1]; % adbf combinations for each figure (adding state in case I want it later)
numCombo = size(adbf,1);
sumPairCorr = zeros(numCombo,numRuns,2);
sumPPSource = zeros(numCombo,numRuns,2);
sumPPSpikes = zeros(numCombo,numRuns,2);
sumCorrSource = zeros(numCombo,3,numRuns);
for combo = 1:numCombo
    cad = adbf(combo,2);
    cbf = adbf(combo,1);
    cns = adbf(combo,3);
    sumPairCorr(combo,:,1) = pwCorrBasal(cad,cns,cbf,:);
    sumPairCorr(combo,:,2) = pwCorrApical(cad,cns,cbf,:);
    sumPPSource(combo,:,1) = ppCorrBasalSource(cad,cns,cbf,:);
    sumPPSource(combo,:,2) = ppCorrApicalSource(cad,cns,cbf,:);
    sumPPSpikes(combo,:,1) = ppCorrBasalSpikes(cad,cns,cbf,:);
    sumPPSpikes(combo,:,2) = ppCorrApicalSpikes(cad,cns,cbf,:);
    
    sumCorrSource(combo,:,:) = pwCorrSources(:,cad,cns,cbf,:);
end
sumCorrSource = sort(sumCorrSource,2,'descend');

xRange = 0.35;
xOffset = 0.15;
figure(2); clf; 
set(gcf,'units','normalized','outerposition',[0.17 0.47 0.67 0.37]);

subplot(1,3,1); hold on;
for combo = 1:numCombo
    %plot(combo - 0.1 + xscat(sumPairCorr(combo,:,1),xRange), sumPairCorr(combo,:,1),'marker','o','linestyle','none','color','k','markersize',4,'markerfacecolor','none','linewidth',1);
    %plot(combo + 0.1 + xscat(sumPairCorr(combo,:,2),xRange), sumPairCorr(combo,:,2),'marker','o','linestyle','none','color','b','markersize',4,'markerfacecolor','none','linewidth',1);
    scatter(combo - xOffset + xscat(sumPairCorr(combo,:,1),xRange), sumPairCorr(combo,:,1),48,'k','marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor','k','markerfacealpha',0.3);
    scatter(combo + xOffset + xscat(sumPairCorr(combo,:,2),xRange), sumPairCorr(combo,:,2),48,'b','marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor','b','markerfacealpha',0.3);
end
legName = {'High/High','Low/Low','High/Low'};
set(gca,'xtick',1:3,'xticklabel',legName);
set(gca,'ytick',0:0.2:1);
xlim([0.5 3.5]);
ylim([0 1]);
ylabel('weighted pairwise input corr');
legend('Basal','Apical','location','southwest');
set(gca,'fontsize',24);

useMetric = sumPPSpikes;
subplot(1,3,2); hold on;
for combo = 1:numCombo
    %plot(combo - 0.1 + xscat(sumPairCorr(combo,:,1),xRange), sumPairCorr(combo,:,1),'marker','o','linestyle','none','color','k','markersize',4,'markerfacecolor','none','linewidth',1);
    %plot(combo + 0.1 + xscat(sumPairCorr(combo,:,2),xRange), sumPairCorr(combo,:,2),'marker','o','linestyle','none','color','b','markersize',4,'markerfacecolor','none','linewidth',1);
    scatter(combo - xOffset + xscat(useMetric(combo,:,1),xRange), useMetric(combo,:,1),48,'k','marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor','k','markerfacealpha',0.3);
    scatter(combo + xOffset + xscat(useMetric(combo,:,2),xRange), useMetric(combo,:,2),48,'b','marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor','b','markerfacealpha',0.3);
end
legName = {'High/High','Low/Low','High/Low'};
set(gca,'xtick',1:3,'xticklabel',legName);
set(gca,'ytick',0:2e-5:8e-5);
xlim([0.5 3.5]);
ylim([0 8e-5]);
ylabel('weighted cov(post, pre)');
legend('Basal','Apical','location','southwest');
set(gca,'fontsize',24);


subplot(1,3,3); hold on;
for combo = 1:numCombo
    nonPreferred = reshape(sumCorrSource(combo,2:3,:),numRuns*2,1);
    scatter(combo - 0*xOffset + xscat(sumCorrSource(combo,1,:),0.8*xRange), sumCorrSource(combo,1,:),48,[0.5,0,1],'marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor',[0.5,0,1],'markerfacealpha',0.3);
    scatter(combo + 0*xOffset + xscat(nonPreferred,0.8*xRange), nonPreferred,48,'k','marker','o','markeredgecolor','none','linewidth',1,'markerfacecolor','k','markerfacealpha',0.3);
end
legName = {'High/High','Low/Low','High/Low'};
set(gca,'xtick',1:3,'xticklabel',legName);
set(gca,'ytick',-2e-3:2e-3:10e-3);
xlim([0.5 3.5]);
ylim([-2e-3 10e-3]);
ylabel('cov(Post, Source)');
legend('Preferred','Non-preferred','location','northwest');
set(gca,'fontsize',24);

print(gcf,'-painters',fullfile(fpath,'complexityCorrelationSummary'),'-depsc');


%% -- Plot Source Loading Matrix and All Run Weight Heat Maps (unsorted) --
ad = 3;
ns = 1;
bf = 1;

b2plot = squeeze(basalPref(:,ad,ns,bf,:));
% b2plot = squeeze(basalWeights(:,ad,ns,bf,:));
b2plot = b2plot ./ max(b2plot,[],1);
a2plot = squeeze(apicalPref(:,ad,ns,bf,:));
% a2plot = squeeze(apicalWeights(:,ad,ns,bf,:));
a2plot = a2plot ./ max(a2plot,[],1);
[~,pkidx] = max(b2plot,[],1);
[~,idx] = sort(pkidx);

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.16 0.45 0.49 0.42]);
subplot(1,3,1); 
imagesc(sourceLoading{ns});
xlabel('Source');
ylabel('Input');
title('Loadings');
set(gca,'fontsize',24);
colormap('hot');

% prmCombo = [2,1; 2,4; 1,4];
% prmTitle = {'Both High','Both Low','B:High, A:Low'};
% for i = 1:3
%     b2plot = squeeze(basalPref(:,prmCombo(i,2),ns,prmCombo(i,1),:));
%     a2plot = squeeze(apicalPref(:,prmCombo(i,2),ns,prmCombo(i,1),:));
% 
%     subplot(1,4,i+1); hold on;
%     plot(normalize(mean(b2plot(:,idx),2),'range',[0,1]),numInputs:-1:1,'color','k','linewidth',1.5);
%     plot(normalize(mean(a2plot(:,idx),2),'range',[0,1]),numInputs:-1:1,'color','b','linewidth',1.5);
%     xlabel('Mean Weight');
%     ylabel('Input');
%     title(prmTitle{i});
%     set(gca,'fontsize',24);
% end
% legend('Basal','Apical','location','southeast');

% print(gcf,'-painters',fullfile(fpath,'SourceLoadingSummary'),'-djpeg');

subplot(1,3,2); 
imagesc(b2plot(:,idx));
xlabel('Run #');
ylabel('Input');
title('Basal Weights');
set(gca,'fontsize',24);

subplot(1,3,3); 
imagesc(a2plot(:,idx));
xlabel('Run #');
ylabel('Input');
title('Apical Weights');
set(gca,'fontsize',24);

% print(gcf,'-painters',fullfile(fpath,'SourceLoadingAllRunsUnsorted_Bhigh_Alow'),'-djpeg');

% subplot(1,5,3); hold on;
% plot(normalize(mean(sourceLoading{ns},2),'range',[0,1]),numInputs:-1:1,'color','k','linewidth',1.5);
% plot(normalize(mean(b2plot(:,idx),2),'range',[0,1]),numInputs:-1:1,'color','r','linewidth',1.5);
% xlabel({'Loading &';'Mean Basal Weight'});
% ylabel('Input');
% title('Basal Weights');
% set(gca,'fontsize',24);

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
ns = 1;
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
AD = 3; 
NS = 1;
BF = 1;
figure(1); clf; 
subplot(2,2,1); imagesc(mean(basalPTraj(:,:,AD,NS,BF,:),6));
subplot(2,2,2); imagesc(mean(apicalPTraj(:,:,AD,NS,BF,:),6));

smWidth = 10;
subplot(2,2,3); hold on;
for in = 1:numInputs
    mnplot = mean(filter(ones(1,smWidth),1,basalPTraj(in,:,AD,NS,BF,:),[],2),6);
    sdplot = std(filter(ones(1,smWidth),1,basalPTraj(in,:,AD,NS,BF,:),[],2),1,6)/2/sqrt(numRuns);
    patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(in,:),'EdgeColor','none','FaceAlpha',0.1);
    plot(1:lengthTraj/dsFactor,mnplot,'color',cc(in,:),'linewidth',1.5);
end
subplot(2,2,4); hold on;
for in = 1:numInputs
    mnplot = mean(filter(ones(1,smWidth),1,apicalPTraj(in,:,AD,NS,BF,:),[],2),6);
    sdplot = std(filter(ones(1,smWidth),1,apicalPTraj(in,:,AD,NS,BF,:),[],2),1,6)/2/sqrt(numRuns);
    patch([1:lengthTraj/dsFactor,fliplr(1:lengthTraj/dsFactor)], [mnplot+sdplot fliplr(mnplot-sdplot)],'k','FaceColor',cc(in,:),'EdgeColor','none','FaceAlpha',0.1);
    plot(1:lengthTraj/dsFactor,mnplot,'color',cc(in,:),'linewidth',1.5);
end










