
%{
Using the integration metrics to analyze these data. 
1. Construct correlation matrix (first empirically, compare with
theoretical, choose which better to use)
2. Measure integration using average weights to the possible inputs
3. Measure integration using full weight matrix
4. (My theoretical intuition is that 2&3 should be the same...
5. Make it pretty!!!

-- work so far --
- I measured integration with just apical and also the full cell (full cell
includes basal)
- I constructed correlation mat (but didn't incorporate the true timing...
just assumed that each stim presentation lasted the same amount of time).
Since we're going for so long, this should be okay???
- Next: compare the correlation mat with true and time compressed^...
- Next: see how conditioning the weights on passing the conductance
threshold affects the results. 
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
baseRate = [sampleData.iaf.baseBasal, sampleData.iaf.baseApical];
driveRate = baseRate + [sampleData.iaf.driveBasal, sampleData.iaf.driveApical];
if baseRate(1)~=baseRate(2) || driveRate(1)~=driveRate(2)
    fprintf(2,'Note that basal and apical rates are not identical...\n');
end
T = sampleData.iaf.T;
lengthTraj = T/1000;
acThresh = sampleData.iaf.apicalCondThresh;
bcThresh = sampleData.iaf.basalCondThresh;
dangles = mean(diff(sampleData.iaf.angles)); 
stimHistBins = sampleData.iaf.angles(1)-dangles/2 : dangles : sampleData.iaf.angles(end) + dangles/2;

% load final weights
keepIdx = lengthTraj-100:lengthTraj;
scaleApicalWeight = 3e-9;
scaleBasalWeight = 35e-9;
dsFactor = 10;
apicalWeights = zeros(numAngles, numPosition, numAD, numEP, numRuns);
basalWeights = zeros(numAngles, numAD, numEP, numRuns);
basalTrajectory = zeros(numAngles, lengthTraj/dsFactor, numAD,numEP,numRuns);
apicalTrajectory = zeros(numAngles, lengthTraj/dsFactor, numPosition, numAD,numEP,numRuns);
cmat = zeros(numAngles*numPosition,numAngles*numPosition,numAD,numEP,numRuns);
smat = zeros(numAngles*numPosition,numAngles*numPosition,numAD,numEP,numRuns);
intCorr = zeros(numAD,numEP,numRuns);
intDot = zeros(numAD,numEP,numRuns);
intCorrF = zeros(numAD,numEP,numRuns);
intDotF = zeros(numAD,numEP,numRuns);
intCorrCond = zeros(numAD,numEP,numRuns);
intDotCond = zeros(numAD,numEP,numRuns);
intCorrFCond = zeros(numAD,numEP,numRuns);
intDotFCond = zeros(numAD,numEP,numRuns);
msg = '';
for ridx = 1:numRuns
    for ad = 1:numAD
        for ep = 1:numEP
            fprintf(repmat('\b',1,length(msg)));
            msg = sprintf('Run %d/%d, AD %d/%d, EP %d/%d...\n',ridx,numRuns,ad,numAD,ep,numEP);
            fprintf(msg);
            cdata = load(fullfile(dpath,nameConvention(ridx,ad,ep)),'apicalWeightTrajectory','basalWeightTrajectory','y','ychange','iaf');
            apicalWeights(:,:,ad,ep,ridx) = mean(cdata.apicalWeightTrajectory(:,keepIdx,:),2);
            basalWeights(:,ad,ep,ridx) = mean(cdata.basalWeightTrajectory(:,keepIdx),2);
            apicalTrajectory(:,:,:,ad,ep,ridx) = permute(mean(reshape(permute(cdata.apicalWeightTrajectory,[4,2,1,3]),dsFactor,lengthTraj/dsFactor,numAngles,numPosition),1),[3,2,4,1]);
            basalTrajectory(:,:,ad,ep,ridx) = permute(mean(reshape(permute(cdata.basalWeightTrajectory,[3,2,1]),dsFactor,lengthTraj/dsFactor,numAngles),1),[3,2,1]);
            
            yidx = round(cdata.y/pi*4) + 4*(0:numPosition-1)';
            amat = baseRate(1)*ones(numAngles*numPosition, size(yidx,2));
            idxManual = yidx + size(amat,1)*(0:size(yidx,2)-1);
            amat(idxManual) = driveRate(1);
            anorm = sqrt(sum(amat.^2,2));
            cmat(:,:,ad,ep,ridx) = corr(amat');
            smat(:,:,ad,ep,ridx) = (amat * amat') ./ (anorm * anorm');
            
            % Compute integration for apical weights only
            aWeightVector = reshape(apicalWeights(:,:,ad,ep,ridx),[],1);
            corrSim = (aWeightVector' * cmat(:,:,ad,ep,ridx) * aWeightVector) ./ sum(aWeightVector)^2;
            dotSim = (aWeightVector' * smat(:,:,ad,ep,ridx) * aWeightVector) ./ sum(aWeightVector)^2;
            intCorr(ad,ep,ridx) = -log2(corrSim);
            intDot(ad,ep,ridx) = -log2(dotSim);
            
            % Compute integration for all weights
            bWeightVector = cat(1, zeros(16,1), basalWeights(:,ad,ep,ridx),zeros(16,1));
            sWeightVector = aWeightVector + bWeightVector;
            corrSimF = (sWeightVector' * cmat(:,:,ad,ep,ridx) * sWeightVector) ./ sum(sWeightVector)^2;
            dotSimF = (sWeightVector' * smat(:,:,ad,ep,ridx) * sWeightVector) ./ sum(sWeightVector)^2;
            intCorrF(ad,ep,ridx) = -log2(corrSimF);
            intDotF(ad,ep,ridx) = -log2(dotSimF);
            
            % -- now do same integration computation -- but for conditioned
            % weights on passing conductance threshold
            [~,~,shBasalBin] = histcounts(cdata.iaf.basalTuneCenter,stimHistBins);
            [~,~,shApicBin] = histcounts(cdata.iaf.apicalTuneCenter,stimHistBins);
            for shc = 1:length(stimHistBins)-1
                bIdxUse = (shBasalBin==shc) & (cdata.iaf.basalWeight >= cdata.iaf.basalCondThresh);
                bWeightVector(shc+16,1) = sum(cdata.iaf.basalWeight(bIdxUse),'all');
                aIdxUse = (shApicBin==shc) & (cdata.iaf.apicalWeight >= cdata.iaf.apicalCondThresh);
                for ix = 1:numPosition
                    idx = numAngles*(ix-1);
                    aWeightVector(idx+shc) = sum(cdata.iaf.apicalWeight(aIdxUse & cdata.iaf.apicalTuneIdx==ix),'all');
                end
            end
            corrSimCond = (aWeightVector' * cmat(:,:,ad,ep,ridx) * aWeightVector) ./ sum(aWeightVector)^2;
            dotSimCond = (aWeightVector' * smat(:,:,ad,ep,ridx) * aWeightVector) ./ sum(aWeightVector)^2;
            intCorrCond(ad,ep,ridx) = -log2(corrSimCond);
            intDotCond(ad,ep,ridx) = -log2(dotSimCond);
            
            sWeightVector = aWeightVector + bWeightVector;
            corrSimFCond = (sWeightVector' * cmat(:,:,ad,ep,ridx) * sWeightVector) ./ sum(sWeightVector)^2;
            dotSimFCond = (sWeightVector' * smat(:,:,ad,ep,ridx) * sWeightVector) ./ sum(sWeightVector)^2;
            intCorrFCond(ad,ep,ridx) = -log2(corrSimFCond);
            intDotFCond(ad,ep,ridx) = -log2(dotSimFCond);
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


%% -- plot summary data for integration --

meanIntCorr = mean(intCorr,3);
meanIntDot = mean(intDot,3);
meanIntCorrF = mean(intCorrF,3);
meanIntDotF = mean(intDotF,3);
meanIntCorr = mean(intCorrCond,3);
meanIntDot = mean(intDotCond,3);
meanIntCorrF = mean(intCorrFCond,3);
meanIntDotF = mean(intDotFCond,3);

epLabel = 0.25:0.25:1;
adLabel = [110,105,102.5,100];

figure(1); clf; 
set(gcf,'Position',[440   215   825   583]);

subplot(2,2,1); 
imagesc(1:numAD,1:numEP,meanIntCorrF');
caxis([0 max(meanIntCorrF,[],'all')]);
colormap('pink');
colorbar;
set(gca,'xtick',1:4, 'xticklabel', cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));
set(gca,'ytick',1:4, 'yticklabel', cellfun(@(c) sprintf('%s',num2str(c)), num2cell(epLabel), 'uni', 0));
xlabel('D/P Ratio');
ylabel('P(edge)');
title('int. (corr) - full cell');
set(gca,'fontsize',16);

subplot(2,2,3); 
imagesc(1:numAD,1:numEP,meanIntCorr');
caxis([0 max(meanIntCorr,[],'all')]);
colormap('pink');
colorbar;
set(gca,'xtick',1:4, 'xticklabel', cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));
set(gca,'ytick',1:4, 'yticklabel', cellfun(@(c) sprintf('%s',num2str(c)), num2cell(epLabel), 'uni', 0));
xlabel('D/P Ratio');
ylabel('P(edge)');
title('int. (corr) - apical only');
set(gca,'fontsize',16);

subplot(2,2,2);
imagesc(1:numAD,1:numEP,meanIntDotF');
caxis([0 max(meanIntDotF,[],'all')]);
colormap('pink');
colorbar;
set(gca,'xtick',1:4, 'xticklabel', cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));
set(gca,'ytick',1:4, 'yticklabel', cellfun(@(c) sprintf('%s',num2str(c)), num2cell(epLabel), 'uni', 0));
xlabel('D/P Ratio');
ylabel('P(edge)');
title('int. (dot)  - full cell');
set(gca,'fontsize',16);

subplot(2,2,4);
imagesc(1:numAD,1:numEP,meanIntDot');
caxis([0 max(meanIntDot,[],'all')]);
colormap('pink');
colorbar;
set(gca,'xtick',1:4, 'xticklabel', cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));
set(gca,'ytick',1:4, 'yticklabel', cellfun(@(c) sprintf('%s',num2str(c)), num2cell(epLabel), 'uni', 0));
xlabel('D/P Ratio');
ylabel('P(edge)');
title('int. (dot)  - apical only');
set(gca,'fontsize',16);

%%

figure(3); clf; 
imagesc(1:numAD,1:numEP,meanIntDotF');
caxis([0 max(meanIntDotF,[],'all')]);
colormap('pink');
cb = colorbar;
set(gca,'xtick',1:4, 'xticklabel', cellfun(@(c) sprintf('%s%%',num2str(c)), num2cell(adLabel), 'uni', 0));
set(gca,'ytick',1:4, 'yticklabel', cellfun(@(c) sprintf('%s',num2str(c)), num2cell(epLabel), 'uni', 0));
xlabel('D/P Ratio');
ylabel('P(edge)');
ylabel(cb,'Integration');
set(cb,'ticks',0:0.01:0.05);
set(gca,'fontsize',16);
% print(gcf,fullfile('/Users/landauland/Dropbox/SabatiniLab/FENS-2022/Figures','integrationFigure'),'-depsc');


%% -- Summary Data -- (#ATL 220630 keeping this in case I want to keep using similar functions)
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
% print(gcf,'-painters',fullfile(fpath,'EdgeProbabilityMap'),'-depsc');















