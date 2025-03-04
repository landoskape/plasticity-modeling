


%% -- collect data --

dpath = '/Users/landauland/Documents/Research/o2/songAbbott/cModel1_data';
savePath = '/Users/landauland/Documents/Research/o2/songAbbott/cModel1';
nameScheme = @(sd,sc,rn) sprintf('iafCM_SD%d_SC%d_Run%d.mat',sd,sc,rn);

T = 1200*1000;

NSD = 3;
NSC = 10;
numRuns = 100;
existData = zeros(NSD,NSC,numRuns);
weight = zeros(25,40,NSD,NSC,numRuns);
latent = cell(NSD,NSC,numRuns);
spkTimes = cell(NSD,NSC,numRuns);
msg = '';
warning('off','all');
for nsd = 1:NSD
    for nsc = 1:NSC
        for rn = 1:numRuns
            fprintf(repmat('\b',1,length(msg)));
            msg = sprintf('%d/%d, %d/%d, %d/%d, %d/%d, working...\n',nsd,NSD,nsc,NSC,rn,numRuns);
            fprintf(msg);
            fName = fullfile(dpath,nameScheme(nsd,nsc,rn));
            if exist(fName,'file')
                existData(nsd,nsc,rn) = 1;
                data = load(fName);
                weight(:,:,nsd,nsc,rn) = data.iaf.ampaWeights;
                latent{nsd,nsc,rn} = data.y(data.spkTimes);
                spkTimes{nsd,nsc,rn} = data.spkTimes;
            end
        end
    end
end
iaf = data.iaf;
warning('on','all');
fprintf(repmat('\b',1,length(msg)));
msg = sprintf('%d/%d, %d/%d, %d/%d, %d/%d, finished!\n',nsd,NSD,nsc,NSC,rn,numRuns);
fprintf(msg);

% save(fullfile(savePath,'allResults.mat'),'weight','latent','spkTimes','iaf');


%% -- analysis --

NSD = 3;
NSC = 10;
numRuns = 100;

numSynapses = 25;
numActive = 39;
numSilent = 1;
sDepArray = [1.1, 1.075, 1.05];
sCorArray = linspace(0.3, 0, NSC);
aCorArray = linspace(0.3, 0, numActive);
sCorIndex = linspace(1,2,NSC);
aCorIndex = linspace(1,2,numActive);
%dendriteCorrPrm = [linspace(0.3,0,numActive),silentCorrelation*ones(1,numSilent)];

avgWeight = squeeze(mean(mean(weight,1),5));
stdWeight = squeeze(std(mean(weight,1),1,5));

redMap = [linspace(0,1,NSC)',zeros(NSC,2)];
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.17 0.55 0.63 0.43]);
for nsd = 1:NSD
    subplot(1,NSD,nsd); hold on;
    concatenateActive = reshape(permute(mean(weight(:,1:numActive,nsd,:,:),1),[4,5,2,3,1]),NSC*numRuns,numActive);
    concatenateSilent = reshape(permute(mean(weight(:,numActive+1:end,nsd,:,:),1),[5,4,2,3,1]),numRuns,NSC);
    plot(fliplr(aCorIndex),mean(concatenateActive,1)/iaf.maxWeight,'color','k','linewidth',1.5);
    plot(fliplr(sCorIndex),mean(concatenateSilent,1)/iaf.maxWeight,'color','b','linewidth',1.5);
    shadedErrorBar(fliplr(aCorIndex),...
        mean(concatenateActive,1)/iaf.maxWeight,std(concatenateActive,1,1)/iaf.maxWeight,...
        {'color','k','linewidth',1.5},1);
    shadedErrorBar(fliplr(sCorIndex),...
        mean(concatenateSilent,1)/iaf.maxWeight,std(concatenateSilent,1,1)/iaf.maxWeight,...
        {'color','b','linewidth',1.5},1);
    % Plot Each Silent Correlation
    %for nsc = 1:NSC
    %    plot(fliplr(aCorIndex), avgWeight(1:numActive,nsd,nsc)/iaf.maxWeight,'color',redMap(nsc,:));
    %end
%     plot(fliplr(sCorIndex), squeeze(mean(avgWeight(numActive+1:end,nsd,:),1))/iaf.maxWeight,'color','b','linewidth',1.5);
    set(gca,'xtick',sCorIndex(1:3:end));
    set(gca,'xticklabel',fliplr(sCorArray(1:3:end)));
    ylim([0, 1]);
    xlabel('Input Correlation');
    ylabel('Relative Weight');
    title({'Active Dep: 1.100'; sprintf('Silent Dep: %.3f', sDepArray(nsd))});
    if nsd==1,legend('Active Dendrites','Silent Dendrites','location','northwest');end
    set(gca,'fontsize',24);
    
end

savePath = '/Users/landauland/Documents/Research/o2/songAbbott/cModel1';
% print(gcf,'-painters',fullfile(savePath,'correlationTraces'),'-djpeg');





%% -- analysis - all in one panel --

NSD = 3;
NSC = 10;
numRuns = 100;

numSynapses = 25;
numActive = 39;
numSilent = 1;
sDepArray = [1.1, 1.075, 1.05];
sCorArray = linspace(0.3, 0, NSC);
aCorArray = linspace(0.3, 0, numActive);
sCorIndex = linspace(1,2,NSC);
aCorIndex = linspace(1,2,numActive);
%dendriteCorrPrm = [linspace(0.3,0,numActive),silentCorrelation*ones(1,numSilent)];

avgWeight = squeeze(mean(mean(weight,1),5));
stdWeight = squeeze(std(mean(weight,1),1,5));

cmap = [zeros(NSD,2), linspace(0,1,NSD)'];
redMap = [linspace(0,1,NSC)',zeros(NSC,2)];



figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.45 0.37 0.53]);
hold on;

for nsd = 1:NSD
    concatenateSilent = reshape(permute(mean(weight(:,numActive+1:end,nsd,:,:),1),[5,4,2,3,1]),numRuns,NSC);
    plot(fliplr(sCorIndex),mean(concatenateSilent,1)/iaf.maxWeight,'color',cmap(nsd,:),'linewidth',1.5);
end

concatenateActive = reshape(permute(mean(weight(:,1:numActive,1,:,:),1),[4,5,2,3,1]),NSC*numRuns,numActive);
shadedErrorBar(fliplr(aCorIndex),...
        mean(concatenateActive,1)/iaf.maxWeight,std(concatenateActive,1,1)/iaf.maxWeight,...
        {'color','k','linewidth',1.5},1);
plot(fliplr(aCorIndex),mean(concatenateActive,1)/iaf.maxWeight,'color','k','linewidth',1.5);
for nsd = 1:NSD
    concatenateSilent = reshape(permute(mean(weight(:,numActive+1:end,nsd,:,:),1),[5,4,2,3,1]),numRuns,NSC);
    plot(fliplr(sCorIndex),mean(concatenateSilent,1)/iaf.maxWeight,'color',cmap(nsd,:),'linewidth',1.5);
    shadedErrorBar(fliplr(sCorIndex),...
        mean(concatenateSilent,1)/iaf.maxWeight,std(concatenateSilent,1,1)/iaf.maxWeight,...
        {'color',cmap(nsd,:),'linewidth',1.5},1);
    set(gca,'xtick',sCorIndex(1:3:end));
    set(gca,'xticklabel',fliplr(sCorArray(1:3:end)));
    ylim([0, 1]);
    xlabel('Input Correlation');
    ylabel('Relative Weight');
    title('Active Dep/Pot: 1.1');
    set(gca,'fontsize',24);
end
set(gca,'ytick',0:0.2:1);
legend('D/P: 1.1','D/P: 1.075','D/P: 1.05','location','northwest');

savePath = '/Users/landauland/Documents/Research/o2/songAbbott/cModel1';
% print(gcf,'-painters',fullfile(savePath,'correlationTraces_onePanel'),'-djpeg');





%% -- analysis - all in one panel --

NSD = 3;
NSC = 10;
numRuns = 100;

numSynapses = 25;
numActive = 39;
numSilent = 1;
sDepArray = [1.1, 1.075, 1.05];
sCorArray = linspace(0.3, 0, NSC);
aCorArray = linspace(0.3, 0, numActive);
sCorIndex = linspace(1,2,NSC);
aCorIndex = linspace(1,2,numActive);
%dendriteCorrPrm = [linspace(0.3,0,numActive),silentCorrelation*ones(1,numSilent)];

avgWeight = squeeze(mean(mean(weight,1),5));
stdWeight = squeeze(std(mean(weight,1),1,5));

cmap = [zeros(NSD,2), linspace(0,1,NSD)'];
redMap = [linspace(0,1,NSC)',zeros(NSC,2)];



figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.45 0.37 0.53]);
hold on;

plot(fliplr(aCorIndex),mean(concatenateActive,1)/iaf.maxWeight,'color','k','linewidth',1.5);
concatenateActive = reshape(permute(mean(weight(:,1:numActive,1,:,:),1),[4,5,2,3,1]),NSC*numRuns,numActive);
shadedErrorBar(fliplr(aCorIndex),...
        mean(concatenateActive,1)/iaf.maxWeight,std(concatenateActive,1,1)/iaf.maxWeight,...
        {'color','k','linewidth',1.5},1);
plot(fliplr(aCorIndex),mean(concatenateActive,1)/iaf.maxWeight,'color','k','linewidth',1.5);
set(gca,'xtick',sCorIndex(1:3:end));
set(gca,'xticklabel',fliplr(sCorArray(1:3:end)));
ylim([0, 1]);
xlabel('Input Correlation');
ylabel('Relative Weight');
title('Active Dep/Pot: 1.1');
set(gca,'fontsize',24);
set(gca,'ytick',0:0.2:1);
legend('D/P: 1.1','location','northwest');

savePath = '/Users/landauland/Documents/Research/o2/songAbbott/cModel1';
% print(gcf,'-painters',fullfile(savePath,'correlationTraces_JustActiveonePanel'),'-djpeg');

