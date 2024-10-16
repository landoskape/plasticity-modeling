


%% -- collect data --

dpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1_data';
savePath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';
nameScheme = @(ad,sd,sc,rn) sprintf('iafOriPos_AD%d_SD%d_OC%d_Run%d.mat',ad,sd,sc,rn);

T = 3600*1000;

NO = 3; % num orientation 
NP = 7; % num position

NAD = 4; % num active dendrite depression
NSD = 4; % num silent dendrite depression
NOC = 4; % num oriented bar correlation
numRuns = 40;
existData = zeros(NAD,NSD,NOC,numRuns);
weight = zeros(25,40,NAD,NSD,NOC,numRuns);
connection = zeros(25,40,NAD,NSD,NOC,numRuns);
aweight = zeros(NO*NP, T/1000, NAD,NSD, NOC, numRuns);
sweight = zeros(NO*NP, T/1000, NAD,NSD, NOC, numRuns);
%y = zeros(NP,T,NSD,NOC,numRuns);
spkTimes = cell(NAD,NSD,NOC,numRuns);
msg = '';
warning('off','all');
for nad = 1:NAD
    for nsd = 1:NSD
        for noc = 1:NOC
            for rn = 1:numRuns
                fprintf(repmat('\b',1,length(msg)));
                msg = sprintf('%d/%d, %d/%d, %d/%d, %d/%d, %d/%d, working...\n',nad,NAD,nsd,NSD,noc,NOC,rn,numRuns);
                fprintf(msg);
                fName = fullfile(dpath,nameScheme(nad,nsd,noc,rn));
                if exist(fName,'file')
                    existData(nad,nsd,noc,rn) = 1;
                    data = load(fName);
                    weight(:,:,nad,nsd,noc,rn) = data.iaf.ampaWeights;
                    connection(:,:,nad,nsd,noc,rn) = data.iaf.inputConnection;
                    spkTimes{nad,nsd,noc,rn} = data.spkTimes;
                    aweight(:,:,nad,nsd,noc,rn) = data.aweights(:,end-3599:end);
                    sweight(:,:,nad,nsd,noc,rn) = data.sweights(:,1:size(sweight,2));
                    %y(:,:,nsd,noc,rn) = data.y;
                end
            end
        end
    end
end
iaf = data.iaf;
y = data.y;
warning('on','all');
fprintf(repmat('\b',1,length(msg)));
msg = sprintf('%d/%d, %d/%d, %d/%d, %d/%d, finished!\n',nad,NSD,nsd,NSD,noc,NOC,rn,numRuns);
fprintf(msg);

% save(fullfile(savePath,'allResults.mat'),'weight','spkTimes','aweight','sweight','iaf','y','-v7.3');




%% -- analysis --

NO = 3; % num orientation 
NP = 7; % num position
NI = NO*NP;

NAD = 2; 
NSD = 3;
NOC = 5;
numRuns = 50;

activeWeight = reshape(squeeze(aweight(:,end,:,:,:)),NO,NP,NAD,NSD,NOC,numRuns);
silentWeight = reshape(squeeze(sweight(:,end,:,:,:)),NO,NP,NAD,NSD,NOC,numRuns);

[~,aIdxPref] = max(sum(activeWeight,2),[],1);
[~,sIdxPref] = max(sum(silentWeight,2),[],1);
aIdxPref = squeeze(aIdxPref);
sIdxPref = squeeze(sIdxPref);

sActiveWeight = zeros(size(activeWeight));
sSilentWeight = zeros(size(silentWeight));
for nad = 1:NAD
    for nsd = 1:NSD
        for noc = 1:NOC
            for rn = 1:numRuns
                [~,cIdx] = sort(sum(activeWeight(:,:,nad,nsd,noc,rn),2),'descend');
                sActiveWeight(:,:,nad,nsd,noc,rn) = activeWeight(cIdx,:,nad,nsd,noc,rn);
                sSilentWeight(:,:,nad,nsd,noc,rn) = silentWeight(cIdx,:,nad,nsd,noc,rn);
            end
        end
    end
end

% -- example figure -> show grid of preferences etc...
%figure(1); clf; 
for nsd = 1:NSD
    for noc = 1:NOC
        % we'll see if I remember this one...
    end
end
        

%% -- example of tuning when correlation is 0 --

fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';

dt = 0.001;
T = 3600;

figure(1); clf;

imagesc(dt:dt:T, 1:18, sqrt(aweight(:,:,2,1,1,1)))
colormap('hot');
set(gca,'xtick',0:1200:T);
set(gca,'xticklabel',{'0min','20min','40min','1hr'});
set(gca,'ytick',[]);
%set(gca,'ytick',2:3:17);
%set(gca,'yticklabel',1:6);
xlabel('Time in Simulation');
ylabel('Input (Position/Orientation)');
set(gca,'fontsize',24);
% print(gcf,'-painters',fullfile(fpath,'tunedTo1'),'-djpeg');


%% -- example of tuning when correlation is higher than 0 --

fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';

corrVal = linspace(0,1,NOC);

useActive = 1; 
useSilent = 3;

idxPlot = [2, 4];
numPlot = length(idxPlot);

pkScaleActive = max(mean(sActiveWeight(1,:,useActive,useSilent,idxPlot,:),6),[],'all');
pkScaleSilent = max(mean(sActiveWeight(1,:,useActive,useSilent,idxPlot,:),6),[],'all')/8;

silScale = pkScaleSilent; 
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.53 0.46 0.47 0.33]);

for np = 1:numPlot
    subplot(1,numPlot,np); hold on;
    plot(1:NP, mean(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/pkScaleActive,6), 'color','k','linewidth',1.5);
    shadedErrorBar(1:NP, mean(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/pkScaleActive,6), std(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/pkScaleActive,1,6)/sqrt(numRuns),...
        {'color','k','linewidth',1.5},1);
    
    xlim([0.5 NP+0.5]);
    ylim([0 1.2]);
    set(gca,'xtick',1:7);
    set(gca,'ytick',0:0.5:1);
    xlabel('Position');
    ylabel('Relative Weight');
    title(sprintf('P(Edge): %.2f',corrVal(idxPlot(np))));
    set(gca,'fontsize',24);
end

subplot(1,numPlot,1);
legend('Active','location','northwest');

print(gcf,'-painters',fullfile(fpath,'justActive_severalCorrelations'),'-djpeg');



%%

fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';

useActive = 2; 
useSilent = 2;

pkScaleActive = max(mean(sActiveWeight(1,:,useActive,useSilent,idxPlot,:),6),[],'all');
pkScaleSilent = max(mean(sActiveWeight(1,:,useActive,useSilent,idxPlot,:),6),[],'all')/8;

silScale = pkScaleSilent; 
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.53 0.46 0.47 0.33]);

for np = 1:numPlot
    subplot(1,numPlot,np); hold on;
    plot(1:NP, mean(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/pkScaleActive,6), 'color','k','linewidth',1.5);
    plot(1:NP, mean(sSilentWeight(1,:,useActive,useSilent,idxPlot(np),:)/silScale,6), 'color','b','linewidth',1.5);
    shadedErrorBar(1:NP, mean(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/pkScaleActive,6), std(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/pkScaleActive,1,6)/sqrt(numRuns),...
        {'color','k','linewidth',1.5},1);
    shadedErrorBar(1:NP, mean(sSilentWeight(1,:,useActive,useSilent,idxPlot(np),:)/silScale,6), std(sActiveWeight(1,:,useActive,useSilent,idxPlot(np),:)/silScale,1,6)/numRuns,...
        {'color','b','linewidth',1.5},1);
    
    xlim([0.5 NP+0.5]);
    ylim([0 1.2]);
    set(gca,'xtick',1:7);
    set(gca,'ytick',0:0.5:1);
    xlabel('Position');
    ylabel('Relative Weight');
    title(sprintf('P(Edge): %.2f',corrVal(idxPlot(np))));
    set(gca,'fontsize',24);
end

subplot(1,numPlot,1);
legend('Active','Silent','location','northwest');

print(gcf,'-painters',fullfile(fpath,sprintf('withSilent_severalCorrelations_AD%d_SD%d',useActive,useSilent)),'-djpeg');




%% -- figure of shadedErrorBar for all parameters - both active & silent
cmap = 'krb';
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.19 0.38 0.63 0.6]);

for nsd = 1:NSD
    for noc = 1:NOC
        subplot(NSD,NOC,NOC*(nsd-1)+noc); hold on;
        shadedErrorBar(1:NP, mean(sActiveWeight(1,:,nsd,noc,:),5), std(sActiveWeight(1,:,nsd,noc,:),1,5)/sqrt(numRuns),...
            {'color','k','linewidth',1.5},1);
        shadedErrorBar(1:NP, mean(sSilentWeight(1,:,nsd,noc,:),5), std(sSilentWeight(1,:,nsd,noc,:),1,5)/sqrt(numRuns),...
            {'color','b','linewidth',1.5},1);
    end
end








%% -- create schematic of stimulus --

fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';

rng(6); 

T = 11;
NP = 7; 
NO = 3;

sBar = 4; 
lWidth = 5; 

cVal = 0.5;
orientation = randi(NO,T,NP);
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.34 0.18 0.26 0.8]);

%line(0.5*[1,1], [T,1], 'color','k','linewidth',1.5);

for t = 1:T
    if rem(t,2)==1
        patch([0.5 NP+1.5 NP+1.5 0.5],[2*t-1 2*t-1 2*t+1 2*t+1],0.7*[1,1,1],'EdgeColor','none');
    end
    if rand()<cVal
        idxStart = randi(NP-sBar+1,1);
        orientation(t,idxStart:idxStart+sBar-1)=1;
    end
    for np = 1:NP
        if orientation(t,np)==1
            line([np, np+1],2*[t,t],'color','k','linewidth',lWidth);
        elseif orientation(t,np)==2
            line([np+0.25,np+0.75],[2*t-0.5, 2*t+0.5],'color','k','linewidth',lWidth);
        else
            line(np+0.5*[1,1],[2*t-0.5, 2*t+0.5],'color','k','linewidth',lWidth);
        end
    end
end

xlim([0.5 NP+1.5]);
ylim([0 T*2+1]);

set(gca,'visible','off');


rng('shuffle');

% print(gcf,'-painters',fullfile(fpath,'stimulusLines'),'-djpeg');







%% -- covariance of input channels --

NO = 3; % num orientation 
NP = 7; % num position
NI = NO*NP;

T = size(y,2); 

inputChannels = zeros(T/2,NI);
for t = 1:T/2
    inputChannels(t,y(:,T/2+t)) = 1;
end
idxOrder = [1:NO:NI, 2:NO:NI, 3:NO:NI];
inputChannels = inputChannels(:,idxOrder);

imagesc(cov(inputChannels))



%% --





%% -- plot schematic ratio with silent / active together --


fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';


NR = 3;
cmap = [zeros(NR,1), linspace(0,1,NR)', zeros(NR,1)];

figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.06 0.5 0.92 0.34]);

subplot(1,4,1); hold on;

allRatios = [1.1, 1.05, 1.025];
for ar = 1:NR
    plot(0.85, allRatios(ar), 'marker','o','color',cmap(ar,:),'markerfacecolor',cmap(ar,:),'markersize',12);
end
text(1.075, 1.1, '- 110%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1.075, 1.05, '- 105%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1.075, 1.0, '- 100%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
line([0.95 1.05]-0.15,[1,1],'color','k','linewidth',1.0);
text(0.975, 1.165, 'Dep/Pot Ratio', 'fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
xlim([0.7 1.2]);
ylim([0.95 1.2]);

set(gca,'visible','off')


subplot(1,4,2); hold on;
useActive = 1; 
useSilent = 1;
pkScale = 1;%max(mean(sSilentWeight(1,:,useActive,useSilent,idxPlot,:),6),[],'all');
for nr = 1:NR
    plot(1:NP, mean(sSilentWeight(1,:,useActive,nr,idxPlot(np),:)/pkScale,6), 'color',cmap(nr,:),'linewidth',1.5);
    shadedErrorBar(1:NP, mean(sSilentWeight(1,:,useActive,nr,idxPlot(np),:)/pkScale,6), std(sActiveWeight(1,:,useActive,nr,idxPlot(np),:)/pkScale,1,6)/numRuns,...
        {'color',cmap(nr,:),'linewidth',1.5},1);
end
xlim([0.5 NP+0.5]);
ylim([0 1.1]);
set(gca,'xtick',1:7);
set(gca,'ytick',0:0.5:1);
xlabel('Position');
ylabel('Relative Weight');
title(sprintf('P(Edge): %.2f',corrVal(idxPlot(np))));
set(gca,'fontsize',24);


subplot(1,4,3); hold on;
T = 100000; 
inputActive = randi(NO, T, NP) + NO*(0:(NP-1));
idxOffset = NO*(0:(NP-1));
oriCorr = 0.75;
for t = 1:T
    if rand()<oriCorr
        % Make a row of three
        idxStart = randi(NP-3);
        inputActive(t,idxStart:idxStart+3) = 1 + idxOffset(idxStart:idxStart+3);
    end
end
iOn = zeros(T,NO*NP);
for t = 1:T
    iOn(t,inputActive(t,:)) = 1;
end
idxOrder = [1:3:NO*NP, 2:3:NO*NP, 3:3:NO*NP];
cv = cov(iOn(:,idxOrder));
cv = cv(1:7, 1:7); 
vv = diag(cv);
vvMat = sqrt(vv .* vv');
cv = cv./vvMat;
%cv = cv - triu(cv);


cvPlot = cv - diag(diag(cv));


imagesc(cvPlot); 
colormap(flipud(redblue(max(abs(cvPlot(:)))*[-1,1],'k')))
caxis(max(abs(cvPlot(:)))*[-1,1])
cb = colorbar;
%ylabel(cb,'r_{i,j}');
set(gca,'ydir','reverse');
xlim([0.5 7.5]);
ylim([0.5 7.5]);
set(gca,'xtick',1:7);
set(gca,'ytick',1:7);
xlabel('Position');
ylabel('Position');
title('correlation');
set(gca,'fontsize',24);

I = eye(7);
idxOffDiag = find(I(:)==0);
subplot(1,4,4); hold on; 
complexity = zeros(1,NR);
for nr = 1:NR
    cmean = mean(sSilentWeight(1,:,useActive,nr,idxPlot(np),:)/pkScale,6);
    cWeightCorr = cmean' .* cmean .* cv;
    %complexity(nr) = - log ( mean( cWeightCorr .* cmean,'all'));
    
    centerAverage = sum(cWeightCorr,'all') / sum(cmean' .* cmean,'all');
    complexity(nr) = - log (centerAverage);
    
%     complexity(nr) = -log(mean(cmean.*sum(cv,1)));
end
for nr = 1:NR
    bar(nr,complexity(nr),'FaceColor',cmap(nr,:))
end
set(gca,'xtick',1:3);
set(gca,'xticklabel',allRatios);
xlabel('D/P Ratio');
ylabel('Complexity');
set(gca,'fontsize',24);


% print(gcf,'-painters',fullfile(fpath,'summaryData_RatiosComplexity'),'-djpeg');

 
%%
figure(2); clf; 
plot(1:7, sum(cv),'color','k','linewidth',1.5)
set(gca,'xtick',1:7);
set(gca,'ytick',0:3);
xlim([0.5 7.5]);
ylim([0 2.6])
xlabel('Position');
ylabel('Marginal Correlation');
set(gca,'fontsize',24);
    
print(gcf,'-painters',fullfile(fpath,'marginalCorr'),'-djpeg');
%complexity = - log * sum(w * w * r);


%%
NO = 3; 
NP = 7; 


T = 10000; 
inputActive = randi(NO, T, NP) + NO*(0:(NP-1));
idxOffset = NO*(0:(NP-1));
oriCorr = 0.5;

for t = 1:T
    if rand()<oriCorr
        % Make a row of three
        idxStart = randi(NP-3);
        inputActive(t,idxStart:idxStart+3) = 1 + idxOffset(idxStart:idxStart+3);
    end
end


iOn = zeros(T,NO*NP);
for t = 1:T
    iOn(t,inputActive(t,:)) = 1;
end

idxOrder = [1:3:NO*NP, 2:3:NO*NP, 3:3:NO*NP];

figure(1); clf;
subplot(1,2,1); 
imagesc(iOn);

subplot(1,2,2);
imagesc(cov(iOn(:,idxOrder)))

   





