
fpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/thesisFigures';

xLim = [-37 7];
yLim = [-53 53];
plotCircleAngle = 0:0.01:2*pi;

radiusSoma = 5;

% Basal Parameters
% numBasal = 1;
% lengthBasal = 8;
% radiusBasal = 0.7;
% lwidthBasal = 2;
% lwidBasCir = 1.2;
% fracCircleBasal = 0.7;
% angleBasal = linspace(3*pi/2-pi*fracCircleBasal,3*pi/2+pi*fracCircleBasal,numBasal);
% % Basal Tuning 
% drad = 2*pi/numBasal;
% basalTuning = -pi+drad : drad : pi;
% % Basal Positions
% xBasal = (radiusSoma+radiusBasal)*cos(angleBasal);
% yBasal = (radiusSoma+radiusBasal)*sin(angleBasal);
% xBasCir = radiusBasal*cos(plotCircleAngle);
% yBasCir = radiusBasal*sin(plotCircleAngle);
% % ColorMap Basal Inputs
% cmapBasal = hsv(numBasal);

% Basal Parameters
% -- just invert apical stuff --

% Apical Parameters
numApical = 1;
apicalWidth = 5;
apicalLength = 40;
apicalSynapses = 4;
apicalOffset = 0.12;
lwidthApical = 2;
apicalPosition = linspace(apicalOffset*apicalLength,apicalLength,apicalSynapses);
apSynRadius = 1;
leftRightOffset = 0.6;
xApCir = apSynRadius*cos(plotCircleAngle);
yApCir = apSynRadius*sin(plotCircleAngle);

% Presynaptic Parameters
preWidth = 5;
preRadius = 2.5;
preOffset = 0.5;
preCenter = -15;
preSynWidth = 1.5;
preXCir = preRadius*cos(plotCircleAngle);
preYCir = preRadius*sin(plotCircleAngle);

% Source Parameters
numSources = 3;
sourceRadius = 2.2;
sourceLWidth = 5;
sourceEWidth = 0.1;
sourceLength = 4;
sourceCircFraction = 0.2;
angleSource = linspace(pi-pi*sourceCircFraction,pi+pi*sourceCircFraction,numSources);
xSourceCir = sourceRadius*cos(plotCircleAngle);
ySourceCir = sourceRadius*sin(plotCircleAngle);
cmapSource = {[0,0.5,1],'r',[0,0.6,0]};

figure(1);clf;
set(gcf,'units','normalized','outerposition',[0.39 0.1 0.27 0.88]);

% Plot Soma
patch(radiusSoma*cos(plotCircleAngle),radiusSoma*sin(plotCircleAngle),'k');

% Plot Basal Inputs
line([0,0],-[radiusSoma, radiusSoma+apicalLength],'color','k','linewidth',apicalWidth);
for asyn = 1:apicalSynapses
    xpos = -(apSynRadius+preOffset+preSynWidth);
    ypos = -(radiusSoma+apicalPosition(asyn));
    patch(xApCir,ypos+yApCir,'k','linewidth',lwidthApical);
    
    line(xpos+[0,preSynWidth],ypos+[0,preSynWidth],'color','k','linewidth',preWidth);
    line(xpos+[0,preSynWidth],ypos-[0,preSynWidth],'color','k','linewidth',preWidth);
    line([preCenter,xpos],ypos*[1,1],'color','k','linewidth',preWidth);
    patch(preCenter+preXCir,ypos+preYCir,'k','linewidth',lwidthApical);
    
    randStrength = rand(numSources,1);
    randStrength = randStrength/sum(randStrength);
    for source = 1:numSources
        xLine = cos(angleSource(source))*[sourceLength+preRadius+randStrength(source)*sourceRadius,preRadius+randStrength(source)*sourceRadius]; 
        yLine = sin(angleSource(source))*[sourceLength+preRadius+randStrength(source)*sourceRadius,preRadius+randStrength(source)*sourceRadius]; 
        line(preCenter+xLine,ypos+yLine,'color',cmapSource{source},'linewidth',randStrength(source)*sourceLWidth)
        xSource = (preRadius + randStrength(source)*sourceRadius)*cos(angleSource(source));
        ySource = (preRadius + randStrength(source)*sourceRadius)*sin(angleSource(source));
        patch(xSource+randStrength(source)*xSourceCir+preCenter, ypos+ySource+randStrength(source)*ySourceCir, cmapSource{source},'linewidth',sourceEWidth);
    end
end


% Plot Apical Inputs and Presynaptic Partners
line([0,0],[radiusSoma, radiusSoma+apicalLength],'color','k','linewidth',apicalWidth);
for asyn = 1:apicalSynapses
    xpos = -(apSynRadius+preOffset+preSynWidth);
    ypos = radiusSoma+apicalPosition(asyn);
    patch(xApCir,ypos+yApCir,'k','linewidth',lwidthApical);
    
    line(xpos+[0,preSynWidth],ypos+[0,preSynWidth],'color','k','linewidth',preWidth);
    line(xpos+[0,preSynWidth],ypos-[0,preSynWidth],'color','k','linewidth',preWidth);
    line([preCenter,xpos],ypos*[1,1],'color','k','linewidth',preWidth);
    patch(preCenter+preXCir,ypos+preYCir,'k','linewidth',lwidthApical);
    
    randStrength = rand(numSources,1);
    tempRand = 4;
    randStrength = exp(tempRand*randStrength)/sum(exp(tempRand*randStrength));%randStrength/sum(randStrength);
    for source = 1:numSources
        xLine = cos(angleSource(source))*[sourceLength+preRadius+randStrength(source)*sourceRadius,preRadius+randStrength(source)*sourceRadius]; 
        yLine = sin(angleSource(source))*[sourceLength+preRadius+randStrength(source)*sourceRadius,preRadius+randStrength(source)*sourceRadius]; 
        line(preCenter+xLine,ypos+yLine,'color',cmapSource{source},'linewidth',randStrength(source)*sourceLWidth)
        xSource = (preRadius + randStrength(source)*sourceRadius)*cos(angleSource(source));
        ySource = (preRadius + randStrength(source)*sourceRadius)*sin(angleSource(source));
        patch(xSource+randStrength(source)*xSourceCir+preCenter, ypos+ySource+randStrength(source)*ySourceCir, cmapSource{source},'linewidth',sourceEWidth);
    end
end
randStrength = [1,1,1];
for source = 1:numSources
    ypos = 0 - 2.5*(source-2);
    xLine = cos(pi)*[sourceLength+preRadius+randStrength(source)*sourceRadius,preRadius+randStrength(source)*sourceRadius]; 
    yLine = sin(pi)*[sourceLength+preRadius+randStrength(source)*sourceRadius,preRadius+randStrength(source)*sourceRadius]; 
    line(-6+preCenter+xLine,ypos+yLine,'color',cmapSource{source},'linewidth',randStrength(source)*sourceLWidth)
    xSource = (preRadius + randStrength(source)*sourceRadius)*cos(pi);
    ySource = (preRadius + randStrength(source)*sourceRadius)*sin(pi);
    patch(-6+xSource+0.5*xSourceCir+preCenter, ypos+ySource+0.5*ySourceCir, cmapSource{source},'linewidth',sourceEWidth);
end
xlim(xLim);
ylim(yLim);
pbaspect([1 diff(yLim)/diff(xLim) 1])

text(preCenter-10,apicalLength/2+radiusSoma,'Apical Synapses','Fontsize',24,'Rotation',90,'HorizontalAlignment','Center');
text(preCenter-10,-apicalLength/2-radiusSoma,'Basal Synapses','Fontsize',24,'Rotation',90,'HorizontalAlignment','Center');
text(preCenter-10-7,0,'Sources','Fontsize',24,'Rotation',90,'HorizontalAlignment','Center');
set(gca,'xtick',[]);
set(gca,'ytick',[]);
set(gca,'visible','off')

% print(gcf,'-painters',fullfile(fpath,'schematicModel'),'-depsc');




%% -- old poirazi version --

%{
figPath = '/Users/landauland/Documents/Research/o2/poirazi/poiraziFigures';

xLim = [-20 20];
yLim = [-20 53];
plotCircleAngle = 0:0.01:2*pi;

radiusSoma = 10;

% Basal Parameters
numBasal = 50;
lengthBasal = 8;
radiusBasal = 0.7;
lwidthBasal = 2;
lwidBasCir = 1.2;
fracCircleBasal = 0.7;
angleBasal = linspace(3*pi/2-pi*fracCircleBasal,3*pi/2+pi*fracCircleBasal,numBasal);
% Basal Tuning 
drad = 2*pi/numBasal;
basalTuning = -pi+drad : drad : pi;
% Basal Positions
xBasal = (radiusSoma+radiusBasal)*cos(angleBasal);
yBasal = (radiusSoma+radiusBasal)*sin(angleBasal);
xBasCir = radiusBasal*cos(plotCircleAngle);
yBasCir = radiusBasal*sin(plotCircleAngle);
% ColorMap Basal Inputs
cmapBasal = hsv(numBasal);

% Apical Parameters
numApical = 1;
apicalBaseLength = 1;
apicalBaseLWidth = 5;
apicalLength = 40;
fracCircleApical = 0;
angleApical = linspace(pi/2-pi*fracCircleApical,pi/2+pi*fracCircleApical,numApical);
apicalSynapses = 56;
apicalOffset = 0.2;
synapsePosition = linspace(apicalOffset*apicalLength,apicalLength,apicalSynapses/2);
apSynRadius = 0.55;
leftRightOffset = 0.6;
xApCir = apSynRadius*cos(plotCircleAngle);
yApCir = apSynRadius*sin(plotCircleAngle);
offsetLeft = leftRightOffset*[-sin(angleApical); cos(angleApical)];
offsetRight = -leftRightOffset*[-sin(angleApical); cos(angleApical)];


figure(1);clf;
set(gcf,'units','normalized','outerposition',[0.39 0.38 0.19 0.6]);

% Plot Soma
patch(radiusSoma*cos(plotCircleAngle),radiusSoma*sin(plotCircleAngle),'k');

% Plot Basal Inputs
for nb = 1:numBasal
    xLine = cos(angleBasal(nb))*[lengthBasal+radiusSoma+radiusBasal,radiusSoma+radiusBasal]; 
    yLine = sin(angleBasal(nb))*[lengthBasal+radiusSoma+radiusBasal,radiusSoma+radiusBasal]; 
    line(xLine,yLine,'color',cmapBasal(nb,:),'linewidth',lwidthBasal)
    patch(xBasal(nb)+xBasCir, yBasal(nb)+yBasCir, cmapBasal(nb,:),'linewidth',lwidBasCir);
end

% Plot Apical Dendrites
line([0,0],[radiusSoma, radiusSoma+apicalBaseLength],'color','k','linewidth',apicalBaseLWidth);
for na = 1:numApical
    line([0 cos(angleApical(na))*apicalLength],...
        radiusSoma+apicalBaseLength+[0 sin(angleApical(na))*apicalLength],'color','k','linewidth',apicalBaseLWidth);
    xSynPos = cos(angleApical(na))*synapsePosition;
    ySynPos = sin(angleApical(na))*synapsePosition + radiusSoma+apicalBaseLength;
    for asyn = 1:apicalSynapses/2
        if rand()<0.5
            patch(offsetLeft(1,na)+xApCir+xSynPos(asyn),offsetLeft(2,na)+yApCir+ySynPos(asyn),'k','linewidth',lwidthBasal);
        else
            patch(offsetLeft(1,na)+xApCir+xSynPos(asyn),offsetLeft(2,na)+yApCir+ySynPos(asyn),'k','FaceColor','none','linewidth',lwidthBasal);
        end
        if rand()<0.5
            patch(offsetRight(1,na)+xApCir+xSynPos(asyn),offsetRight(2,na)+yApCir+ySynPos(asyn),'k','linewidth',lwidthBasal);
        else
            patch(offsetRight(1,na)+xApCir+xSynPos(asyn),offsetRight(2,na)+yApCir+ySynPos(asyn),'k','FaceColor','none','linewidth',lwidthBasal);
        end
    end 
end
xlim(xLim);
ylim(yLim);
pbaspect([1 diff(yLim)/diff(xLim) 1])

set(gca,'xtick',[]);
set(gca,'ytick',[]);
set(gca,'visible','off')
% print(gcf,'-painters',fullfile(figPath,'schematicModel'),'-djpeg');

%}

%% -- plot stimuli -- 

% good one: T=1000, shift=3000
T = 2000;
Tshift = 5000;

warning('off','all');
iafData = load('/Users/landauland/Documents/Research/o2/poirazi/botTopSS1_data_set2/poiraziBotTopSS_AD2_SN1_Run5.mat','iaf');
stimData = load('/Users/landauland/Documents/Research/o2/poirazi/botTopSS1/sameStim4.mat');
warning('on','all');
iaf = iafData.iaf; 
iafSave = iaf;
iaf.inhRate = 18;
stimValue = stimData.stimValue(:,end-T+1-Tshift:end-Tshift);
clear iafData stimData


%%

figPath = '/Users/landauland/Documents/Research/o2/poirazi/poiraziFigures';

vm = iaf.vm*ones(1,T);
spike = zeros(1,T);
for t = 2:T
    iaf = stepBotTopSS(iaf,stimValue(:,t));
    vm(t) = iaf.vm;
    spike(t) = iaf.spike;
end
spkTimes = find(spike)*iaf.dt;

% patchify stimulus
idx1 = [1, find(diff(stimValue(1,:))~=0)+1];
idx2 = [1, find(diff(stimValue(2,:))~=0)+1];
stim1 = stimValue(1,idx1);
stim2 = stimValue(2,idx2);
idx1 = [idx1, T]*iaf.dt; 
idx2 = [idx2, T]*iaf.dt; 

nBins = 20;
stim1Bins = linspace(-pi, pi, nBins);
stim1BinCenters = mean([stim1Bins(1:end-1); stim1Bins(2:end)],1);
[~,~,stim1BinLocation] = histcounts(stim1, stim1Bins);
cmapStim = hsv(nBins-1);


% Plot Stimuli Only
figure(4); clf; 
set(gcf,'units','normalized','outerposition',[0.14 0.68 0.52 0.26]);
for st1 = 1:length(stim1)
    xPatch = [idx1(st1) idx1(st1) idx1(st1+1) idx1(st1+1)];
    yPatch = [-80 -75 -75 -80];
    patch(xPatch,yPatch,cmapStim(stim1BinLocation(st1),:),'linestyle','none');
end
text(iaf.dt * -150,-77.5,'STIMULUS','Fontsize',24,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','middle');
text(iaf.dt * -150,-82.5,'STATE','Fontsize',24,'FontWeight','Bold','HorizontalAlignment','center','VerticalAlignment','middle');
for st2 = 1:length(stim2)
    xPatch = [idx2(st2) idx2(st2) idx2(st2+1) idx2(st2+1)];
    yPatch = [-85 -80 -80 -85];
    if stim2(st2)==1
        patch(xPatch,yPatch,'k');
    else
        patch(xPatch,yPatch,'w');
    end
end
set(gca,'visible','off')
% print(gcf,'-painters',fullfile(figPath,'stimStateVisualization'),'-djpeg');


%%

apAmplitude = -10;
tv = (1:T)*iaf.dt;
figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.01 0.45 0.5 0.4]);

plot(tv, 1000*vm, 'color','k','linewidth',1.5);
for st = 1:length(spkTimes)
    line(spkTimes(st)*[1,1],[-70 apAmplitude],'color','r','linewidth',1.5);
end
for st1 = 1:length(stim1)
    xPatch = [idx1(st1) idx1(st1) idx1(st1+1) idx1(st1+1)];
    yPatch = [-80 -75 -75 -80];
    patch(xPatch,yPatch,cmapStim(stim1BinLocation(st1),:),'linestyle','none');
end
for st2 = 1:length(stim2)
    xPatch = [idx2(st2) idx2(st2) idx2(st2+1) idx2(st2+1)];
    yPatch = [-85 -80 -80 -85];
    if stim2(st2)==1
        patch(xPatch,yPatch,'k');
    else
        patch(xPatch,yPatch,'w');
    end
end
xlim([-100 T+100]*iaf.dt);
ylim([-90 apAmplitude]);
set(gca,'xtick',0:2);
set(gca,'ytick',[-70 -50 -30]);
xlabel('Time (ms)');
ylabel('mV');
set(gca,'fontsize',24);

% print(gcf,'-painters',fullfile(figPath,'iafTrajectory'),'-djpeg');


[~,~,tcBinLocation] = histcounts(iafSave.tuningCenter(:),stim1Bins);
averageWeight = zeros(1, nBins-1);
for nb = 1:nBins-1
    mnToAverage = iafSave.basalWeight(tcBinLocation==nb)/iaf.maxBasalWeight;
    averageWeight(nb) = mean(mnToAverage);
end
    
figure(2); clf;
set(gcf,'units','normalized','outerposition',[0.51 0.45 0.24 0.4]);
polarscatter(iafSave.tuningCenter(:),iafSave.basalWeight(:)/iaf.maxBasalWeight);
hold on;
% for n = 1:numel(iafSave.tuningCenter)
%     polarscatter(iafSave.tuningCenter(n),iafSave.basalWeight(n)/iaf.maxBasalWeight,...
%         'markeredgecolor',cmapStim(tcBinLocation(n),:),'markerfacecolor',cmapStim(tcBinLocation(n),:),...
%         'markerfacealpha',1);
% end
p = polarplot([stim1BinCenters stim1BinCenters(1)], [averageWeight, averageWeight(1)],...
    'color','k','linewidth',2);
set(gca,'thetaGrid','off');
set(gca,'thetaticklabel',[]);
set(gca,'RTickLabel',[]);

% print(gcf,'-painters',fullfile(figPath,'stimulusWeights'),'-djpeg');



figure(3); clf;
set(gcf,'units','normalized','outerposition',[0.75 0.45 0.24 0.4]);
xRange = 0.2;
hold on;
patch([0.5 0.5 1.5 1.5],[0 1 1 0],'k','facealpha',0.5,'linewidth',1.5);
patch(1+[0.5 0.5 1.5 1.5],[0 1 1 0],'k','facealpha',0.0,'linewidth',1.5);
plot(1+xscat(iaf.apicalWeight(iaf.apicalTuning==1),xRange),iaf.apicalWeight(iaf.apicalTuning==1)/iaf.maxApicalWeight,...
    'marker','o','linestyle','none','color','k','markerfacecolor','k');
plot(2+xscat(iaf.apicalWeight(iaf.apicalTuning==2),xRange),iaf.apicalWeight(iaf.apicalTuning==2)/iaf.maxApicalWeight,...
    'marker','o','linestyle','none','color','k','markerfacecolor','k');
xlim([0.5,2.5]);
set(gca,'visible','off');

% print(gcf,'-painters',fullfile(figPath,'stateWeights'),'-djpeg');
    

%%

randWeights = rand(size(iafSave.tuningCenter));
averageWeightRand = zeros(1, nBins-1);
for nb = 1:nBins-1
    mnToAverage = randWeights(tcBinLocation==nb);
    averageWeightRand(nb) = mean(mnToAverage);
end

figure(2); clf;
set(gcf,'units','normalized','outerposition',[0.51 0.45 0.24 0.4]);
polarscatter(iafSave.tuningCenter(:),randWeights(:));
hold on;
% for n = 1:numel(iafSave.tuningCenter)
%     polarscatter(iafSave.tuningCenter(n),randWeights(n),...
%         'markeredgecolor',cmapStim(tcBinLocation(n),:),'markerfacecolor',cmapStim(tcBinLocation(n),:),...
%         'markerfacealpha',1);
% end
p = polarplot([stim1BinCenters stim1BinCenters(1)], [averageWeightRand, averageWeightRand(1)],...
    'color','k','linewidth',2);
set(gca,'thetaGrid','off');
set(gca,'thetaticklabel',[]);
set(gca,'RTickLabel',[]);

% print(gcf,'-painters',fullfile(figPath,'stimulusWeightsRand'),'-djpeg');


randWeights = rand(100,1);
figure(3); clf;
set(gcf,'units','normalized','outerposition',[0.75 0.45 0.24 0.4]);
xRange = 0.2;
hold on;
patch([0.5 0.5 1.5 1.5],[0 1 1 0],'k','facealpha',0.5,'linewidth',1.5);
patch(1+[0.5 0.5 1.5 1.5],[0 1 1 0],'k','facealpha',0.0,'linewidth',1.5);
plot(1+xscat(randWeights(1:50),xRange),randWeights(1:50),...
    'marker','o','linestyle','none','color','k','markerfacecolor','k');
plot(2+xscat(randWeights(51:100),xRange),randWeights(51:100),...
    'marker','o','linestyle','none','color','k','markerfacecolor','k');
xlim([0.5,2.5]);
set(gca,'visible','off');

% print(gcf,'-painters',fullfile(figPath,'stateWeightsRand'),'-djpeg');
    





%% -- legend for covariance matrices

N = size(cmapStim,1);
H = 0.005;

xPatch = [0 0 1/N/2 1/N/2];
yPatch = [0 H H 0];

figure(7); clf; 
set(gcf,'units','normalized','outerposition',[0.14 0.68 0.52 0.26]);
for sn = 1:N
    patch(xPatch+(sn-1)/N/2,yPatch+H,cmapStim(sn,:),'linestyle','none');
end
for sn = 1:N
    patch(0.5+xPatch+(sn-1)/N/2,yPatch+H,cmapStim(sn,:),'linestyle','none');
end
p = patch([0 0 0.5 0.5],yPatch,'k');
patch([0.5 0.5 1 1],yPatch,'w');
set(gca,'visible','off')

% print(gcf,'-painters',fullfile(figPath,'covPlotLabelDouble'),'-djpeg');


























