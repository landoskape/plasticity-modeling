
%{
Note: 200413 -- all 50 runs from all 8 tunings were completed and saved
%}

% script to analyze results from a run where we varied tuning width
lmPath = '/Users/landauland/Documents/Research/SabatiniLab/presentations/LabMeeting/200417';
hpath = '/Users/landauland/Documents/Research/o2/stdpModels/circularFeature1';

resultDir = dir(fullfile(hpath,'data'));
resultNames = {resultDir(:).name};
resultNames = resultNames(cellfun(@(c) any(strfind(c, '.mat')), resultNames, 'uni', 1));

vonmises = @(x,u,ts) exp(ts.*cos(x-u))./(2*pi*besseli(0,ts));
tuningSharpness=[0.1,1,2,3,5,10,0.3,0.5];
NT = length(tuningSharpness);
NS = 200;
numRuns = 50;
NExc = 1200;
smFactor = 100;
dt = 0.0001;
simTime = 10000;
nsamples = simTime/dt;

spkRateSamples = nsamples*dt;
synWeights = zeros(NExc,NT,numRuns);
allWeights = zeros(NExc,NS,NT,numRuns);
spikeRate = zeros(spkRateSamples,NT,numRuns);
% Also save higher resolution from convergent time window end of sim...
cnvTime = 10; 
cnvResolution = 0.01;
cnvSamples = cnvTime/cnvResolution;
cnvFullSamples = cnvTime/dt;
cnvRate = zeros(cnvSamples,NT,numRuns);
cnvStim = zeros(cnvFullSamples,NT,numRuns);
cnvSampleOffset = (simTime-cnvTime)/dt;
% SpikeTriggeredStimulusAverage
stTime = 0.05;
stSamples = stTime/dt;
stAll = cell(NT,numRuns);
stAvg = zeros(stSamples,NT,numRuns);
stStd = zeros(stSamples,NT,numRuns);
for tidx = 1:NT
    for ridx = 1:numRuns
        fprintf('Tuning %d/%d, Run %d/%d, ...\n',tidx,NT,ridx,numRuns);
        name = sprintf('OneFeatureModel_1_Tuning%d_Run%d.mat',tidx,ridx);
        try
            cdata = load(fullfile(hpath, 'data',name),'gA');%,'spkTimes','stimValue');
            synWeights(:,tidx,ridx) = cdata.gA(:,end);
            allWeights(:,:,tidx,ridx) = cdata.gA;
%             cspk=false(nsamples,1);
%             cspk(cdata.spkTimes)=true;
%             cpsth = sum(reshape(cspk,nsamples/spkRateSamples,spkRateSamples),1)*nsamples/spkRateSamples*dt;
%             spikeRate(:,tidx,ridx) = cpsth;
%             % Measure convergent spikes in high res time window
%             cnvSpkTimes = cdata.spkTimes(cdata.spkTimes>cnvSampleOffset);
%             cnvSpk = false(cnvFullSamples,1);
%             cnvSpk(cnvSpkTimes-cnvSampleOffset)=true;
%             cnvpsth = sum(reshape(cnvSpk,cnvResolution/dt,cnvFullSamples/(cnvResolution/dt)),1)/cnvResolution;
%             cnvRate(:,tidx,ridx)=cnvpsth;
%             cnvStim(:,tidx,ridx)=cdata.stimValue(end-cnvFullSamples+1:end);
%             % Get spk triggered stimulus average
%             NCS = length(cnvSpkTimes);
%             cstavg = zeros(stSamples,NCS);
%             for ncs = 1:NCS
%                 cCnvSpk = cnvSpkTimes(ncs);
%                 cstavg(:,ncs)=cdata.stimValue(cCnvSpk-stSamples+1:cCnvSpk);
%             end
%             stAll{tidx,ridx}=cstavg;
%             stAvg(:,tidx,ridx)=circ_mean(cstavg,[],2);
%             stStd(:,tidx,ridx)=circ_std(cstavg,[],[],2);
        catch
            disp([name, ' failed...']);
        end
    end
end

smoothWeights = csmooth(synWeights,smFactor);


% save('results_200413_longRun','allWeights','synWeights','spikeRate','cnvRate','cnvStim','stAll','stAvg','stStd','tuningSharpness');

%{
What do I want to represent. First, that graph below, vector strength as a
function of time, we'll do a shaded error bar for each tuningSharpness and
also check to see how variable the onset of strong tuning is, i.e. the
appearance of the bump in the colormap image.

Also show PSTHs, we'll do an average rate (yoked across all
stimuli,choosing a window that gives a uniform distribution of angle
visits), and a stimulus evoked firing rate. We can even do a spike
triggered stimulus average. I wonder if the unidirectional jumps will
affect it at all?

I want to show the bump emergence and some how capture the shape based on a
time-locked average, time-locked to the point the bump emerges

Can we somehow determine a dga/dt based estimate of tuning emergence?
Vector strength might not be quite enough actually...

And _very_ importantly, can we figure out if there's an explanation to why
it starts to happen wherever it does?

%}
% Check Tuning Angle
% excAng = linspace(-pi,pi,100)';
% cosTune = cos(excAng);
% sinTune = sin(excAng);
% angEst = atan(sinTune./cosTune)-pi*(cosTune<0&sinTune<0)+pi*(cosTune<0&sinTune>0);
% scatter(excAng,angEst,'k')
% refline(1,0)


%% Estimate Tuning Angle

xst = dt:dt:stTime;
excAng = linspace(-pi,pi,NExc)';
cosTune = squeeze(mean(synWeights.*cos(excAng),1));
sinTune = squeeze(mean(synWeights.*sin(excAng),1));
angEstimate = atan(sinTune./cosTune)-pi*(cosTune<0&sinTune<0)+pi*(cosTune<0&sinTune>0);

cmap = varycolor(NT);
stCenter = squeeze(circ_mean(stAvg));

figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.26 0.51 0.72]);
subplot(2,2,1); hold on;
plotAvg = reshape(circ_mean(reshape(cstavg-angEstimate(end,end),numel(cstavg),1),[],2),size(cstavg,1),size(cstavg,2));
plot((xst-max(xst))*1000, plotAvg, 'color',[0 0 0 0.02],'linewidth',1);
xlim([-50 0]); ylim([-pi pi]);
set(gca,'ytick',-2*pi:pi:2*pi); set(gca,'yticklabel',{'-2\pi','-\pi','0','\pi','2\pi'});
xlabel('Time before spike (ms)');
ylabel('Stim \theta Centered on Neural Tuning');
title('Example Cell Spike-Triggered Stim Average');
set(gca,'fontsize',16);

subplot(2,2,3); hold on;
for nt = 1:NT
    p=plot(angEstimate(nt,:),stCenter(nt,:),'color',[cmap(nt,:),0.5],'linestyle','none',...
        'marker','o','markerfacecolor',cmap(nt,:));
end
set(gca,'xtick',-2*pi:pi:2*pi); set(gca,'xticklabel',{'-2\pi','-\pi','0','\pi','2\pi'});
set(gca,'ytick',-2*pi:pi:2*pi); set(gca,'yticklabel',{'-2\pi','-\pi','0','\pi','2\pi'});
xlim([-pi pi]); ylim([-pi pi]);
xlabel('Synaptic Tuning'); ylabel('Spike-Triggered Tuning');
title('Tuning & Spk-Triggered Stimulus Agree');
set(gca,'fontsize',16);

subplot(2,2,2); hold on;
for nt = 1:NT
    plot((xst-max(xst))*1000,squeeze(stAvg(:,nt,:))-angEstimate(nt,:),'color',[cmap(nt,:),0.5]);
end
ylim([-2*pi 2*pi]);
set(gca,'ytick',-2*pi:pi:2*pi); set(gca,'yticklabel',{'-2\pi','-\pi','0','\pi','2\pi'});
xlabel('Time before spike (ms)');
ylabel('Stim \theta Centered on Neural Tuning');
title('Spike-Triggered Stimulus Average');
set(gca,'fontsize',16);

subplot(2,2,4); hold on;
for nt = 1:NT
    plot((xst-max(xst))*1000,squeeze(stStd(:,nt,:)),'color',[cmap(nt,:),0.5]);
end
ylim([0 pi/2]);
set(gca,'ytick',[0 pi/4 pi/2]); set(gca,'yticklabel',{'0','\pi/4','\pi/2'});
xlabel('Time before spike (ms)');
ylabel('Stim Standard Deviation');
title('Spike-Triggered Stimulus Deviation');
set(gca,'fontsize',16);

% print(gcf,'-painters',fullfile(lmPath,'spkTriggeredStimAverage'),'-djpeg');




%% Plot Stimulus Features

NExc = 1200;

[~,idxTS] = sort(compiledResults.prmValues);
idxTS = idxTS(1:6);
tuningSharpness = compiledResults.prmValues(idxTS);
synWeights=squeeze(compiledResults.allWeights(:,end,:,idxTS));
NT = length(tuningSharpness);
dt = 0.0001;

stAvg = compiledResults.stAvg(:,:,idxTS);
stStd = compiledResults.stStd(:,:,idxTS);

ex = [7 6];
stAll = compiledResults.stAll(:,idxTS);
cstavg = stAll{ex(1),ex(2)};
stTime = 0.050;

xst = dt:dt:stTime;
excAng = linspace(-pi,pi,NExc)';
cosTune = squeeze(mean(synWeights.*cos(excAng),1));
sinTune = squeeze(mean(synWeights.*sin(excAng),1));
angEstimate = atan(sinTune./cosTune)-pi*(cosTune<0&sinTune<0)+pi*(cosTune<0&sinTune>0);

cmap = varycolor(NT);
stCenter = squeeze(circ_mean(stAvg));

figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.29 0.6 0.61 0.38]);

subplot(1,3,1); hold on;
ex2plot = reshape(circ_mean(reshape(cstavg-angEstimate(ex(1),ex(2))...
    ,numel(cstavg),1),[],2),size(cstavg,1),size(cstavg,2));
plot((xst-max(xst))*1000, ex2plot, 'color',[0 0 0 0.05],'linewidth',2);
xlim([-50 0]); ylim([-pi pi]);
set(gca,'ytick',-2*pi:pi:2*pi); set(gca,'yticklabel',{'-2\pi','-\pi','0','\pi','2\pi'});
xlabel('Time before spike (ms)');
ylabel('Stim \theta Centered on Neural Tuning');
title('Example Cell Spike-Triggered Stim Average');
set(gca,'fontsize',16);


subplot(1,3,2); hold on;
for nt = 1:NT, plot((xst-max(xst))*1000,stStd(:,1,nt),'color',[cmap(nt,:),0.5],'linewidth',1.5); end
for nt = 1:NT
    plot((xst-max(xst))*1000,stStd(:,:,nt),'color',[cmap(nt,:),0.5],'linewidth',1.5);
end
xlim([-50 0]);
ylim([0 pi/2]);
set(gca,'ytick',[0 pi/4 pi/2]); set(gca,'yticklabel',{'0','\pi/4','\pi/2'});
xlabel('Time before spike (ms)');
ylabel('Stim Standard Deviation');
title('Spike-Triggered Stimulus Deviation');
legend(cellfun(@(c) sprintf('\\kappa: %.1f',c), num2cell(tuningSharpness), 'uni', 0),'location','southwest');
set(gca,'fontsize',16);



drive    =   40; % Maximum rate of stimulus driven pre-synaptic APs
baseRate =    5; % Baseline pre-synaptic AP Rate
vonmises = @(stim,u,ts) exp(ts.*cos(stim-u))./(2*pi*besseli(0,ts));
exRate = @(stimValue,u,ts) baseRate+vonmises(stimValue,u,ts)*drive;

subplot(1,3,3); hold on;
for nt = 1:NT
    plot(excAng, csmooth(exRate(excAng,0,tuningSharpness(nt))/100,380)*380, 'color',cmap(nt,:),'linewidth',1.5);
end
for nt = 1:NT
    shadedErrorBar(excAng, csmooth(exRate(excAng,0,tuningSharpness(nt))/100,380)*380,...
        sqrt(csmooth(exRate(excAng,0,tuningSharpness(nt))/100,380)*380),{'color',cmap(nt,:),'linewidth',1.5},1);
end
xlim([-pi pi]); ylim([0 90]);
set(gca,'xtick',[-pi 0 pi]); set(gca,'xticklabel',{'-\pi','0','\pi'});
xlabel('Stimulus Angle');
ylabel('Spikes');
title('Spikes / 10ms / 380 Inputs');
set(gca,'fontsize',16);

print(gcf,'-painters',fullfile(lmPath,'spkTriggeredStimAverage'),'-djpeg');


%% Plot of Example Emergence of Tuning across all synapses & time

exTrajectory = [3 2];
gA = compiledResults.allWeights(:,:,exTrajectory(1),exTrajectory(2));
angle = linspace(-pi,pi,1200);

T = 10000;
NS = 200;
dns = T/NS;
t = (dns:dns:T)/3600;

dsFactorTime = 5;
dsFactorAngle = 20;

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.47 0.56 0.49 0.4]);

s1=subplot(1,2,1); 
smoothGA = dsarray(dsarray(gA(:,1:dsFactorTime*floor(size(gA,2)/dsFactorTime))',dsFactorTime)',dsFactorAngle)';
imagesc(t,angle,smoothGA');
set(gca,'YDir','normal');
colormap(s1,'hot');
set(gca,'xtick',0:3);
set(gca,'ytick',[-pi 0 pi]); set(gca,'yticklabel',{'-\pi','0','\pi'});
xlabel('Hours');
ylabel('Angle of Input');
title('Synaptic Strength');
set(gca,'fontsize',16);

s2=subplot(1,2,2); 
dga = diff(gA,1,2)';
dsdga = dsarray(dga(1:dsFactorTime*floor(size(dga,1)/dsFactorTime),:),dsFactorTime);
dsdsga = dsarray(dsdga',dsFactorAngle);
dsdsga = dsdsga(:,2:end);
imagesc(t,angle,dsdsga);
set(gca,'ydir','normal')
colormap(s2,redblue);
caxis([-1 1]*max(abs([min(dsdsga(:)) max(dsdsga(:))])));
set(gca,'xtick',0:3);
set(gca,'ytick',[-pi 0 pi]); set(gca,'yticklabel',{'-\pi','0','\pi'});
xlabel('Hours');
ylabel('Angle of Input');
title('Changes In Synaptic Strength');
set(gca,'fontsize',16);



% print(gcf,'-painters',fullfile(lmPath,'changesSynapticStrengthExample'),'-djpeg');




%% Plot of all angle strength trajectories

[~,idxTS] = sort(compiledResults.prmValues);
idxTS = idxTS(1:6);
tuningSharpness = compiledResults.prmValues(idxTS);
NT = length(tuningSharpness);

NExc = 1200;
dt = 0.0001;
T = 10000;
nsamples = T/dt;
NS = 200;
xAx = (1:NS)*(nsamples*dt)*(1/NS);

excAng = linspace(-pi,pi,NExc)';
cosTune=compiledResults.allWeights(:,:,:,idxTS).*cos(excAng);
sinTune=compiledResults.allWeights(:,:,:,idxTS).*sin(excAng);
avgTune = cat(4, squeeze(mean(cosTune,1)), squeeze(mean(sinTune,1)));
angleStrength = sqrt(sum(avgTune.^2,4))/0.02; % relative vector strength to max possible;

cmap = varycolor(NT);
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.61 0.46 0.37]);
subplot(1,2,1);
hold on;
for nt = 1:NT
    plot(xAx/60/60, squeeze(angleStrength(:,:,nt)),'color',cmap(nt,:));
end
xlim([0 T/60/60]);
set(gca,'xtick',0:1:3); set(gca,'ytick',0:0.1:0.3);
xlabel('Hours');
ylabel('Vector Strength');
title('Acquisition of Tuning');
set(gca,'fontsize',16);

subplot(1,2,2);
hold on;
for nt = 1:NT
    plot(xAx/60/60, mean(angleStrength(:,:,nt),2),'color',cmap(nt,:),'linewidth',1.5);
end
for nt = 1:NT
    shadedErrorBar(xAx/60/60,mean(angleStrength(:,:,nt),2),std(angleStrength(:,:,nt),1,2),{'color',cmap(nt,:)},1);
end
xlim([0 T/60/60]);
set(gca,'xtick',0:1:3); set(gca,'ytick',0:0.1:0.3);
xlabel('Hours');
ylabel('Vector Strength');
title('Acquisition of Tuning');
legend(cellfun(@(c) sprintf('\\kappa: %.1f',c), num2cell(tuningSharpness), 'uni', 0),'location','northeast');
set(gca,'fontsize',16);
% print(gcf,'-painters',fullfile(lmPath,'vectorStrength_acquisitionTuning'),'-djpeg');



figure(2); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.15 0.61 0.24 0.37]);
vonmises = @(stim,u,ts) exp(ts.*cos(stim-u))./(2*pi*besseli(0,ts));
baseRate = 5; driveRate = 40;
neuron.exSynapses.exRate = @(stimValue) neuron.exSynapses.baseRate + ...
    vonmises(stimValue,neuron.exSynapses.tuningCenter)*neuron.exSynapses.drive;
x = linspace(-pi,pi,200);
dRate = driveRate*vonmises(x,0,1);
ts = tuningSharpness;
for t = 1:length(ts)
    patch([x fliplr(x)],[driveRate*vonmises(x,0,ts(t))+baseRate, baseRate*zeros(1,length(x))],...
        cmap(t,:),'FaceAlpha',0.2,'EdgeColor',cmap(t,:),'LineWidth',1.5);
end
line([-pi pi],baseRate*[1 1],'color','k','linewidth',1,'linestyle','--');
set(gca,'xtick',[-pi 0 pi]); set(gca,'xticklabel',{'-\pi','0','\pi'});
set(gca,'ytick',0:5:30);
xlim([-pi pi]);
xlabel('Stimulus Angle');
ylabel('Presynaptic Firing Rate (hz)');
title('Input');
text(0,2,'Base Rate','HorizontalAlignment','Center','FontSize',16)
text(0,8,'Driven Rate','HorizontalAlignment','Center','FontSize',16);
legend(cellfun(@(c) sprintf('\\kappa=%.1f',c),num2cell(ts),'uni',0),'location','northeast');
set(gca,'fontsize',16);
% print(gcf,'-painters',fullfile(lmPath,'vectorStrength_vonMisesCurves'),'-djpeg');


%% Plot Centered Tuning Traces

[~,idxTS] = sort(compiledResults.prmValues);
idxTS = idxTS(1:6);
synWeights=squeeze(compiledResults.allWeights(:,end,:,idxTS));
angles = linspace(-pi,pi,size(synWeights,1))';
rmAngles = angles.*ones(size(synWeights));
centerTune = squeeze(circ_mean(rmAngles,synWeights));
NT=size(synWeights,3);
numRuns = size(synWeights,2);
centerWeights = zeros(size(synWeights));
for nt = 1:NT
    for r = 1:numRuns
        idxShift = 600-find(angles>=centerTune(r,nt),1,'first');
        centerWeights(:,r,nt)=circshift(synWeights(:,r,nt),idxShift);
    end
end

ts = compiledResults.prmValues(idxTS);
cmap = varycolor(NT);
figure(1); clf; hold on;
for nt = 1:NT
    plot(angles, mean(centerWeights(:,:,nt),2)/0.02,'color',[cmap(nt,:),0.5+log(1+0.2*ts(nt))],'linewidth',1.5);
end
set(gca,'xtick',[-pi 0 pi]); set(gca,'xticklabel',{'-\pi','0','\pi'});
set(gca,'ytick',0:0.5:1);
xlim([-pi pi]);
xlabel('Stimulus Angle');
ylabel('Postsynaptic Strength');
title('Centered Tuning Curves');
legend(cellfun(@(c) sprintf('\\kappa=%.1f',c),num2cell(ts),'uni',0),'location','northeast');
set(gca,'fontsize',16);

% print(gcf,'-painters',fullfile(lmPath,'tuningCurve'),'-djpeg');

% for nt = 2:NT
%     plot(angles, squeeze(centerWeights(:,:,nt)),'color',[cmap(nt,:),0.005],'linewidth',1);
% end
% for nt = 2:NT
%     plot(angles, mean(centerWeights(:,:,nt),2),'color',cmap(nt,:),'linewidth',1.5);
% end


%% Plot of PSTHs

[~,idxTS] = sort(compiledResults.prmValues);
idxTS = idxTS(1:6);
NT = length(idxTS);
tuningSharpness=compiledResults.prmValues(idxTS);

spikeRate = compiledResults.spikeRate;

xpsth = (1:10000)/3600;

cmap = varycolor(NT);
figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.39 0.61 0.36 0.37]);
%subplot(1,2,1);
hold on;
for nt = 1:NT
    plot(xpsth, mean(spikeRate(:,:,nt),2),'color',cmap(nt,:),'linewidth',1.5);
end
for nt = 1:NT
    shadedErrorBar(xpsth,mean(spikeRate(:,:,nt),2),std(spikeRate(:,:,nt),1,2),{'color',cmap(nt,:)},1);
end
xlim([0 1]);
set(gca,'xtick',0:0.25:1);
set(gca,'xticklabel',0:15:60);
xlabel('Minutes');
ylabel('Spike Rate (hz)');
title('PSTH');
legend(cellfun(@(c) sprintf('\\kappa=%.1f',c), num2cell(tuningSharpness), 'uni', 0),'location','northeast');
set(gca,'fontsize',16);
% print(gcf,'-painters',fullfile(lmPath,'spikeRate'),'-djpeg');

%% -- 
dRad = 2*pi/1200;
excAng = (dRad:dRad:2*pi)';
cosTune = synWeights.*cos(excAng);
sinTune = synWeights.*sin(excAng);
avgTune = cat(3, squeeze(sum(cosTune,1)), squeeze(sum(sinTune,1)));
angleStrength = sqrt(sum(avgTune.^2,3));

figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.07 0.44 0.22 0.34]);
hold on;
xRange = 0.4;
for nt = 1:NT
    plot(nt+xscat(angleStrength(nt,:),xRange),angleStrength(nt,:),'color','k','linestyle','none','marker','o');
    line(nt+0.7*xRange*[-1 1],mean(angleStrength(nt,:))*[1 1],'color','k','linewidth',3);
end
set(gca,'xtick',1:NT);
set(gca,'xticklabel',tuningSharpness);



%% - Display Figure - 

excAng = linspace(-pi,pi,NExc)';
cosTune = synWeights.*cos(excAng);
sinTune = synWeights.*sin(excAng);
avgTune = cat(3, squeeze(mean(cosTune,1)), squeeze(mean(sinTune,1)));
angleStrength = sqrt(sum(avgTune.^2,3));


example=randi(numRuns,NT,1);
figure(2); clf;
set(gcf,'units','normalized','outerposition',[0.15 0.4 0.8 0.47]);
subplot(1,3,1); hold on;
subplot(1,3,2); hold on;
subplot(1,3,3); hold on;
cmap = varycolor(NT);
xPlot = -pi:0.1:pi+0.1;
xRange = 0.4;
for ts = 1:NT
    subplot(1,3,1);
    plot(xPlot, vonmises(xPlot,0,tuningSharpness(ts)),'color',cmap(ts,:),'linewidth',1.5);
    
    subplot(1,3,2);
    plot(ts+xscat(angleStrength(ts,:),xRange),angleStrength(ts,:),'color',cmap(ts,:),'linestyle','none','marker','o');
    line(ts+0.7*xRange*[-1 1],mean(angleStrength(ts,:))*[1 1],'color','k','linewidth',3);
    
    subplot(1,3,3);
    p=plot(1:NExc, synWeights(:,ts,example(ts)),'color',cmap(ts,:),'linewidth',0.01);
    p.Color = [p.Color, 0.18];
end
for ts = 1:NT
    subplot(1,3,3);
    plot(1:NExc, smoothWeights(:,ts,example(ts)),'color',cmap(ts,:),'linewidth',2.5);
end
subplot(1,3,1); 
set(gca,'xtick',[-pi,0,pi]); set(gca,'xticklabel',{'\pi',0,'\pi'});
xlim([-pi pi]); 
xlabel('Tuning Angle'); ylabel('Input Drive'); title('Tuning Curve (centered at 0)');
set(gca,'fontsize',16);

subplot(1,3,2);
set(gca,'xtick',1:NT); set(gca,'xticklabel',tuningSharpness);
xlabel('TuningSharpness'); ylabel('Vector Strength'); title('Tuning Cohesion'); set(gca,'fontsize',16);

subplot(1,3,3);
set(gca,'xtick',[0 NExc/2 NExc]); set(gca,'xticklabel',{'\pi',0,'\pi'});
xlim([0 NExc]); xlabel('Tuning Angle'); ylabel('Synaptic Strength');
title('Example Cells, Raw and Smoothed'); set(gca,'fontsize',16);



%% -- explain vector strength --

excAng = linspace(-pi,pi,NExc);

%idxShow = [5 40 120];
idxShow=200;
NIS = length(idxShow);

%gA = allWeights(:,:,2,21);
gA = cdata.gA;

figure(11); clf; 
set(gcf,'units','normalized','outerposition',[0.35 0.25 0.56 0.63]);
for idx=1:NIS
    cosTune = gA(:,idxShow(idx)).*cos(excAng');
    sinTune = gA(:,idxShow(idx)).*sin(excAng');
    avgTune = cat(3, squeeze(mean(cosTune,1)), squeeze(mean(sinTune,1)));
    
    subplot(2,NIS,idx); hold on;
    plot(excAng,gA(:,idxShow(idx))/0.02);
    xlim([-pi pi]);
    set(gca,'xtick',[-pi,0,pi]); set(gca,'xticklabel',{'-\pi','0','\pi'});
    set(gca,'ytick',[0 1]);
    xlabel('Presynaptic Tuning');
    ylabel('Postsynaptic Strength'); 
    title('Tuning Curve');
    set(gca,'fontsize',16);
    
    subplot(2,NIS,NIS+idx); hold on;
    line([0 cosTune(1)],[0 sinTune(1)],'color','k','linewidth',0.3);
    line([0 avgTune(1,1,1)],[0 avgTune(1,1,2)],'color','r','linewidth',2.5);
    for i = 1:NExc
        line([0 cosTune(i)],[0 sinTune(i)],'color','k','linewidth',0.3);
    end
    line([0 avgTune(1,1,1)],[0 avgTune(1,1,2)],'color','r','linewidth',2.5);
    xlim([-0.02 0.02]); ylim([-0.02 0.02]);
    set(gca,'xtick',[-0.02 0 0.02]); set(gca,'ytick',[-0.02 0 0.02]);
    set(gca,'xticklabel',[-1 0 1]); set(gca,'yticklabel',[-1 0 1]);
    xlabel('TuningStrength'); ylabel('TuningStrength');
    title('Full Tuning Field');
    if idx==3
        legend('Input Vectors','Average','location','northeast');
    end
    set(gca,'fontsize',16);
end
%print(gcf,'-painters',fullfile(lmPath,'vectorStrengthExplanation'),'-djpeg');

figure(12); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.13 0.44 0.22 0.34]);
plot(xAx,angleStrength(:,2,21),'color','k','linewidth',1.5);
for idx=1:NIS
    plot(xAx(idxShow(idx)),angleStrength(idxShow(idx),2,21),...
        'color','r','marker','o','markersize',10,'markerfacecolor','r');
end
xlabel('Time (s)');
ylabel('Vector Strength');
title('Development of Tuning');
set(gca,'fontsize',16);
%print(gcf,'-painters',fullfile(lmPath,'vectorStrengthExplanationTrajectory'),'-djpeg');


%%
exTrajectory = [2 21];
gA = allWeights(:,:,exTrajectory(1),exTrajectory(2));

excAng = linspace(-pi,pi,NExc)';
cosTune = mean(gA.*cos(excAng),1);
sinTune = mean(gA.*sin(excAng),1);
angEstimate = atan(sinTune./cosTune)-pi*(cosTune<0&sinTune<0)+pi*(cosTune<0&sinTune>0);
angleStrength = sqrt(sum([cosTune;sinTune].^2,1));


xSavePoints = 1:size(gA,2);
smoothKernel=[1,5,10,50];
dsFactorTime = 5;
dsFactorAngle = 30;

figure(1); clf;
set(gcf,'units','normalized','outerposition',[0.47 0.34 0.49 0.62]);

subplot(2,2,1); hold on;
plot(angleStrength,'color','k','linewidth',1.5);
xlabel('SavePoint'); 
ylabel('Angle Strength'); 
title('Tuning'); 
set(gca,'fontsize',16);

h = 2;
subplot(2,2,2); hold on;
plot(1.5:size(gA,2), sqrt(mean(diff(gA,1,2).^2,1)),'color','k','linewidth',1.5);
plot(1+2*h:size(gA,2)-2*h, sqrt(mean((fivePointDer(gA',h)/2/h).^2,2)),'color','r','linewidth',1.5); 
xlabel('SavePoint');
ylabel('RMS Change In Weight');
title('Net Plasticity For Cell');
legend('diffOnEachPoint','fivePointDiff','location','northeast');
set(gca,'fontsize',16);


s3=subplot(2,2,3); 
smoothGA = dsarray(dsarray(gA(:,1:dsFactorTime*floor(size(gA,2)/dsFactorTime))',dsFactorTime)',dsFactorAngle)';
i1=imagesc(smoothGA');
set(gca,'YDir','normal');
colormap(s3,'hot');
%caxis([-1 1]*max(abs([min(dsdsga(:)) max(dsdsga(:))])));
%colorbar('Location','northoutside');
set(gca,'xticklabel',get(gca,'xtick')*dsFactorTime);
set(gca,'ytick',[]);
xlabel('SavePoint');
ylabel('Angle of Input');
title('Synaptic Strength');
set(gca,'fontsize',16);

s4=subplot(2,2,4); 
dga = diff(gA,1,2)';
dsdga = dsarray(dga(1:dsFactorTime*floor(size(dga,1)/dsFactorTime),:),dsFactorTime);
dsdsga = dsarray(dsdga',dsFactorAngle);
dsdsga = dsdsga(:,2:end);
imagesc(dsdsga);
set(gca,'ydir','normal')
colormap(s4,redblue);
caxis([-1 1]*max(abs([min(dsdsga(:)) max(dsdsga(:))])));
set(gca,'xticklabel',get(gca,'xtick')*dsFactorTime);
set(gca,'ytick',[]);
xlabel('SavePoint');
ylabel('Angle of Input');
title('Changes In Synaptic Strength');
set(gca,'fontsize',16);


figure(12); clf;
set(gcf,'units','normalized','outerposition',[0.02 0.49 0.44 0.41]);
subplot(1,2,1); 
smoothGA = dsarray(dsarray(gA(:,1:dsFactorTime*floor(size(gA,2)/dsFactorTime))',dsFactorTime)',dsFactorAngle)';
imagesc(smoothGA');
set(gca,'ydir','normal');
colormap('hot');
%caxis([-1 1]*max(abs([min(dsdsga(:)) max(dsdsga(:))])));
%colorbar('Location','northoutside');
set(gca,'xticklabel',get(gca,'xtick')*dsFactorTime);
set(gca,'ytick',[]);
xlabel('SavePoint');
ylabel('Angle of Input');
title('Synaptic Strength');
set(gca,'fontsize',16);

xVal = linspace(-pi,pi,NExc)';
[~,pkidx]=max(csmooth(gA(:,end),200));
vmCurve = vonmises(xVal,xVal(pkidx),2);
vmCurve = 0.02*vmCurve/max(vmCurve);

subplot(1,2,2); hold on;
plot(gA(:,end),xVal,'color','k','linewidth',1);
plot(vmCurve,xVal,'color','r','linewidth',2);
ylim([-pi pi]);
xlabel('Synaptic Weight');
ylabel('Tuning Angle');
title('Width SynDist \propto TuningWidth');
legend('SynapticWeights','VonMisesFunction','location','northeast');
set(gca,'fontsize',16);





































