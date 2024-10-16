function [iaf,spkTimes,smallBasalWeight,smallApicalWeight] = runo2_corrD(runIdx,apDepIdx,numStateIdx,basalDepFollow)
%function runo2_corrD(runIdx,apDepIdx,numStateIdx)
rng('shuffle')

apicDepArray = [1.1, 1.05, 1.025, 1.0];
apicalDepression = apicDepArray(apDepIdx);
basalDepression = 1.1;

numStateArray = [3];
numState = numStateArray(numStateIdx);

if basalDepFollow==1
    basalDepression = apicalDepression;
end

runsInEach = 1;
T = 19200*1000;

% -- things that worked: --
% maxBasal 1e-9
% maxApical 200e-12
% numBasal = 100
% numApical 9*50

options.dt = 0.001;
options.T = T; 
options.maxBasalWeight = 300e-12; % 150 pS (20mV/100MOhm)/DF/numSynapses
options.maxApicalWeight = 100e-12; % 50 pS (less than basal) --- ***** APICAL OFF FOR TESTING *****
options.loseSynapseRatio = 0.01; % ratio to max weight (for basal and apical) that initiates a new synaptic connection
options.newSynapseRatio = 0.01; % starting weight (ratio to max) of new synapse
options.basalDepression = basalDepression; 
options.apicalDepression = apicalDepression; 
options.numBasal = 300; % remember to account for max synaptic weight!!!
options.numApical = 100; % there are 9 indices 
options.plasticityRate = 0.01;
options.conductanceThreshold = 0.1; % threshold for counting synaptic conductance (ratio to max weight) (offset relu)
% Homeostasis
%{
tau_r * r' = -r + instantaneousRate
h = r_h / r : ratio of ideal homeostatic rate to current rate estimate
dw/dt_homeo = -h*w
%}
options.homeostasisTau = 20; % seconds (this is tau of firing rate estimate)
options.homeostasisRate = 20; % spikes/sec

pRateIncrease = [1, T/12, T/2];  % pRateIncrease(1)=1 turns this off
pRate = @(t) options.plasticityRate + ...
    (pRateIncrease(1)-1)*options.plasticityRate*(1/(1+exp(-(t-pRateIncrease(3))/pRateIncrease(2))));

% Stimulus 
options.numInputs = 100;
options.numSignals = numState;
options.sourceMethod = 'gauss'; % 'divide' or 'gauss'
options.sourceStrength = 3;
switch options.sourceMethod
    case 'divide'
        numInputPerSignal = options.numInputs/options.numSignals;
        sourceLoading = cell2mat(arrayfun(@(signal) arrayfun(@(input) options.sourceStrength*(input>(signal-1) & input<=signal), (1:options.numInputs)/numInputPerSignal, 'uni', 1), (1:options.numSignals)','uni', 0));
    case 'divideSoft'
        numInputPerSignal = options.numInputs/options.numSignals;
        sourceLoading = cell2mat(arrayfun(@(signal) arrayfun(@(input) options.sourceStrength*(input>(signal-1) & input<=signal), (1:options.numInputs)/numInputPerSignal, 'uni', 1), (1:options.numSignals)','uni', 0));
        idxSoft = rem(floor((0:options.numInputs-1)/numInputPerSignal*2),2)==1;
        sourceLoading(:,idxSoft) = sourceLoading(:,idxSoft)/2;
    case 'gauss'
        % Use a gaussian source window (allowing overlap), distribute
        shiftInputPerSignal = round(options.numInputs/options.numSignals);
        firstInputSignalIdx = round(shiftInputPerSignal/2);
        idxGauss = (1:options.numInputs)-options.numInputs/2;
        widthGauss = 2/5*shiftInputPerSignal;
        gaussLoading = exp(-idxGauss.^2/(2*widthGauss^2));
        idxPeakGauss = find(gaussLoading==max(gaussLoading),1);
        gaussLoading = circshift(gaussLoading,firstInputSignalIdx - idxPeakGauss);
        sourceLoading = cell2mat(arrayfun(@(signal) circshift(gaussLoading,(signal-1)*shiftInputPerSignal), (1:options.numSignals)', 'uni', 0));
end
options.sourceLoading = sourceLoading;
options.varAdjustment = sqrt(sum(sourceLoading.^2,1)+1);
options.rateStd = 10;
options.rateMean = 20;


tauStim = round(0.01/options.dt); % time constant of stim (in samples)

for runNum = 1:runsInEach
    iaf = build_corrD(options); % construct model
    
    needInput = true; % generate input signal
    y = zeros(options.numInputs,0); % keep track of input values
    ychange = zeros(1,0); % keep track of when input changes
    spikes = zeros(1,T);
    %vm = zeros(1,T);
    smallBasalWeight = zeros(iaf.numInputs,T/1000);
    smallApicalWeight = zeros(iaf.numInputs,T/1000);
    msg = ''; % Initialize msg
    keepTime = tic;
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/1000)==0
            timePer_t = toc(keepTime)/t;
            estTimeRemaining = (T-t)*timePer_t + T*timePer_t*(runsInEach-runNum);
            %fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Sample %.1f/%d seconds, %.1f Percent, EstTimeRemaining: %.1f seconds ...\n',...
                t*options.dt,T*options.dt,100*t/T,estTimeRemaining);
            fprintf(msg);
        end
        
        % Update Plasticity Rate
        iaf.apicalPotValue = pRate(t) * iaf.maxApicalWeight;
        iaf.apicalDepValue = pRate(t) * options.apicalDepression * iaf.maxApicalWeight;
        iaf.basalPotValue = pRate(t) * iaf.maxBasalWeight;
        iaf.basalDepValue = pRate(t) * options.basalDepression * iaf.maxBasalWeight;

        % Generate Input
        if needInput
            interval = ceil(exprnd(tauStim))+1; % duration of interval for current rates
            input = (randn(1,options.numInputs) + randn(1,options.numSignals)*options.sourceLoading)./options.varAdjustment;
            rate = options.rateStd * input + options.rateMean;
            rate(rate<0) = 0;
            y(:,end+1) = rate(:);
            ychange(end+1) = t;
            trackInterval = interval - 1;
            if trackInterval>0
                needInput = false;
            end
        else
            y(:,t) = y(:,t-1); % ***** for saving input/updates *****
            trackInterval = trackInterval - 1;
            if trackInterval==0
                needInput = true;
            end
        end
        
        iaf = step_corrD(iaf,rate(:));
        %vm(t) = iaf.vm;
        spikes(t) = iaf.spike;
        
        if rem(t,1000)==0
            smallBasalWeight(:,t/1000) = arrayfun(@(inputIdx) sum(iaf.basalWeight(iaf.basalTuneIdx==inputIdx)), 1:iaf.numInputs, 'uni', 1);
            smallApicalWeight(:,t/1000) = arrayfun(@(inputIdx) sum(iaf.apicalWeight(iaf.apicalTuneIdx==inputIdx)), 1:iaf.numInputs, 'uni', 1);
        end 
    end
    spkTimes = find(spikes);
    % Save data
    %name = sprintf('correlation1_AD%d_NS%d_Run%d',...
    %  apDepIdx,numStateIdx,(runIdx-1)*runsInEach+runNum);
    %savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/stdpModels/correlation1',name);
    %save(savepath,'iaf','spkTimes','y','ychange','smallApicalWeight','smallBasalWeight','-v7.3')
end
