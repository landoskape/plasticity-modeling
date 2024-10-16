function [iaf,spkTimes,smallBasalWeight,smallApicalWeight] = runo2_corrD(runIdx,apDepIdx,numStateIdx,basalDepFollow)
%function runo2_corrD(runIdx,apDepIdx,numStateIdx)
%
% for running o2, use second function line because no output needed
% --


rng('shuffle') % necessary for all o2 runs because the rng initializes to 0 if you don't manually shuffle

apicDepArray = [1.1, 1.05, 1.025, 1.0]; % apical depression parameters to use
apicalDepression = apicDepArray(apDepIdx);
basalDepression = 1.1;

numStateArray = [3]; % number of independent components (used to have options, now just using 3
numState = numStateArray(numStateIdx);

% Is basalDepression = 1.1 or equal to whatever apicalDepression is?
if basalDepFollow==1
    basalDepression = apicalDepression;
end

runsInEach = 1;
T = 19200*1000; % as I've shared it, the model stablizes around T=10000*1000, but isn't at equilibrium yet 

options.dt = 0.001; % time step
options.T = T; % number of time steps
options.maxBasalWeight = 300e-12; % can roughly think about in terms of conductance (300 pS)
options.maxApicalWeight = 100e-12; % (100 pS)
options.loseSynapseRatio = 0.01; % ratio to max weight (for basal and apical) that initiates a new synaptic connection
options.newSynapseRatio = 0.01; % starting weight (ratio to max) of new synapse
options.basalDepression = basalDepression; % fraction of max depression to max potentiation
options.apicalDepression = apicalDepression; % fraction of max depression to max potentiation
options.numBasal = 300; % number of basal synapses
options.numApical = 100; % number of apical synapses
options.plasticityRate = 0.01; % 0.01 is about the upper limit, it gets too noisy otherwise
options.conductanceThreshold = 0.1; % threshold for counting synaptic conductance (ratio to max weight) (like an offset relu)

% Homeostasis
%{
tau_r * r' = -r + instantaneousRate
h = r_h / r : ratio of ideal homeostatic rate to current rate estimate
dw/dt_homeo = -h*w

In practice the ending firing rate is never quite equal to the homeostatic
set point because it is in equilibrium with the STDP rule!!! (this is
interesting on it's own...?)
%}
options.homeostasisTau = 20; % seconds (this is the tau of firing rate estimate)
options.homeostasisRate = 20; % spikes/sec (this sets the homeostatic set point) 

% A mechanism to increase or decrease the rate of plasticity throughout
%pRateIncrease = [1, T/12, T/2];  % pRateIncrease(1)=1 turns this off
%pRate = @(t) options.plasticityRate + ...
%    (pRateIncrease(1)-1)*options.plasticityRate*(1/(1+exp(-(t-pRateIncrease(3))/pRateIncrease(2))));

% Stimulus 
options.numInputs = 100; % 100 input "types" with their own loading
options.numSignals = numState; % number of independent components
options.sourceMethod = 'gauss'; % 'divide' or 'gauss' % -- see switch below
options.sourceStrength = 3; % SNR ratio

% The following constructs sourceLoading matrices in several ways. 
% use imagesc(sourceLoading) to  see what they do. In my thesis I used
% 'gauss'
switch options.sourceMethod
    case 'divide'
        % Disjoint loading onto each source (each input only has 1 source)
        numInputPerSignal = options.numInputs/options.numSignals;
        sourceLoading = cell2mat(arrayfun(@(signal) arrayfun(@(input) options.sourceStrength*(input>(signal-1) & input<=signal), (1:options.numInputs)/numInputPerSignal, 'uni', 1), (1:options.numSignals)','uni', 0));
    case 'divideSoft'
        % Disjoint, but half of inputs have weaker SNR
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
options.varAdjustment = sqrt(sum(sourceLoading.^2,1)+1); % measures total variance from source and noise
options.rateStd = 10; % scale rates to have this standard deviation
options.rateMean = 20; % shift rates to have this mean 

tauStim = round(0.01/options.dt); % time constant of stim (in samples), better to be less than STDP time constants

for runNum = 1:runsInEach
    iaf = build_corrD(options); % construct model
    
    needInput = true; % generate input signal
    % y & ychange keep track of inputs, this is a compressed way to do it
    % that only records input values and when the input changes, rather
    % than every time bin which has lots of redundancy
    y = zeros(options.numInputs,0); % keep track of input values
    ychange = zeros(1,0); % keep track of when input changes
    spikes = zeros(1,T); % Keep track of when model spikes
    %vm = zeros(1,T);
    smallBasalWeight = zeros(iaf.numInputs,T/1000); % net synaptic weight to each input type for basal inputs
    smallApicalWeight = zeros(iaf.numInputs,T/1000);
    msg = ''; % Initialize msg
    keepTime = tic;
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/1000)==0
            timePer_t = toc(keepTime)/t;
            estTimeRemaining = (T-t)*timePer_t + T*timePer_t*(runsInEach-runNum);
            fprintf(repmat('\b',1,length(msg))); % -- comment out for o2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            msg = sprintf('Sample %.1f/%d seconds, %.1f Percent, EstTimeRemaining: %.1f seconds ...\n',...
                t*options.dt,T*options.dt,100*t/T,estTimeRemaining);
            fprintf(msg);
        end
        
        % Update Plasticity Rate - this wasn't very helpful
        %iaf.apicalPotValue = pRate(t) * iaf.maxApicalWeight;
        %iaf.apicalDepValue = pRate(t) * options.apicalDepression * iaf.maxApicalWeight;
        %iaf.basalPotValue = pRate(t) * iaf.maxBasalWeight;
        %iaf.basalDepValue = pRate(t) * options.basalDepression * iaf.maxBasalWeight;

        % Generate Input
        % Start with an exponentially distributed duration
        % Pick an input rate vector randomly
        % Keep it until end of random duration
        if needInput
            interval = ceil(exprnd(tauStim))+1; % duration of interval for current rates
            % Generate sample of input activity
            input = (randn(1,options.numInputs) + randn(1,options.numSignals)*options.sourceLoading)./options.varAdjustment;
            rate = options.rateStd * input + options.rateMean; % Translate to desired distribution
            rate(rate<0) = 0; % No negative rates!
            y(:,end+1) = rate(:); % Update new rate
            ychange(end+1) = t; % Note time of update
            trackInterval = interval - 1; % Counter for this stimulus
            if trackInterval>0
                needInput = false;
            end
        else
            trackInterval = trackInterval - 1; % update counter
            if trackInterval==0
                needInput = true;
            end
        end
        
        iaf = step_corrD(iaf,rate(:)); % step model
        %vm(t) = iaf.vm;
        spikes(t) = iaf.spike; % record spikes
        
        if rem(t,1000)==0
            % Every 100 samples, record how much net weight has accumulated
            % for each presynaptic input
            smallBasalWeight(:,t/1000) = arrayfun(@(inputIdx) sum(iaf.basalWeight(iaf.basalTuneIdx==inputIdx)), 1:iaf.numInputs, 'uni', 1);
            smallApicalWeight(:,t/1000) = arrayfun(@(inputIdx) sum(iaf.apicalWeight(iaf.apicalTuneIdx==inputIdx)), 1:iaf.numInputs, 'uni', 1);
        end 
    end
    spkTimes = find(spikes);
    
    % Save data - (for o2)
    %name = sprintf('correlation1_AD%d_NS%d_Run%d',...
    %  apDepIdx,numStateIdx,(runIdx-1)*runsInEach+runNum);
    %savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/stdpModels/correlation1',name);
    %save(savepath,'iaf','spkTimes','y','ychange','smallApicalWeight','smallBasalWeight','-v7.3')
end
