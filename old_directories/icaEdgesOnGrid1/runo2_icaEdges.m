function [iaf,spikes,y,aweights,sweights,vm,homRateEstimate,numInputs] = runo2_icaEdges(runIdx,sDepIdx,oCorIdx)
rng('shuffle');

sDepArray = [1.1, 1.05, 1.025];
silentDepression = sDepArray(sDepIdx);
oCorArray = linspace(0, 1, 5);
oriCorrelation = oCorArray(oCorIdx);

T = 2400*1000;

options.dt = 0.001;
options.maxWeight = 0.012; % 150pS
options.inhWeight = 0.05; % 500pS
options.propNecessaryNMDA = inf;
options.loseSynapseRatio = 0.01;
options.newSynapseRatio = 0.01;
options.numActive = 39;
options.numSilent = 2;
options.numSynapses = 25;
options.conductanceThreshold = 0.2;
options.plasticityRate = [0.01, 0.01];
options.depValue = [1.1, silentDepression];
options.potentiationTau = 0.02;
options.depressionTau = 0.02;
% Homeostasis
%{
tau_r * r' = -r + instantaneousRate
tau_r : tau of firing rate estimate
r' = change in firing rate estimate (dr/dt)
instantaneousRate = 1(spike)/dt
^ this estimates the firing rate ^

h = r_h / r : ratio of ideal homeostatic rate to current rate estimate
dw/dt_homeo = -hw
%}
options.homeostasisTau = 20; % seconds (this is tau of firing rate estimate)
options.homeostasisRate = 20; % spikes/sec

activeIdx = [true(1,options.numActive), false(1,options.numSilent)];
silentIdx = [false(1,options.numActive), true(1,options.numSilent)];

% Setup Inputs
tau_c = 20; % time steps (should be ms!!!!)
options.numOrientation = 3;
options.gridLength = 3;
options.edgeLength = 3;
options.numEdge = 1; % ##not used yet...
options.oriCorr = oriCorrelation;

NO = options.numOrientation;
L = options.gridLength;
eL = options.edgeLength; 
NP = L*L; 

saveTime = 1000;
keepTime = tic;
runsInEach = 1;
for runNum = 1:runsInEach
    % Setup new model
    iaf = buildIcaEdges(options);

    needInput = true;
    spikes = zeros(1,T);
    homRateEstimate = zeros(1,T);
    %homScale = zeros(1,T); 
    y = zeros(NP,T);
    aweights = zeros(NO*NP,T/saveTime);
    sweights = zeros(NO*NP,T/saveTime);
    vm = iaf.rest*ones(1,T);
    numInputs = zeros(NO*NP,T/saveTime);
    %numReplace = zeros(1,T);
    idxSave = 0;
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/100)==0
            %fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Sample %d/%d seconds, %d Percent, EstRemaining: %.1f seconds ...\n',...
                t*options.dt,T*options.dt,100*t/T,(T-t)/T*toc(keepTime)*T/t*runsInEach/runNum);
            fprintf(msg);
        end

        % Generate Input
        if needInput
            interval = ceil(exprnd(tau_c))+1; % duration of interval for current rates
            inputActive = randi(NO,NP,1); 
            if rand<options.oriCorr
                edgeOri = randi(NO);
                [edgeIdx,~] = getEdgeIdx(edgeOri,L,eL);
                inputActive(edgeIdx) = edgeOri; 
                %inputActive(1:5) = 1;
            end
            inputActive = inputActive + NO*(0:NP-1)';
            y(:,t) = inputActive;
            trackInterval = interval - 1;
            if trackInterval>0
                needInput = false;
            end
        else
            y(:,t) = y(:,t-1);
            trackInterval = trackInterval - 1;
            if trackInterval==0
                needInput = true;
            end
        end

        iaf = stepIcaEdges(iaf, inputActive);
        spikes(t) = iaf.spike;
        vm(t) = iaf.vm;
        homRateEstimate(t) = iaf.homRateEstimate;
        %homScale(t) = iaf.homScale;

        if rem(t, saveTime)==0
            idxSave = idxSave + 1;
            for i = 1:(NO*NP)
                aweights(i,idxSave) = sum(iaf.ampaWeights(iaf.inputConnection==i & activeIdx));
                sweights(i,idxSave) = sum(iaf.ampaWeights(iaf.inputConnection==i & silentIdx));
            end
            numInputs(:,idxSave) = histcounts(iaf.inputConnection,0.5:NO*NP+0.5);
        end
    end
    spkTimes = find(spikes);

    % Save data
    name = sprintf('iafOriPos_SD%d_Run%d',sDepIdx,(runIdx-1)*runsInEach+runNum);
    savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/songAbbott/oriPosMoving1',name);
    % savepath = fullfile('~/Documents/Research/o2/stdpModels/cf_withSilentData',name);
    % save(savepath,'iaf','spkTimes','aweights','sweights','y')
end
