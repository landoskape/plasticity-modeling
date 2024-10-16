function [iaf,spkTimes,vm,smallBasalWeight,smallApicalWeight] = runo2_cba(runIdx,apDepIdx,eProbIdx)
rng('shuffle')

apicDepArray = [1.1, 1.05, 1.025, 1.0, 0.9];
apicalDepression = apicDepArray(apDepIdx);
basalDepression = 1.1;

edgeProbArray = linspace(0,1,5); % [0, 0.25, 0.5, 0.75, 1];
edgeProb = edgeProbArray(eProbIdx); 

runsInEach = 1;
T = 400*1000;

% - 3x3 grid where each position provides an orientation each sample.
% - the basal inputs only receive input from the central grid position
% - the apical inputs receive input from all positions
% - each input (B&A) is circular gaussian tuned to orientation
% - inputs can be replaced 

% -- things that worked: --
% maxBasal 1e-9
% maxApical 200e-12
% numBasal = 100
% numApical 9*50

options.dt = 0.001;
options.T = T;
options.basalSharpness = 1; 
options.apicalSharpness = 1; 
options.maxBasalWeight = 500e-12; % 150 pS (20mV/100MOhm)/DF/numSynapses
options.maxApicalWeight = 100e-12; % 50 pS (less than basal) --- ***** APICAL OFF FOR TESTING *****
options.loseSynapseRatio = 0.01; % ratio to max weight (for basal and apical) that initiates a new synaptic connection
options.newSynapseRatio = 0.01; % starting weight (ratio to max) of new synapse
options.basalDepression = basalDepression; 
options.apicalDepression = apicalDepression; 
options.numBasal = 100; % remember to account for max synaptic weight!!!
options.numApical = 9*50; % there are 9 indices 
options.plasticityRate = [0.01 0.01];
options.conductanceThreshold = 0.2; % threshold for counting synaptic conductance (ratio to max weight) (offset relu)
% Homeostasis
%{
tau_r * r' = -r + instantaneousRate
h = r_h / r : ratio of ideal homeostatic rate to current rate estimate
dw/dt_homeo = -h*w
%}
options.homeostasisTau = 20; % seconds (this is tau of firing rate estimate)
options.homeostasisRate = 20; % spikes/sec

% Stimulus 
options.basalIndex = 5; % basal inputs always look at stim idx 5 (central idx of 3x3 grid)
options.apicalIndices = 1:9; % apical inputs look at stim idx 1:9 (all of grid)
options.edgeProb = edgeProb; % probability of an edge appearing in any stimulus reset
tauStim = round(0.02/options.dt); % time constant of stim (in samples)
numAngles = 4; % between 0 and 180 (because it's orientation and not direction)
stimSize = 3; % stimSize x stimSize grid (tested on 3x3)
numIdx = stimSize^2;
options.numAngles = numAngles; % store for iaf
options.angles = (1:numAngles)/numAngles*pi; % angles to choose tuning from (0, pi]
dangles = mean(diff(options.angles)); 
stimHistBins = options.angles(1)-dangles/2 : dangles : options.angles(end) + dangles/2;
stimHistCenters = options.angles;

for runNum = 1:runsInEach
    iaf = build_cba(options); % construct model
    
    needInput = true; % create orientation image
    spikes = zeros(1,T);
    vm = zeros(1,T);
    basalWeightTrajectory = zeros(numAngles,T);
    apicalWeightTrajectory = zeros(numAngles,T,numIdx);
    
    msg = ''; % Initialize msg
    keepTime = tic;
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/1000)==0
            timePer_t = toc(keepTime)/t;
            estTimeRemaining = (T-t)*timePer_t + T*timePer_t*(runsInEach-runNum);
            fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Sample %.1f/%d seconds, %.1f Percent, EstTimeRemaining: %.1f seconds ...\n',...
                t*options.dt,T*options.dt,100*t/T,estTimeRemaining);
            fprintf(msg);
        end

        % Generate Input
        if needInput
            interval = ceil(exprnd(tauStim))+1; % duration of interval for current rates
            inputOri = randi(numAngles,stimSize^2,1)/numAngles*pi; % (0, pi] % could also try any angle from 0 to pi...
            if rand()<iaf.edgeProb
                edgeOri = randi(4); % only 4 options!!! for this!!!
                edgeIdx = getEdgeIdx(edgeOri, stimSize, stimSize);
                inputOri(edgeIdx) = edgeOri/4*pi - 1/4*pi; 
            end            
            %im = cell2mat(cellfun(@(c) drawEdge(c,51), num2cell(reshape(inputOri,3,3)), 'uni', 0));
            %figure(1);clf;imagesc(im)
            trackInterval = interval - 1;
            if trackInterval>0
                needInput = false;
            end
        else
            % y(:,t) = y(:,t-1); % ***** for saving input/updates *****
            trackInterval = trackInterval - 1;
            if trackInterval==0
                needInput = true;
            end
        end
        
        iaf = step_cba(iaf,inputOri);
        vm(t) = iaf.vm;
        spikes(t) = iaf.spike;
        
        [~,~,shBasalBin] = histcounts(iaf.basalTuneCenter,stimHistBins);
        [~,~,shApicBin] = histcounts(iaf.apicalTuneCenter,stimHistBins);
        for shc = 1:length(stimHistCenters)
            basalWeightTrajectory(shc,t) = sum(iaf.basalWeight(shBasalBin==shc),'all');
            for ix = 1:numIdx
                apicalWeightTrajectory(shc,t,ix) = sum(iaf.apicalWeight(shApicBin==shc & iaf.apicalTuneIdx==ix),'all');
            end
        end
    end
    smallBasalWeight = permute(mean(reshape(permute(...
        basalWeightTrajectory,[3 2 1]),1000,T/1000,length(stimHistCenters)),1),[3 2 1]);
    smallApicalWeight = repmat(smallBasalWeight,1,1,numIdx);
    for ix = 1:numIdx
        smallApicalWeight(:,:,ix) = permute(mean(reshape(permute(...
            apicalWeightTrajectory(:,:,ix),[3 2 1]),1000,T/1000,length(stimHistCenters)),1),[3 2 1]);
    end
    spkTimes = find(spikes);
    % Save data
    %name = sprintf('bt1dMultiState1_AD%d_NS%d_AT%d_Run%d',...
    %   apDepIdx,numStateIdx,apicThreshIdx,(runIdx-1)*runsInEach+runNum);
    %savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/poirazi/bt1dMultiState1',name);
    %save(savepath,'iaf','spkTimes','dspkTimes','smallApicalWeight','smallBasalWeight','-v7.3')
end
