function [iaf,spkTimes,y,basalWeightTrajectory,apicalWeightTrajectory] = runo2_cba(runIdx,apDepIdx,eProbIdx)
% function runo2_cba(runIdx,apDepIdx,eProbIdx)
rng('shuffle')

apicDepArray = [1.1, 1.05, 1.025, 1.0];
apicalDepression = apicDepArray(apDepIdx);
basalDepression = 1.1;

edgeProbArray = [0.25,0.5,0.75,1];%linspace(0,1,5); % [0, 0.25, 0.5, 0.75, 1];
edgeProb = edgeProbArray(eProbIdx); 

runsInEach = 1;
T = 800*1000;

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
options.maxBasalWeight = 300e-12; % 150 pS (20mV/100MOhm)/DF/numSynapses
options.maxApicalWeight = 100e-12; % 50 pS (less than basal) --- ***** APICAL OFF FOR TESTING *****
options.loseSynapseRatio = 0.01; % ratio to max weight (for basal and apical) that initiates a new synaptic connection
options.newSynapseRatio = 0.01; % starting weight (ratio to max) of new synapse
options.basalDepression = basalDepression; 
options.apicalDepression = apicalDepression; 
options.numBasal = 300; % remember to account for max synaptic weight!!!
options.numApical = 100;%9*20; % there are 9 indices 
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
tauStim = round(0.01/options.dt); % time constant of stim (in samples)
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
    y = zeros(numIdx,0); % save input values
    ychange = zeros(1,0); % save input change times
    spikes = zeros(1,T);
    %vm = zeros(1,T);
    basalWeightTrajectory = zeros(numAngles,T/1000);
    apicalWeightTrajectory = zeros(numAngles,T/1000,numIdx);
    
    msg = ''; % Initialize msg
    keepTime = tic;
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/1000)==0
            timePer_t = toc(keepTime)/t;
            estTimeRemaining = (T-t)*timePer_t + T*timePer_t*(runsInEach-runNum);
            %fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Run %d/%d, Sample %.1f/%d seconds, %.1f Percent, EstTimeRemaining: %.1f seconds ...\n',...
                runNum,runsInEach,t*options.dt,T*options.dt,100*t/T,estTimeRemaining);
            fprintf(msg);
        end

        % Generate Input
        if needInput
            interval = ceil(exprnd(tauStim))+1; % duration of interval for current rates
            inputOri = randi(numAngles,stimSize^2,1)/numAngles*pi; % (0, pi] % could also try any angle from 0 to pi...
            if rand()<iaf.edgeProb
                edgeOri = randi(4); % only 4 options!!! for this!!!
                edgeIdx = getEdgeIdx(edgeOri, stimSize, stimSize);
                inputOri(edgeIdx) = edgeOri/4*pi; 
            end
            y(:,end+1) = inputOri;
            ychange(end+1) = t;
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
        %vm(t) = iaf.vm;
        spikes(t) = iaf.spike;
        
        if rem(t,1000)==0
            tidx = round(t/1000);
            [~,~,shBasalBin] = histcounts(iaf.basalTuneCenter,stimHistBins);
            [~,~,shApicBin] = histcounts(iaf.apicalTuneCenter,stimHistBins);
            for shc = 1:length(stimHistCenters)
                basalWeightTrajectory(shc,tidx) = sum(iaf.basalWeight(shBasalBin==shc),'all');
                for ix = 1:numIdx
                    apicalWeightTrajectory(shc,tidx,ix) = sum(iaf.apicalWeight(shApicBin==shc & iaf.apicalTuneIdx==ix),'all');
                end
            end
        end
    end
    spkTimes = find(spikes);
    
    % Save data
    name = sprintf('cba1_AD%d_EP%d_Run%d',...
      apDepIdx,eProbIdx,(runIdx-1)*runsInEach+runNum);
    savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/stdpModels/cba1',name);
    %save(savepath,'iaf','spkTimes','y','ychange','basalWeightTrajectory','apicalWeightTrajectory','-v7.3')
end
