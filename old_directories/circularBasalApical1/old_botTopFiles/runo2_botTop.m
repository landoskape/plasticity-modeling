function [iaf,spkTimes,dspkTimes,smallApicalWeight,smallBasalWeight] = runo2_botTop(runIdx,apDepIdx,numStateIdx,apicThreshIdx)
rng('shuffle')

apicalDepressionArray = [1.1, 1.0, 0.9, 0.7];
apicalDepression = apicalDepressionArray(apDepIdx);

numStateArray = [2,3,5,8];
numState = numStateArray(numStateIdx);

apicThreshArray = [1/2, 1/3, 1/5];
apicThresh = apicThreshArray(apicThreshIdx);

runsInEach = 1;

% Stimulus Paramaters
stimulus.bvalue = 0;
stimulus.avalue = 1;
stimulus.basalChangeProb = 0.05;
stimulus.apicalChangeProb = 0.005;
stimulus.numDims = numState;

T = 800*1000;

options.dt = 0.001;
options.tuningSharpness = 1;
options.propBasalNecessary = 0.05;
options.apicalMaxPercThreshold = 0.5; % Apical Dendrites can only get to her$
options.loseSynapseRatio = 0.01;
options.newSynapseRatio = 0.1;
options.apicalAttenuation = 2;
options.apicalDepression = apicalDepression;
options.numBasal = 10;
options.numApical = 1;
options.numSynapses = 100;
options.numDims = numState;
options.allowApicalAMPA = 1;
options.plasticityRate = [0.01 0.01];
options.apicalThresholdFraction = apicThresh;
options.conductanceThreshold = 0.2;
options.arcMethod = 'none';

stimHistBins = linspace(-pi, pi, 12);
stimHistCenters = mean([stimHistBins(1:end-1); stimHistBins(2:end)],1);

keepTime = tic;
for runNum = 1:runsInEach
    iaf = build_botTop(options);

    spikes = zeros(1,T);
    dspikes = zeros(options.numApical,T);
    apicalWeightTrajectory = zeros(numState, options.numApical, T);
    basalWeightTrajectory = zeros(length(stimHistCenters), T);

    msg = '';
    for t = 1:T
        % Print some updates to the screen
        if rem(t,T/1000)==0
            fprintf(repmat('\b',1,length(msg))); % -- can't do this on o2...
            msg = sprintf('Sample %.1f/%d seconds, %.1f Percent, EstTime: %.1f seconds ...\n',...
                t*options.dt,T*options.dt,100*t/T,toc(keepTime)*T/t*runsInEach/runNum);
            fprintf(msg);
        end

        [iaf,stimulus] = step_botTop(iaf,stimulus);
        spikes(t) = iaf.spike;
        dspikes(:,t) = iaf.apicalSpiking;
        for apDend = 1:options.numApical
            for stateNum = 1:numState
                idx = iaf.apicalTuning(:,apDend)==stateNum;
                apicalWeightTrajectory(stateNum,apDend,t) = sum(iaf.apicalWeight(idx,apDend));
            end
        end
        
        [~,~,shBin] = histcounts(iaf.tuningCenter,stimHistBins);
        for shc = 1:length(stimHistCenters)
            basalWeightTrajectory(shc,t) = sum(iaf.basalWeight(shBin==shc),'all');
        end
    end
    smallApicalWeight = permute(mean(reshape(permute(...
        apicalWeightTrajectory,[4 3 1 2]),1000,T/1000,numState,options.numApical),1),[3 4 2 1]);
    smallBasalWeight = permute(mean(reshape(permute(...
        basalWeightTrajectory,[3 2 1]),1000,T/1000,length(stimHistCenters)),1),[3 2 1]);

    spkTimes = find(spikes);
    dspkTimes = find(dspikes);

    % Reduce Stim Vector
    %idxChange1 = [1,find(diff(cStimValue(1,:))~=0)+1];
    %stimVal1 = cStimValue(1,idxChange1);
    %idxChange2 = [1,find(diff(cStimValue(2,:))~=0)+1];
    %stimVal2 = cStimValue(2,idxChange2);

    % Save data
    name = sprintf('bt1dMultiState1_AD%d_NS%d_AT%d_Run%d',...
       apDepIdx,numStateIdx,apicThreshIdx,(runIdx-1)*runsInEach+runNum);
    savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/poirazi/bt1dMultiState1',name);
    %save(savepath,'iaf','spkTimes','dspkTimes','smallApicalWeight','smallBasalWeight','-v7.3')
end
