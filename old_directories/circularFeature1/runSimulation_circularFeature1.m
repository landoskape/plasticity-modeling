function runSimulation_circularFeature1(numRuns,tuningSharpnessIdx)
% Andrew Landau, modified from class exercise developed by Rachel Wilson+
rng('shuffle');

tuningSharpnessArray=[0.1,1,2,3,5,10];
tuningSharpness=tuningSharpnessArray(tuningSharpnessIdx);

%% Simulation Parameters
simPrm.simLength    =  10;         % Time to simulate, (sec)
simPrm.stepSize     = .0001;        % Time step resolution, (sec)
simPrm.nTimePoints = round(simPrm.simLength/simPrm.stepSize);  % # of points in simulation
simPrm.savePoints = 10/simPrm.stepSize;
simPrm.nSaves = simPrm.nTimePoints/simPrm.savePoints;

%% Stimulus Parameters
stimulus.numFeatures = 1;
stimulus.circular = 1;

stimulus.dNudge = pi/4;
stimulus.nudgeRate = 1000;

stimulus.dJump = pi;
stimulus.jumpRate = 50;

stimulus.value = 0;

%% Run Model 
for r = 1:numRuns
    aNeuron = modelNeuronFeatures(tuningSharpness); % Make a modelNeuron structure

    % Change some of the default parameters of the simulation
    % -- nothing to do --

    % Run simulation and save output
    spikes = zeros(1,simPrm.nTimePoints);
    stimValue = zeros(1,simPrm.nTimePoints);
    gA = zeros(aNeuron.Nex,simPrm.nSaves);
    saveIdx = 0;
    msg = '';
    for n=1:simPrm.nTimePoints
        localIdx = rem(n,simPrm.savePoints);
        if localIdx==0
            localIdx=simPrm.savePoints;
        elseif localIdx==1
            saveIdx=saveIdx+1;
        end
        if rem(n,10000)==0
            fprintf(repmat('\b',1,length(msg)));
            msg = sprintf('time point %d/%d ...\n',n,simPrm.nTimePoints);
            fprintf(msg);
        end
        [aNeuron,stimulus] = stepTimeFeatures(aNeuron,stimulus,simPrm.stepSize);  % Advance the sim time by 1 step
        spikes(n) = aNeuron.spike;
        stimValue(n) = stimulus.value;
        if localIdx==1
            gA(:,saveIdx) = aNeuron.exSynapses.gA;
        end
    end
    
    spkTimes = find(spikes);
    spkISI = simPrm.stepSize*diff(spkTimes);
    instantRate = 1./spkISI;
    
    %% Save data
    name = sprintf('OneFeatureModel_1_Tuning%d_Run%d',tuningSharpnessIdx,r);
    %savepath = fullfile('/n/data1/hms/neurobio/sabatini/andrew/stdpModels/circularFeature1',name);
    savepath = fullfile('~/Documents/Research/o2/stdpModels/circularFeature1/data',name);
    save(savepath,'aNeuron','simPrm','stimulus','spikes','stimValue','instantRate');
    
    % Reset stimulusvalue before running again
    stimulus.value = 0;
end



