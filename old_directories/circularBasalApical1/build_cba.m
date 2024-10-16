function iaf = build_cba(options)

iaf = struct();
iaf.dt = options.dt; % s
iaf.T = options.T; % total time (use to reconstruct signals if necessary)
iaf.tau = 20e-3; % s
iaf.resistance = 100e6; % Ohm
iaf.rest = -70e-3; % V
iaf.thresh = -50e-3; % V
iaf.vm = iaf.rest;

iaf.exRev = 0; % V
iaf.excTau = 20e-3; %s
iaf.driveBasal = 45; % HZ
iaf.baseBasal = 5; % HZ (baseline rate of firing) 
iaf.driveApical = 45;
iaf.baseApical = 5;

% Specify synaptic structure
iaf.numBasal = options.numBasal;
iaf.numApical = options.numApical;
iaf.basalIndex = options.basalIndex;
iaf.apicalIndices = options.apicalIndices(:); 

% Stimulus Structure
iaf.numAngles = options.numAngles;
iaf.angles = options.angles(:);
iaf.edgeProb = options.edgeProb;

% ---------------------- BASAL -------------------------
% Specify Basal Weights
iaf.maxBasalWeight = options.maxBasalWeight;
iaf.minBasalWeight = iaf.maxBasalWeight * options.loseSynapseRatio;
iaf.basalStartWeight = iaf.maxBasalWeight * options.newSynapseRatio;
iaf.basalCondThresh = iaf.maxBasalWeight * options.conductanceThreshold;
iaf.basalWeight = iaf.maxBasalWeight .* rand(iaf.numBasal, 1);
iaf.basalConductance = 0; % store total basal conductance

% Basal Tunings  
iaf.basalTuneIdx = iaf.basalIndex; % index of tuning for all basal!
iaf.basalTuneCenter = iaf.angles(randi(options.numAngles,iaf.numBasal,1));
iaf.basalSharpness = options.basalSharpness; 
% getRateBasal: takes as input the input orientation grid, and the vector 
% of basal tuning centers "btc". Then computes the double angle von mises 
% function for each. (You index to input ori once!!!) 
iaf.getRateBasal = @(inputOri, btc) iaf.baseBasal + iaf.driveBasal * ... % compute orientation 
    exp(iaf.basalSharpness * cos(2*(inputOri(iaf.basalIndex) - btc)))/(2*pi*besseli(0,iaf.basalSharpness)); 


% ---------------------- APICAL -------------------------
% Specify Apical Weights
iaf.maxApicalWeight = options.maxApicalWeight;
iaf.minApicalWeight = iaf.maxApicalWeight * options.loseSynapseRatio;
iaf.apicalStartWeight = iaf.maxApicalWeight * options.newSynapseRatio;
iaf.apicalCondThresh = iaf.maxApicalWeight * options.conductanceThreshold;
iaf.apicalWeight = iaf.maxApicalWeight .* rand(iaf.numApical, 1);
iaf.apicalConductance = 0; % store total apical conductance

% Apical Tunings  
numIdx = length(iaf.apicalIndices);
% *** In the following two lines, I either randomize apicalTuneIdx or
% distribute it equally. The equal distribution one should be paired with a
% change to step_cba that replaces tune idx sometimes
% iaf.apicalTuneIdx = reshape(repmat(1:numIdx,iaf.numApical/numIdx,1),iaf.numApical,1);
iaf.apicalTuneIdx = iaf.apicalIndices(randi(length(iaf.apicalIndices),iaf.numApical,1)); % index of tuning for all apical!
iaf.apicalTuneCenter = iaf.angles(randi(options.numAngles,iaf.numApical,1)); 
iaf.apicalSharpness = options.apicalSharpness; 
% getRateApical: takes as input the input orientation grid, and the vector 
% of apical tuning centers "atc". For apical, the input index can also
% change so we give the apical tuning index "ati". Then computes the double
% angle von mises function for each. 
iaf.getRateApical = @(inputOri, atc, ati) iaf.baseApical + iaf.driveApical * ... % compute orientation 
    exp(iaf.apicalSharpness * cos(2*(inputOri(ati) - atc)))/(2*pi*besseli(0,iaf.apicalSharpness)); 

% STDP Stuffs
iaf.basalPotentiation = zeros(size(iaf.basalWeight));
iaf.basalDepression = 0;
iaf.basalPotValue = options.plasticityRate(1) * iaf.maxBasalWeight;
iaf.basalDepValue = options.plasticityRate(1) * options.basalDepression * iaf.maxBasalWeight;
iaf.apicalPotentiation = zeros(size(iaf.apicalWeight));
iaf.apicalDepression = 0;
iaf.apicalPotValue = options.plasticityRate(2) * iaf.maxApicalWeight;
iaf.apicalDepValue = options.plasticityRate(2) * options.apicalDepression * iaf.maxApicalWeight;
iaf.potTau = 0.02;
iaf.depTau = 0.02;
iaf.homTau = options.homeostasisTau;
iaf.homRate = options.homeostasisRate;
iaf.homRateEstimate = iaf.homRate; % start at homRate so as to avoid blowups 

% Make Some Stupid Inhibitory Synapses
% -- note, will make this slow recurrent from excitatory input --
iaf.numInhibitory = 200;
iaf.inhRate = 20;
iaf.inhWeight = 100e-12;
iaf.gabaTau = 20e-3;
iaf.gabaConductance = 0;
