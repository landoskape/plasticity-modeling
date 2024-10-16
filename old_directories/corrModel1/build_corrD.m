function iaf = build_corrD(options)

iaf = struct();
iaf.dt = options.dt; % in seconds
iaf.T = options.T; % total time (use to reconstruct signals if necessary)
iaf.tau = 20e-3; % s
iaf.resistance = 100e6; % Ohm (without capacitance it's kind of weird but this is useful for scaling max synaptic weight allowed in intuitive values
iaf.rest = -70e-3; % V
iaf.thresh = -50e-3; % V
iaf.vm = iaf.rest; % initial membrane voltage

iaf.exRev = 0; % V
iaf.excTau = 20e-3; %s

% Specify synaptic structure
iaf.numBasal = options.numBasal;
iaf.numApical = options.numApical;

% Stimulus Structure
iaf.numInputs = options.numInputs;
iaf.numSignals = options.numSignals;
iaf.sourceMethod = options.sourceMethod;
iaf.sourceStrength = options.sourceStrength;
iaf.sourceLoading = options.sourceLoading;
iaf.varAdjustment = options.varAdjustment;
iaf.rateStd = options.rateStd;
iaf.rateMean = options.rateMean;

% ---------------------- BASAL -------------------------
% Specify Basal Weights
iaf.maxBasalWeight = options.maxBasalWeight;
iaf.minBasalWeight = iaf.maxBasalWeight * options.loseSynapseRatio;
iaf.basalStartWeight = iaf.maxBasalWeight * options.newSynapseRatio;
iaf.basalCondThresh = iaf.maxBasalWeight * options.conductanceThreshold;
iaf.basalWeight = iaf.maxBasalWeight .* rand(iaf.numBasal, 1);
iaf.basalConductance = 0; % store total basal conductance
iaf.basalTuneIdx = randi(iaf.numInputs,iaf.numBasal,1); % index of tuning for all basal!

% ---------------------- APICAL -------------------------
% Specify Apical Weights
iaf.maxApicalWeight = options.maxApicalWeight;
iaf.minApicalWeight = iaf.maxApicalWeight * options.loseSynapseRatio;
iaf.apicalStartWeight = iaf.maxApicalWeight * options.newSynapseRatio;
iaf.apicalCondThresh = iaf.maxApicalWeight * options.conductanceThreshold;
iaf.apicalWeight = iaf.maxApicalWeight .* rand(iaf.numApical, 1);
iaf.apicalConductance = 0; % store total apical conductance
iaf.apicalTuneIdx = randi(iaf.numInputs,iaf.numApical,1); %index of tuning for all apical

% STDP Stuffs
iaf.basalPotentiation = zeros(size(iaf.basalWeight)); % potentiation tracker required for every synapse
iaf.basalDepression = 0; % depression tracker is global
iaf.basalPotValue = options.plasticityRate * iaf.maxBasalWeight;
iaf.basalDepValue = options.plasticityRate * options.basalDepression * iaf.maxBasalWeight;

iaf.apicalPotentiation = zeros(size(iaf.apicalWeight));
iaf.apicalDepression = 0;
iaf.apicalPotValue = options.plasticityRate * iaf.maxApicalWeight;
iaf.apicalDepValue = options.plasticityRate * options.apicalDepression * iaf.maxApicalWeight;

iaf.potTau = 0.02;
iaf.depTau = 0.02;
iaf.homTau = options.homeostasisTau;
iaf.homRate = options.homeostasisRate;
iaf.homRateEstimate = iaf.homRate; % start at homRate so as to avoid blowups 

% Make Some Stupid Inhibitory Synapses
% -- note, should make this slow recurrent inhibition from excitatory input eventually --
iaf.numInhibitory = 200;
iaf.inhRate = 20;
iaf.inhWeight = 100e-12;
iaf.gabaTau = 20e-3;
iaf.gabaConductance = 0;
