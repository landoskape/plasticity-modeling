function iaf = build_botTop(options)

iaf = struct();
iaf.dt = options.dt; % s
iaf.tau = 20e-3; % s
iaf.resistance = 100e6; % Ohm
iaf.rest = -70e-3; % V
iaf.thresh = -50e-3; % V
iaf.vm = iaf.rest;

iaf.exRev = 0; % V
iaf.ampaTau = 20e-3; %s
iaf.nmdaTau = 20e-3; %s
iaf.driveBasal = 45; % HZ
iaf.baseBasal = 5; % HZ (baseline rate of firing)
iaf.driveApical = 45;
iaf.baseApical = 5;
iaf.apicalAttenuation = options.apicalAttenuation;
iaf.arcMethod = options.arcMethod;

% Specify synaptic structure
iaf.numBasal = options.numBasal;
iaf.numApical = options.numApical;
iaf.numSynapses = options.numSynapses;
iaf.apicalDims = options.numDims;

% ---------------------- BASAL -------------------------
% Specify Basal Weights
totalBasalSynapses = iaf.numBasal * iaf.numSynapses;
iaf.propBasalNecessary = options.propBasalNecessary; % How many necessary to generate spike?
iaf.maxBasalWeight = (iaf.thresh-iaf.rest)/...
    ((iaf.exRev-iaf.rest) * iaf.resistance * (totalBasalSynapses*iaf.propBasalNecessary));
iaf.minBasalWeight = iaf.maxBasalWeight * options.loseSynapseRatio;
iaf.basalStartWeight = iaf.maxBasalWeight * options.newSynapseRatio;
iaf.basalCondThresh = options.conductanceThreshold * iaf.maxBasalWeight;
iaf.basalWeight = iaf.maxBasalWeight .* rand(iaf.numSynapses, iaf.numBasal);
iaf.basalAMPA = zeros(1,iaf.numBasal);


% Specify Basal NMDARs
iaf.basalThreshold = iaf.maxBasalWeight * iaf.numSynapses / 2;
iaf.basalNWeight = iaf.maxBasalWeight * iaf.numSynapses;
iaf.basalNMDA = zeros(1,iaf.numBasal);

% Basal Tunings
iaf.tuningCenter = 2*pi*rand(size(iaf.basalWeight)) - pi;
iaf.tuningSharpness = options.tuningSharpness;
iaf.getRateBasal = @(stim) iaf.baseBasal + iaf.driveBasal * ...
    exp(options.tuningSharpness.*cos(stim - iaf.tuningCenter))./(2*pi*besseli(0,options.tuningSharpness));

% STDP Stuffs
iaf.basalPotentiation = zeros(size(iaf.basalWeight));
iaf.basalDepression = 0;
iaf.basalPotValue = options.plasticityRate(1) * iaf.maxBasalWeight;
iaf.basalDepValue = options.plasticityRate(1) * 1.1 * iaf.maxBasalWeight;
iaf.basalHomeostasis = 0.1 * iaf.basalDepValue;
iaf.potTau = 0.02;
iaf.depTau = 0.02;
iaf.homTau = 20;

% ---------------------- APICAL -------------------------
% Specify Apical Weights - only let spikes get to soma! (to start with)
totalApicalSynapses = iaf.numApical * iaf.numSynapses;
% only let apicals depolarize to some amount of threshold
iaf.maxApicalWeight = options.apicalMaxPercThreshold*(iaf.thresh-iaf.rest)/...
    ((iaf.exRev-iaf.rest) * iaf.resistance * totalApicalSynapses);
iaf.minApicalWeight = iaf.maxApicalWeight * options.loseSynapseRatio;
iaf.apicalStartWeight = iaf.maxApicalWeight * options.newSynapseRatio;
iaf.apicalCondThresh = options.conductanceThreshold * iaf.maxApicalWeight;
iaf.apicalWeight = iaf.maxApicalWeight .* rand(iaf.numSynapses, iaf.numApical);
iaf.apicalAMPA = zeros(1,iaf.numApical);
iaf.allowApicAMPA = options.allowApicalAMPA;

% Specify Apical NMDARs
iaf.apicalThreshold = iaf.maxApicalWeight * iaf.numSynapses * options.apicalThresholdFraction;
iaf.apicalNWeight = iaf.maxApicalWeight * iaf.numSynapses;
iaf.apicalNMDA = zeros(1,iaf.numApical);

% Apical Tunings
iaf.apicalTuning = randi(options.numDims,size(iaf.apicalWeight));
iaf.getRateApical = @(dimStim) iaf.baseApical + iaf.driveApical * (iaf.apicalTuning==dimStim);

% STDP Stuffs
iaf.apicalPotentiation = zeros(size(iaf.apicalWeight));
iaf.apicalDepression = 0;
iaf.apicalPotValue = options.plasticityRate(2) * iaf.maxApicalWeight;
iaf.apicalDepValue = options.plasticityRate(2) * options.apicalDepression * iaf.maxApicalWeight;
iaf.apicalHomeostasis = 0.1 * iaf.apicalDepValue;

% Make Some Stupid Inhibitory Synapses
% -- note, will make this slow recurrent from excitatory input --
iaf.numInhibitory = 200;
iaf.inhRate = 20;
iaf.inhWeight = iaf.maxBasalWeight * 2;
iaf.gabaTau = 20e-3;
iaf.gabaConductance = 0;
