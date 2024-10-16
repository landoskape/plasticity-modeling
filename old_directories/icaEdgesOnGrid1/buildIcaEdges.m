function iaf = buildIafOriPos(options)

iaf = struct();
iaf.dt = options.dt; % s
iaf.tau = 20e-3; % s
iaf.resistance = 100e6; % Ohm
iaf.rest = -70e-3; % V
iaf.thresh = -54e-3; % V
iaf.reset = -60e-3; 
iaf.vm = iaf.rest;

iaf.exRev = 0; % V
iaf.ampaTau = 5e-3; %s
iaf.nmdaTau = 20e-3; %s

% Specify synaptic structure
iaf.numActive = options.numActive;
iaf.numSilent = options.numSilent;
iaf.numDendrites = options.numActive + options.numSilent;
iaf.numSynapses = options.numSynapses;

% -- setup synaptic weights --
iaf.baseRate = 5; 
iaf.driveRate = 25; 
iaf.maxWeight = options.maxWeight; 
iaf.minWeight = iaf.maxWeight * options.loseSynapseRatio; 
iaf.startWeight = iaf.maxWeight * options.newSynapseRatio;
iaf.conductanceThreshold = options.conductanceThreshold * iaf.maxWeight;

iaf.numOrientation = options.numOrientation;
iaf.numPosition = options.gridLength^2;
iaf.numInputs = iaf.numOrientation * iaf.numPosition; 

iaf.inputConnection = randi(iaf.numInputs, iaf.numSynapses, iaf.numDendrites);
iaf.ampaWeights = iaf.maxWeight .* rand(iaf.numSynapses, iaf.numDendrites);
iaf.ampaConductance = zeros(1,iaf.numDendrites);

iaf.nmdaThreshold = iaf.maxWeight * iaf.numSynapses * options.propNecessaryNMDA;
iaf.nmdaMaxWeight = iaf.maxWeight * iaf.numSynapses;
iaf.nmdaWeights = zeros(1,iaf.numDendrites);
iaf.nmdaConductance = zeros(1,iaf.numDendrites);

% -- setup STDP --
iaf.potEligibility = zeros(size(iaf.ampaWeights));
iaf.depEligibility = zeros(1, iaf.numDendrites);
iaf.potIncrement = iaf.maxWeight * options.plasticityRate(1);
activeDepIncrement = iaf.maxWeight * options.plasticityRate(1) * options.depValue(1);
silentDepIncrement = iaf.maxWeight * options.plasticityRate(1) * options.depValue(2);
iaf.depIncrement = [activeDepIncrement * ones(1,iaf.numActive), silentDepIncrement * ones(1,iaf.numSilent)];
iaf.potTau = options.potentiationTau;
iaf.depTau = options.depressionTau;
iaf.homTau = options.homeostasisTau;
iaf.homRate = options.homeostasisRate;
iaf.homRateEstimate = iaf.homRate; % start at homRate so as to avoid blowups 

% Make Some Stupid Inhibitory Synapses
% -- note, will make this slow recurrent from excitatory input --
iaf.numInhibitory = 200;
iaf.inhRate = 10;
iaf.inhWeight = options.inhWeight; % Just make it in pSiemens
iaf.gabaTau = 5e-3;
iaf.gabaConductance = 0;

end

