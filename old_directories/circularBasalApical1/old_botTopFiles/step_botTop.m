function [iaf,stimulus] = step_botTop(iaf,stimulus)

if iaf.vm > iaf.thresh
    iaf.spike = true; % spiking!
    iaf.vm = iaf.rest; % reset to rest

    % Plasticity on Basal Synapses
    iaf.basalDepression = iaf.basalDepression - iaf.basalDepValue; % Update Depression Term
    iaf.basalWeight = iaf.basalWeight + iaf.basalPotentiation; % Potentiate AMPA Components

    iaf.basalWeight = min(iaf.basalWeight, iaf.maxBasalWeight);
    iaf.basalWeight = max(iaf.basalWeight, 0);

    % Plasticity on Apical Synapses
    iaf.apicalDepression = iaf.apicalDepression - iaf.apicalDepValue;

    % Implement Arc Redistribution
    if strcmp(iaf.arcMethod,'full')
        % Compute actual potentiation (ignore potentiation above maximum)
        actualPotentiation = min(iaf.apicalWeight+iaf.apicalPotentiation, iaf.maxApicalWeight) - iaf.apicalWeight;

        [sortPotentiation,idx] = sort(actualPotentiation); % Sort potentiation to get index, make distribution easy
        [~,inverseIdx] = sort(idx); % get reverse index
        arcRedistribution = -flipud(sortPotentiation); % account for potentiation
        arcDepression = arcRedistribution(inverseIdx) / 10; % index back to correct order
        iaf.apicalWeight = iaf.apicalWeight + actualPotentiation + arcDepression; % implement
    elseif strcmp(iaf.arcMethod,'buffer')
        % the idea here would be to use a homeostatic type Arc "buffer",
        % where the dendrite would have a constant level of weight, and
        % depression potentiation would move weight to/from the buffer.
        % Over time, the distribution of weight in the synapses and in the
        % buffer may change - but it could never exceed it.
        % -
	% One way to implement would be "instant", where weight is taken
        % from the buffer to potentiate, placed in buffer during
        % depression, and remaining weight instantly moved after updates.
    else
        iaf.apicalWeight = iaf.apicalWeight + iaf.apicalPotentiation;
    end

    iaf.apicalWeight = min(iaf.apicalWeight, iaf.maxApicalWeight);
    iaf.apicalWeight = max(iaf.apicalWeight, 0);
else
    iaf.spike = false; % not spiking!

    % Vm is current vm - leak + conductance*DF*Resistance/tau --
    % --- -which is effectively conductance*DF/capacitance (which is fine)
    leakTerm = (iaf.rest - iaf.vm) * iaf.dt/iaf.tau;
    excDF = iaf.exRev - iaf.vm;
    inhDF = iaf.rest - iaf.vm;
    basalConductance = iaf.basalAMPA + iaf.basalNMDA;
    apicalConductance = iaf.allowApicAMPA * iaf.apicalAMPA + iaf.apicalNMDA;
    totalExcConductance = sum(basalConductance) + iaf.apicalAttenuation*sum(apicalConductance);
    excDV = totalExcConductance * excDF * iaf.resistance * iaf.dt/iaf.tau;
    inhDV = iaf.gabaConductance * inhDF * iaf.resistance * iaf.dt/iaf.tau;
    iaf.vm = iaf.vm + leakTerm + excDV + inhDV;
end

% Update Stimulus
basalChangeValueProb = rand();
stimulus.bvalue = stimulus.bvalue*(basalChangeValueProb > stimulus.basalChangeProb) + ...
    (2*pi*rand()-pi)*(basalChangeValueProb<=stimulus.basalChangeProb);
apicalChangeValueProb = rand();
stimulus.avalue = stimulus.avalue*(apicalChangeValueProb > stimulus.apicalChangeProb) + ...
    randi(stimulus.numDims)*(apicalChangeValueProb <= stimulus.apicalChangeProb);

% Generate Basal Input Conductances
basalRate = iaf.getRateBasal(stimulus.bvalue);
basalSpikes = rand(size(iaf.basalWeight))<(basalRate * iaf.dt); % Randomly generate presynaptic spikes
basalConductance = sum(basalSpikes.*iaf.basalWeight.*(iaf.basalWeight>iaf.basalCondThresh), 1); % sum input conductance in each dendrite

iaf.basalAMPA = iaf.basalAMPA + basalConductance - iaf.basalAMPA * iaf.dt/iaf.ampaTau; % Update basal AMPA
iaf.basalSpiking = iaf.basalAMPA > iaf.basalThreshold; % Find spiking basals
iaf.basalAMPA(iaf.basalSpiking) = 0; % Reset ampa conductance
iaf.basalNMDA = iaf.basalNMDA + iaf.basalNWeight.*iaf.basalSpiking - iaf.basalNMDA*iaf.dt/iaf.nmdaTau; % move to NMDA

% Generate Apical Input Conductances
apicalRate = iaf.getRateApical(stimulus.avalue);
apicalSpikes = rand(size(iaf.apicalWeight))<(apicalRate * iaf.dt); % Randomly generate presynaptic spikes
apicalConductance = sum(apicalSpikes.*iaf.apicalWeight.*(iaf.apicalWeight>iaf.apicalCondThresh), 1); % sum input conductance in each dendrite

iaf.apicalAMPA = iaf.apicalAMPA + apicalConductance - iaf.apicalAMPA * iaf.dt/iaf.ampaTau; % Update apical AMPA
iaf.apicalSpiking = iaf.apicalAMPA > iaf.apicalThreshold; % Find spiking apicals
iaf.apicalAMPA(iaf.apicalSpiking) = 0; % Reset ampa conductance
iaf.apicalNMDA = iaf.apicalNMDA + iaf.apicalNWeight.*iaf.apicalSpiking - iaf.apicalNMDA*iaf.dt/iaf.nmdaTau; % move to NMDA

% Generate Inhibitory Conductances
inhPreSpikes = rand(iaf.numInhibitory,1) < (iaf.inhRate * iaf.dt);
inhConductance = iaf.inhWeight*sum(inhPreSpikes);
iaf.gabaConductance = iaf.gabaConductance + inhConductance - iaf.gabaConductance *iaf.dt/iaf.gabaTau;

% Do depression with active inputs
iaf.basalWeight = iaf.basalWeight + basalSpikes .* iaf.basalDepression;
replaceSynapse = iaf.basalWeight < iaf.minBasalWeight;
numReplace(1) = sum(replaceSynapse,'all');
iaf.tuningCenter(replaceSynapse) = 2*pi*rand(numReplace,1) - pi;
iaf.basalWeight(replaceSynapse) = iaf.basalStartWeight;

% Depression with apical
iaf.apicalWeight = iaf.apicalWeight + apicalSpikes .* iaf.apicalDepression;
replaceSynapse = iaf.apicalWeight < iaf.minApicalWeight;
numReplace(2) = sum(replaceSynapse,'all');
iaf.apicalTuning(replaceSynapse) = randi(iaf.apicalDims,sum(replaceSynapse,'all'),1);
iaf.apicalWeight(replaceSynapse) = iaf.apicalStartWeight;

% Update Potentiation/Depression Terms BASAL
iaf.basalPotentiation = iaf.basalPotentiation + ...
    iaf.basalPotValue * basalSpikes - iaf.basalPotentiation * iaf.dt/iaf.potTau;
iaf.basalDepression = iaf.basalDepression - iaf.basalDepression*iaf.dt/iaf.depTau;

% Update Potentiation/Depression Terms APICAL
iaf.apicalPotentiation = iaf.apicalPotentiation + ...
    iaf.apicalPotValue * apicalSpikes - iaf.apicalPotentiation * iaf.dt/iaf.potTau;
iaf.apicalDepression = iaf.apicalDepression - iaf.apicalDepression*iaf.dt/iaf.depTau;

% Do Homeostasis
iaf.basalWeight = iaf.basalWeight - iaf.basalHomeostasis * iaf.dt/iaf.homTau;
iaf.apicalWeight = iaf.apicalWeight - iaf.apicalHomeostasis * iaf.dt/iaf.homTau;

% Prevent weights from leaving range
iaf.basalWeight = min(iaf.basalWeight, iaf.maxBasalWeight);
iaf.basalWeight = max(iaf.basalWeight, 0);


%{
% ---------------------- APICAL -------------------------
% Specify Apical Weights - only let spikes get to soma! (to start with)
totalApicalSynapses = iaf.numApical * iaf.numSynapses;
iaf.propApicalNecessary = options.propApicalNecessary;
iaf.maxApicalWeight = 0.5*(iaf.thresh-iaf.rest)/... only let apicals depolarize to half threshold (without spiking)
    ((iaf.exRev-iaf.rest) * iaf.resistance * (totalApicalSynapses*iaf.propApicalNecessary));
iaf.minApicalWeight = iaf.maxApicalWeight * options.loseSynapseRatio;
iaf.startWeight = iaf.maxApicalWeight * options.newSynapseRatio;
iaf.apicalWeight = iaf.maxApicalWeight .* rand(iaf.numSynapses, iaf.numApical);
iaf.apicalAMPA = zeros(1,iaf.numApical);

% Specify Apical NMDARs
iaf.apicalThreshold = iaf.maxApicalWeight * iaf.numSynapses / 2;
iaf.apicalNWeight = iaf.maxApicalWeight * iaf.numSynapses;
iaf.apicalNMDA = zeros(1,iaf.numapical);

% Basal Tunings
iaf.apicalTuning = randi(options.numDims,size(iaf.apicalWeight));
iaf.getRateApical = @(dimStim) iaf.baseApical + iaf.driveApical * (iaf.apicalTuning==dimStim);

% STDP Stuffs
iaf.apicalPotentiation = zeros(size(iaf.apicalWeight));
iaf.apicalDepression = 0;
iaf.apicalPotValue = 0.01 * iaf.maxApicalWeight;
iaf.apicalDepValue = 0.01 * options.apicalDepression * iaf.maxApicalWeight;
%}