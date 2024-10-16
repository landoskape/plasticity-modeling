function [iaf,bweight,aweight] = step_corrD(iaf,inputRates)

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
    iaf.apicalWeight = iaf.apicalWeight + iaf.apicalPotentiation;

    iaf.apicalWeight = min(iaf.apicalWeight, iaf.maxApicalWeight);
    iaf.apicalWeight = max(iaf.apicalWeight, 0);
else
    iaf.spike = false; % not spiking!

    % Vm is current vm - leak + conductance*DF*Resistance/tau --
    % --- -which is effectively conductance*DF/capacitance (which is fine)
    leakTerm = (iaf.rest - iaf.vm) * iaf.dt/iaf.tau;
    excDF = iaf.exRev - iaf.vm;
    inhDF = iaf.rest - iaf.vm;
    excDV = (iaf.basalConductance + iaf.apicalConductance) * excDF * iaf.resistance * iaf.dt/iaf.tau;
    inhDV = iaf.gabaConductance * inhDF * iaf.resistance * iaf.dt/iaf.tau;
    iaf.vm = iaf.vm + leakTerm + excDV + inhDV;
end

% Generate Basal Input Conductances
basalRate = inputRates(iaf.basalTuneIdx);
basalSpikes = rand(size(iaf.basalWeight))<(basalRate * iaf.dt); % Randomly generate presynaptic spikes
cBasalConductance = sum(basalSpikes.*iaf.basalWeight.*(iaf.basalWeight>iaf.basalCondThresh), 1); % sum input conductance in each dendrite
dBasalConductance = cBasalConductance - iaf.basalConductance*iaf.dt/iaf.excTau; % delta basal conductance
iaf.basalConductance = iaf.basalConductance + dBasalConductance; % Update basal conductance

% Generate Apical Input Conductances
apicalRate = inputRates(iaf.apicalTuneIdx);
apicalSpikes = rand(size(iaf.apicalWeight))<(apicalRate * iaf.dt); % Randomly generate presynaptic spikes
cApicalConductance = sum(apicalSpikes.*iaf.apicalWeight.*(iaf.apicalWeight>iaf.apicalCondThresh), 1); % sum input conductance in each dendrite
dApicalConductance = cApicalConductance - iaf.apicalConductance*iaf.dt/iaf.excTau; % delta apical conductance
iaf.apicalConductance = iaf.apicalConductance + dApicalConductance; % Update apical AMPA

% Generate Inhibitory Conductances
inhPreSpikes = rand(iaf.numInhibitory,1) < (iaf.inhRate * iaf.dt);
inhConductance = iaf.inhWeight*sum(inhPreSpikes);
iaf.gabaConductance = iaf.gabaConductance + inhConductance - iaf.gabaConductance *iaf.dt/iaf.gabaTau;

% Do depression and replacement with basal inputs
iaf.basalWeight = iaf.basalWeight + basalSpikes .* iaf.basalDepression;
basalReplace = iaf.basalWeight < iaf.minBasalWeight;
iaf.basalTuneIdx(basalReplace) = randi(iaf.numInputs,sum(basalReplace),1);
iaf.basalWeight(basalReplace) = iaf.basalStartWeight;

% Do depression and replacement with apical inputs
iaf.apicalWeight = iaf.apicalWeight + apicalSpikes .* iaf.apicalDepression;
apicalReplace = iaf.apicalWeight < iaf.minApicalWeight;
iaf.apicalTuneIdx(apicalReplace) = randi(iaf.numInputs,sum(apicalReplace),1);
iaf.apicalWeight(apicalReplace) = iaf.apicalStartWeight;

% Update Potentiation/Depression Terms BASAL
iaf.basalPotentiation = iaf.basalPotentiation + ...
    iaf.basalPotValue * basalSpikes - iaf.basalPotentiation * iaf.dt/iaf.potTau;
iaf.basalDepression = iaf.basalDepression - iaf.basalDepression*iaf.dt/iaf.depTau;

% Update Potentiation/Depression Terms APICAL
iaf.apicalPotentiation = iaf.apicalPotentiation + ...
    iaf.apicalPotValue * apicalSpikes - iaf.apicalPotentiation * iaf.dt/iaf.potTau;
iaf.apicalDepression = iaf.apicalDepression - iaf.apicalDepression*iaf.dt/iaf.depTau;

% Do Homeostasis
iaf.homRateEstimate = iaf.homRateEstimate + iaf.dt/iaf.homTau * (1*iaf.spike/iaf.dt - iaf.homRateEstimate); 
iaf.homScale = iaf.homRate / max(0.1,iaf.homRateEstimate) - 1; % fraction change (positive or negative)
iaf.basalWeight = iaf.basalWeight + iaf.homScale * iaf.basalWeight * iaf.dt/iaf.homTau;
iaf.apicalWeight = iaf.apicalWeight + iaf.homScale * iaf.apicalWeight * iaf.dt/iaf.homTau;

% Prevent weights from leaving range
iaf.basalWeight = min(iaf.basalWeight, iaf.maxBasalWeight);
iaf.basalWeight = max(iaf.basalWeight, 0);
iaf.apicalWeight = min(iaf.apicalWeight, iaf.maxApicalWeight);
iaf.apicalWeight = max(iaf.apicalWeight, 0);

bweight = sum(iaf.basalWeight);
aweight = sum(iaf.apicalWeight);