function [iaf,basalSpikes,apicalSpikes] = stepFitModels(iaf,inputRates)
% step a fully fit model without plasticity

if iaf.vm > iaf.thresh
    iaf.spike = true; % spiking!
    iaf.vm = iaf.rest; % reset to rest
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