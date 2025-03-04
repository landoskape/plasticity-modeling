function iaf = stepIafCM(iaf,inputRates)

if iaf.vm > iaf.thresh
    iaf.spike = true; % spiking!
    iaf.vm = iaf.reset; % reset to rest
    
    % Plasticity on Basal Synapses
    iaf.depEligibility = iaf.depEligibility - iaf.depIncrement; % Update Depression Term 
    iaf.ampaWeights = iaf.ampaWeights + iaf.potEligibility; % Potentiate Synaptic Weights
    
    iaf.ampaWeights = min(iaf.ampaWeights, iaf.maxWeight);
    iaf.ampaWeights = max(iaf.ampaWeights, 0);
    
else
    iaf.spike = false; % not spiking!
    
    % Vm is current vm - leak + conductance*DF*Resistance/tau --
    % --- -which is effectively conductance*DF/capacitance (which is fine)
    leakTerm = (iaf.rest - iaf.vm) * iaf.dt/iaf.tau;
    conductance = sum(iaf.ampaConductance + iaf.nmdaConductance);
    excDV = conductance * (iaf.exRev - iaf.vm) * iaf.dt/iaf.tau;
    inhDV = iaf.gabaConductance * (iaf.rest - iaf.vm) * iaf.dt/iaf.tau;
    iaf.vm = iaf.vm + leakTerm + excDV + inhDV;
end

% Generate Basal Input Conductances
preSpikes = rand(size(inputRates))<(inputRates * iaf.dt);
% sum input conductance in each dendrite
conductance = sum(preSpikes.*iaf.ampaWeights.*(iaf.ampaWeights>iaf.conductanceThreshold), 1); 
% Update ampa conductance in each dendrite
iaf.ampaConductance = iaf.ampaConductance + conductance - iaf.ampaConductance * iaf.dt/iaf.ampaTau; 
iaf.dendriteSpiking = iaf.ampaConductance > iaf.nmdaThreshold; % Find spiking dendrites
iaf.ampaConductance(iaf.dendriteSpiking) = 0; % Reset ampa conductance in spiking dendrites
iaf.nmdaConductance = iaf.nmdaConductance + iaf.nmdaWeights.*iaf.dendriteSpiking - iaf.nmdaConductance*iaf.dt/iaf.nmdaTau; % move to NMDA

% Generate Inhibitory Conductances
inhPreSpikes = rand(iaf.numInhibitory,1) < (iaf.inhRate * iaf.dt);
inhConductance = iaf.inhWeight*sum(inhPreSpikes);
iaf.gabaConductance = iaf.gabaConductance + inhConductance - iaf.gabaConductance *iaf.dt/iaf.gabaTau;

% Do depression with active inputs
iaf.ampaWeights = iaf.ampaWeights + preSpikes .* iaf.depEligibility;

% Update Potentiation/Depression Terms BASAL
iaf.potEligibility = iaf.potEligibility + ...
    iaf.potIncrement * preSpikes - iaf.potEligibility * iaf.dt/iaf.potTau; 
iaf.depEligibility = iaf.depEligibility - iaf.depEligibility*iaf.dt/iaf.depTau; 

% Do Homeostasis
totalWeight = sum(iaf.ampaWeights,'all');
scalingFactor = iaf.homLimit - totalWeight;
iaf.ampaWeights = iaf.ampaWeights + iaf.dt/iaf.homTau * scalingFactor * iaf.ampaWeights;

% Prevent weights from leaving range
iaf.ampaWeights = min(iaf.ampaWeights, iaf.maxWeight);
iaf.ampaWeights = max(iaf.ampaWeights, 0);






















































