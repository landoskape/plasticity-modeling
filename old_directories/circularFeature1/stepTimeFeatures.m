function [neuron,stimulus] = stepTimeFeatures(neuron, stimulus, dT)

% Determine if the neuron will spike, or not
if (neuron.Vm > neuron.Vthresh)
    neuron.spike = true;        % Note there's a spike
    % Reset Vm to the reset voltage
    neuron.Vm = neuron.Vreset;
    
    % Update learning rule M
    neuron.exSynapses.M = neuron.exSynapses.M - neuron.exSynapses.Aminus;
    
    % Update synaptic conductances as a result of the learning rule applied
    % to post-synaptic spikes.
    neuron.exSynapses.gA = neuron.exSynapses.gA + neuron.exSynapses.Pa.*neuron.exSynapses.gMax;
    
    % Don't allow conductances out of the range [0,gMax]
    neuron.exSynapses.gA = neuron.exSynapses.gA - ...
        (neuron.exSynapses.gA > neuron.exSynapses.gMax).*...
        (neuron.exSynapses.gA - neuron.exSynapses.gMax);
    neuron.exSynapses.gA = neuron.exSynapses.gA - ...
        (neuron.exSynapses.gA < 0).*...
        (neuron.exSynapses.gA);
    
else % If it doesn't spike...
    neuron.spike = false;       % Note there's no spike
    % Update membrane voltage based on conductances
    dV = (dT/neuron.tauM)*(neuron.Vrest - neuron.Vm ...
        + neuron.gEx*(neuron.Eex - neuron.Vm) ...
        + neuron.gIn*(neuron.Ein - neuron.Vm) );
    neuron.Vm = neuron.Vm + dV;
end

% Conductances to decay exponentially
dgEx = -neuron.gEx*dT/neuron.tauEx;
dgIn = -neuron.gIn*dT/neuron.tauIn;
neuron.gEx = neuron.gEx + dgEx;
neuron.gIn = neuron.gIn + dgIn;

% Update Stimulus
stimulus = changeStimulus(stimulus,dT);

% Generate excitatory presynaptic spikes, 1 for spike, 0 for none
stimulusDependentRate = neuron.exSynapses.exRate(stimulus.value); % set in "modelNeuronFeatures()"
neuron.exSynapses.preSynapticSpike = (rand(neuron.Nex,1) < dT*stimulusDependentRate(:));

% Generate inhibitory presynaptic spikes, 1 for spike, 0 for none
neuron.inSynapses.preSynapticSpike = (rand(neuron.Nin,1) < dT*neuron.inSynapses.rate);

% Presynaptic spikes generate conductances in the post-synaptic cell
exCond = neuron.exSynapses.preSynapticSpike.*neuron.exSynapses.gA;
inCond = neuron.inSynapses.preSynapticSpike.*neuron.inSynapses.gA;
neuron.gEx = neuron.gEx + sum(exCond);
neuron.gIn = neuron.gIn + sum(inCond);

% Update learning rule: Pa increases conductances on post-synaptic spiking,
% and is incremented on pre-synaptic spiking. 
neuron.exSynapses.Pa = neuron.exSynapses.Pa +...
    neuron.exSynapses.preSynapticSpike.*neuron.exSynapses.Aplus;

% Update the conductances as a result of the learning rule
% applied to pre-synaptic spikes.
neuron.exSynapses.gA = neuron.exSynapses.gA + ...
    neuron.exSynapses.preSynapticSpike.*neuron.exSynapses.M*neuron.exSynapses.gMax;

% Don't allow conductances out of the range [0,gMax]
neuron.exSynapses.gA = neuron.exSynapses.gA - ...
    (neuron.exSynapses.gA > neuron.exSynapses.gMax).*...
    (neuron.exSynapses.gA - neuron.exSynapses.gMax);
neuron.exSynapses.gA = neuron.exSynapses.gA - ...
    (neuron.exSynapses.gA < 0).*...
    (neuron.exSynapses.gA);

% The learning rule functions M and Pa decay exponentially
dM = -neuron.exSynapses.M*dT/neuron.exSynapses.tauMinus;
neuron.exSynapses.M = neuron.exSynapses.M + dM;
dPa = -neuron.exSynapses.Pa.*dT/neuron.exSynapses.tauPlus;
neuron.exSynapses.Pa = neuron.exSynapses.Pa + dPa;

scaleFactor = sum(neuron.exSynapses.gA)-neuron.exSynapses.hLimit;
dHomeostatic = -(dT/neuron.exSynapses.hTau)*scaleFactor*neuron.exSynapses.gA;
neuron.exSynapses.gA = neuron.exSynapses.gA + dHomeostatic;

end % End stepTime()