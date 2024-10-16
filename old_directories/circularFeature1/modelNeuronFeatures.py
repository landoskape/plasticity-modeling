# modelNeuron.m
#
# This function defines a model neuron with STDP, as described in
# Song, et. al., 2000.  
# 
# The model neuron is a structure which we call 'neuron' inside this
# function. Note that when you call this function your structure will be
# called whatever you set the output of the function as.  For example, if
# you call:
#   
#         aNeuron = modelNeuron() 
# 
# your structure will be called aNeuron. 
# 
#
# This structure has many fields.  In this function we set the default
# values for these fields in this way: 
# 
#         neuron.Vm = -58;    % Membrane voltage, mV
#
# This sets the 'default' membrane voltage to -58mV. 
# 
# Note: you should not change the default values for a field in this
# document. Instead you should first create a default model neuron by
# calling aNeuron = modelNeuron(). Then in a separate script you should
# change the values of your example neuron with code that looks like this: 

import numpy as np
from scipy.special import jv as besseli


class createNeuron(tuningSharpness):
    def __init__(self):
        self.setVariables()
        self.setProperties()
        self.createSynapses(tuningSharpness)
        
    def setVariables(self):
        self.vm = -60
        self.gEx = 0
        self.gIn = 0
        self.spike = False
        
    def setProperties(self):
        self.tauM    =  .020    # Membrane time constant, sec
        self.Vrest   =   -70    # Resting membrane voltage, mV
        self.Eex     =     0    # Excitatory reversal potential, mV
        self.Ein     =   -70    # Inhibitory reversal potential, mV
        self.tauEx   =  .005    # Time constant of excitatory conductances
        self.tauIn   =  .005    # Time constant of inhibitory conductances
        self.Vthresh =   -55    # Spike threshold voltage, mV
        self.Vreset  =   -60    # Post-spike reset voltage, mV

    def createSynapses(self,tuningSharpness):
        self.Nex      = 1200 # Number of excitatory synapses
        self.exDrive  = 40 # Maximum rate of stimulus driven pre-synaptic APs
        self.exBaseRate = 5 # Baseline pre-synaptic AP Rate
        self.exGMax = .02 # Maximum peak excitatory conductance
        self.ex_gA    = self.exGMax * np.random.uniform(0,1,self.Nex)
        self.exHLim = 3 # homeostatic limit of total conductance
        self.exHTau = np.inf
        
        self.exTuneSharpness = tuningSharpness
        dRad = 2*np.pi / self.Nex
        self.exTuningCenter = np.arange(dRad,2*np.pi+dRad,dRad) - np.pi
        
        vonmises = lambda x,u: np.exp(tuningSharpness * np.cos(x-u))/(2*np.pi*besseli(0,tuningSharpness))
        self.getExRate = lambda stimValue: self.exBaseRate + self.exDrive*vonmises(stimValue, self.exTuningCenter)
        
        self.exPreSpike = np.zeros(self.Nex)
        self.exPa = np.zeros(self.Nex)
        self.exM = 0
        self.exAPlus = 0.005
        self.exAMinus = 1.05*0.005
        self.exTauPlus = 0.02
        self.exTauMinus = 0.02

        # M and Pa are house keeping variables for implementing STDP
        # rule.  Both decay exponentially toward zero.
        #    Pa is positive, used to increase the strength of
        #       synapses, and is incremented on pre-synaptic spiking.
        #       On postsynaptic spiking, gA -> gA + P*gMax
        #    M is negative, used to decrease the strength of synapses,
        #       and is decremented on post-synaptic spiking.
        #       On presynaptic spiking,  gA -> gA + M*gMax
        
        self.Nin = 200
        self.inRate = 10
        self.inPreSpike = np.zeros(self.Nin)
        self.in_gA = 0.05 * np.ones(self.Nin)
        
        

def stepTime(modelNeuron,dT):
    if modelNeuron.vm > modelNeuron.Vthresh:
        modelNeuron.spike = 1
        modelNeuron.vm = modelNeuron.Vreset
        
        modelNeuron.exM = modelNeuron.exM - modelNeuron.exAm # Update learning rule M
        
        # Update conductances
        modelNeuron.ex_gA = modelNeuron.ex_gA + np.multiply(modelNeuron.exPa,modelNeuron.exGMax)

        # Don't allow conductances out of the range [0,gMax]
        modelNeuron.ex_gA = modelNeuron.ex_gA - \
            np.multiply((modelNeuron.ex_gA > modelNeuron.exGMax),(modelNeuron.ex_gA - modelNeuron.exGMax))
        modelNeuron.ex_gA = modelNeuron.ex_gA - \
            np.multiply((modelNeuron.ex_gA < 0),modelNeuron.ex_gA)
        
    else: # If it doesn't spike...
        modelNeuron.spike = 0
        
        # Update voltage based on conductance
        dV = (dT/modelNeuron.tauM)*(modelNeuron.Vrest - modelNeuron.vm + \
                                    modelNeuron.gEx*(modelNeuron.Eex - modelNeuron.vm) + \
                                    modelNeuron.gIn*(modelNeuron.Ein - modelNeuron.vm) )
        modelNeuron.vm = modelNeuron.vm + dV

    # Allow conductances to decay exponentially
    dgEx = -modelNeuron.gEx*dT/modelNeuron.tauEx
    dgIn = -modelNeuron.gIn*dT/modelNeuron.tauIn
    modelNeuron.gEx = modelNeuron.gEx + dgEx
    modelNeuron.gIn = modelNeuron.gIn + dgIn

    # Generate Poisson presynaptic spikes, 1 for spike, 0 for none
    modelNeuron.exPreSpike = (np.random.uniform(0,1,(modelNeuron.Nex,1)) < dT*modelNeuron.exRate)
    modelNeuron.inPreSpike = (np.random.uniform(0,1,(modelNeuron.Nin,1)) < dT*modelNeuron.inRate)

    # Presynaptic spikes generate conductances in the post-synaptic cell
    exCond = np.multiply(modelNeuron.exPreSpike, modelNeuron.ex_gA)
    inCond = np.multiply(modelNeuron.inPreSpike, modelNeuron.in_gA)
    modelNeuron.gEx = modelNeuron.gEx + np.sum(exCond)
    modelNeuron.gIn = modelNeuron.gIn + np.sum(inCond)

    #% Update learning rule: Pa increases conductances on post-synaptic spiking, and is incremented on pre-synaptic spiking.
    modelNeuron.exPa = modelNeuron.exPa + np.multiply(modelNeuron.exPreSpike, modelNeuron.exAp)

    # Update the conductances as a result of the learning rule applied to pre-synaptic spikes.
    modelNeuron.ex_gA = modelNeuron.ex_gA + np.multiply(modelNeuron.exPreSpike,  modelNeuron.exM) * modelNeuron.exGMax
    
    # Don't allow conductances out of the range [0,gMax]
    modelNeuron.ex_gA = modelNeuron.ex_gA - \
        np.multiply((modelNeuron.ex_gA > modelNeuron.exGMax),(modelNeuron.ex_gA - modelNeuron.exGMax))
    modelNeuron.ex_gA = modelNeuron.ex_gA - \
        np.multiply((modelNeuron.ex_gA < 0),modelNeuron.ex_gA)

    # The learning rule functions M and Pa decay exponentially
    dM = -modelNeuron.exM*dT/modelNeuron.exTauM
    modelNeuron.exM = modelNeuron.exM + dM;
    dPa = -modelNeuron.exPa * dT/modelNeuron.exTauP
    modelNeuron.exPa = modelNeuron.exPa + dPa;
    
    return modelNeuron

