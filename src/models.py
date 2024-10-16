from copy import copy
import numpy as np


class IaF:
    """
    This is a basic integrate and fire neuron for running conductance models of STDP.
    """

    def __init__(
        self,
        num_inputs,
        basal_depression_ratio,
        apical_depression_ratio,
        num_basal=300,
        num_apical=100,
    ):
        # basic parameters
        self.dt = 0.001  # s
        self.tau = 20e-3  # s
        self.r_m = 100e6  # Ohm
        self.v_rest = -70e-3  # Volts
        self.v_thresh = -50e-3  # Volts
        self.exc_rev = 0e-3  # Volts
        self.exc_tau = 20e-3  # seconds
        self.vm = copy(self.v_rest)  # initial membrane potential (Volts)

        self.lose_synapse_ratio = 0.01
        self.new_synapse_ratio = 0.01
        self.cond_threshold = 0.1

        # NOTE had a note saying that I'll "make this slow from recurrent excitatory"
        self.num_inhibitory = 0
        self.inh_rate = 20
        self.inh_weight = 200e-12  # Amps
        self.inh_tau = 20e-3
        self.inh_conductance = 0

        self.num_inputs = num_inputs

        # basal synaptic structure
        self.num_basal = num_basal
        self.max_basal_weight = 300e-12  # pS
        self.min_basal_weight = self.max_basal_weight * self.lose_synapse_ratio
        self.basal_start_weight = self.max_basal_weight * self.new_synapse_ratio
        self.basal_cond_threshold = self.max_basal_weight * self.cond_threshold
        self.basal_weights = self.max_basal_weight * np.random.rand(self.num_basal)
        self.basal_conductance = 0
        self.basal_tune_index = np.random.randint(0, self.num_inputs, self.num_basal)

        self.num_apical = num_apical
        self.max_apical_weight = 100e-12  # pS
        self.min_apical_weight = self.max_apical_weight * self.lose_synapse_ratio
        self.apical_start_weight = self.max_apical_weight * self.new_synapse_ratio
        self.apical_cond_threshold = self.max_apical_weight * self.cond_threshold
        self.apical_weights = self.max_apical_weight * np.random.rand(self.num_apical)
        self.apical_conductance = 0
        self.apical_tune_index = np.random.randint(0, self.num_inputs, self.num_apical)

        # STDP Parameters
        self.plasticity_rate = 0.01
        self.potentiation_tau = 0.02  # ms
        self.depression_tau = 0.02  # ms
        self.basal_depression_ratio = basal_depression_ratio
        self.apical_depression_ratio = apical_depression_ratio

        self.basal_potentiation = np.zeros(num_basal)
        self.basal_depression = 0
        self.basal_pot_value = self.plasticity_rate * self.max_basal_weight
        self.basal_dep_value = self.plasticity_rate * self.max_basal_weight * self.basal_depression_ratio

        self.apical_potentiation = np.zeros(num_apical)
        self.apical_depression = 0
        self.apical_pot_value = self.plasticity_rate * self.max_apical_weight
        self.apical_dep_value = self.plasticity_rate * self.max_apical_weight * self.apical_depression_ratio

        # Homeostasis Parameters
        self.homeo_tau = 20  # seconds
        self.homeo_rate = 20  # spikes / second
        self.homeo_rate_estimate = copy(self.homeo_rate)

    def step(self, input):
        if self.vm > self.v_thresh:
            self.spike = True
            self.vm = self.v_rest  # reset to rest

            # Implement plasticity on basal synapses
            self.basal_depression -= self.basal_dep_value
            self.basal_weights += self.basal_potentiation
            self.basal_weights = np.minimum(self.basal_weights, self.max_basal_weight)
            self.basal_weights = np.maximum(self.basal_weights, 0)

            # Implement plasticity on apical synapses
            self.apical_depression -= self.apical_dep_value
            self.apical_weights += self.apical_potentiation
            self.apical_weights = np.minimum(self.apical_weights, self.max_apical_weight)
            self.apical_weights = np.maximum(self.apical_weights, 0)

        else:
            self.spike = False

            # Vm is current vm - leak + conductance*DF*Resistance/tau --
            # --- -which is effectively conductance*DF/capacitance (which is fine)
            leak_term = (self.v_rest - self.vm) * self.dt / self.tau
            exc_df = self.exc_rev - self.vm
            inh_df = self.v_rest - self.vm
            exc_dv = (self.basal_conductance + self.apical_conductance) * exc_df * self.r_m * self.dt / self.tau
            inh_dv = self.inh_conductance * inh_df * self.r_m * self.dt / self.tau
            self.vm = self.vm + leak_term + exc_dv + inh_dv

        # Generate Basal Input Conductances
        basal_rate = input[self.basal_tune_index]
        basal_spikes = np.random.rand(*self.basal_weights.shape) < (basal_rate * self.dt)
        c_basal_conductance = np.sum(basal_spikes * self.basal_weights * (self.basal_weights > self.basal_cond_threshold))
        self.basal_conductance = c_basal_conductance - self.basal_conductance * self.dt / self.exc_tau

        # Generate Apical Input Conductances
        apical_rate = input[self.apical_tune_index]
        apical_spikes = np.random.rand(*self.apical_weights.shape) < (apical_rate * self.dt)
        c_apical_conductance = np.sum(apical_spikes * self.apical_weights * (self.apical_weights > self.apical_cond_threshold))
        self.apical_conductance += c_apical_conductance - self.apical_conductance * self.dt / self.exc_tau

        # Generate Inhibitory Conductances
        inh_pre_spikes = np.random.rand(self.num_inhibitory) < (self.inh_rate * self.dt)
        inh_conductance = self.inh_weight * np.sum(inh_pre_spikes)
        self.inh_conductance += inh_conductance - self.inh_conductance * self.dt / self.inh_tau

        # Do depression and replacement with basal inputs
        self.basal_weights += basal_spikes * self.basal_depression
        basal_replace = self.basal_weights < self.min_basal_weight
        self.basal_tune_index[basal_replace] = np.random.randint(0, self.num_inputs, np.sum(basal_replace))
        self.basal_weights[basal_replace] = self.basal_start_weight

        # Do depression and replacement with apical inputs
        self.apical_weights += apical_spikes * self.apical_depression
        apical_replace = self.apical_weights < self.min_apical_weight
        self.apical_tune_index[apical_replace] = np.random.randint(0, self.num_inputs, np.sum(apical_replace))
        self.apical_weights[apical_replace] = self.apical_start_weight

        # Update Potentiation/Depression Terms BASAL
        self.basal_potentiation += self.basal_pot_value * basal_spikes - self.basal_potentiation * self.dt / self.potentiation_tau
        self.basal_depression -= self.basal_depression * self.dt / self.depression_tau

        # Update Potentiation/Depression Terms APICAL
        self.apical_potentiation += self.apical_pot_value * apical_spikes - self.apical_potentiation * self.dt / self.potentiation_tau
        self.apical_depression -= self.apical_depression * self.dt / self.depression_tau

        # Do Homeostasis
        self.homeo_rate_estimate += (1 * self.spike / self.dt - self.homeo_rate_estimate) * self.dt / self.homeo_tau
        self.homeo_scale = self.homeo_rate / max(0.1, self.homeo_rate_estimate) - 1  # fraction change
        self.basal_weights += self.homeo_scale * self.basal_weights * self.dt / self.homeo_tau
        self.apical_weights += self.homeo_scale * self.apical_weights * self.dt / self.homeo_tau

        # Prevent weights from leaving range
        self.basal_weights = np.minimum(self.basal_weights, self.max_basal_weight)
        self.basal_weights = np.maximum(self.basal_weights, 0)
        self.apical_weights = np.minimum(self.apical_weights, self.max_apical_weight)
        self.apical_weights = np.maximum(self.apical_weights, 0)


class StimulusICA:
    """
    A class to store / generate information about the stimuli.

    Will probably refactor to have a base stimulus class that I fill out for different stim types.
    """

    def __init__(
        self,
        num_inputs=99,
        num_latent=3,
        source_method="gaussian",
        source_strength=3,
        rate_std=10,
        rate_mean=20,
        update_tau=20,
    ):
        self.num_inputs = num_inputs
        self.num_latent = num_latent
        self.source_method = source_method
        self.source_strength = source_strength
        self.rate_std = rate_std
        self.rate_mean = rate_mean
        self.update_tau = update_tau

        self.source_loading = self._build_source_loading()
        self.var_adjustment = np.sqrt(np.sum(self.source_loading**2, axis=0) + 1)

        self.need_input = True
        self.update_buffer = 0
        self.rate = None

    def step(self):

        if self.need_input or self.rate is None:
            input_changed = True
            interval = np.ceil(np.random.exponential(self.update_tau)) + 1  # duration of interval for current rates
            noise = np.random.randn(self.num_inputs)
            signal = np.sum(self.source_loading * np.random.randn(self.num_latent).reshape(-1, 1), axis=0)
            input = (noise + signal) / self.var_adjustment
            self.rate = self.rate_std * input + self.rate_mean
            self.rate[self.rate < 0] = 0
            self.update_buffer = interval - 1

        else:
            input_changed = False
            self.update_buffer = self.update_buffer - 1

        self.need_input = self.update_buffer == 0
        return self.rate, input_changed

    def _build_source_loading(self):

        # always use an even number for symmetry
        num_input_per_signal = int(np.floor(self.num_inputs / self.num_latent))

        if self.source_method == "divide":
            loading_base = np.concatenate(
                (self.source_strength * np.ones(num_input_per_signal), np.zeros((self.num_latent - 1) * num_input_per_signal)),
            )

        elif self.source_method == "gaussian":
            centered_x = np.arange(self.num_inputs)
            centered_x = centered_x - np.mean(centered_x)
            width_gaussian = 2 / 5 * num_input_per_signal
            loading_base = np.roll(np.exp(-(centered_x**2) / (2 * width_gaussian**2)), -num_input_per_signal)

        source_loading = np.stack([np.roll(loading_base, ilatent * num_input_per_signal) for ilatent in range(self.num_latent)])

        extra_inputs = self.num_inputs - num_input_per_signal * self.num_latent
        source_loading = np.concatenate((source_loading, np.zeros((self.num_latent, extra_inputs))), axis=1)

        return source_loading
