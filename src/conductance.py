import numpy as np
from scipy.constants import R, physical_constants
F = physical_constants['Faraday constant'][0]

class VGCC:
    """Voltage-Gated Calcium Channel (VGCC) model with activation and inactivation gates.
    
    Rate constants for activation (m) and inactivation (h) gates:
    alpha_m(V) = 0.055(-27-V)/(e^((-27-V)/3.8)-1) [ms^-1 mV^-1]
    beta_m(V) = 0.94e^((-75-V)/17) [ms^-1]
    alpha_h(V) = 0.000457e^((-13-V)/50) [ms^-1]
    beta_h(V) = 0.0065/(e^((-V-15)/28)+1) [ms^-1]
    
    Time constant and open probability:
    tau(V) = 1/(alpha_V + beta_V)
    P_open(V) = m^2 * h
    """
    def __init__(self):
        """Initialize VGCC with parameters"""
        pass
        
    def _alpha_m(self, V):
        """Forward rate constant for activation gate"""
        return 0.055 * (-27 - V) / (np.exp((-27 - V) / 3.8) - 1)
    
    def _beta_m(self, V):
        """Backward rate constant for activation gate"""
        return 0.94 * np.exp((-75 - V) / 17)
    
    def _alpha_h(self, V):
        """Forward rate constant for inactivation gate"""
        return 0.000457 * np.exp((-13 - V) / 50)
    
    def _beta_h(self, V):
        """Backward rate constant for inactivation gate"""
        return 0.0065 / (np.exp((-V - 15) / 28) + 1)
    
    def time_constant(self, V):
        """Compute the time constant of the VGCC activation & inactivation gates
        
        Args:
            V (float): Membrane potential in mV

        Returns:
            tuple: Time constants for activation and inactivation gates in ms
        """
        alpha_V_m = self._alpha_m(V)
        beta_V_m = self._beta_m(V)
        alpha_V_h = self._alpha_h(V)
        beta_V_h = self._beta_h(V)
        time_constant_m = 1 / (alpha_V_m + beta_V_m)
        time_constant_h = 1 / (alpha_V_h + beta_V_h)
        return time_constant_m, time_constant_h
    
    def open_probability_activation(self, V):
        """Compute the open probability of the VGCC activation gate"""
        alpha_V_m = self._alpha_m(V)
        beta_V_m = self._beta_m(V)
        return alpha_V_m / (alpha_V_m + beta_V_m)
    
    def open_probability_inactivation(self, V):
        """Compute the open probability of the VGCC inactivation gate"""
        alpha_V_h = self._alpha_h(V)
        beta_V_h = self._beta_h(V)
        return alpha_V_h / (alpha_V_h + beta_V_h)
    
    def open_probability(self, V):
        """Compute the open probability of the VGCC
        
        Args:
            V (float): Membrane potential in mV
        
        Returns:
            float: Open probability
        """
        m = self.open_probability_activation(V)
        h = self.open_probability_inactivation(V)
        return m**2 * h
    
    def dmdt(self, V, m):
        """Compute the time derivative of the activation gate"""
        alpha_V_m = self._alpha_m(V)
        beta_V_m = self._beta_m(V)
        return alpha_V_m * (1 - m) - beta_V_m * m
    
    def dhdt(self, V, h):
        """Compute the time derivative of the inactivation gate"""
        alpha_V_h = self._alpha_h(V)
        beta_V_h = self._beta_h(V)
        return alpha_V_h * (1 - h) - beta_V_h * h
        

class NMDAR:
    """N-Methyl-D-Aspartate Receptor (NMDAR) model with voltage-dependent Mg2+ block.
    
    Magnesium block kinetics:
    k_off = e^(0.017V + 0.96) [ms^-1]
    k_on = [Mg^2+]e^(-0.045V - 6.97) [ms^-1 µM^-1]
    
    Time constant and open probability:
    tau(V) = 1/(k_on + k_off)
    P_open(V) = k_off/(k_on + k_off)
    """
    def __init__(self, mg_conc=1000):
        """Initialize NMDAR with parameters
        
        Args:
            mg_conc (float): Magnesium concentration in µM (default: 1 µM)
        """
        self.mg_conc = mg_conc
    
    def _k_off(self, V):
        """Off rate for Mg2+ block"""
        return np.exp(0.017 * V + 0.96)
    
    def _k_on(self, V):
        """On rate for Mg2+ block"""
        return self.mg_conc * np.exp(-0.045 * V - 6.97)
    
    def time_constant(self, V):
        """Compute the time constant of the NMDAR
        
        Args:
            V (float): Membrane potential in mV
        Returns:
            float: Time constant in ms
        """
        return 1 / (self._k_on(V) + self._k_off(V))
    
    def open_probability(self, V):
        """Compute the open probability of the NMDAR
        
        Args:
            V (float): Membrane potential in mV
        Returns:
            float: Open probability
        """
        return self._k_off(V) / (self._k_on(V) + self._k_off(V))
    
    def dndt(self, V, n):
        """Compute the time derivative of the NMDAR activation gate"""
        k_off = self._k_off(V)
        k_on = self._k_on(V)
        return k_off * (1 - n) - k_on * n

def compute_current(V, p_open, ca_in, ca_out, temp=310.15):
    """Compute relative calcium current using modified Goldman-Hodgkin-Katz equation:
    
    I_Ca = P_open * V * ([Ca]_in - [Ca]_out * e^(-2VF/RT))/(1 - e^(-2VF/RT))

    This function does not include the maximum conductance of the channel, which is
    technically required to compute the current. We can think of this as the current 
    per unit of conductance -- similar to the current density (but not quite). It's 
    posed like this because the simulations that use this model are concerned with 
    what conditions open the channel, rather than the specific conductance. 
    
    where:
    - F is Faraday's constant
    - R is gas constant
    - T is temperature in Kelvin
    """
    V_volts = V / 1000  # Convert mV to V
    numer = ca_in - ca_out * np.exp(-2 * V_volts * F / (R * temp))
    denom = 1 - np.exp(-2 * V_volts * F / (R * temp))
    return p_open * V_volts * (numer / denom)

class AP:
    """Action potential waveform model using quadratic function.
    
    Voltage equation:
    V(t) = V_base + V_amp - (at)^2  for -V_dur/2 < t < V_dur/2
    V(t) = V_base                    otherwise
    
    where 'a' is calculated as:
    a = sqrt(4*V_amp/V_dur^2)
    """
    def __init__(self, v_amp, v_dur, v_base=-70):
        """Initialize action potential waveform
        
        Args:
            v_amp (float): AP amplitude in mV
            v_dur (float): AP duration in ms
            v_base (float): Baseline voltage in mV (default: -70 mV)
        """
        self.v_amp = v_amp
        self.v_dur = v_dur
        self.v_base = v_base
        self.a = np.sqrt(v_amp) * 2 / v_dur
    
    def voltage(self, t, t_peak):
        """Compute the voltage at times t
        
        Args:
            t (np.ndarray): Time in ms
            t_peak (float): Time of peak voltage in ms
        Returns:
            np.ndarray: Voltage in mV
        """
        voltage = self.v_base * np.ones_like(t)
        t_ap_idx = np.where(abs(t-t_peak) <= self.v_dur/2)[0]
        voltage[t_ap_idx] += self.v_amp - (self.a * (t[t_ap_idx]-t_peak))**2
        return voltage
    
