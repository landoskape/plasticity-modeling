import numpy as np
from scipy.constants import R, physical_constants
import matplotlib.pyplot as plt
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
    

if __name__ == "__main__":
    # Simulation parameters
    v_range = np.linspace(-80, 40, 200)  # Voltage range for steady-state plots
    dt = 0.01  # Time step for numerical integration (ms)
    t_start = 0
    t_end = 3
    t_range = np.linspace(t_start, t_end, int((t_end-t_start)/dt))    
    ap_peak_time = 1  # Time of AP peak in ms
    ap_amplitudes = np.linspace(10, 100, 10)  # AP amplitudes to test
    
    
    # Initialize channels
    nmdar = NMDAR()
    vgcc = VGCC()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    
    # Subplot 1: Open probabilities
    ax1.plot(v_range, nmdar.open_probability(v_range), 'k', label='NMDAR')
    ax1.plot(v_range, vgcc.open_probability_activation(v_range), 'b', label='VGCC')
    ax1.set_xlabel('Membrane Potential (mV)')
    ax1.set_ylabel('Open Probability')
    ax1.set_title('Channel Open Probabilities')
    ax1.legend()
    
    # Subplot 2: Time constants
    ax2.plot(v_range, nmdar.time_constant(v_range), 'k', label='NMDAR')
    tau_m, tau_h = vgcc.time_constant(v_range)
    ax2.plot(v_range, tau_m, 'b', label='VGCC')
    ax2.set_xlabel('Membrane Potential (mV)')
    ax2.set_ylabel('Time Constant (ms)')
    ax2.set_title('Channel Time Constants')
    ax2.legend()
    
    plt.tight_layout()
    plt.show(block=True)

    # Create figure: Response to action potentials
    plt.figure(figsize=(12, 4))
    
    # Generate voltage traces and responses for different AP amplitudes
    for amp in ap_amplitudes:
        # Create AP waveform
        ap = AP(v_amp=amp, v_dur=1, v_base=-70)
        v_trace = ap.voltage(t_range, ap_peak_time)
        
        # Initialize state variables to steady state at baseline voltage
        m = vgcc.open_probability_activation(v_trace[0])
        h = vgcc.open_probability_inactivation(v_trace[0])
        n = nmdar.open_probability(v_trace[0])
        
        # Arrays to store results
        m_trace = np.zeros_like(t_range)
        h_trace = np.zeros_like(t_range)
        n_trace = np.zeros_like(t_range)
        m_trace[0] = m
        h_trace[0] = h
        n_trace[0] = n
        
        # Numerical integration using Euler's method
        for i in range(1, len(t_range)):
            m += dt * vgcc.dmdt(v_trace[i-1], m)
            h += dt * vgcc.dhdt(v_trace[i-1], h)
            n += dt * nmdar.dndt(v_trace[i-1], n)
            
            m_trace[i] = m
            h_trace[i] = h
            n_trace[i] = n
        
        # Calculate open probabilities
        vgcc_p = m_trace**2 * h_trace  # VGCC open probability
        nmdar_p = n_trace  # NMDAR open probability
        
        # Plot results
        plt.subplot(131)
        plt.plot(t_range, v_trace, color='k', label=f'{amp:.0f} mV')
        
        plt.subplot(132)
        plt.plot(t_range, nmdar_p, color='k', label=f'{amp:.0f} mV')
        
        plt.subplot(133)
        plt.plot(t_range, vgcc_p, color='k', label=f'{amp:.0f} mV')
    
    # Format AP response plots
    plt.subplot(131)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Action Potentials')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(132)
    plt.xlabel('Time (ms)')
    plt.ylabel('Open Probability')
    plt.title('NMDAR Response')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(133)
    plt.xlabel('Time (ms)')
    plt.ylabel('Open Probability')
    plt.title('VGCC Response')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.show(block=True)
