import numpy as np

p = {'P': 0.4, 'E': 0.1, 'I': 0.8, 'S': 0.4, 'F': 0.1, 'L': 0.4}
blanchard_model_timestep = 62
# ms_per_timestep = 1 # (1/16)*1000 # 16 frames per second

# p = exp(-1/tau)
def p_scaled_calc(ms_per_timestep, tc_amp):
    tau = []
    p_scaled = {}
    for i, value in enumerate(p.values()):
        tau_new = - ((blanchard_model_timestep/ms_per_timestep) / (np.log(value)) * tc_amp)
        tau.append(tau_new)
        p_scaled.update({list(p.keys())[i] : np.exp(-1/tau_new)})
    return p_scaled, tau_new

# p_scaled, tau = p_scaled_calc(1, 10)