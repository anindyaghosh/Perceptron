import argparse
import numpy as np
import os
import pickle
import sys

from pygenn.genn_model import (create_custom_neuron_class,
                                create_custom_weight_update_class,
                                GeNNModel, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, required=True)
parser.add_argument('-b_i', '--b2i', type=float, default=10)
parser.add_argument('-i_b', '--i2b', type=float, default=10)
parser.add_argument('-tc', '--time_constant_factor', type=int, default=10)
parser.add_argument('-d', '--delay_factor', type=int, default=10)
parser.add_argument('-so', '--save_output', action='store_true', default=False)
parser.add_argument('-nt', '--noise_threshold', type=float, default=3)

args = parser.parse_args()

class GA_search():
    def __init__(self, filename):
        self.B_I = 1.5 * args.b2i/10
        self.I_B = -4.0 * args.i2b/10
        self.tc_range_factor = args.time_constant_factor
        self.delay = args.delay_factor/10
        self.noise_nt = args.noise_threshold
        
        self.filename = os.path.join(os.getcwd(), '..', 'do_dataset', 'polarity_add', filename)

        self.original_resolution = np.array([640, 480])
        self.new_resolution = np.array([20, 20]) # original_resolution
        
        if any(x in self.filename for x in ['synthSpikes', '20_20', 'test']):
            self.original_resolution = self.new_resolution
        
        self.timesteps_per_frame = 1
        self.tc_range = np.hstack((np.arange(1, step=0.1), np.logspace(0, 1, 11)))

    def import_files(self, version, noise_nt):
        sys.path.append('../')
        from p_calc import p_scaled_calc
        from extractCleanSourceSpikes import extractSourceSpikes
        
        if '20_20' in self.filename:
            version = 'global'
            noise_nt = 0
        else:
            version = 'v2'
            noise_nt = self.noise_nt
            
        self.time, neuron_spike_times_pos, neuron_spike_times_neg = extractSourceSpikes(self.filename, self.new_resolution, 
                                                                                    version, noise_nt).read_input_spikes()
        
        p_scaled, _ = p_scaled_calc(self.timesteps_per_frame, self.tc_range[self.tc_range_factor])
        
        return p_scaled, neuron_spike_times_pos, neuron_spike_times_neg
    
    def LGMD_model(self):
            
        p_scaled, _, neuron_spike_times1 = self.import_files('v2', 3)
        
        B_PARAMS = {"p": p_scaled['P'], "theta": 0.3, "alpha": 0.5, "RefracPeriod": 0.0}
        I_PARAMS = {"p": p_scaled['I']}
        
        delay_param = 62
        
        # ----------------------------------------------------------------------------
        # Custom GeNN models
        # ----------------------------------------------------------------------------
        # Very simple integrate-and-fire neuron model
        lt_model = create_custom_neuron_class(
            "lt_model",
            param_names=["p"],
            var_name_types=[("V", "scalar")],
            sim_code="""
            $(V) = $(p)*$(V) + $(Isyn);
            """)
            
        lif_model = create_custom_neuron_class(
            "lif_model",
            param_names=["p", "theta", "alpha", "RefracPeriod"],
            var_name_types=[("V", "scalar"), ("_i", "scalar"), ("RefracTime", "scalar")],
            sim_code="""
            if ($(RefracTime) <= 0.0) {
                $(_i) = $(Isyn);
                $(V) = $(p)*$(V) + $(Isyn);
            }
            else {
                $(RefracTime) -= DT;
            }
            """,
            reset_code="""
            $(V) -= $(alpha);
            $(RefracTime) = $(RefracPeriod);
            """,
            threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(theta)", is_auto_refractory_required=False)
        
        graded_synapse_model = create_custom_weight_update_class(
            "graded_synapse",
            param_names=["theta"],
            var_name_types=[("w", "scalar")],
            event_threshold_condition_code="$(V_pre) >= $(theta)",
            event_code="$(addToInSyn, $(w)*$(V_pre));")
        
        # ----------------------------------------------------------------------------
        # Build model
        # ----------------------------------------------------------------------------
        # Create GeNN model
        model = GeNNModel("float", "LGMD_GA_search", selectGPUByDeviceID=True, deviceSelectMethod=DeviceSelect_MANUAL)
        model.dT = float(self.timesteps_per_frame)
        sim_time = int(self.time + (model.dT * 100))
        PRESENT_TIMESTEPS = int(np.ceil(sim_time/model.dT))
        
        # Initial values to initialise all neurons to
        lt_init = {"V": 0.0}
        lif_init = {"V": 0.0, "_i": 0.0, "RefracTime": 0.0}
        
        # Create first neuron layer
        B1_pop = model.add_neuron_population("B1_cell", self.new_resolution[0] * self.new_resolution[1],
                                                    lif_model, B_PARAMS, lif_init)
        
        I1_pop = model.add_neuron_population("I1_cell", self.new_resolution[0] * self.new_resolution[1],
                                                    lt_model, I_PARAMS, lt_init)
        
        # SpikeSourceArray
        # The times in each list need to be sorted
        flat_neuron_spike_times = np.concatenate(neuron_spike_times1)
    
        # Calculate number of spikes per neuron
        num_spikes_per_neuron = [len(n) for n in neuron_spike_times1]
        
        # Calculate cumulative sum of spike counts which gives end index into flat_neuron_spike_times
        num_spikes_per_neuron_cumulative = np.cumsum(num_spikes_per_neuron)
        
        # We can now initialise the start and end spike variable for each neuron
        # (the first spike for the first neuron is at flat_neuron_spike_times[0])
        stim_ini1 = {"startSpike": np.append(0, num_spikes_per_neuron_cumulative[:-1]), "endSpike": num_spikes_per_neuron_cumulative}
        
        # Create GeNN model
        P1 = model.add_neuron_population("Stim", len(neuron_spike_times1), "SpikeSourceArray", {}, stim_ini1)
        
        # Set spike time array
        P1.set_extra_global_param("spikeTimes", flat_neuron_spike_times)
        
        P1.spike_recording_enabled = True
        B1_pop.spike_recording_enabled = True
        
        w_I_B_WEIGHT = self.I_B
        w_B_I_WEIGHT = (self.B_I/10) / (62 / model.dT)
        
        delay = round(delay_param * self.delay)
        
        model.add_synapse_population(
            "p1_b1synapse", "SPARSE_GLOBALG", NO_DELAY, P1, B1_pop,
            "StaticPulse", {}, {"g": 1.0}, {}, {},
            "DeltaCurr", {}, {},
            init_connectivity("OneToOne", {}))
        
        model.add_synapse_population(
            "b1_i1synapse", "SPARSE_GLOBALG", NO_DELAY, B1_pop, I1_pop,
            "StaticPulse", {}, {"g": w_B_I_WEIGHT}, {}, {},
            "DeltaCurr", {}, {},
            init_connectivity("OneToOne", {}))
        
        i1_b1 = model.add_synapse_population(
            "i1_b1synapse", "SPARSE_GLOBALG", delay, I1_pop, B1_pop,
            graded_synapse_model, {"theta": 0.0}, {"w": w_I_B_WEIGHT}, {}, {}, # 0.3
            "DeltaCurr", {}, {})
        
        i1_b1.set_sparse_connections(np.arange(self.new_resolution[0] * self.new_resolution[1]), 
                                   np.arange(self.new_resolution[0] * self.new_resolution[1]))
        
        model.build()
        model.load(num_recording_timesteps=PRESENT_TIMESTEPS)
        
        Bv1 = np.empty((PRESENT_TIMESTEPS, B1_pop.size))
        Iv1 = np.empty((PRESENT_TIMESTEPS, I1_pop.size))
        
        Bv1_view = B1_pop.vars["V"].view
        Iv1_view = I1_pop.vars["V"].view
        
        while model.t < sim_time:
            model.step_time()
            
            model.pull_recording_buffers_from_device()
            
            B1_pop.pull_var_from_device("V")
            I1_pop.pull_var_from_device("V")
            
            Bv1[model.timestep - 1,:] = Bv1_view.copy()
            Iv1[model.timestep - 1,:] = Iv1_view.copy()
            
        B1_spike_times, B1_spike_ids = B1_pop.spike_recording_data
        P1_spike_times, P1_spike_ids = P1.spike_recording_data
        
        return {"P1": (P1_spike_times, P1_spike_ids), "B1": (B1_spike_times, B1_spike_ids)}
    
    def save_spikes(self):
        spike_info = self.LGMD_model()

        if args.save_output:
            filehead = args.filename.split('.')[0]
            save_dir = os.path.join(os.getcwd(), 'grid_search_self_inhibition_agg', filehead)
            
            os.makedirs(save_dir, exist_ok=True)
            
            param_dict = {'b_i':args.b2i, 'i_b':args.i2b, 'tc':args.time_constant_factor, 
                          'd':args.delay_factor, 'nt':args.noise_threshold}
            
            param_dict.update((key, int(value)) for key, value in param_dict.items())
            
            save_name = os.path.join(save_dir, '_'.join([f'bi{param_dict["b_i"]}', f'ib{param_dict["i_b"]}',
                                                         f'tc{param_dict["tc"]}', f'd{param_dict["d"]}', f'nt{param_dict["nt"]}']))
            
            pickle.dump(spike_info, open(save_name + '.p', "wb"))
            
    def ds_version(self):
        ds_filename = '_'.join([*args.filename.split('.')[0].split('_')[:-3], 'frames_downsampled_20_20']) + '.spikes'
        _, _, neuron_spike_times_ds = GA_search(ds_filename).import_files('global', 0)
        
        ds_dir_name = '_'.join(ds_filename.split('.')[0].split('_')[:-2])
        save_dir = os.path.join(os.getcwd(), 'grid_search_self_inhibition', 'downsampled_stims')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save only neuron spike times of downsampled here
        pickle.dump(neuron_spike_times_ds, open(os.path.join(save_dir, ds_dir_name + '.p'), "wb"))
        
x = GA_search(args.filename)
x.save_spikes()
# x.LGMD_model()
# x.ds_version()

class params_cycle:
    def __init__(self, filename):
        self.filename = filename
        
    def coarse(self):
        with open(self.filename, 'w') as file:
            for b_i in [12, 15, 20, 25]:
                for i_b in [10, 12, 15, 20]:
                    for d in [3, 5, 7]:
                        for tc in [10, 12, 15]:
                            for nt in range(3, 7):
                                string = f'-b_i {b_i} -i_b {i_b} -tc {tc} -d {d} -so -nt {nt}'
                                file.write(string + '\n')
                            
# p = params_cycle('arguments_self_inhib_agg.txt')
# p.coarse()