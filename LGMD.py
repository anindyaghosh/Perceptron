import argparse
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import os
import skimage.measure
import sys

# ----------------------------------------------------------------------------
# Warning - disable once scipy >= 1.8

# import warnings
# warnings.simplefilter('ignore', UserWarning)
from scipy import sparse
# ----------------------------------------------------------------------------

from pygenn.genn_model import (create_custom_neuron_class,
                                create_custom_weight_update_class,
                                GeNNModel, init_connectivity)
from pygenn.genn_wrapper import NO_DELAY
# from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL

from qian_dataset.extractSourceSpikes import read_input_spikes
# from ag_dataset.extractCleanSourceSpikes_dvs import read_input_spikes_show_dvs
from ag_dataset.integrateSpikes_dvs import temporal_downsampling_dvs
from dvx_dataset.extractCleanSourceSpikes_dvx_v1 import read_input_spikes_show_dvx
# from dvx_dataset.integrateSpikes_dvx import temporal_downsampling_dvx
from do_dataset.extractCleanSourceSpikes_do_v1 import read_input_spikes_show_do_v1
# from do_dataset.integrateSpikes_do import temporal_downsampling_do
from p_calc import p_scaled_calc
from extractCleanSourceSpikes import extractSourceSpikes

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, required=True)
parser.add_argument('-c', '--cell_type', type=str, default='S', help='IF cell type to view')
parser.add_argument('-s', '--show_plot', type=bool)
parser.add_argument('-e', '--E_syn', type=float, default=6)
parser.add_argument('-i', '--I_syn', type=float, default=2)
parser.add_argument('-k', '--kernel_factor', type=float, default=10)
parser.add_argument('-tc', '--time_constant_factor', type=int, default=10)
parser.add_argument('-d', '--delay_factor', type=float, default=10)
parser.add_argument('-ffk', '--feedforward_strength', type=float, default=10)
parser.add_argument('-ffd', '--feedforward_delay', type=float, default=10)
parser.add_argument('-so', '--save_output', type=bool)
parser.add_argument('-n', '--code_num', type=int)
parser.add_argument('-nt', '--noise_threshold', type=float)

subparsers = parser.add_subparsers(title='connections',
                                    description='Valid synaptic connections to be saved')
connection_parser = subparsers.add_parser('syn_save', help='connection_choice polarity_choice')
connection_parser.add_argument('--connection', choices=['E', 'I', 'S'], required=True)
connection_parser.add_argument('--polarity', choices=['+', '-'], required=True)

args = parser.parse_args()

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

filename = args.filename
filehead, _ = filename.split('.')

# filename = sys.argv[1]

# def pol_split(filename):
#     ending = '_reverse.spikes'
#     if filename.endswith(ending):
#         filehead, ext = filename.split(ending)
#         filename_pos = filehead + '_+' + ending
#         filename_neg = filehead + '_-' + ending
#     else:
#         filehead, ext = filename.split('.')
#         filename_pos = filehead + '_+.' + ext
#         filename_neg = filehead + '_-.' + ext
#     return filename_pos, filename_neg

# filename_pos, filename_neg = pol_split(filename)

# filehead, ext = filename.split('.')
# filename1 = filehead + '_reverse.' + ext
filetype = 'polarity'
original_resolution = np.array([640, 480])
new_resolution = np.array([20, 20]) # original_resolution

if any(x in filename for x in ['synthSpikes', '20_20', 'test']):
    original_resolution = new_resolution

timesteps_per_frame = 1
# sharpen_factor = 1 # int(sys.argv[3])
# filename = "dx-b_f_0.spikes"

def filename_decision(filename_):
    if os.path.exists(os.path.join(os.getcwd(), "qian_dataset", filename_)):
        time, neuron_spike_times = read_input_spikes(filename_)
    elif os.path.exists(os.path.join(os.getcwd(), "ag_dataset", filetype, filename_)):
        print('Not ag_dataset')
        # time, neuron_spike_times = read_input_spikes_show_dvs(filename_, filetype, new_resolution)
        # file = temporal_downsampling_dvs(filename_, filetype, new_resolution, timesteps_per_frame)
        # time, neuron_spike_times = file.integrate_spikes()
    elif os.path.exists(os.path.join(os.getcwd(), "dvx_dataset", "polarity_add", filename_)):
        time, neuron_spike_times_pos, neuron_spike_times_neg = extractSourceSpikes(filename_, new_resolution, 
                                                                                   'v2', args.noise_threshold).read_input_spikes()
        # file = temporal_downsampling_dvx(filename_, filetype, new_resolution, timesteps_per_frame)
        # time, neuron_spike_times = file.integrate_spikes()
    elif os.path.exists(os.path.join(os.getcwd(), "do_dataset", "polarity_add", filename_)):
        time, neuron_spike_times_pos, neuron_spike_times_neg = extractSourceSpikes(filename_, new_resolution, 
                                                                                   'v2', args.noise_threshold).read_input_spikes()
        # file = temporal_downsampling_dvx(filename_, filetype, new_resolution, timesteps_per_frame)
        # time, neuron_spike_times = file.integrate_spikes()
    else:
        sys.exit("Cannot find spike file!")
    return time, neuron_spike_times_pos, neuron_spike_times_neg

time, neuron_spike_times, neuron_spike_times1 = filename_decision(filename)
time1 = time

# if filetype == 'original_files':
#     time, neuron_spike_times = filename_decision(filename)
#     time1, neuron_spike_times1 = filename_decision(filename1)
# else:
#     time, neuron_spike_times = filename_decision(filename_pos)
#     time1, neuron_spike_times1 = filename_decision(filename_neg)



# synth_multi.spikes is do-f_f_0.spikes, then b_f_0, then f_f_0

tc_range = np.hstack((np.arange(1, step=0.1), np.logspace(0, 1, 11)))

p_scaled, _ = p_scaled_calc(timesteps_per_frame, tc_range[args.time_constant_factor])
# p_scaled = p_scaled_calc(timesteps_per_frame, 1)

# E_PARAMS = {"p": 0.1} # Params for 5 cm/s
# I_PARAMS = {"p": 0.8}
# S_PARAMS = {"p": 0.4, "theta": 0.5, "alpha": 0.5} # 0.2 0.4
# F_PARAMS = {"p": 0.1}
# LGMD_PARAMS = {"p": 0.4, "theta": 0.25, "alpha": 0.25}

T_PARAMS = {"p": p_scaled['E']}
E_PARAMS = {"p": p_scaled['E']} # Params for 5 cm/s
I_PARAMS = {"p": p_scaled['I']}
S_PARAMS = {"p": p_scaled['S'], "theta": 0.5, "alpha": 0.5, "RefracPeriod": 0.0} # 0.2 0.4 0.5
F_PARAMS = {"p": p_scaled['F']}
LGMD_PARAMS = {"p": p_scaled['L'], "theta": 0.25, "alpha": 0.25, "RefracPeriod": 0.0}
DCMD_PARAMS = {"p": 0.995, "theta": 2.0, "alpha": 2.0, "RefracPeriod": 0.0}

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
    var_name_types=[("V", "scalar"), ("i", "scalar"), ("RefracTime", "scalar")],
    sim_code="""
    if ($(RefracTime) <= 0.0) {
        $(i) = $(Isyn);
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

i_sgraded_synapse_model = create_custom_weight_update_class(
    "i_sgraded_synapse",
    param_names=["theta"],
    var_name_types=[("w", "scalar"), ("d", "int")],
    event_threshold_condition_code="$(V_pre) >= $(theta)",
    event_code="$(addToInSynDelay, $(w)*$(V_pre), $(d));")

# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", f"LGMD_{args.code_num}", selectGPUByDeviceID=True) # , deviceSelectMethod=DeviceSelect_MANUAL)
model.dT = float(timesteps_per_frame)
# if 'circle' or 'square' in filename:
#     print('p2')
#     print(filename)
#     if '60' in filename:
#         sim_time = 1750
#     elif '479' in filename:
#         sim_time = 14100
#     else:
#         sim_time = 3500
# else:
#     print('p1')
sim_time = int(time + (model.dT * 100))
PRESENT_TIMESTEPS = int(np.ceil(sim_time/model.dT))

# Initial values to initialise all neurons to
lt_init = {"V": 0.0}
lif_init = {"V": 0.0, "i": 0.0, "RefracTime": 0.0}

# Create first neuron layer
T_pop = model.add_neuron_population("TmA_cell", new_resolution[0] * new_resolution[1],
                                            lt_model, E_PARAMS, lt_init)

E_pop = model.add_neuron_population("E_cell", new_resolution[0] * new_resolution[1],
                                            lt_model, E_PARAMS, lt_init)

I_pop = model.add_neuron_population("I_cell", new_resolution[0] * new_resolution[1],
                                            lt_model, I_PARAMS, lt_init)

S_pop = model.add_neuron_population("S_cell", new_resolution[0] * new_resolution[1],
                                            lif_model, S_PARAMS, lif_init) # only central 256 cells

F_pop = model.add_neuron_population("F_cell", 1,
                                            lt_model, F_PARAMS, lt_init)

DCMD_pop = model.add_neuron_population("DCMD_cell", 1,
                                            lif_model, DCMD_PARAMS, lif_init)

LGMD_pop = model.add_neuron_population("LGMD_cell", 1,
                                            lif_model, LGMD_PARAMS, lif_init)

T1_pop = model.add_neuron_population("TmA1_cell", new_resolution[0] * new_resolution[1],
                                            lt_model, E_PARAMS, lt_init)

E1_pop = model.add_neuron_population("E1_cell", new_resolution[0] * new_resolution[1],
                                            lt_model, E_PARAMS, lt_init)

I1_pop = model.add_neuron_population("I1_cell", new_resolution[0] * new_resolution[1],
                                            lt_model, I_PARAMS, lt_init)

S1_pop = model.add_neuron_population("S1_cell", new_resolution[0] * new_resolution[1],
                                            lif_model, S_PARAMS, lif_init) # only central 256 cells

F1_pop = model.add_neuron_population("F1_cell", 1,
                                            lt_model, F_PARAMS, lt_init)

L1_pop = model.add_neuron_population("L1_cell", 1,
                                            lif_model, LGMD_PARAMS, lif_init)

D1_pop = model.add_neuron_population("D1_cell", 1,
                                            lif_model, DCMD_PARAMS, lif_init)


##############################################################################

# List of spike times for each neuron
# The times in each list need to be sorted
flat_neuron_spike_times = np.concatenate(neuron_spike_times)

# Calculate number of spikes per neuron
num_spikes_per_neuron = [len(n) for n in neuron_spike_times]

# Calculate cumulative sum of spike counts which gives end index into flat_neuron_spike_times
num_spikes_per_neuron_cumulative = np.cumsum(num_spikes_per_neuron)

# We can now initialise the start and end spike variable for each neuron
# (the first spike for the first neuron is at flat_neuron_spike_times[0])
stim_ini = {"startSpike": np.append(0, num_spikes_per_neuron_cumulative[:-1]), "endSpike": num_spikes_per_neuron_cumulative}

# Create GeNN model
P = model.add_neuron_population("Stim", len(neuron_spike_times), "SpikeSourceArray", {}, stim_ini)

# Set spike time array
P.set_extra_global_param("spikeTimes", flat_neuron_spike_times)

##############################################################################

flat_neuron_spike_times1 = np.concatenate(neuron_spike_times1)

# Calculate number of spikes per neuron
num_spikes_per_neuron1 = [len(n) for n in neuron_spike_times1]

# Calculate cumulative sum of spike counts which gives end index into flat_neuron_spike_times
num_spikes_per_neuron_cumulative1 = np.cumsum(num_spikes_per_neuron1)

# We can now initialise the start and end spike variable for each neuron
# (the first spike for the first neuron is at flat_neuron_spike_times[0])
stim_ini1 = {"startSpike": np.append(0, num_spikes_per_neuron_cumulative1[:-1]), "endSpike": num_spikes_per_neuron_cumulative1}

# Create GeNN model
P1 = model.add_neuron_population("Stim1", len(neuron_spike_times1), "SpikeSourceArray", {}, stim_ini1)

# Set spike time array
P1.set_extra_global_param("spikeTimes", flat_neuron_spike_times1)

# Turn on spike recording
P.spike_recording_enabled = True
LGMD_pop.spike_recording_enabled = True
S_pop.spike_recording_enabled = True
DCMD_pop.spike_recording_enabled = True

P1.spike_recording_enabled = True
L1_pop.spike_recording_enabled = True
S1_pop.spike_recording_enabled = True
D1_pop.spike_recording_enabled = True

##############################################################################

# E_factor = float(sys.argv[2])/10
# I_factor = float(sys.argv[3])/10
# w_P_T = np.eye(new_resolution * new_resolution) * 0.6 / 62
# w_T_E = np.eye(new_resolution * new_resolution) * -0.3
# w_I_T = np.eye(new_resolution * new_resolution) * -4 / 62

w_P_E_WEIGHT = args.E_syn/10 / (62 / model.dT) # 0.6 / (62 / model.dT)
w_P_I_WEIGHT = args.I_syn/10 / (62 / model.dT) # 0.2 / (62 / model.dT)
w_E_S_WEIGHT = 1.0

# w_P_E = np.eye(new_resolution * new_resolution) * w_P_E_WEIGHT # E_factor
# w_P_I = np.eye(new_resolution * new_resolution) * w_P_I_WEIGHT # I_factor # 0.3 0.22
# w_E_S = np.zeros((new_resolution * new_resolution, new_resolution * new_resolution)) # 0.8 0.82
# w_I_S = np.zeros((new_resolution[0] * new_resolution[1], new_resolution[0] * new_resolution[1]))
# d_I_S = np.zeros((new_resolution[0] * new_resolution[1], new_resolution[0] * new_resolution[1]))
w_P_F = np.zeros((new_resolution[0] * new_resolution[1], 1)) # NaN
w_S_LGMD = np.zeros((new_resolution[0] * new_resolution[1], 1))
w_F_LGMD = -5.0 * args.feedforward_strength/10 # 5
w_LGMD_DCMD = 0.95

# Find pixels cells that S is synaptically connected to
# pix = np.hstack([[r * new_resolution + c for c in np.arange(0.1*new_resolution, new_resolution-0.1*new_resolution)] 
#                  for r in np.arange(0.1*new_resolution, new_resolution-0.1*new_resolution)])

# pix = list(map(np.int32, pix))

ax_range = np.asarray([np.arange(0.1*n, n-0.1*n).astype(int) for n in new_resolution], dtype=object)

x = np.asarray(np.meshgrid(ax_range[0], ax_range[1]), dtype=int)
pix = np.hstack(np.ravel_multi_index((x[1], x[0]), (new_resolution[1], new_resolution[0])))
# pix = np.hstack([(r * new_resolution[0]) + ax_range for r in ax_range])

# For E-S, P-F and S-LGMD, only central 256 cells are synaptically connected
for i in pix:
    # w_E_S[i, i] = w_E_S_WEIGHT
    w_P_F[i] = 0.04 * 0.2 / (62 / model.dT)
    w_S_LGMD[i] = 0.04 * 2

# w = {"a": -0.4, "b": -0.32, "c": -0.2, "d": -1.5}
w = {"a": -0.0, "b": -0.0, "c": -0.0, "d": -1.5}
w.update((key, value * args.kernel_factor/10) for key, value in w.items())

d = {"a": 1, "b": 1, "c": 2, "d": 1}
d.update((key, value * args.delay_factor/10) for key, value in d.items())

d_P_F = 1

def delay_upscaling(my_dict, d_P_F):
    if model.dT == 1.0:
        for key, value in my_dict.items():
            tup = np.array(np.multiply((value, value), delay_param / (new_resolution / 20)))
            my_dict.update({key : tup})
        # for key in my_dict:
        #     my_dict[key] *= int(delay_param / np.mean(new_resolution / 20))
        d_P_F *= round(delay_param * (args.feedforward_delay/10))
    elif model.dT == 62.0:
        pass
    else:
        sys.exit('Delays for unforeseen model.dT!')
    return d_P_F

d_P_F = delay_upscaling(d, d_P_F)

# # All 256 cells of S
# ind_s = [[(r, c) for c in np.arange(0.1*new_resolution, new_resolution-0.1*new_resolution)]
#           for r in np.arange(0.1*new_resolution, new_resolution-0.1*new_resolution)]

# ind_s = list(map(np.int32, ind_s))

# # Mapping I-S cells
# def coord_to_pixel(arr):
#     out = []
#     for row in arr:
#         out.append([(x[0] * new_resolution) + x[1] for x in row])
#     return out

# for i, cell_row in enumerate(ind_s):
#     for j in np.arange(len(cell_row)):
#         r, c = cell_row[j]
#         i_ring_buf_ = [[(r, c-1), (r-1, c), (r, c+1), (r+1, c)],
#                           [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)],
#                           [(r-2, c), (r, c-2), (r, c+2), (r+2, c)]]
        
#         s_ = (r * new_resolution) + c
#         i_ring = coord_to_pixel(i_ring_buf_)
#         i_ring_ = np.concatenate(i_ring)
        
#         for k, i_ in enumerate(i_ring_):
#             if np.floor_divide(k, 4) == 0:
#                 w_I_S[i_, s_] = w["a"]
#                 d_I_S[i_, s_] = d["a"]
#             elif np.floor_divide(k, 4) == 1:
#                 w_I_S[i_, s_] = w["b"]
#                 d_I_S[i_, s_] = d["b"]
#             elif np.floor_divide(k, 4) == 2:
#                 w_I_S[i_, s_] = w["c"]
#                 d_I_S[i_, s_] = d["c"]

c_ij = ["c",
        "b", "a", "b",
        "c", "a", "d", "a", "c",
        "b", "a", "b",
        "c"]

# Row offset
i_i = np.asarray([-2,
                  -1, -1, -1,
                  0, 0, 0, 0, 0,
                  1, 1, 1,
                  2])

# Column offset
i_j = np.asarray([0,
                  -1, 0, 1,
                  -2, -1, 0, 1, 2,
                  -1, 0, 1,
                  0])

I_S_kernel = list(zip(i_i, i_j))

d_ij = []
for i, ind in enumerate(I_S_kernel):
    if (ind[0] * ind[1]):
        d_ij.append(np.mean(d[c_ij[i]]))
    elif ind[0]:
        d_ij.append(d[c_ij[i]][1])
    elif ind[1]:
        d_ij.append(d[c_ij[i]][0])
    elif not np.prod(ind):
        d_ij.append(np.mean(d[c_ij[i]]))
        
d_ij = np.array(np.rint(d_ij), dtype=int)

w_ij = np.asarray([w[c] for c in c_ij])
# d_ij = np.asarray([d[c] for c in c_ij])

g_idx = []

# Loop through central cells of S
for s_i in np.arange(int(np.ceil(0.1*new_resolution[1])), int(new_resolution[1]-0.1*new_resolution[1])):
    for s_j in np.arange(int(np.ceil(0.1*new_resolution[0])), int(new_resolution[0]-0.1*new_resolution[0])):
        
        # Convert s_i, s_j into flat index for S
        s_idx = np.ravel_multi_index([s_i, s_j], (new_resolution[1], new_resolution[0]))
        
        # Convert add offsets to s_i and s_j to get kernel indices
        i_idx = np.ravel_multi_index([s_i + i_i, s_j + i_j], (new_resolution[1], new_resolution[0]))
        
        # Stick weights and delays into kernel
        g_idx.append([i_idx, s_idx])
        # w_I_S[i_idx, s_idx] = w_ij
        # d_I_S[i_idx, s_idx] = d_ij

g_idx = np.asarray(g_idx, dtype=object)

# Returns indices of non-zero elements of w_I_S
# I_ind, S_ind = np.nonzero(w_I_S)

I_ind = np.concatenate([i[0] for i in g_idx]).astype(np.int32)
S_ind = np.concatenate([[i[1]] * len(w_ij) for i in g_idx]).astype(np.int32)

w_I_S = np.concatenate([w_ij] * int(len(I_ind) / len(w_ij)))
d_I_S = np.concatenate([d_ij] * int(len(I_ind) / len(d_ij)))

w_I_S_ = np.copy(w_I_S)
d_I_S_ = np.copy(d_I_S)
w_E_S_WEIGHT_ = deepcopy(w_E_S_WEIGHT)

##############################################################################

if 'connection' in args:
    if args.connection == 'E':
        w_I_S_ *= 0
        d_I_S_ *= 0
        w_E_S_WEIGHT_ = w_E_S_WEIGHT
    elif args.connection == 'I':
        w_I_S_ = w_I_S
        d_I_S_ = d_I_S
        w_E_S_WEIGHT_ = 0
    elif args.connection == 'S':
        w_I_S_ = w_I_S
        d_I_S_ = d_I_S
        w_E_S_WEIGHT_ = w_E_S_WEIGHT
else:
    w_I_S_ = w_I_S
    d_I_S_ = d_I_S
    w_E_S_WEIGHT_ = w_E_S_WEIGHT

# model.add_synapse_population(
#     "p_tsynapse", "DENSE_INDIVIDUALG", NO_DELAY, P, T_pop,
#     "StaticPulse", {}, {"g": w_P_T.flatten()}, {}, {},
#     "DeltaCurr", {}, {})

# model.add_synapse_population(
#     "t_esynapse", "DENSE_INDIVIDUALG", NO_DELAY, T_pop, E_pop,
#     graded_synapse_model, {"theta": 0.0}, {"w": w_T_E.flatten()}, {}, {},
#     "DeltaCurr", {}, {})

# model.add_synapse_population(
#     "i_tsynapse", "DENSE_INDIVIDUALG", NO_DELAY, I_pop, T_pop,
#     graded_synapse_model, {"theta": 0.0}, {"w": w_I_T.flatten()}, {}, {},
#     "DeltaCurr", {}, {})

model.add_synapse_population(
    "p_esynapse", "SPARSE_GLOBALG", NO_DELAY, P, E_pop,
    "StaticPulse", {}, {"g": w_P_E_WEIGHT}, {}, {},
    "DeltaCurr", {}, {},
    init_connectivity("OneToOne", {}))

model.add_synapse_population(
    "p_isynapse", "SPARSE_GLOBALG", NO_DELAY, P, I_pop,
    "StaticPulse", {}, {"g": w_P_I_WEIGHT}, {}, {},
    "DeltaCurr", {}, {},
    init_connectivity("OneToOne", {}))

# i_i = model.add_synapse_population(
#     "self_inhib_isynapse", "SPARSE_GLOBALG", int(np.min(d_ij)), I_pop, I_pop,
#     graded_synapse_model, {"theta": 0.0}, {"w": w_I_I_WEIGHT_}, {}, {},
#     "DeltaCurr", {}, {})

# i_i.set_sparse_connections(pix, pix)

e_s = model.add_synapse_population(
    "e_ssynapse", "SPARSE_GLOBALG", NO_DELAY, E_pop, S_pop,
    graded_synapse_model, {"theta": 0.0}, {"w": w_E_S_WEIGHT_}, {}, {}, # 0.3
    "DeltaCurr", {}, {})

e_s.set_sparse_connections(pix, pix)

# i_s = model.add_synapse_population(
#     "i_ssynapse", "SPARSE_INDIVIDUALG", NO_DELAY, I_pop, S_pop,
#     i_sgraded_synapse_model, {"theta": 0.0}, {"w": w_I_S[I_ind, S_ind], "d": d_I_S[I_ind, S_ind]}, {}, {},
#     "DeltaCurr", {}, {})

i_s = model.add_synapse_population(
    "i_ssynapse", "SPARSE_INDIVIDUALG", NO_DELAY, I_pop, S_pop,
    i_sgraded_synapse_model, {"theta": 0.0}, {"w": w_I_S_, "d": d_I_S_}, {}, {},
    "DeltaCurr", {}, {})

i_s.pop.set_max_dendritic_delay_timesteps(int(max(d_ij)) + 1)
i_s.set_sparse_connections(I_ind, S_ind)

## DENSE as SPARSE overheads make the switch non cost-effective

model.add_synapse_population(
    "p_fsynapse", "DENSE_INDIVIDUALG", NO_DELAY, P, F_pop,
    "StaticPulse", {}, {"g": w_P_F.flatten()}, {}, {},
    "DeltaCurr", {}, {})

model.add_synapse_population(
    "s_LGMDsynapse", "DENSE_INDIVIDUALG", NO_DELAY, S_pop, LGMD_pop,
    "StaticPulse", {}, {"g": w_S_LGMD.flatten()}, {}, {},
    "DeltaCurr", {}, {})

model.add_synapse_population(
    "f_LGMDsynapse", "DENSE_INDIVIDUALG", d_P_F, F_pop, LGMD_pop,
    graded_synapse_model, {"theta": 0.15}, {"w": w_F_LGMD}, {}, {}, # 0.15 0.03
    "DeltaCurr", {}, {})

# model.add_synapse_population(
#     "LGMD_DCMDsynapse", "DENSE_INDIVIDUALG", NO_DELAY, LGMD_pop, DCMD_pop,
#     "StaticPulse", {}, {"g": w_LGMD_DCMD}, {}, {},
#     "DeltaCurr", {}, {})

##############################################################################

# model.add_synapse_population(
#     "p1_t1synapse", "DENSE_INDIVIDUALG", NO_DELAY, P1, T1_pop,
#     "StaticPulse", {}, {"g": w_P_T.flatten()}, {}, {},
#     "DeltaCurr", {}, {})

# model.add_synapse_population(
#     "t1_e1synapse", "DENSE_INDIVIDUALG", NO_DELAY, T1_pop, E1_pop,
#     graded_synapse_model, {"theta": 0.0}, {"w": w_T_E.flatten()}, {}, {},
#     "DeltaCurr", {}, {})

# model.add_synapse_population(
#     "i1_t1synapse", "DENSE_INDIVIDUALG", NO_DELAY, I1_pop, T1_pop,
#     graded_synapse_model, {"theta": 0.0}, {"w": w_I_T.flatten()}, {}, {},
#     "DeltaCurr", {}, {})

model.add_synapse_population(
    "p1_e1synapse", "SPARSE_GLOBALG", NO_DELAY, P1, E1_pop,
    "StaticPulse", {}, {"g": w_P_E_WEIGHT}, {}, {},
    "DeltaCurr", {}, {},
    init_connectivity("OneToOne", {}))

model.add_synapse_population(
    "p1_i1synapse", "SPARSE_GLOBALG", NO_DELAY, P1, I1_pop,
    "StaticPulse", {}, {"g": w_P_I_WEIGHT}, {}, {},
    "DeltaCurr", {}, {},
    init_connectivity("OneToOne", {}))

# i1_i1 = model.add_synapse_population(
#     "self_inhib_i1synapse", "SPARSE_GLOBALG", int(np.min(d_ij)), I1_pop, I1_pop,
#     graded_synapse_model, {"theta": 0.0}, {"w": w_I_I_WEIGHT_}, {}, {},
#     "DeltaCurr", {}, {})

# i1_i1.set_sparse_connections(pix, pix)

e1_s1 = model.add_synapse_population(
    "e1_s1synapse", "SPARSE_GLOBALG", NO_DELAY, E1_pop, S1_pop,
    graded_synapse_model, {"theta": 0.0}, {"w": w_E_S_WEIGHT_}, {}, {}, # 0.3
    "DeltaCurr", {}, {})

e1_s1.set_sparse_connections(pix, pix)

i1_s1 = model.add_synapse_population(
    "i1_s1synapse", "SPARSE_INDIVIDUALG", NO_DELAY, I1_pop, S1_pop,
    i_sgraded_synapse_model, {"theta": 0.0}, {"w": w_I_S_, "d": d_I_S_}, {}, {},
    "DeltaCurr", {}, {})

i1_s1.pop.set_max_dendritic_delay_timesteps(int(max(d_ij)) + 1)
i1_s1.set_sparse_connections(I_ind, S_ind)

# model.add_synapse_population(
#     "i1self", "SPARSE_GLOBALG", NO_DELAY, I1_pop, I1_pop,
#     i_sgraded_synapse_model, {"theta": 0.0}, {"w": w_I_S_, "d": d_I_S_}, {}, {},
#     "DeltaCurr", {}, {},
#     init_connectivity("OneToOne", {}))

## DENSE as SPARSE overheads make the switch non cost-effective

model.add_synapse_population(
    "p1_f1synapse", "DENSE_INDIVIDUALG", NO_DELAY, P1, F1_pop,
    "StaticPulse", {}, {"g": w_P_F.flatten()}, {}, {},
    "DeltaCurr", {}, {})

model.add_synapse_population(
    "s1_LGMD1synapse", "DENSE_INDIVIDUALG", NO_DELAY, S1_pop, L1_pop,
    "StaticPulse", {}, {"g": w_S_LGMD.flatten()}, {}, {},
    "DeltaCurr", {}, {})

model.add_synapse_population(
    "f1_LGMDsynapse", "DENSE_INDIVIDUALG", d_P_F, F1_pop, L1_pop,
    graded_synapse_model, {"theta": 0.15}, {"w": w_F_LGMD}, {}, {}, # 0.15
    "DeltaCurr", {}, {})

# # model.add_synapse_population(
# #     "LGMD1_DCMD1synapse", "DENSE_INDIVIDUALG", NO_DELAY, L1_pop, D1_pop,
# #     "StaticPulse", {}, {"g": w_LGMD_DCMD}, {}, {},
# #     "DeltaCurr", {}, {})

##############################################################################

model.build()
model.load(num_recording_timesteps=PRESENT_TIMESTEPS)

##############################################################################

# Tv = np.empty((PRESENT_TIMESTEPS, T_pop.size))
Ev = np.empty((PRESENT_TIMESTEPS, E_pop.size))
Iv = np.empty((PRESENT_TIMESTEPS, I_pop.size))
Sv = np.empty((PRESENT_TIMESTEPS, S_pop.size))
Fv = np.empty((PRESENT_TIMESTEPS, F_pop.size))
Lv = np.empty((PRESENT_TIMESTEPS, LGMD_pop.size))
# Dv = np.empty((PRESENT_TIMESTEPS, DCMD_pop.size))
i_s_curr_pos = np.empty((PRESENT_TIMESTEPS, S_pop.size))

# Tv1 = np.empty((PRESENT_TIMESTEPS, T1_pop.size))
Ev1 = np.empty((PRESENT_TIMESTEPS, E1_pop.size))
Iv1 = np.empty((PRESENT_TIMESTEPS, I1_pop.size))
Sv1 = np.empty((PRESENT_TIMESTEPS, S1_pop.size))
Fv1 = np.empty((PRESENT_TIMESTEPS, F1_pop.size))
Lv1 = np.empty((PRESENT_TIMESTEPS, L1_pop.size))
# Dv1 = np.empty((PRESENT_TIMESTEPS, D1_pop.size))
i_s_curr_neg = np.empty((PRESENT_TIMESTEPS, S1_pop.size))

# Tv_view = T_pop.vars["V"].view
Ev_view = E_pop.vars["V"].view
Iv_view = I_pop.vars["V"].view
Sv_view = S_pop.vars["V"].view
Fv_view = F_pop.vars["V"].view
Lv_view = LGMD_pop.vars["V"].view
# Dv_view = DCMD_pop.vars["V"].view
i_s_view_pos = S_pop.vars["i"].view

# Tv1_view = T1_pop.vars["V"].view
Ev1_view = E1_pop.vars["V"].view
Iv1_view = I1_pop.vars["V"].view
Sv1_view = S1_pop.vars["V"].view
Fv1_view = F1_pop.vars["V"].view
Lv1_view = L1_pop.vars["V"].view
# Dv1_view = D1_pop.vars["V"].view
i_s_view_neg = S_pop.vars["i"].view

while model.t < sim_time:
    model.step_time()
    
    model.pull_recording_buffers_from_device();
    
    # T_pop.pull_var_from_device("V")
    E_pop.pull_var_from_device("V")
    I_pop.pull_var_from_device("V")
    S_pop.pull_var_from_device("V")
    F_pop.pull_var_from_device("V")
    LGMD_pop.pull_var_from_device("V")
    # DCMD_pop.pull_var_from_device("V")
    S_pop.pull_var_from_device("i")
    
    # T1_pop.pull_var_from_device("V")
    E1_pop.pull_var_from_device("V")
    I1_pop.pull_var_from_device("V")
    S1_pop.pull_var_from_device("V")
    F1_pop.pull_var_from_device("V")
    L1_pop.pull_var_from_device("V")
    S1_pop.pull_var_from_device("i")
    # D1_pop.pull_var_from_device("V")
    
    # Tv[model.timestep - 1,:] = Tv_view[:]
    Ev[model.timestep - 1,:] = Ev_view[:]
    Iv[model.timestep - 1,:] = Iv_view[:]
    Sv[model.timestep - 1,:] = Sv_view[:]
    Fv[model.timestep - 1,:] = Fv_view[:]
    Lv[model.timestep - 1,:] = Lv_view[:]
    # Dv[model.timestep - 1,:] = Dv_view[:]
    i_s_curr_pos[model.timestep - 1,:] = i_s_view_pos[:]
    
    # Tv1[model.timestep - 1,:] = Tv1_view[:]
    Ev1[model.timestep - 1,:] = Ev1_view[:]
    Iv1[model.timestep - 1,:] = Iv1_view[:]
    Sv1[model.timestep - 1,:] = Sv1_view[:]
    Fv1[model.timestep - 1,:] = Fv1_view[:]
    Lv1[model.timestep - 1,:] = Lv1_view[:]
    # Dv1[model.timestep - 1,:] = Dv1_view[:]
    i_s_curr_neg[model.timestep - 1,:] = i_s_view_neg[:]
    
# ----------------------------------------------------------------------------
# Saving synaptic weights Isyn

if 'connection' and 'polarity' in args:
    if args.polarity == '+':
        # Synaptic current and post-synaptic membrane potential of S
        i_s_curr = i_s_curr_pos
        post_syn_mp = Sv
    else:
        i_s_curr = i_s_curr_neg
        post_syn_mp = Sv1
    save_string = '_'.join(['i_s_curr', args.polarity, args.connection, filehead]) + '.npz'
    os.makedirs('currents', exist_ok=True)
    np.savez(os.path.join('currents', save_string), new_resolution=new_resolution, i_syn=i_s_curr, post_v=post_syn_mp)

# ----------------------------------------------------------------------------

P_spike_times, P_spike_ids = P.spike_recording_data
L_spike_times, L_spike_ids = LGMD_pop.spike_recording_data
S_spike_times, S_spike_ids = S_pop.spike_recording_data
# D_spike_times, D_spike_ids = DCMD_pop.spike_recording_data

P1_spike_times, P1_spike_ids = P1.spike_recording_data
L1_spike_times, L1_spike_ids = L1_pop.spike_recording_data
S1_spike_times, S1_spike_ids = S1_pop.spike_recording_data
# D1_spike_times, D1_spike_ids = D1_pop.spike_recording_data

# Tv = np.mean(Tv, axis=1)
def neuron_pop_activity(neuron_pop):
    val = []
    for i in range(neuron_pop.shape[0]):
        neuron_pop_hist = neuron_pop[i,:].reshape(new_resolution[1], new_resolution[0])
        val.append(np.nanmean(skimage.measure.block_reduce(neuron_pop_hist, (2,2), np.median)))
    val = np.asarray(val, dtype=float)
    return val

Ev_smooth = neuron_pop_activity(Ev)
Iv_smooth = neuron_pop_activity(Iv)
Sv_smooth = neuron_pop_activity(Sv)
# Ev = np.mean(Ev, axis=1)
# Iv = np.mean(Iv, axis=1)
# Sv = np.mean(Sv, axis=1)

# Tv1 = np.mean(Tv1, axis=1)
Ev1_smooth = neuron_pop_activity(Ev1)
Iv1_smooth = neuron_pop_activity(Iv1)
Sv1_smooth = neuron_pop_activity(Sv1)
# i_s1_curr = np.mean(i_s1_curr, axis=1)

# print([int(float) for float in L1_spike_times], time1)
# print(len(L1_spike_times))

##############################################################################
    
def numSpikes(spike_times):
    if not spike_times.any():
        n_spk = 0
    else:
        if '800' in args.filename:
            clip_pos = 800
        elif '1000' in args.filename:
            clip_pos = 1000
        else:
            clip_pos = 0
        spk_corr = len(np.unique(np.clip(spike_times, clip_pos, None))) - 1 # spike_times[0]
        n_spk = spk_corr
    return n_spk

fig, axes = plt.subplots(7, sharex=True)

t_plot = np.arange(sim_time, step=model.dT)

axes[0].scatter(P_spike_times, P_spike_ids, s=1); axes[0].set_title("P")
axes[1].plot(t_plot, Ev_smooth[:], c = 'g', label='E')
axes[1].plot(t_plot, Iv_smooth[:], c = 'r', label='I'); axes[1].legend(loc="upper right")
# axes[1].plot(t_plot, Tv[:], c = 'c', label='T'); 
ax_s = axes[2].plot(t_plot, Sv_smooth[:], c = 'g', label='S')

ax = axes[2].twinx()
ax_f = ax.plot(t_plot, Fv[:], c = 'r', label='F');
axs = ax_s + ax_f
labs = [l.get_label() for l in axs]
axes[2].legend(axs, labs, loc="upper right")

axes[3].scatter(S_spike_times, S_spike_ids, s=1); axes[3].set_title("S")
axes[4].plot(t_plot, Lv[:]); axes[4].set_title("LGMD")
axes[5].scatter(L_spike_times, L_spike_ids, s=1); axes[5].set_title("LGMD")

if len(L_spike_times > 0):
    bin_nums = int(np.ceil((np.max(L_spike_times) - np.min(L_spike_times))/100))
    num = np.max(L_spike_times) - np.min(L_spike_times)
    if bin_nums == 0:
        bin_nums = 1
        num = 0.0
else:
    bin_nums = 1
    num = 'Null'

histy, _, _ = axes[6].hist(L_spike_times, bins=bin_nums)
axes[6].minorticks_on(); axes[6].grid()

print(numSpikes(L_spike_times), time, histy.max())
# print(numSpikes(D_spike_times), bin_nums)

fig.supxlabel("Time [ms]")

print('w: ', args.kernel_factor/10)
print('d: ', args.delay_factor/10)
print('E: ', (args.E_syn/10), 'I: ', (args.I_syn/10))
print('Tc: ', args.time_constant_factor)

# if '60' in filename:
#     if 'reverse' in filename:
#         t_ind = np.array([0, 100])
#     else:
#         t_ind = np.array([1450, 1750])
# elif '479' in filename:
#     if 'reverse' in filename:
#         t_ind = np.array([0, 150])
#     else:
#         t_ind = np.array([13950, 14100])
# else:    
#     if 'reverse' in filename:
#         t_ind = np.array([0, 200])
#     else:
#         t_ind = np.array([3250, 3500])
    
# if len(L_spike_times):
#     ll_array = np.where(L_spike_times >= t_ind[0])[0]
#     ul_array = np.where(L_spike_times <= t_ind[1])[0]
    
#     if ll_array.size > 0 and ul_array.size > 0:
#         ll = ll_array[0]
#         ul = ul_array[-1] + 1
#         L_opt_size = len(L_spike_times[ll:ul])
#     else:
#         L_opt_size = 0
        
#     L_score = L_opt_size / len(L_spike_times)
#     print(L_score)
# else:
#     print('No LGMD spikes!')
#     print(0)

# fig, axes = plt.subplots(3, sharex=True)

# t_plot = np.arange(sim_time, step=model.dT)

# axes[0].scatter(P_spike_times, P_spike_ids, s=1); axes[0].set_title("P")
# ax_s = axes[1].plot(t_plot, Sv[:], c = 'g', label='S')

# ax = axes[1].twinx()
# ax_f = ax.plot(t_plot, Fv[:], c = 'r', label='F');
# axs = ax_s + ax_f
# labs = [l.get_label() for l in axs]
# axes[1].legend(axs, labs, loc="upper right")

# axes[2].plot(t_plot, Lv[:]); axes[2].set_title("LGMD")

# fig.supxlabel("Time [ms]")

#############################################################################

fig1, axes1 = plt.subplots(7, sharex=True)

axes1[0].scatter(P1_spike_times, P1_spike_ids, s=1); axes1[0].set_title("P1")
axes1[1].plot(t_plot, Ev1_smooth[:], c = 'g', label='E1')
axes1[1].plot(t_plot, Iv1_smooth[:], c = 'r', label='I1'); axes1[1].legend(loc="upper right")
# axes[1].plot(t_plot, Tv[:], c = 'c', label='T'); 
ax_s1 = axes1[2].plot(t_plot, Sv1_smooth[:], c = 'g', label='S1')

ax1 = axes1[2].twinx()
ax_f1 = ax1.plot(t_plot, Fv1[:], c = 'r', label='F1');
axs1 = ax_s1 + ax_f1
labs1 = [l.get_label() for l in axs1]
axes1[2].legend(axs1, labs1, loc="upper right")

axes1[3].scatter(S1_spike_times, S1_spike_ids, s=1); axes1[3].set_title("S1")
axes1[4].plot(t_plot, Lv1[:]); axes1[4].set_title("LGMD1")
axes1[5].scatter(L1_spike_times, L1_spike_ids, s=1); axes1[5].set_title("LGMD1")

if len(L1_spike_times > 0):
    bin_nums = int(np.ceil((np.max(L1_spike_times) - np.min(L1_spike_times))/100))
    num = np.max(L1_spike_times) - np.min(L1_spike_times)
    if bin_nums == 0:
        bin_nums = 1
        num = 0.0
else:
    bin_nums = 1
    num = 'Null'

histy1, _, _ = axes1[6].hist(L1_spike_times, bins=bin_nums)
axes1[6].minorticks_on(); axes1[6].grid()

print(numSpikes(L1_spike_times), time, histy1.max())
# print(numSpikes(D_spike_times), bin_nums)

fig1.supxlabel("Time [ms]")

print('w: ', args.kernel_factor/10)
print('d: ', args.delay_factor/10)
print('E: ', (args.E_syn/10), 'I: ', (args.I_syn/10))
print('Tc: ', args.time_constant_factor)

#############################################################################

# fig1, axes1 = plt.subplots(3, sharex=True)

# t_plot = np.arange(sim_time, step=model.dT)

# axes1[0].scatter(P1_spike_times, P1_spike_ids, s=1); axes1[0].set_title("P")
# ax1_s = axes1[1].plot(t_plot, Sv1[:], c = 'g', label='S')

# ax1 = axes1[1].twinx()
# ax1_f = ax1.plot(t_plot, Fv1[:], c = 'r', label='F');
# ax1s = ax1_s + ax1_f
# labs1 = [l.get_label() for l in ax1s]
# axes1[1].legend(ax1s, labs1, loc="upper right")

# axes1[2].plot(t_plot, Lv1[:]); axes1[2].set_title("LGMD")

# fig1.supxlabel("Time [ms]")

##############################################################################

dfp = pd.DataFrame({'P_spike_times': P_spike_times, 'P_spike_ids': P_spike_ids})
dfs = pd.DataFrame({'S_spike_times': S_spike_times, 'S_spike_ids': S_spike_ids})
dfl = pd.DataFrame({'L_spike_times': L_spike_times, 'L_spike_ids': L_spike_ids})

dfp1 = pd.DataFrame({'P1_spike_times': P1_spike_times, 'P1_spike_ids': P1_spike_ids})
dfs1 = pd.DataFrame({'S1_spike_times': S1_spike_times, 'S1_spike_ids': S1_spike_ids})
dfl1 = pd.DataFrame({'L1_spike_times': L1_spike_times, 'L1_spike_ids': L1_spike_ids})

if args.save_output:
    
    save_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), 'lgmd_output/grid_search_bio_plausible', filehead)
    
    os.makedirs(save_dir, exist_ok=True)
    
    param_dict = {'e':args.E_syn, 'i':args.I_syn, 'k':args.kernel_factor,
                  'tc':args.time_constant_factor, 'd':args.delay_factor,
                  'nt':args.noise_threshold, 'ffk':args.feedforward_strength,
                  'ffd':args.feedforward_delay}
    
    param_dict.update((key, int(value)) for key, value in param_dict.items())
    
    save_name = os.path.join(save_dir, '_'.join([filehead, f'e{param_dict["e"]}', f'i{param_dict["i"]}', f'k{param_dict["k"]}',
                                                 f'tc{param_dict["tc"]}', f'd{param_dict["d"]}', f'nt{param_dict["nt"]}',
                                                 f'ffk{param_dict["ffk"]}', f'ffd{param_dict["ffd"]}']))
    
    ds = pd.concat([dfp, dfs, dfl, dfp1, dfs1, dfl1], axis=1)
    ds.to_csv(save_name + '.csv')
    np.savez(save_name, Ev=Ev, Iv=Iv, Ev1=Ev1, Iv1=Iv1, Sv=Sv, Sv1=Sv1, 
             Fv=Fv, Fv1=Fv1, Lv=Lv, Lv1=Lv1, time=t_plot)
    
# df.to_pickle("./S_spikes.pkl")

timesteps_per_frame_visualize = 1

if args.show_plot:
    def read_show_spikes(cell_type):
        
        cell_resolution = new_resolution
        
        if cell_type == "P":
            df = dfp1
        elif cell_type == "S":
            df = dfs1
            if not dfs1.size > 0:
                df = dfp1
                print("Cell type changed to P")
                if not dfp1.size > 0:
                    df = dfp
                    print("Cell changed to ON events")
        elif cell_type == "L":
            df = dfl1
            cell_resolution = np.array([1, 1])
    
        spike_list = []
        
        timesteps = list(np.unique(df.iloc[:,0]))
        timesteps = [int(x) for x in timesteps]
        
        [spike_list.append(list(df.iloc[np.where(df.iloc[:,0] == t)[0],1])) for t in timesteps]
        frames = []
            
        for i in np.arange(len(spike_list)):
            frame_input_spikes = np.asarray(spike_list[i], dtype=int)
            time = timesteps[i]
            frame = time // timesteps_per_frame_visualize
            
            # Split into X and Y
            frame_input_x = np.floor_divide(frame_input_spikes, cell_resolution[0])
            frame_input_y = np.remainder(frame_input_spikes, cell_resolution[0])
            
            # Take histogram so as to assemble frame
            if np.sum(cell_resolution) > 1:
                frame_input_image = np.histogram2d(frame_input_x, frame_input_y,
                                                [np.arange(cell_resolution[1] + 1), 
                                                  np.arange(cell_resolution[0] + 1)])[0]
            else:
                frame_input_image = np.histogram(frame_input_x,
                                                bins=np.arange(cell_resolution[1] + 1))[0]
            
            # Add frame to list
            frames.append((frame, frame_input_image))
            
            num_frames = frames[-1][0] + 1
            timestep_inputs = np.zeros((cell_resolution[1], cell_resolution[0], num_frames), dtype=float)
            for i, f in frames:
                timestep_inputs[:,:,i] += f
        return timestep_inputs
    
    if (dfp.size or dfs.size) > 0:
        timestep_inputs = read_show_spikes(args.cell_type)
        
        fig, axis = plt.subplots()
        
        input_image_data = timestep_inputs[:,:,0]
        input_image = axis.imshow(input_image_data, interpolation="nearest", cmap="jet", vmin=0.0, 
                                  vmax=np.max(timestep_inputs))# float(timesteps_per_frame))
        
        fig.colorbar(input_image, ax=axis)
        
        class Index:
            def __init__(self):
                self.ind = 0
                self.paused = False
        
            def _next(self, event):
                self.ind += 1
                i = self.ind % timestep_inputs.shape[2]
                input_image_data = timestep_inputs[:,:,i]
                input_image.set_array(input_image_data)
                time.set_text(str("Time = %u ms" %(i * timesteps_per_frame_visualize)))
                plt.draw()
        
            def _prev(self, event):
                self.ind -= 1
                i = self.ind % timestep_inputs.shape[2]
                input_image_data = timestep_inputs[:,:,i]
                input_image.set_array(input_image_data)
                time.set_text(str("Time = %u ms" %(i * timesteps_per_frame_visualize)))
                plt.draw()
        
            def _toggle_pause(self, event):
                if self.paused:
                    ani.resume()
                else:
                    ani.pause()
                self.paused = not self.paused
        
        callback = Index()
        axprev = plt.axes([0.1, 0.9, 0.1, 0.07])
        axnext = plt.axes([0.21, 0.9, 0.1, 0.07])
        axpause = plt.axes([0.32, 0.9, 0.1, 0.07])
        
        bnext = Button(axnext, 'Next')
        bprev = Button(axprev, 'Previous')
        bpause = Button(axpause, 'Pause')
        
        bnext.on_clicked(callback._next)
        bprev.on_clicked(callback._prev)
        bpause.on_clicked(callback._toggle_pause)
        
        axtext = fig.add_axes([0.5, 0.95, 0.4, 0.05]) # plt.axes([left, bottom, width, height])
        axtext.axis("off")
        time = axtext.text(0.5, 0.5, str("Time = 0 ms"), ha="left", va="top")
        
        def updatefig(frame):
            global input_image_data, input_image, timesteps_per_frame_visualize
            
            # Decay image data
            # input_image_data *= 0.9
        
            # Loop through all timesteps that occur within frame
            input_image_data = timestep_inputs[:,:,frame]
        
            # Set image data
            input_image.set_array(input_image_data)
            
            time.set_text(str("Time = %u ms" %(frame*timesteps_per_frame_visualize)))
        
            # Return list of artists which we have updated
            # **YUCK** order of these dictates sort order
            # **YUCK** score_text must be returned whether it has
            # been updated or not to prevent overdraw
            return input_image, time
        
        # Play animation
        ani = animation.FuncAnimation(fig, updatefig, range(timestep_inputs.shape[2]), 
                                      interval=timesteps_per_frame_visualize, blit=True, repeat=True)
        
    else:
        print('No P spikes!')
        
    plt.show()

############################################################################

# writergif = animation.PillowWriter(fps=33)
# ani.save("animation_Sv_updated.gif", writer = writergif)

# a = np.asarray(spike_list, dtype="object")
# np.savetxt("foo.csv", a, delimiter=",", fmt="%s")
