import argparse
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy.integrate import simpson
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--shape', type=str, required=True)
parser.add_argument('-l', '--l_v', type=str, required=True)
parser.add_argument('-viz', '--visualise', action='store_true', default=False)
parser.add_argument('-v', '--view', type=str)

args = parser.parse_args()

class evaluation():
    def __init__(self, save_folder):
        self.save_folder = save_folder
        self.stims_dir = '_'.join(['events', args.shape, args.l_v, '120_original_stimuli_frames_1000_aug'])
        self.ds_stims = os.path.join(os.getcwd(), 'grid_search_self_inhibition', 
                                     'downsampled_stims', '_'.join(['events', args.shape, args.l_v, 
                                                                   '120_original_stimuli_frames_downsampled'])) + '.p'
        
    def ddf(self, x, sigma):
        delta = 1 / (self.sigma * np.sqrt(2*np.pi)) * np.exp(-(x**2) / (2 * self.sigma**2))
        return delta
    
    def superposition(self, x, FIR):
        
        sum_FIR = np.zeros(self.time + 1)
        d = x.ravel()
        f = lambda s: np.unravel_index(s.index, x.shape)
        inds = pd.Series(d).groupby(d).apply(f)
        
        for i, t in enumerate(inds.index):
            if t <= self.time:
                t = int(t)
                sum_FIR[t] = np.sum(FIR[inds.iloc[i]])
        
        t_sum = np.arange(self.time + 1)
        
        return t_sum, np.array(sum_FIR)
    
    def spike_density_function(self, t_i, r, sigma):
        self.sigma = sigma
        
        # fig, axis = plt.subplots()
        sdf = []
        for i, t in enumerate([t_i]):
            x1, x2 = (j + t for j in (r, -r))
            x = np.linspace(x1, x2, 201)
            FIR = self.ddf(x - t, self.sigma) # finite energy impulse response
            
            # Clipping Gaussians to time of collision
            # FIR[np.where(x > len(self.t_vals))] = 0
            
            # axis.plot(x, FIR)
            if FIR.size > 0:
                t_sum, sum_FIR = self.superposition(x, FIR)
            else:
                t_sum, sum_FIR = np.array([]), np.array([])

            sdf.append((t_sum, sum_FIR))
            
            # Proof
            if FIR.size > 0:
                area = simpson(sum_FIR, t_sum)
                # print(area)
                
        return sdf
        
    def stims_sweep(self):
        self.stim_dir_path = os.path.join(os.getcwd(), self.save_folder, self.stims_dir)
        self.stims = [stim for stim in os.listdir(self.stim_dir_path)]
    
    def get_sdfs(self):
        # Get original 
        neuron_ds = self.get_original()
        # SDF of down-sampled version
        sdf_ds = self.spike_density_function(neuron_ds[:,1], 100, sigma=20)
        t_sum_ds, sum_FIR_ds = sdf_ds[0]
        
        # Get paths of all pickled sweeps for that stimulus
        self.stims_sweep()
        
        if not args.visualise:
            # Score matrix
            score_mat = []
            
            # Load spike_info from pickle
            for stim in self.stims:
                # Obtain hyperparams and their values
                hypers = [re.split(r'(\d+)', s)[0:2] for s in stim.split('.')[0].split('_')]
                
                spike_info = pickle.load(open(os.path.join(self.stim_dir_path, stim), "rb"))
                sdf = self.spike_density_function(spike_info["B1"][0], 100, sigma=20)
                # Meant for ON and OFF channels but only looking at OFF channel here
                t_sum, sum_FIR = sdf[0]
                
                t_common, sdf_ind, t_ds_ind = np.intersect1d(t_sum.astype(int), t_sum_ds.astype(int), return_indices=True)
                rho = np.corrcoef(sum_FIR[sdf_ind], sum_FIR_ds[t_ds_ind])
                score = rho[0, 1] * 1
                
                # print(score)
                
                labels = [l[0] for l in hypers]
                params = [l[1] for l in hypers]
                
                score_mat.append((*np.asarray(params, dtype=int), score))
        
        else:
            # Load specific spike file
            args_load = {"bi":25, "ib":20, "tc":10, "d":3, "nt":6}
            specific_file = '_'.join([(k + str(v)) for (k, v) in args_load.items()]) + '.p'
            
            spike_info = pickle.load(open(os.path.join(self.stim_dir_path, specific_file), "rb"))
            sdf = self.spike_density_function(spike_info["B1"][0], 100, sigma=20)
            # Meant for ON and OFF channels but only looking at OFF channel here
            t_sum, sum_FIR = sdf[0]
            
            t_common, sdf_ind, t_ds_ind = np.intersect1d(t_sum.astype(int), t_sum_ds.astype(int), return_indices=True)
            rho = np.corrcoef(sum_FIR[sdf_ind], sum_FIR_ds[t_ds_ind])
            score = rho[0, 1] * 1
            
            print(score)
            
            # Show plots of sdfs
            fig, axes = plt.subplots(2, sharex=True)
            axes[0].plot(t_sum[sdf_ind], sum_FIR[sdf_ind]); axes[0].set_title('Full resolution')
            axes[1].plot(t_sum_ds[t_ds_ind], sum_FIR_ds[t_ds_ind]); axes[1].set_title('Down-sampled')
            plt.xlabel('Time (ms)')
            plt.ylabel('SDF (Hz)')
            plt.savefig('SDF fig', dpi=250)
            
            fig, axes = plt.subplots(len(spike_info)+1, sharex=True, sharey=True)
            for i, (name, neuron) in enumerate(spike_info.items()):
                axes[i].scatter(neuron[0], neuron[1], s=1); axes[i].set_title(name)
                axes[-1].scatter(neuron_ds[:,1], neuron_ds[:,0], s=1)
            
            plt.savefig('Spike rasters fig', dpi=250)
            plt.show()
            sys.exit('Just visualising!')
        
        # Save folder
        pickle_dir = os.path.join(os.getcwd(), self.save_folder, 'df', self.stims_dir)
        os.makedirs(os.path.split(pickle_dir)[0], exist_ok=True)
        
        df = pd.DataFrame(np.asarray(score_mat), columns=[*labels, 'Score'])
        df.to_pickle(pickle_dir)
        return df
        
    def get_original(self):
        neuron_spike_times_ds = pickle.load(open(self.ds_stims, "rb"))
        
        neuron_arr = []
        for time, neuron in enumerate(neuron_spike_times_ds):
            if neuron:
                neuron_arr.append([(time, n) for n in neuron])
                
        neuron_ds = np.concatenate(neuron_arr)
        self.time = max(neuron_ds[:,1])
        
        return neuron_ds
    
    def hierarchical_colormap(self):
        # Hierarchical colormap
        self.df = pd.read_pickle(os.path.join(os.getcwd(), self.save_folder, 'df', self.stims_dir))
        
        # TODO: Make sure reshape is better. Problem originates from the order parameters are saved in
        
        # Print row of maximum score value
        max_score_idx = self.df['Score'].idxmax()
        lst = list(zip(params.keys(), self.df.iloc[max_score_idx].to_numpy(dtype=int)[:-1]))
        string = [':'.join([i[0], str(i[1])]) for i in lst]
        text = " ".join(string)
        print(text)
        
        self.x_layers = ['tc', 'd', 'nt']
        self.y_layers = ['bi', 'ib']
        
        xaxis = self.df[self.x_layers].drop_duplicates().to_numpy()
        yaxis = self.df[self.y_layers].drop_duplicates().to_numpy()
        
        self.shape = np.array([yaxis.shape[0], xaxis.shape[0]])
        
        scores = self.df['Score'].to_numpy().reshape(self.shape)
        
        return scores
    
    def cluster_plot(self, params):
        # Imports
        sys.path.append('../')
        from p_calc import p_scaled_calc
        
        # Load scores
        self.scores = self.hierarchical_colormap()
        
        # Hierarchical colormap
        self.fig, self.ax = plt.subplots()
        # self.im, self.annot = [], []
            
        # self.im = self.ax.imshow(self.stim, cmap='coolwarm')
        self.im = self.ax.pcolormesh(self.scores, cmap='coolwarm')
        size = 8
        pad = 0.5
        plt.gca().invert_yaxis()
        
        x_points = self.scores.shape[1]
        self.ax.set_xticks((np.arange(x_points) + pad).tolist())
        xlabels_0 = (np.arange(len(params[self.x_layers[-1]])) + 1).tolist() * int(x_points / len(params[self.x_layers[-1]]))
        self.ax.set_xticklabels([f'${self.x_layers[-1]}_{str(int)}$' for int in xlabels_0], fontsize=size)
        
        for j in range(x_points):
            shift_x1 = len(params[self.x_layers[-1]])
            shift_x2 = int(x_points / len(params[self.x_layers[0]]))
            
            x_labels1 = (np.arange(len(params[self.x_layers[1]])) + 1).tolist() * round(
                x_points / np.prod([len(params[k]) for k in self.x_layers[1:]]))
            x_labels2 = (np.arange(len(params[self.x_layers[0]])) + 1).tolist() * round(
                x_points / np.prod([len(params[k]) for k in self.x_layers[:]]))
            
            if j % shift_x1 == 0:
                it = j // shift_x1
                self.ax.annotate(f'${self.x_layers[1]}_{str(x_labels1[it])}$', xy=(j+shift_x1-1+pad, 0), 
                                  xycoords=('data', 'axes fraction'), xytext=(0, -18), 
                                  textcoords='offset points', va='top', ha='center', fontsize=size)
            if j % shift_x2 == 0:
                it = j // shift_x2
                self.ax.annotate(f'${self.x_layers[0]}_{str(x_labels2[it])}$', xy=(j+shift_x2-1+pad, 0), 
                                  xycoords=('data', 'axes fraction'), xytext=(0, -32), 
                                  textcoords='offset points', va='top', ha='center', fontsize=size)
        
        y_points = self.scores.shape[0]
        self.ax.set_yticks((np.arange(y_points) + pad).tolist())
        ylabels_0 = (np.arange(len(params[self.y_layers[-1]])) + 1).tolist() * int(y_points / len(params[self.y_layers[-1]]))
        self.ax.set_yticklabels([f'${self.y_layers[-1]}_{str(int)}$' for int in ylabels_0], fontsize=size)
        
        for j in range(y_points):
            shift_y2 = int(y_points / len(params[self.y_layers[0]]))
            
            y_labels2 = (np.arange(len(params[self.y_layers[0]])) + 1).tolist() * round(
                y_points / np.prod([len(params[k]) for k in self.y_layers[:]]))
            
            if j % shift_y2 == 0:
                it = j // shift_y2
                self.ax.annotate(f'${self.y_layers[0]}_{str(y_labels2[it])}$', xy=(0, j+shift_y2-1+pad), 
                                 xycoords=('axes fraction', 'data'), xytext=(-28, 0), 
                                 textcoords='offset points', va='center', ha='center', fontsize=size)
        
        self.ax.grid()
        plt.colorbar(self.im, orientation="horizontal", ax=self.ax)
        
        string = [np.array((keys, [*values]), dtype=object) for keys, values in params.items()]
        
        tc_range = np.hstack((np.arange(1, step=0.1), np.logspace(0, 1, 11)))
        for enum, s in enumerate(string):
            s[1] = np.array(s[1])
            if s[0] in ['bi', 'ib']:
                s[1] = s[1]/10
            elif s[0] in ['d']:
                s[1] = np.asarray(np.round(s[1]/10 * 62), dtype=int)
            elif s[0] in ['tc']:
                _, tau = p_scaled_calc(1, tc_range[s[1]])
                s[1] = np.round(tau, 2)
            string[enum] = ': '.join([s[0], ', '.join(map(str, s[1]))])
            
        self.params_dict = {}
        for i in string:
            key = i[:i.find(':')]
            values = list(map(float, i[i.find(':'):].replace(':', '').split(',')))
            self.params_dict[key] = values
            
        text = '\n'.join(i for i in string)
        plt.text(0.5, 0.05, text, transform=plt.gcf().transFigure)
        
        self.annot = (self.ax.annotate("", xy=(0, 0), 
                                       xytext=(20, 20), textcoords="offset points", 
                                       bbox=dict(boxstyle="round", fc="w")))
        
        self.annot.set_visible(False)
        
    # On-click functionality
    def connect(self):
        self.fig.canvas.mpl_connect('axes_enter_event', self.enter_axes)
        self.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.active_axes = None
    
    def update_annot(self, ind):
        stim = self.scores
        
        cmap = mpl.colormaps['coolwarm']
        norm = plt.Normalize(np.min(stim), np.max(stim))
        
        pos = ind["ind"][0]
        coord_xy = np.unravel_index([pos], (self.shape))
        
        if not self.active_axes == None:
            self.annot.xy = np.flip(coord_xy)
            
            values = self.df.iloc[pos].to_numpy(dtype=int)[:-1]
            
            values_new = []
            for i, v in enumerate(params.values()):
                param_idx = v.index(values[i])
                values_new.append(list(self.params_dict.values())[i][param_idx])
            
            
            lst = list(zip(params.keys(), values_new))
            string = [':'.join([i[0], str(i[1])]) for i in lst]
            text = " ".join(string)
            self.annot.set_text(text)
            self.annot.get_bbox_patch().set_facecolor(cmap(norm(stim[coord_xy])))
            self.annot.get_bbox_patch().set_alpha(0.4)
            self.ax.add_patch(Rectangle(self.annot.xy, 1, 1, 
                                                          fill=False, edgecolor='black', lw=1))
    
    def enter_axes(self, event):
        self.active_axes = True
        
    def leave_axes(self, event):
        self.active_axes = None
    
    def on_click(self, event):
        if not self.active_axes == None:
            [p.remove() for p in reversed(self.ax.patches)]
            vis = self.annot.get_visible()
            cont, ind = self.im.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
            
if args.view == 'coarse':
    params = {'bi':(10, 12, 15, 5), 'ib':(10, 12, 5, 7), 'tc':(10, 12, 5), 'd':(10, 12, 5, 7), 'nt':(2, 3, 4)}
    save_folder = 'grid_search_self_inhibition'
else:
    # params = {'bi':(12, 15, 20), 'ib':(10, 12, 15), 'tc':(10, 12, 15), 'd':(3, 5, 7), 'nt':(3, 4, 5, 6)}
    params = {'bi':(12, 15, 20, 25), 'ib':(10, 12, 15, 20), 'tc':(10, 12, 15), 'd':(3, 5, 7), 'nt':(3, 4, 5, 6)}
    save_folder = 'grid_search_self_inhibition_agg'
df = evaluation(save_folder).get_sdfs()
# grid = evaluation(save_folder)
# grid.cluster_plot(params)
# grid.connect()
# plt.show()