##################################################################################
# Make_figs2.py - This code generates figures for place cells, silent cells 
# (both soma and dendrites) and for inhibitory cells (both somatic-targeting and 
# dendritic targeting)
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Jan 2017
##################################################################################


# Clear everything!
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]


clearall()

# Import libraries ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
# from scipy.optimize import curve_fit
from matplotlib.pyplot import cm
import matplotlib as mpl
import pickle

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

import sys as sys

args = sys.argv

# Edit the formatting of matplotlib ----------------------------------------------

mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['font.weight'] = 900
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.edgecolor'] = 3 * [0.5]
mpl.rcParams['xtick.color'] = 3 * [0.5]
mpl.rcParams['ytick.color'] = 3 * [0.5]
mpl.rcParams['xtick.major.pad'] = 5
mpl.rcParams['ytick.major.pad'] = 5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('font', weight=400)
plt.ion()

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"


def hide_frame(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def hide_frame_all(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


cmap = cm.inferno

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import the data

# Choose the directory and load the files ----------------------------------------

while True:
    try:
        my_run = args[1]
        print('')
        Dir = 'my_runs/' + my_run + '/Data_trials/'
        fnames = os.listdir(Dir)
        break
    except KeyboardInterrupt:
        print(' Interrupted!\n')
        exit()
    except:
        print('>> No run with this number. Please try again:')

# --------------------------------------------------------------------------------
fnames.sort()

# load all the data from the file
with open(Dir + fnames[0], 'rb') as pickle_in:
    data_all = pickle.load(pickle_in)
    pickle_in.close()

# extract the firing rates for soma and dendrites
Soma_FRs = np.array([data_all[tr_id]["soma"] for tr_id in data_all.keys()])
Dends_FRs = np.array([data_all[tr_id]["dendrites"] for tr_id in data_all.keys()])
Syn_weights = np.array([data_all[tr_id]["Wpre_to_dend"] for tr_id in data_all.keys()])
ExtraCurr = np.array([data_all[tr_id]["ExtraCurr"] for tr_id in data_all.keys()])
n_laps = np.array([data_all[tr_id]["n_laps"] for tr_id in data_all.keys()])[0]

# take an average over all trials
Soma_FRs_ave = np.mean(Soma_FRs, axis=0)
Dends_FRs_ave = np.mean(Dends_FRs, axis=0)
Syn_weights_ave = np.mean(Syn_weights, axis=0)
ExtraCurr_ave = np.mean(ExtraCurr, axis=0)

# create a vector with all the time points
tVec = np.linspace(0, data_all[1]["t_explore"], Soma_FRs.shape[1])
tVec2 = np.repeat([tVec], 50, axis=0)

label_font_size = 20

# --------------------------------------------------------------------------------
# Save figure

fig_dir = r'./Figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)



#%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 3F ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mpl.rcParams['axes.edgecolor'] = 3 * [0.]
mpl.rcParams['xtick.color'] = 3 * [0.]
mpl.rcParams['ytick.color'] = 3 * [0.]
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)


import json

config_file = 'my_runs/' + my_run + '/config.json'

with open(config_file) as f:
    data = json.load(f)

n_laps = data['sim_pars']['n_laps']


soma = Soma_FRs_ave.reshape((n_laps, -1))
dend = Dends_FRs_ave[:,0].reshape((n_laps, -1))

soma_mean = np.mean(soma, axis=1)
dend_mean = np.mean(dend, axis=1)

fig = plt.figure(num=4, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')

ax = plt.subplot()

hide_frame(ax)
plt.ylabel('Mean activity', fontsize=1.6*label_font_size)
plt.xlabel('Lap', fontsize=1.6*label_font_size)
plt.plot(dend_mean, 'k--', lw=2, label='Dendrite')
plt.plot(soma_mean, 'k-', lw=2, label='Soma')



leg1 = plt.legend(loc=0, fontsize=18)
leg1.get_frame().set_linewidth(0.0)
leg1.get_frame().fill = False

plt.savefig(fig_dir + 'Run_{0:03d}-Mean_activity.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()

#%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 3F - zoom +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mpl.rcParams['axes.edgecolor'] = 3 * [0.]
mpl.rcParams['xtick.color'] = 3 * [0.]
mpl.rcParams['ytick.color'] = 3 * [0.]
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)


import json

config_file = 'my_runs/' + my_run + '/config.json'

with open(config_file) as f:
    data = json.load(f)

n_laps = data['sim_pars']['n_laps']


soma = Soma_FRs_ave.reshape((n_laps, -1))
dend = Dends_FRs_ave[:,0].reshape((n_laps, -1))

soma_mean = np.mean(soma, axis=1)
dend_mean = np.mean(dend, axis=1)

fig = plt.figure(num=4, figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')

ax = plt.subplot()

hide_frame(ax)
plt.ylabel('Mean activity', fontsize=1.6*label_font_size)
plt.xlabel('Lap', fontsize=1.6*label_font_size)
plt.plot(dend_mean, 'k--', lw=2, label='Dendrite')
plt.plot(soma_mean, 'k-', lw=2, label='Soma')
plt.xlim((0,10))


leg1 = plt.legend(loc=0, fontsize=18)
leg1.get_frame().set_linewidth(0.0)
leg1.get_frame().fill = False

plt.savefig(fig_dir + 'Run_{0:03d}-Mean_activity_zoom.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()

