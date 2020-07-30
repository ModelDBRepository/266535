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
# mpl.rcParams['axes.titlepad'] = 20

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
# plt.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}\bfseries\boldmath"]
plt.rc('font', weight=400)
plt.ion()

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
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
        #		my_run = input('\n>> Which run do you want to use?\n>> ')
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions to create figures ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

label_font_size = 30


def plot_FR(ax, tVec, FR, color, xlabel='Position', ylabel='Somatic activity', label=' ', alpha=1):
    '''
    This function creates a plot of the firing rate over time by default. The x and y
    labels can be modified
    '''
    hide_frame(ax)
    plt.ylabel(ylabel, fontsize=label_font_size)
    plt.xlabel(xlabel, fontsize=label_font_size)
    plt.xlim((0, 50.01))
    plt.ylim(-0.01, 2.)
    plt.plot(tVec, FR, lw=2, label=label, color=color, alpha=alpha)

    #	leg1 = plt.legend(loc=0,fontsize=13)
    #	leg1.get_frame().set_linewidth(0.0)
    #	leg1.get_frame().fill = False
    plt.tick_params(axis="y", labelcolor="k")
    plt.tick_params(axis="x", labelcolor="k")


cmap = cm.viridis




def plot_RF(FR, n_laps):
    points_lap = int(FR.shape[0] / n_laps)
    RF_develop = FR.reshape((-1, points_lap))

    plt.xlabel("Position", fontsize=label_font_size)
    plt.ylabel('Lap', fontsize=label_font_size)

    ymin = RF_develop.shape[0]
    ymax = 0
    xmin = 0
    xmax = 50

    plt.imshow(RF_develop, interpolation='nearest', aspect='auto', cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    cbar = plt.colorbar(shrink=0.9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='k')
    cbar.ax.tick_params(labelsize=10, width=0.5, size=2)
    cbar.outline.set_visible(False)
    plt.tick_params(axis="y", labelcolor="k")
    plt.tick_params(axis="x", labelcolor="k")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig_dir = r'./Figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

#%% --------------------------------------------------------------------------------
# Plot figures - soma

fig = plt.figure(num=1, figsize=(4, 4), dpi=100, facecolor='w', edgecolor='k')
gs1 = GridSpec(1, 1)

ax1 = plt.subplot(gs1[0, 0])
plot_RF(Soma_FRs_ave, n_laps)
plt.title('Soma', fontsize=label_font_size)
plt.yticks([0,10,100])

plt.savefig(fig_dir + 'Run_{0:03d}-Soma.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()

#%% --------------------------------------------------------------------------------
# Plot figures - soma

fig = plt.figure(num=1, figsize=(4, 4), dpi=100, facecolor='w', edgecolor='k')
gs1 = GridSpec(1, 1)

ax1 = plt.subplot(gs1[0, 0])
plot_RF(Soma_FRs_ave, n_laps)
plt.title('Soma', fontsize=label_font_size)
plt.ylim((10,0))


plt.savefig(fig_dir + 'Run_{0:03d}-Soma_zoom.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()

#%% --------------------------------------------------------------------------------
# Plot figures - dendrites

fig = plt.figure(num=2, figsize=(4, 4), dpi=100, facecolor='w', edgecolor='k')
gs1 = GridSpec(1, 1)

ax2 = plt.subplot(gs1[0, 0])
plot_RF(Dends_FRs_ave[:, 0], n_laps)
plt.title('Dendrite', fontsize=label_font_size)
plt.yticks([0,10,100])


plt.savefig(fig_dir + 'Run_{0:03d}-Dend.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()


#%% --------------------------------------------------------------------------------
# Plot figures - dendrites

fig = plt.figure(num=2, figsize=(4, 4), dpi=100, facecolor='w', edgecolor='k')
gs1 = GridSpec(1, 1)



ax2 = plt.subplot(gs1[0, 0])
plot_RF(Dends_FRs_ave[:, 0], n_laps)
plt.title('Dendrite', fontsize=label_font_size)
plt.ylim((10,0))

plt.savefig(fig_dir + 'Run_{0:03d}-Dend_zoom.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 3 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

FR = Soma_FRs_ave

points_lap = int(FR.shape[0] / n_laps)
RF_develop = FR.reshape((-1, points_lap))

Dend_RF = Dends_FRs_ave[:, 0].reshape((-1, points_lap))


#%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 3 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import json

config_file = 'my_runs/' + my_run + '/config.json'

with open(config_file) as f:
    data = json.load(f)

n_laps = data['sim_pars']['n_laps']



pos = np.linspace(0, 50, points_lap)

fig = plt.figure(num=10, figsize=(5, 2), dpi=100, facecolor='w', edgecolor='k')
ax = plt.subplot()
plot_FR(ax, pos, RF_develop[0], 'b', ylabel='')
plt.savefig(fig_dir + 'Run_{0:03d}-RF1.pdf'.format(int(my_run)), dpi=300, transparent=True)

fig = plt.figure(num=11, figsize=(5, 2), dpi=100, facecolor='w', edgecolor='k')
ax = plt.subplot()
plot_FR(ax, pos, RF_develop[5], '#990099', ylabel='')
plt.savefig(fig_dir + 'Run_{0:03d}-RF2.pdf'.format(int(my_run)), dpi=300, transparent=True)

fig = plt.figure(num=12, figsize=(5, 2), dpi=100, facecolor='w', edgecolor='k')
ax = plt.subplot()
# plot_FR(ax, pos, RF_develop[0], 'b', alpha=0.2)
plot_FR(ax, pos, RF_develop[n_laps-1], '#FF9900', ylabel='')
plt.savefig(fig_dir + 'Run_{0:03d}-RF3.pdf'.format(int(my_run)), dpi=300, transparent=True)

fig = plt.figure(num=20, figsize=(5, 2), dpi=100, facecolor='w', edgecolor='k')
ax = plt.subplot()
plot_FR(ax, pos, Dend_RF[0], 'b', ylabel='')
plt.savefig(fig_dir + 'Run_{0:03d}-Dend_RF1.pdf'.format(int(my_run)), dpi=300, transparent=True)

fig = plt.figure(num=21, figsize=(5, 2), dpi=100, facecolor='w', edgecolor='k')
ax = plt.subplot()
plot_FR(ax, pos, Dend_RF[5], '#990099', ylabel='')
plt.savefig(fig_dir + 'Run_{0:03d}-Dend_RF2.pdf'.format(int(my_run)), dpi=300, transparent=True)

fig = plt.figure(num=22, figsize=(5, 2), dpi=100, facecolor='w', edgecolor='k')
ax = plt.subplot()
# plot_FR(ax, pos, Dend_RF[0], 'b', alpha=0.2)
plot_FR(ax, pos, Dend_RF[n_laps-1], '#FF9900', ylabel='')
plt.savefig(fig_dir + 'Run_{0:03d}-Dend_RF3.pdf'.format(int(my_run)), dpi=300, transparent=True)

plt.show()
