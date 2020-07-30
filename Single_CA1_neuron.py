# -----------------------------------------------------------------------
# Code designed to simulate one single neuron with 3 compartments (2 parallel
# dendrites + 1 soma) receiving place-tuned input. the following properties are 
# implemented:
# 1. Each dendritic compartment is activated following a nonlinear function of 
#    the weighted sum of inputs
# 2. Input neurons are simulated. Their firing rate will depend on the
#    animal’s location and will be determined by a non-gaussian place field.
# 3. The learning rule is local on the dendrites. It depends on presynaptic
#    activity and dendritic activity. Therefore, inputs coming to different 
#    dendrites will learn independently while inputs coming to the same dendrite 
#    will have a correlated learning rule. 
#
# -----------------------------------------------------------------------
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Jan 2020


# Clear everything!
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]


clearall()

# ------------------------------------- Import libraries ----------------------------------------------------

import numpy as np
import os
import sys
import pickle
import scipy.io as sio
import multiprocessing as mp
from time import time as time_now
from scipy.optimize import curve_fit

from sacred import Experiment
import shutil
import tempfile



ex = Experiment("Exploring_three_levels_of_plasticity")



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parameters to be saved for each the experiment -------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@ex.config
def config():
    sim_pars = {
        # Parameters ---------------------------------------------------------------------------------
        'T_length': 50.,     # Length of the track
        'N_pre': 10,         # Number of neurons in the presynaptic layer
        'Nth': 1.0,          # "Spiking threshold" (there is actually no threshold - rate-based neurons)
        'N_dend': 2,         # Number of dendritic compartments
        'ExtraCurr_0': 1.,   # Extra current injected at the soma

        # Simulation parameters --------------------------------------------------------------
        'dt': 1.,            # [ms] Simulation time step

        # Excitatory synaptic plasticity -----------------------------------------------------
        'eta_Extra': 0.e-4,  # [ms^-1] Learning rate for the extra currents onto pyramidal neurons
        'eta_input': 2e-4,   # [ms^-1] Learning rate for the input weights from prelayer neurons
        'eta_homeo': 2e-4,   # [ms^-1] Learning rate for homeostatic plasticity

        # Novelty signal ---------------------------------------------------------------------
        'Idend_target': 8.5, # Dendritic inhibition in familiar environments
        'Isoma_target': 0.,  # Somatic inhibition in familiar environments
        'Idend_0': 0.8,      # Initial dendritic inhibition
        'Isoma_0': 1.2,      # Initial somatic inhibition
        'tau_nov': 100e3,    # [ms] Novelty signal time constant

        # Presynaptic place field ------------------------------------------------------------
        'PF_amp': 2.2,       # Amplitude of presynaptic place fields

        # Experiment parameters --------------------------------------------------------------
        'n_laps': 100,       # Number of laps the subjets runs for each trial
        'v_run': 1e-2,       # Running speed
        'noise_input': 0.05, # amplitude of noise for input neurons

        # Firing rate parameters -------------------------------------------------------------
        'eta_FR': 2.e-1,     # [ms^-1] Learning rate for the firing rates

        # Number of trials for multi-trial experiment ----------------------------------------
        'NTrials': 1,

        'message': ''
    }


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Some functions to be used throughout the code --------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _rect(x): return x * (x > 0.)  # Rectification


def _upper(x, max): return x - (x - max) * (x > max)  # apply an upper bound


def _lower(x, min): return x - (x - min) * (x < min)  # apply a lower bound


@ex.capture(prefix="sim_pars")
def Pre_layer_input(pos, N_pre, T_length, PF_amp):
    '''
    This function returns the firing rate of each neuron in the presynaptic layer as a function of
    the animal's position
    '''
    sigma = 5.  # width of place fields of presynaptic neurons
    amp = PF_amp  # amplitude of place fields of presynaptic neurons
    place_input = np.zeros(N_pre)

    # Receptive fields (Feedforward Excitatory synaptic weights):
    for n in np.arange(1, N_pre + 1):
        pos0 = (n - 1) * T_length / N_pre
        dist = np.abs((pos0 - pos + T_length / 2.) % T_length - T_length / 2.)
        place_input[(n - 1):n] = amp * np.exp(-(dist) ** 2 / (2. * sigma ** 2))

    return place_input


def g_dend(x):
    '''
    This is the gain function of dendritic compartments for a given input current x

    input:
        x: input current (N_dend,)
    output:
        g_dend: firing rate of the dendritic compartments (N_dend,)
    '''
    linear_length = 5.
    x = 2 * x

    g1 = _rect(np.tanh(x / linear_length))
    g2 = 0.5 * np.tanh((x - linear_length) * 2.) + 0.5
    g_dend = 2 * (2 * g1 + 1 * g2) / 3.

    return g_dend


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialization of variables --------------------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Network(object):
    """
    Network of neurons

    inputs:
        pars is a dictionary with the following keys:
            N_pre  : Number of neurons in the presynaptic layer
            N_dend : Number of dendritic compartments

    Attributes:
        Wpre_d: Synaptic weights from presynaptic layer to dendrites
        ExtraCurr: Extra currents controlling the excitability of neurons (plastic)
        RtES: [Hz] Firing rate of somatic compartment
        RtED: [Hz] Firing rates of dendritic compartments
    """

    def __init__(self, pars):
        """
        Return a Network object
        """
        self.Wpre_d = np.zeros((pars['N_dend'], pars['N_pre']))
        self.Wpre_d[0, 4] = 1.
        self.Wpre_d[0, :] += _rect(np.random.normal(0, 0.01, pars['N_pre']))
        self.ExtraCurr = pars['ExtraCurr_0']

        # time-dependent variables
        self.RtES = 0.
        self.RtED = np.zeros(pars['N_dend'])

        # Dendritic and somatic currents
        self.I_dend = np.zeros(pars['N_dend'])
        self.I_soma = 0.


@ex.command
def my_vars(sim_pars):
    np.random.seed()

    # Variables that will change with time ---------------------------------------------------------------
    net = Network(sim_pars)

    return net


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculations to be performed at every integration step -----------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@ex.command
def SimStep(net, Pre_input, sim_pars):
    # Simulation parameters ------------------------------------------------------------------------------
    eta_FR = sim_pars['eta_FR']
    Nth = sim_pars['Nth']
    eta_Extra = sim_pars['eta_Extra']
    eta_input = sim_pars['eta_input']
    dt = sim_pars['dt']
    Idend_target = sim_pars['Idend_target']
    Isoma_target = sim_pars['Isoma_target']
    tau_nov = sim_pars['tau_nov']
    eta_homeo = sim_pars['eta_homeo']
    noise_input = sim_pars['noise_input']

    # Get the values from the network --------------------------------------------------------------------
    Wpre_d = net.Wpre_d
    RESin = net.RtES
    REDin = net.RtED
    ExtraCurr = net.ExtraCurr
    I_soma = net.I_soma
    I_dend = net.I_dend

    # Compute the values for the next time step ----------------------------------------------------------

    Pre_input += noise_input * np.random.normal(0, 1, Pre_input.shape[0])
    Pre_input = _rect(Pre_input)

    # Calculate the firing rate for the postsynaptic neuron (and rectify it when negative)----------------
    RtES = RESin + eta_FR * dt * (-RESin + _rect(1 * np.sum(REDin) + ExtraCurr - Nth - I_soma))
    RtED = REDin + eta_FR * dt * (-REDin + g_dend(np.dot(Wpre_d, Pre_input) - I_dend))

    RtES = RtES * (RtES > 0.)
    RtED = RtED * (RtED > 0.)

    # Update the feedforward weights ---------------------------------------------------------------------
    R_dend_mat = REDin.reshape((-1, 1))
    target_norm = 3.
    Wpre_d = Wpre_d + eta_input * dt * (np.dot(R_dend_mat, Pre_input.reshape((1, -1))))
    Wpre_d = (Wpre_d.T - eta_homeo * dt * R_dend_mat.T * (np.sum(Wpre_d, axis=1) - target_norm)).T

    Wpre_d = Wpre_d * (Wpre_d > 0.)  # rectify synaptic weights
    Wpre_d = _upper(Wpre_d, 3)

    # Update inhibition to simulate a novel environment becoming familiar --------------------------------
    I_dend = I_dend - (1. / tau_nov) * dt * (I_dend - Idend_target)
    I_soma = I_soma - (1. / tau_nov) * dt * (I_soma - Isoma_target)

    # Return the new values for the network --------------------------------------------------------------
    net.RtES = RtES
    net.RtED = RtED
    net.ExtraCurr = ExtraCurr
    net.Wpre_d = Wpre_d
    net.I_soma = I_soma
    net.I_dend = I_dend

    return net


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the steps to be done during the experiment ----------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@ex.capture
def PM_plast_one_trial(sim_pars):  # place map plasticity
    """
    """

    # Extract parameters from configuration --------------------------------------------------------------
    N_pre = sim_pars['N_pre']
    N_dend = sim_pars['N_dend']
    T_length = sim_pars['T_length']
    dt = sim_pars['dt']
    Idend_0 = sim_pars['Idend_0']
    Isoma_0 = sim_pars['Isoma_0']
    v_run = sim_pars['v_run']

    # Create the network ---------------------------------------------------------------------------------
    net = my_vars()

    # -------------------------------------------------------------------------------------------------
    # Start actual experiment and let the animal explore the environment ------------------------------
    # -------------------------------------------------------------------------------------------------

    v = v_run   # [ms^-1] speed of the animal (the unit of length is abitrary)
    n_laps = sim_pars['n_laps']  # number of laps
    t_explore = n_laps * T_length / v  # total time of experiment
    explore_steps = int(t_explore / dt)  # number of steps for integration

    # Track all the postsynaptic firing rates ---------------------------------------------------------
    RESs = np.zeros((explore_steps))  # somatic firing rate
    REDs = np.zeros((explore_steps, N_dend))  # dendritic firing rate
    Wpre_ds = np.zeros((explore_steps, N_dend, N_pre))
    ExtraCurrs = np.zeros((explore_steps))

    def pos_exp(t): return v * t

    # Initialize inhibition for novel environments (low dend inhibition + high somatic inhibition)
    net.I_dend = 0 * net.I_dend + Idend_0
    net.I_soma = 0 * net.I_soma + Isoma_0

    for step in range(explore_steps):
        t_bin = step * dt
        Pre_input = Pre_layer_input(pos_exp(t_bin))

        # Call the simulation step to calculate all the values for next time step
        net = SimStep(net, Pre_input)

        # Save the results
        RESs[step] = net.RtES
        REDs[step] = net.RtED
        Wpre_ds[step] = net.Wpre_d
        ExtraCurrs[step] = net.ExtraCurr

    return RESs, REDs, Wpre_ds, ExtraCurrs, t_explore, n_laps


# -----------------------------------------------------------------------------------------------------
# Call the function with the experiment and save the results (define one trial)

def One_trial(ids):  # place map plasticity
    '''
    ids should be a tuple containing:
    (1) The trial id, which will be specific for each core running the simulation
    (2) The run id, which will be defined by sacred and depends on the id of the experiment being run
    '''

    trial_id, run_id, tempdir = ids

    time_in = time_now()

    # Call the function with the experiment:
    RESs, REDs, Wpre_ds, ExtraCurrs, t_explore, n_laps = PM_plast_one_trial()

    time_final = time_now()
    total_time = time_final - time_in

    # Same the result for this trial in a temporary file:
    fname = '/Firing_rates_Place_cell_soma_and_dends_trial_{0:03d}.pickle'.format(int(trial_id))
    fpath = tempdir.name + fname

    print(' >> ' + fname + ' time = ' + '{0:4.1f}'.format(total_time))

    with open(fpath, 'wb') as pickle_out:
        data = {"t_explore": t_explore,
                "soma": RESs,
                "dendrites": REDs,
                "Wpre_to_dend": Wpre_ds,
                "ExtraCurr": ExtraCurrs,
                "n_laps": n_laps}
        pickle.dump(data, pickle_out)
        pickle_out.close()

    return fpath


# -----------------------------------------------------------------------------------------------------
# Define the main funtion which will call other functions and run it in parallel

from sacred.observers import FileStorageObserver

ex.observers.append(FileStorageObserver.create('my_runs'))


@ex.automain
def my_main(sim_pars, _run):
    tempdir = tempfile.TemporaryDirectory()

    _run.info["Description"] = sim_pars['message']  # write message in a file called "info.json"

    print('\n Starting simulations...\n')

    run_id = str(_run._id)  # id of the experiment (sacred)

    NTrials = sim_pars['NTrials']
    Trials = [(i + 1, run_id, tempdir) for i in range(NTrials)]

    # Run code in parallel...
    # quant_proc = np.max((mp.cpu_count() - 1, 1))
    # pool = mp.Pool(processes=quant_proc)

    fnames = map(One_trial, Trials)  # return the names of the output files

    # Export the final file with all the data for all trials
    exp_dir = r'./my_runs/' + run_id + '/Data_trials/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    fname = exp_dir + '/Firing_rates_Place_cell_soma_and_dends_all_trials.pickle'

    data = {}
    for i, name in enumerate(fnames):
        i = i + 1
        with open(name, 'rb') as pickle_in:
            datai = pickle.load(pickle_in)
            data[i] = datai
            pickle_in.close()
        #			os.remove(name)
        with open(fname, 'wb') as pickle_out:
            pickle.dump(data, pickle_out)
            pickle_out.close()

    tempdir.cleanup()
