# %% Import modules -------------------------------------------------------------------------------------
import subprocess
import os
import json
import sys
import numpy as np
import multiprocessing as mp



# %% Run the main code with sacred to track every experiment -------------------------------------------

# from Single_CA1_neuron import ex
# ex.run_command('my_main')


# %% Run the main code to create the figures -----------------------------------------------------------

n_sims = 1


for i in range(n_sims):
    f_names = os.listdir('my_runs/')
    f_names.sort()

    f_names = [int(fname) for fname in f_names[1:-1]]
    f_names.sort()

    if len(f_names) == 0:
        f_names = [1]

    sys.argv = ['0', '{0:d}'.format(f_names[-1-i])]
    exec(open("./Make_figs0.py").read())
    exec(open("./Make_figs1.py").read())

    plt.show()

