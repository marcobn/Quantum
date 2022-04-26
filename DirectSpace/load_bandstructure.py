""" Validate model by testing analytical band structure. """

import numpy as np
import json
from qctb.bandstructure import BandStructure

import qctb.computer.google
import qctb.optimization.scipy
import qctb.model.chain
import qctb.model.graphene
import qctb.model.polonium

import qctb.hamiltonian.singlebody


##############################################################################
#                              LOAD BAND STRUCTURE
filetag = "Polonium_sp_ΓRXMRΓXMΓ.Direct_MN.ShotSimulator.COBYLA"

with open(f"json/{filetag}.json", encoding='utf-8') as fp:
    band = BandStructure.from_json(json.load(fp))




##############################################################################
#                              PLOT ENERGIES

import matplotlib
import matplotlib.pyplot

# PRELIMINARY CONFIGURATION
matplotlib.rc('font', family='serif')
matplotlib.rcParams['axes.linewidth'] = 0       # MANUALLY DRAW PLOT BORDERS
matplotlib.rcParams['mathtext.fontset'] = 'cm'

fig, ax = band.plot()

# DISPLAY PLOT
matplotlib.pyplot.show()
