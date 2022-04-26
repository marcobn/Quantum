""" Validate model by testing analytical band structure. """

import numpy as np
import json
from qctb.model.chain import SimpleChain, AlternatingChain
from qctb.model.graphene import Graphene_s
from qctb.model.polonium import Polonium_sp
from qctb.hamiltonian.singlebody   import OrbitalHamiltonian, SiteHamiltonian
from qctb.bandstructure import BandStructure
from qctb.optimization.scipy import BFGS, COBYLA
from qctb.computer.google import BasicShotSimulator, StatevectorSimulator



##############################################################################
#                              SELECT MODEL

# model = SimpleChain()
# sym = ["π", "Γ", "π"]

# model = AlternatingChain()
# sym = ["π", "Γ", "π"]
#
# model = Graphene_s()
# sym = ["Γ", "K", "M", "Γ"]

model = Polonium_sp(
    Es  = -14,
    Ep  = 0,
    Vss = 0,
    Vsp = 2,
    Vpσ = 2,
    Vpπ = 0,
)
sym = ["Γ", "R", "X", "M", "R", "Γ", "X", "M", "Γ"]
# sym = ["Γ", "X", "M", "Γ"]


##############################################################################
#                              SELECT HAMILTONIAN

# H = OrbitalHamiltonian(model)
H = SiteHamiltonian(model, [3]*model.d)

##############################################################################
#                              SELECT COMPUTER

computer  = BasicShotSimulator(shots=8096)
# computer  = StatevectorSimulator()


##############################################################################
#                              SELECT OPTIMIZER

# optimizer = None        # Analytically locates optimal angles.
# optimizer = BFGS()
optimizer = COBYLA()


##############################################################################
#                              CALCULATE ENERGIES

# band = BandStructure(H, sym, computer, optimizer, N=3, discrete=False)
band = BandStructure(H, sym, computer, optimizer, N=H.N, discrete=True)

filetag = f"{model.id}_{''.join(sym)}.{H.id}.{computer.id}.{optimizer.id if optimizer else 'Optimal'}"

for E, angles, nfev in band.solve():
    print(f"Solved E={E:.3f}, now tackling i={band.i}, l={band.l}")

    with open(f"json/{filetag}.json", "w", encoding='utf-8') as fp:
        # print (band.to_json())
        json.dump(band.to_json(), fp, ensure_ascii=False, indent=4)




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
