""" Validate model by testing analytical band structure. """

import numpy
import json

import cirq

import ortho.optimization.scipy
import ortho.experiment.reciprocal
import ortho.experiment.compact


import matplotlib
import matplotlib.pyplot

# PRELIMINARY CONFIGURATION
matplotlib.rc('font', family='serif')
# matplotlib.rcParams['axes.linewidth'] = 0       # MANUALLY DRAW PLOT BORDERS
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# COLOR SCHEME
COLORS = {
    "quantumviolet": "#53257F",
    "quantumgray": "#555555",
    "my1": "#9E6240",
    "my2": "#55D6BE",
    "my3": "#E4D9FF",
    "my4": "#E7BB41",
}

##############################################################################
#                         PLOT ANALYTICAL ENERGIES


# INITIALIZE FIGURE
fig, ax = matplotlib.pyplot.subplots(
    dpi=100,
    figsize=[7,4.75],
)

# CALCULATE ANALYTICAL ENERGIES
expmt = ortho.experiment.reciprocal.Reciprocal()
xv, bv = expmt.lattice.vertices(expmt.sym)
x0, b0 = expmt.lattice.traverse(expmt.sym)
E0 = numpy.array([expmt.calculate_energies(x) for x in x0])

# CALCULATE y-LIMITS
Δ = 1
ylim = [
    numpy.min(E0) - Δ,
    numpy.max(E0) + Δ
]

# CONFIGURE y-AXIS
ax.set_ylim(ylim)
ax.set_ylabel("Energy (eV)")
for lim in ylim:
    ax.axhline(lim, color='black', lw=1, clip_on=False)

# CONFIGURE x-AXIS AND MARK HIGH-SYMMETRY POINTS
ax.set_xlim([0,1])
ax.set_xticks(xv)
ax.set_xticklabels(expmt.sym)
for x in xv:
    ax.axvline(x, color='black', lw=1, clip_on=False)

# PLOT ANALYTICAL ENERGIES
for l in range(E0.shape[1]):
    ax.plot(
        x0, E0[:,l],
        ls='-', lw=1.5, c='black',
    )



##############################################################################
#                        LOAD EXPERIMENTAL DATA

expmt = ortho.experiment.reciprocal.Reciprocal()    # FOR ANALYTICAL VALUES

filetags = [
    "reciprocal.COBYLA.r=1",
    # "compact.BFGS.r=0",
    "compact.COBYLA.r=1",
    "compact.COBYLA.r=2",
    "compact.COBYLA.r=3",
]

dat_λ = {}
dat_E = {}
dat_E0= {}
dat_ε = {}

for filetag in filetags:
    with open(f"json/{filetag}.json", encoding='utf-8') as fp:
        experiment = json.load(fp)

    λ_ = []
    E_ = []
    E0_= []

    for point in experiment["points"]:
        λ = point["λ"]
        E = [result["E"] for result in point["results"]]
        E = sorted(E)       # NOTE: This step is important!
        E0= expmt.calculate_energies(λ)

        λ_ .append(λ)
        E_ .append(E)
        E0_.append(E0)

    dat_λ [filetag] = numpy.array(λ_)
    dat_E [filetag] = numpy.array(E_)
    dat_E0[filetag] = E0_
    dat_ε [filetag] = numpy.array(E_) - E0_


##############################################################################
#                        PLOT BANDS

labels = {
    "reciprocal.COBYLA.r=1": "Single-Body",
    # "compact.BFGS.r=0": "Compact",
    "compact.COBYLA.r=1": "Compact r=1",
    "compact.COBYLA.r=2": "Compact r=2",
    "compact.COBYLA.r=3": "Compact r=3",
}

colors = {
    "reciprocal.COBYLA.r=1": COLORS["my1"],
    # "compact.BFGS.r=0": COLORS["my3"],
    "compact.COBYLA.r=1": COLORS["my2"],
    "compact.COBYLA.r=2": COLORS["my4"],
    "compact.COBYLA.r=3": COLORS["quantumviolet"],
}

# INITIALIZE FIGURE
fig, ax = matplotlib.pyplot.subplots(
    dpi=100,
    figsize=[7,4.75],
)

# CALCULATE ANALYTICAL ENERGIES
expmt = ortho.experiment.reciprocal.Reciprocal()
xv, bv = expmt.lattice.vertices(expmt.sym)
x0, b0 = expmt.lattice.traverse(expmt.sym)
E0 = numpy.array([expmt.calculate_energies(x) for x in x0])

# CALCULATE y-LIMITS
Δ = 1
ylim = [
    numpy.min(E0) - Δ,
    numpy.max(E0) + Δ
]

# CONFIGURE y-AXIS
ax.set_ylim(ylim)
ax.set_ylabel("Energy (eV)")
for lim in ylim:
    ax.axhline(lim, color='black', lw=1, clip_on=False)

# CONFIGURE x-AXIS AND MARK HIGH-SYMMETRY POINTS
ax.set_xlim([0,1])
ax.set_xticks(xv)
ax.set_xticklabels(expmt.sym)
for x in xv:
    ax.axvline(x, color='black', lw=1, clip_on=False)

# PLOT ANALYTICAL ENERGIES
for l in range(E0.shape[1]):
    ax.plot(
        x0, E0[:,l],
        ls='-', lw=1.5, c='black',
    )

# PLOT EXPERIMENTAL ENERGIES
for filetag in filetags:
    for l in range(4):
        ax.plot(
            dat_λ[filetag], dat_E[filetag][:,l],
            ls='', marker='d', lw=1.5, c=colors[filetag], label=labels[filetag],
            clip_on=False,
        )

fig.savefig("fig/band_plot.png", bbox_inches='tight')

##############################################################################
#                        PLOT ERRORS

labels = {
    "reciprocal.COBYLA.r=1": "Single-Body",
    # "compact.BFGS.r=0": "Full Hilbert Space",
    "compact.COBYLA.r=1": "Compact r=1",
    "compact.COBYLA.r=2": "Compact r=2",
    "compact.COBYLA.r=3": "Compact r=3",
}

colors = {
    "reciprocal.COBYLA.r=1": COLORS["my1"],
    # "compact.BFGS.r=0": COLORS["my3"],
    "compact.COBYLA.r=1": COLORS["my2"],
    "compact.COBYLA.r=2": COLORS["my4"],
    "compact.COBYLA.r=3": COLORS["quantumviolet"],
}


# INITIALIZE FIGURE
fig, ax = matplotlib.pyplot.subplots(
    dpi=100,
    figsize=[7,4.75],
)

# CONFIGURE AXES
ax.set_ylabel("Max Log |Error|")
ax.set_xlabel("Band Index ($l$)")
ax.set_xlim(0,3)
ax.set_xticks(range(4))
for x in range(4):
    ax.axvline(x, color='black', lw=1, clip_on=False)

# PLOT ANALYTICAL ENERGIES
for filetag in filetags:
    λ_ = dat_λ[filetag]
    E_ = dat_E[filetag]
    ε_ = dat_ε[filetag]

    N = E_.shape[1]
    logε = numpy.max(numpy.log(numpy.abs(ε_)), axis=0)

    ax.plot(
        range(N), logε,
        ls='-', marker='d', lw=1.5, c=colors[filetag], label=labels[filetag],
        clip_on=False,
    )
ax.legend()

fig.savefig("fig/band_error.png", bbox_inches='tight')

# DISPLAY PLOT
matplotlib.pyplot.show()
