""" Validate model by testing analytical band structure. """

import json

import numpy
numpy.set_printoptions(3, suppress=True)
rng = numpy.random.default_rng(0)


import cirq

import ortho.experiment.hydrogen

import matplotlib
import matplotlib.pyplot

# PRELIMINARY CONFIGURATION
matplotlib.rc('font', family='serif')
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


# ##############################################################################
# #                       PLOT ANALYTICAL ENERGIES
#
# # CALCULATE ENERGIES
# expmt = ortho.experiment.hydrogen.Hydrogen()
# λ0 = numpy.linspace(0,1,101)
# E0 = numpy.array([expmt.calculate_energies(λ) for λ in λ0])
#
# # INITIALIZE FIGURE
# fig, ax = matplotlib.pyplot.subplots(
#     dpi=100,
#     figsize=[7,4.75],
# )
#
# # CONFIGURE AXES
# ax.set_ylabel("Energy (eV)")
# ax.set_xlim([expmt.r0,expmt.rΩ])
#
# # PLOT ANALYTICAL ENERGIES
# r0 = λ0*(expmt.rΩ - expmt.r0) + expmt.r0
# for l in range(E0.shape[1]):
#     ax.plot(
#         r0, E0[:,l],
#         ls='-', lw=1.5, c='black',
#     )
#
# # TODO: Maybe distinguish each curve by its multiplicity?
# # TODO: Throw in r=0 markers to establish validity.

##############################################################################
#                        LOAD EXPERIMENTAL DATA

expmt = ortho.experiment.hydrogen.Hydrogen()    # FOR ANALYTICAL VALUES

filetags = [
    # "hydrogen.COBYLA.r=0",
    "hydrogen.COBYLA.r=1",
    "hydrogen.COBYLA.r=2",
    "hydrogen.COBYLA.r=3",
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

labels = {
    # "hydrogen.COBYLA.r=0": "Analytical",
    "hydrogen.COBYLA.r=1": "r=1",
    "hydrogen.COBYLA.r=2": "r=2",
    "hydrogen.COBYLA.r=3": "r=3",
}

colors = {
    # "hydrogen.COBYLA.r=0": COLORS["my1"],
    "hydrogen.COBYLA.r=1": COLORS["my2"],
    "hydrogen.COBYLA.r=2": COLORS["my4"],
    "hydrogen.COBYLA.r=3": COLORS["quantumviolet"],
}



# CALCULATE ENERGIES
expmt = ortho.experiment.hydrogen.Hydrogen()
λ0 = numpy.linspace(0,1,101)
E0 = numpy.array([expmt.calculate_energies(λ) for λ in λ0])

# INITIALIZE FIGURE
fig, ax = matplotlib.pyplot.subplots(
    dpi=100,
    figsize=[7,4.75],
)

# CONFIGURE AXES
ax.set_ylabel("Energy (Ha)")
ax.set_xlabel("Bond Length (A)")
ax.set_xlim([expmt.r0,expmt.rΩ])

# PLOT ANALYTICAL ENERGIES
r0 = λ0*(expmt.rΩ - expmt.r0) + expmt.r0
for l in range(E0.shape[1]):
    ax.plot(
        r0, E0[:,l],
        ls='-', lw=1.5, c='black',
    )

# PLOT EXPERIMENTAL ENERGIES
for filetag in filetags:
    for l in range(6):
        r = dat_λ[filetag]*(expmt.rΩ - expmt.r0) + expmt.r0
        ax.plot(
            r, dat_E[filetag][:,l],
            ls='', marker='d', lw=1.5, c=colors[filetag], label=labels[filetag],
            clip_on=False,
        )

fig.savefig("fig/hydrogen_plot.png", bbox_inches='tight')



##############################################################################
#                        PLOT ERRORS


labels = {
    # "hydrogen.COBYLA.r=0": "Analytical",
    "hydrogen.COBYLA.r=1": "r=1",
    "hydrogen.COBYLA.r=2": "r=2",
    "hydrogen.COBYLA.r=3": "r=3",
}

colors = {
    # "hydrogen.COBYLA.r=0": COLORS["my1"],
    "hydrogen.COBYLA.r=1": COLORS["my2"],
    "hydrogen.COBYLA.r=2": COLORS["my4"],
    "hydrogen.COBYLA.r=3": COLORS["quantumviolet"],
}

# INITIALIZE FIGURE
fig, ax = matplotlib.pyplot.subplots(
    dpi=100,
    figsize=[7,4.75],
)

# CONFIGURE AXES
ax.set_ylabel("Max Log |Error|")
ax.set_xlabel("Excitation Index ($l$)")
ax.set_xlim(0,5)
ax.set_xticks(range(6))
for x in range(6):
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

fig.savefig("fig/hydrogen_error.png", bbox_inches='tight')




# DISPLAY PLOT
matplotlib.pyplot.show()
