""" Generate band structure with VQD. """

# IMPORTS
import os

import numpy as np
np.set_printoptions(3, suppress=True)
import qiskit

import qctb
import tightbinding


# SELECT SYSTEM
################################################################################

# SELECT QUANTUM COMPUTER
QC = qctb.ibmq.CAL("ibmq_athens", True)
# QC = qctb.ibmq.RAW("ibmq_athens", True)
QC = qctb.ibmq.SVM
prefix = "json"

# SELECT HAMILTONIAN
H = tightbinding.polonium.Polonium(Vsp=2)

# SELECT OPTIMIZER
OPT = qiskit.aqua.components.optimizers.COBYLA( )

# NUMBER OF OPTIMIZATION RUNS PER VQE RUN
nopt = 1

# PREPARE ANALYTICAL CURVE
################################################################################

# PREPARE PATH
pathlengths = np.linalg.norm( np.diff(H.path, axis=0), axis=1 )
path = []                               # LIST OF kpts TO TRAVERSE OVER
ipath = [0]                             # INDICES OF EACH H.path ELEMENT IN path
for i in range(len(H.path)-1):
    ires = int(np.ceil( 100 * pathlengths[i] / sum(pathlengths) ))
    ipath.append( ipath[-1] + ires )
    for λ in np.linspace(0,1, ires, endpoint=False):
        kpt = λ*(H.path[i+1]-H.path[i]) + H.path[i]
        path.append( kpt )
path.append( H.path[-1] )

# EVALUATE ENERGIES
plot0 = [[] for l in range(H.M)]
for ii, kpt in enumerate(path):
    E0 = H.E0(kpt)
    for l in range(H.M):                # PLOT EACH ENERGY
        plot0[l].append([ii, E0[l]])
plot0 = np.array(plot0)


#                                  LOAD DATA
################################################################################

plotx = [[] for l in range(H.M)]
for filename in os.listdir(prefix):
    try:
        VQD = qctb.vqd.load(f"{prefix}/{filename}")
    except:
        continue    # SKIP IF INVALID JSON

    # SKIP IF SETTINGS DON'T MATCH
    if not (
        VQD.n is None
        and VQD.nopt == nopt
        and VQD.H.toJSON() == H.toJSON()
        and VQD.QC.toJSON() == QC.toJSON()
        and qctb.serializeoptimizer.toJSON(VQD.OPT) ==
            qctb.serializeoptimizer.toJSON(OPT)
    ): continue

    for point in VQD.points:
        # LOCATE COORDINATE OF kpt ALONG path
        kpt = np.array(point['ka'])
        for i in range(len(H.path)-1):

            # FIND SCALING
            Δk = kpt - H.path[i]
            ΔB = H.path[i+1] - H.path[i]
            λ = None
            offbranch = False
            for j in range(H.D):
                if offbranch: break             # SKIP POINT

                if ΔB[j] == 0 and Δk[j] == 0:
                    continue                            #  ON INVARIANT POINT
                elif ΔB[j] == 0:
                    offbranch = True                    # OFF INVARIANT POINT
                else:
                    λj = Δk[j] / ΔB[j]
                    if λ is None:   λ = λj              # REGISTER SCALAR
                    elif λ == λj:   continue                # ACCEPT SAME SCALAR
                    else:           offbranch = True    # NON-SCALAR SCALING

            if offbranch: continue              # SKIP POINT
            if λ is None:   λ = 0               # COMPLETELY INVARIANT BRANCH

            # APPLY SCALING TO PATH INDEX
            pt = ipath[i] + λ*(ipath[i+1]-ipath[i])

            # REGISTER ENERGIES
            for l, level in enumerate(point['levels']):
                plotx[l].append([pt, level['E']])
            break                               # DON'T PERMIT MULTIPLE BRANCHES

plotx = [np.array(plotx[x]) for x in range(len(plotx))]


#                                  PLOT DATA
################################################################################
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# PLOT ANALYTICAL CURVES
for l in range(len(plot0)):
    ax.plot( plot0[l,:,0], plot0[l,:,1], 'k')


# PLOT DATA
for l in range(len(plotx)):
    ax.plot( plotx[l][:,0], plotx[l][:,1], 'x')


plt.show()
