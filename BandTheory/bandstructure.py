""" Generate band structure with VQD. """

# IMPORTS
import os
import time

import numpy as np
np.set_printoptions(3, suppress=True)
import qiskit

import qctb
import tightbinding

#                                 SETTINGS
################################################################################

# QPU SELECTION
qpu = "svm"
# qpu = "qvm"
# qpu = "ibmq_athens"

qvm = True

QC = qctb.ibmq.SVM
# QC = qctb.ibmq.QVM
# QC = qctb.ibmq.RAW(qpu, qvm)
# QC = qctb.ibmq.CAL(qpu, qvm)

# PERIODIC SYSTEM
H = tightbinding.chain.Chain()
H = tightbinding.chain.AlternatingChain()
H = tightbinding.graphene.Graphene()
# H = tightbinding.silicon.Silicon()
# H = tightbinding.polonium.Polonium(Vsp=2)


# CLASSICAL OPTIMIZATION
OPT = qiskit.aqua.components.optimizers.COBYLA()

# FILE PREFIX
jsonprefix = f"json/{H.id}.{'sim.' if qvm else ''}{qpu}"
dataprefix = f"data/{H.id}.{'sim.' if qvm else ''}{qpu}"

# NUMBER OF OPTIMIZATION RUNS PER VQE RUN
nopt = 1

# (MIN) NUMBER OF POINTS TO PLOT ALONG PATH
res = 3
res = 10

################################################################################


# FIND SMALLEST AVAILABLE TRIAL IDs
T = 0                   # JSON FILE
while os.path.isfile(f"{jsonprefix}.{T}.json"):
    T += 1
jsonfile = f"{jsonprefix}.{T}.json"
print (f"Writing data to {jsonfile}")

T = 0                   # DATA FILE
while os.path.isfile(f"{dataprefix}.{T}.data"):
    T += 1
datafile = f"{dataprefix}.{T}.data"
print (f"Writing data to {datafile}")

# CREATE VQD OBJECT
VQD = qctb.vqd.VQD(H, QC, OPT, nopt=nopt, filename=datafile)

# PREPARE PLOTS
plot0=None; plotx=None
# import vpython
# graph = vpython.graph(fast=False)
# plot0 = [vpython.gcurve(graph=graph) for l in range(VQD.H.M)]
# plotx =  vpython.gdots (graph=graph)


#                           BAND STRUCTURE EXPERIMENT
################################################################################
# EXTEND PLOTTERS TO LISTS FOR CONVENIENCE
if plot0 is not None:
    if not isinstance(plot0, list):
        plot0 = [plot0] * VQD.H.M
if plotx is not None:
    if not isinstance(plotx, list):
        plotx = [plotx] * VQD.H.M

# PREPARE PATH
if res:             # COLLECT NEW DATA
    pathlengths = np.linalg.norm( np.diff(VQD.H.path, axis=0), axis=1 )
    path = []
    for i in range(len(VQD.H.path)-1):
        ires = int(np.ceil( res * pathlengths[i] / sum(pathlengths) ))
        for λ in np.linspace(0,1, ires, endpoint=False):
            kpt = λ*(VQD.H.path[i+1]-VQD.H.path[i]) + VQD.H.path[i]
            path.append( kpt )
    path.append( VQD.H.path[-1] )
else:               # PLOT EXISTING DATA
    path = [np.array(point['kpt']) for point in VQD.points]

# ANALYTICAL CURVES
if plot0:
    for i, kpt in enumerate(path[:-1]):
        for λ in np.linspace(0,1,10, endpoint=False): # AMPLIFY RESOLUTION
            kpt = λ*(path[i+1]-path[i]) + path[i]
            E0 = VQD.H.E0(kpt)
            for l in range(VQD.H.M):                    # PLOT EACH ENERGY
                plot0[l].plot(i+λ, E0[l])

# RUN EXPERIMENT
for i, kpt in enumerate(path):
    if res:
        start = time.time()
        point = VQD.run(kpt.tolist())           # RUN POINT
        print (f"Finished: {i}, {kpt}")
        print (f"\tTook {time.time()-start} s")
        if jsonfile:
            VQD.save(jsonfile)                  # SAVE TO FILE
    else:
        point = VQD.points[i]                   # ACCESS EXISTING POINT

    if plotx:
        for l in range(VQD.H.M):                # PLOT EACH ENERGY
            plotx[l].plot(i, point["levels"][l]["E"])

print ("fin")
