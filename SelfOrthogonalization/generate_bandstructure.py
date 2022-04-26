""" Validate model by testing analytical band structure. """

import numpy
import json

import cirq

import ortho.optimization.scipy
import ortho.experiment.reciprocal
import ortho.experiment.compact


# SETUP RUN PARAMETERS
r = 1

from datetime import datetime
for r in range(1,4):
    print (f"\n===== Starting r={r} at {datetime.now()} =====\n")

    # SETUP CONTROL OBJECTS
    expmt = ortho.experiment.compact.Compact()
    optimizer = ortho.optimization.scipy.COBYLA()
    qreg = cirq.LineQubit.range(expmt.n)
    rng = numpy.random.default_rng(0 if r is None else r)

    # SETUP REPORT OBJECTS
    filetag = f"{expmt.id}.{optimizer.id}.r={0 if r is None else r}"
    points = []

    print ("Starting experiment!")

    λ_, b_ = expmt.lattice.traverse(expmt.sym, 2)
    for i, λ in enumerate(λ_):
        points.append({
            "λ": λ,
            "results": [],
        })

        xs = []
        for l in range(expmt.N):
            x0 = rng.random(expmt.num_parameters(l))

            costfn = expmt.construct_cost_function(λ, qreg, l, xs, r=r)

            result = optimizer.optimize(costfn, x0)

            xs.append(result.x)                             # FOR ORTHOGONALIZATION
            points[-1]["results"].append(result.to_json())  # FOR FILE RECORD
            print (f"------------------------")             # FOR USER SANITY
            print (f"Found energy: {result.E}")
            print (f"Parameters: {result.x}")
            print (f"Functions: {result.nfev}")
            print (f"Converged: {result.converged}")

            # WRITE FILE
            with open(f"json/{filetag}.json", "w", encoding='utf-8') as fp:
                json.dump({
                    "experiment":   expmt.to_json(),
                    "optimizer":    optimizer.to_json(),
                    "r":            r,
                    "points":       points,
                }, fp, ensure_ascii=False, indent=4)

    print ("Experiment complete!")


# ##############################################################################
# #                         PLOT RECIPROCAL CALCULATIONS
#
# # CALCULATE ANALYTICAL ENERGIES
# rng = numpy.random.default_rng(0)
# optimizer = ortho.optimization.scipy.COBYLA()
# expmt = ortho.experiment.reciprocal.Reciprocal()
# qreg = cirq.LineQubit.range(expmt.n)
#
# λv, bv = expmt.lattice.vertices(expmt.sym)
# λ_, b_ = expmt.lattice.traverse(expmt.sym, 2)
# E_ = numpy.zeros((len(λ_),expmt.N))
#
# for i, λ in enumerate(λ_):
#     xs = []
#     for l in range(expmt.N):
#         # x0 = numpy.zeros(expmt.num_parameters(l))
#         x0 = rng.random(expmt.num_parameters(l))
#
#         costfn = expmt.construct_cost_function(λ, qreg, l, xs)
#         result = optimizer.optimize(costfn, x0)
#         xs.append(result.x)
#         E_[i,l] = result.E
#
#         print (f"Found energy: {result.E}")
#         print (f"Parameters: {result.x}")
#         print (f"Functions: {result.nfev}")
#         print (f"Converged: {result.converged}")
#


# ##############################################################################
# #                         PLOT COMPACT CALCULATIONS
#
# # CALCULATE ANALYTICAL ENERGIES
# rng = numpy.random.default_rng(0)
# optimizer = ortho.optimization.scipy.COBYLA()   # NOTE: Use BFGS for r=None run.
# expmt = ortho.experiment.compact.Compact()
# qreg = cirq.LineQubit.range(expmt.n)
#
# λv, bv = expmt.lattice.vertices(expmt.sym)
# λ_, b_ = expmt.lattice.traverse(expmt.sym, 2)
# E_ = numpy.zeros((len(λ_),expmt.N))
#
# for i, λ in enumerate(λ_):
#     xs = []
#     for l in range(expmt.N):
#         # x0 = numpy.zeros(expmt.num_parameters(l))
#         x0 = rng.random(expmt.num_parameters(l))
#
#         costfn = expmt.construct_cost_function(λ, qreg, l, xs, r=1)
#         result = optimizer.optimize(costfn, x0)
#         xs.append(result.x)
#         E_[i,l] = result.E
#
#         print (f"Found energy: {result.E}")
#         print (f"Parameters: {result.x}")
#         print (f"Functions: {result.nfev}")
#         print (f"Converged: {result.converged}")
#
