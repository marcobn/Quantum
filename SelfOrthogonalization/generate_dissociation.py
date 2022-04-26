""" Validate model by testing analytical band structure. """

import json

import numpy
numpy.set_printoptions(3, suppress=True)

import cirq

import ortho.optimization.scipy
import ortho.experiment.hydrogen

# SETUP RUN PARAMETERS
r = 1

from datetime import datetime
for r in range(1,6):
    print (f"\n===== Starting r={r} at {datetime.now()} =====\n")

    # SETUP CONTROL OBJECTS
    expmt = ortho.experiment.hydrogen.Hydrogen()
    optimizer = ortho.optimization.scipy.COBYLA()
    qreg = cirq.LineQubit.range(expmt.n)
    rng = numpy.random.default_rng(0 if r is None else r)

    # SETUP REPORT OBJECTS
    filetag = f"{expmt.id}.{optimizer.id}.r={0 if r is None else r}"
    points = []

    print ("Starting experiment!")

    λ_ = numpy.linspace(0,1,11)
    for i, λ in enumerate(λ_):
        points.append({
            "λ": λ,
            "results": [],
        })

        xs = []
        for l in range(expmt.N):
            x0 = rng.random(expmt.num_parameters(l))

            # USE HARTREE-FOCK GROUND-STATE DIRECTLY
            if l == 0:  x0 = numpy.zeros(expmt.num_parameters(0))
            # NOTE: This turns out to actually be bad for BFGS optimizer,
            #   since it finds HF state *is* a minimum in its initial scan.

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
