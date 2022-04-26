""" Check that we understand precisely what each eigenstate is in hydrogen. """

import numpy
numpy.set_printoptions(3, suppress=True)

import openfermion
import ortho.experiment.hydrogen

from datetime import datetime
from ortho.ansatz import isoparticle
import ortho.binary


# CONSTRUCT BASIS VECTORS, AS CREATION/ANNIHILATION OPERATORS
basis = isoparticle.basisstates(4,2)
N = isoparticle.dimension(4,2)
A, At = [], []
for i, zi in enumerate(basis):
    A.append(openfermion.FermionOperator([
        (q,0) for q in range(4)
            if ortho.binary.binarize(zi,4)[q]
    ]))
    At.append(openfermion.FermionOperator([
        (q,1) for q in reversed(range(4))
            if ortho.binary.binarize(zi,4)[q]
    ]))

# DISPLAY BASIS VECTORS
print ("-- BASIS --")
for i, zi in enumerate(basis):
    print (i, ortho.binary.binarize(zi,4))


# SETUP CONTROL OBJECTS
expmt = ortho.experiment.hydrogen.Hydrogen()

print ("Starting experiment!")
λ_ = numpy.linspace(0,1,11)
for i, λ in enumerate(λ_):

    # CALCULATE BOND SEPARATION FROM PATH COORDINATE
    r = (expmt.rΩ - expmt.r0) * λ + expmt.r0

    # LOAD MOLECULE
    molecule = ortho.experiment.hydrogen.molecular_hydrogen(r)

    # CONSTRUCT THE FERMION OPERATOR
    interop = molecule.get_molecular_hamiltonian()
    fermiop = openfermion.get_fermion_operator(interop)

    # CONSTRUCT CONFIGURATION-INTERACTION MATRIX, PRESERVING 2 ELECTRONS
    CI = numpy.zeros((len(basis),len(basis)), dtype=complex)
    for i, zi in enumerate(basis):
        for j, zj in enumerate(basis):
            term = openfermion.normal_ordered( A[i] * fermiop * At[j] )
            CI[i,j] = term.constant

    # DIAGONALIZE MATRIX
    Λ, U = numpy.linalg.eigh(CI)

    # DISPLAY RESULTS
    print (f"------------------------")
    for l in range(6):
        print (f"E={Λ[l]:.3f}  {U[:,l]}")

print ("Experiment complete!")
