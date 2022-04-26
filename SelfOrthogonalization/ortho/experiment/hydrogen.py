""" Implement Experiment for hydrogen dissociation using JW basis. """

import numpy as np
import sympy

import cirq
import openfermion
import openfermionpyscf

import ortho.binary
from ortho.experiment.experiment import Experiment, register
from ortho.ansatz import isoparticle


def molecular_hydrogen(r):
    """ Construct H2 as a MolecularData object and load/calculate integrals. """
    # CONSTRUCT GEOMETRY
    molecule = openfermion.MolecularData(
        geometry=[('H', (0,0,0)), ('H', (0,0,r))],
        charge=0,
        basis='sto-3g',
        multiplicity=1,
        description=f"H2_r-{r:.3f}",
    )

    # NOW HAVE pyscf CALCULATE INTEGRALS
    try:
        # raise OSError()
        molecule.load()
    except OSError:
        molecule = openfermionpyscf.run_pyscf(
            molecule,
            run_scf=True,
            run_cisd=True,
        )
        molecule.save()

    return molecule





class Hydrogen(Experiment):
    id = "hydrogen"

    # HARD-CODE BOUNDS OF DISSOCIATION CURVE
    r0 = 0.3
    rΩ = 2.3

    def __init__(self):
        """ Encapsulates functionality for constructing an interesting plot.

        This plot is a dissociation curve of molecular hydrogen,
            including excited state curves.
        In such a minimal basis, one cannot get much use out of excited states,
            but one can observe each excited state is unbounded.
        Most importantly it lets us show off our functional circuit...

        """
        super().__init__(
            self.id,
            isoparticle.dimension(4,2),
            4,
            isoparticle.basisstates(4,2),
        )

    def calculate_energies(self, λ):
        """ Analytically calculate energies, without quantum computation.

        This method relies on integral calculations by pyscf,
            and openfermion's operator algebra.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot

        Returns
        -------
        E (list): energies of each level, at the given point in the plot

        """
        # CALCULATE BOND SEPARATION FROM PATH COORDINATE
        r = (self.rΩ - self.r0) * λ + self.r0

        # LOAD MOLECULE
        molecule = molecular_hydrogen(r)

        # CONSTRUCT THE FERMION OPERATOR
        interop = molecule.get_molecular_hamiltonian()
        fermiop = openfermion.get_fermion_operator(interop)

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

        # CONSTRUCT CONFIGURATION-INTERACTION MATRIX, PRESERVING 2 ELECTRONS
        CI = np.zeros((len(basis),len(basis)), dtype=complex)
        for i, zi in enumerate(basis):
            for j, zj in enumerate(basis):
                term = openfermion.normal_ordered( A[i] * fermiop * At[j] )
                CI[i,j] = term.constant

        # DIAGONALIZE MATRIX
        return np.linalg.eigh(CI)[0]

    def construct_pauli_sum(self, λ):
        """ Construct the Hamiltonian as a Pauli sum.

        This method relies on integral calculations by pyscf,
            and openfermion's transformation methods.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot

        Returns
        -------
        H (cirq.PauliSum): operator where ⟨H⟩ gives energy of a quantum state

        """
        # CALCULATE BOND SEPARATION FROM PATH COORDINATE
        r = (self.rΩ - self.r0) * λ + self.r0

        # LOAD MOLECULE
        molecule = molecular_hydrogen(r)

        # CONSTRUCT THE PAULI OPERATOR, VIA A CHAIN OF TRANSFORMS
        interop = molecule.get_molecular_hamiltonian()
        fermiop = openfermion.get_fermion_operator(interop)
        qubitop = openfermion.jordan_wigner(fermiop)

        qreg = cirq.LineQubit.range(4)
        H = openfermion.qubit_operator_to_pauli_sum(qubitop, qreg)

        return H

    def construct_ansatz(self, qreg, l, x, r=1):
        """ Construct the self-orthogonalizing circuit.

        Use the UCC-like decomposition preserving particle number.

        Parameters
        ----------
        qreg (list of cirq.Qubit): quantum register to operate on
        l (int): which energy level
        x (list): numerical values of each free parameter (no symbols x_x)
        r (int): number of Trotter steps, when relevant

        Returns
        -------
        ct (cirq.Circuit): the circuit Ω[l]
            unless r=None, in which case ct is the 2darray form of Ω[l]

        """
        # CONSTRUCT THE t AND α ARRAYS
        Np = self.num_parameters(l)     # THIS SHOULD ALSO BE len(x)
        t = np.concatenate(([0]*l, x[:Np//2]))
        α = np.concatenate(([0]*l, x[Np//2:]))

        # CONSTRUCT THE ANALYTICAL OPERATOR
        if r is None: return isoparticle.operator(4,2,t,α,l)

        # CONSTRUCT THE HERMITIAN OPERATOR
        H = isoparticle.hamiltonian(4,2,t,α,l)

        # CONSTRUCT THE TROTTERIZED CIRCUIT
        return isoparticle.trottercircuit(qreg, H, r, -np.pi)


register(Hydrogen)
