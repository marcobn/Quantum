""" Implement Experiment for polonium band structure using compact basis. """

import numpy as np

import cirq

from ortho.experiment.experiment import Experiment, register
import ortho.lattice
import ortho.model.polonium
import ortho.ansatz.compact

class Compact(Experiment):
    id = "compact"

    # HARD-CODE THE PHYSICAL SYSTEM
    lattice = ortho.lattice.SimpleCubic()
    sym = ["Γ", "R", "X", "M", "R", "Γ", "X", "M", "Γ"]
    model = ortho.model.polonium.Polonium_sp(
        Es  = -14,
        Ep  = 0,
        Vss = 0,
        Vsp = 2,
        Vpσ = 2,
        Vpπ = 0,
    )

    def __init__(self):
        """ Encapsulates functionality for constructing an interesting plot.

        This plot is a band-structure of a simple-cubic lattice,
            obtained with the compact qubit basis.

        Parameters
        ----------
        id (str): unique identifier for a given experiment

        """
        super().__init__(self.id, 4, 2, list(range(4)))

    def calculate_energies(self, λ):
        """ Analytically calculate energies, without quantum computation.

        This method shunts all the work to lattices and models.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot

        Returns
        -------
        E (list): energies of each level, at the given point in the plot

        """
        b = self.lattice.frompath(self.sym, λ)  # IDENTIFY RECIPROCAL COORDINATE
        return self.model.E0(b)                 # CALCULATE ENERGIES

    def construct_pauli_sum(self, λ):
        """ Construct the Hamiltonian as a Pauli sum.

        This method constructs Hk from the model,
            then decomposes with the Hilbert-Schmidt inner product.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot

        Returns
        -------
        H (cirq.PauliSum): operator where ⟨H⟩ gives energy of a quantum state

        """
        # CALCULATE COORDINATE-SPECIFIC TERMS
        b = self.lattice.frompath(self.sym, λ)  # IDENTIFY RECIPROCAL COORDINATE
        Hk = self.model.Hk(b)                   # MATRIX FORM OF HAMILTONIAN

        # CONSTRUCT PAULI SUM
        qubits = cirq.LineQubit.range(self.n)
        H = 0
        for i in range(1<<(2*self.n)):
            pauli = ortho.ansatz.compact.pauliword(self.n, i)
            c = np.trace(Hk @ pauli.matrix(qubits)) / (1<<self.n)
            H += pauli * c

        return H

    def construct_ansatz(self, qreg, l, x, r=1):
        """ Construct the self-orthogonalizing circuit.

        Use the UCC-like decomposition considering all basis states.

        Parameters
        ----------
        qreg (list of cirq.Qubit): quantum register to operate on
        l (int): which energy level
        x (list): numerical values of each parameter (no symbols x_x)
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
        if r is None: return ortho.ansatz.compact.operator(t,α,l)

        # CONSTRUCT THE HERMITIAN OPERATOR
        H = ortho.ansatz.compact.hamiltonian(self.n,t,α,l)

        # CONSTRUCT THE TROTTERIZED CIRCUIT
        return ortho.ansatz.compact.trottercircuit(qreg, H, r, -np.pi)

register(Compact)
