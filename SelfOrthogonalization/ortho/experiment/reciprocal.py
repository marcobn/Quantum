""" Implement Experiment for polonium band structure using reciprocal basis. """

import numpy as np
from numpy import pi as π

import cirq

from ortho.experiment.experiment import Experiment, register
import ortho.lattice
import ortho.model.polonium
import ortho.ansatz.gard

class Reciprocal(Experiment):
    id = "reciprocal"

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
            obtained with the reciprocal orbital qubit basis.

        Parameters
        ----------
        id (str): unique identifier for a given experiment

        """
        super().__init__(self.id, 4, 4, [8, 4, 2, 1])

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

        This method manually assembles the Pauli Sum from our paper.

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

        # HANDY ALIASES
        from cirq import PauliString as PW
        qreg = cirq.LineQubit.range(self.n)
        I = [cirq.I(qreg[q]) for q in range(self.n)]
        X = [cirq.X(qreg[q]) for q in range(self.n)]
        Y = [cirq.Y(qreg[q]) for q in range(self.n)]
        Z = [cirq.Z(qreg[q]) for q in range(self.n)]

        # MANUALLY CONSTRUCT PAULI SUM
        H = 0
        for α in range(self.n):
            H += 0.5 * Hk[α,α].real * ( PW(I[α]) - PW(Z[α]) )
            for β in range(α+1, self.n):
                H += 0.5 * Hk[α,β].real * ( PW([X[α],X[β]]) + PW([Y[α],Y[β]]) )
                H += 0.5 * Hk[α,β].imag * ( PW([Y[α],X[β]]) - PW([X[α],Y[β]]) )

        return H

    def construct_ansatz(self, qreg, l, x, r=1):
        """ Construct the self-orthogonalizing circuit.

        Use the one-particle Gard circuit, with A gates.

        Parameters
        ----------
        qreg (list of cirq.Qubit): quantum register to operate on
        l (int): which energy level
        x (list): numerical values of each parameter (no symbols x_x)
        r (int): number of Trotter steps, when relevant
                                    (not relevant in this class)

        Returns
        -------
        ct (cirq.Circuit): the circuit Ω[l]
            unless r=None, in which case ct is the 2darray form of Ω[l]

        """
        # CONSTRUCT THE t AND α ARRAYS
        Np = self.num_parameters(l)     # THIS SHOULD ALSO BE len(x)
        t = x[:Np//2]
        α = x[Np//2:]

        # STITCH TOGETHER ALL THE A-GATES
        ct = cirq.Circuit()
        for q in range(l):
            ct.append( cirq.I(qreg[q]) )
        if l == self.n-1:
            ct.append( cirq.I(qreg[l]) )
        for q in range(l,self.n-1):
            θ = π/2 * t[q-l]
            ϕ = 2*π * α[q-l]

            ct.append( ortho.ansatz.gard.A(θ,ϕ).on(qreg[q], qreg[q+1]) )

            # I like these better but we use A so we can hearken to DS paper.
            # ct.append( cirq.givens(θ).on(qreg[q], qreg[q+1]) )
            # ct.append( cirq.X.on(qreg[q]) )
            # ct.append((cirq.CZ ** (ϕ/π) ).on(qreg[q], qreg[q+1]) )
            # ct.append( cirq.X.on(qreg[q]) )

        return cirq.unitary(ct) if r is None else ct




register(Reciprocal)
