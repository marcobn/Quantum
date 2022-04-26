""" Define interface for any energy plot over a path. """

import numpy as np
import cirq
import ortho.binary

class Experiment:
    def __init__(self, id, N, n, basis):
        """ Encapsulates functionality for constructing an interesting plot.

        Parameters
        ----------
        id (str): unique identifier for a given experiment
        N (int): number of basis vectors in eigenspace
        n (int): number of qubits needed to simulate
        basis (list of int): ordered list of basis states in active space

        """
        self.id = id
        self.N = N
        self.n = n
        self.basis = basis

    def calculate_energies(self, λ):
        """ Analytically calculate energies, without quantum computation.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot

        Returns
        -------
        E (list): energies of each level, at the given point in the plot

        """
        return NotImplemented

    def construct_pauli_sum(self, λ):
        """ Construct the Hamiltonian as a Pauli sum.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot

        Returns
        -------
        H (cirq.PauliSum): operator where ⟨H⟩ gives energy of a quantum state

        """
        return NotImplemented

    def construct_ansatz(self, qreg, l, x, r=1):
        """ Construct the self-orthogonalizing circuit.

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
        return NotImplemented

    def num_parameters(self, l):
        """ Calculate number of free parameters needed for a given energy level.

        Parameters
        ----------
        l (int): which energy level

        Returns
        -------
        Np (int): number of parameters

        """
        return 2*(self.N-1 - l)     # EACH LEVEL CUTS OUT ONE COMPLEX NUMBER

    def construct_cost_function(self, λ, qreg, l, xs, r=1):
        """ Construct a cost-function suitable to optimize the energy surface.

        Parameters
        ----------
        λ (float): value in [0,1] giving location along plot
        qreg (list of cirq.Qubit): quantum register to operate on
        l (int): which energy level
        xs (list of list of float): x[i] gives list of parameters for level i
        r (int): number of Trotter steps, when relevant

        Returns
        -------
        costfn (callable): takes a list of float parameters and returns a float
            Parameter list should have num_parameters(l) floats.

        """
        # PREPARE THE HAMILTONIAN PAULI SUM TO MEASURE
        H = self.construct_pauli_sum(λ)

        # PREPARE THE ORTHOGONALIZATION CIRCUIT
        Ω = [self.construct_ansatz(qreg, i, xs[i], r=r) for i in range(l)]
        ΠΩ = np.eye(1<<self.n) if r is None else cirq.Circuit()
        for i in range(l):
            if r is None:   ΠΩ = ΠΩ @ Ω[i]
            else:           ΠΩ = Ω[i] + ΠΩ

        # PREPARE CIRCUIT TO PREPARE STATE |l⟩ (not needed in matrix mode)
        if r is not None:
            zbin = ortho.binary.binarize(self.basis[l], self.n)

            # APPLY X TO EACH "ON" BIT
            prep_l = cirq.Circuit()
            for q, qbit in enumerate(qreg):
                if zbin[q]: prep_l += cirq.X(qbit)
                else:       prep_l += cirq.I(qbit)

        # PREPARE SIMULATOR (not needed in matrix mode)
            sim = cirq.Simulator()

        # PREPARE QUBIT MAP FOR PAULI EVALUATION
        qubit_map = {q: i for i, q in enumerate(qreg)}

        counter = np.array([0])
        def costfn(x):
            print (f"\tTrial {l}.{counter[0]}")
            counter[0] += 1


            # CONSTRUCT PARAMETERIZED
            Ωl = self.construct_ansatz(qreg, l, x, r=r)

            # ORTHOGONALIZE
            if r is None:   ansatz = ΠΩ @ Ωl
            else:           ansatz = Ωl + ΠΩ

            # EXECUTE ANSATZ ON |l⟩
            if r is None:   Ψ = ansatz[:,self.basis[l]]
            else:           Ψ = sim.simulate(prep_l+ansatz).final_state_vector

            # EVALUATE ENERGY
            Ψ /= np.sqrt(np.vdot(Ψ,Ψ))
            E = H.expectation_from_state_vector(Ψ, qubit_map=qubit_map).real
            return E
        return costfn

    def to_json(self):
        """ Construct serializable dict. """
        return {
            "id": self.id,
            # NOT NEEDED FOR SERIALIZATION, BUT GOOD TO SEE IN FILE:
            "N": self.N,
            "n": self.n,
        }

    def from_json(json):
        """ Construct Experiment object from serializable dict `json`. """
        return REGISTRY[json["id"]]()


REGISTRY = {}
def register(cls):
    REGISTRY[cls.id] = cls
