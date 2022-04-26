import numpy
from numpy import pi

class Model:
    def __init__(self, id, parameters):
        """ Encapsulate orbitals and hopping parameters.

        Parameters
        ----------
        id: str
            specify which model to use
        parameters: dict
            model-specific parameters

        """
        self.id         = id
        self.parameters = parameters
        self.lattice    = self._construct_lattice_(**parameters)
        self.a          = self._construct_orbitals(**parameters)
        self.t          = self._construct_singles_(**parameters)
        # self.h          = self._construct_doubles_(**parameters)
        self.M          = len(self.a)
        self.d          = self.lattice.d

    def _construct_lattice_(self, **parameters):
        """
        Returns
        -------
        lattice: tightbinding.lattice.Lattice

            The underlying Bravais lattice, with dimension d.

        """
        return NotImplemented

    def _construct_orbitals(self, **parameters):
        """
        Returns
        -------
        a: M×d array

            Crystal coordinates of each orbital.
            a[α] is the crystal coordinate of orbital α

        """
        return NotImplemented

    def _construct_singles_(self, **parameters):
        """
        Returns
        -------
        t: list with len(M) of dict[tuple(int) with len(1+d)]: float

            Single-body hopping parameters of model.

            t[α][(β,δ)] is the coefficient for a†[(α,0)] a[(β,δ)]

            Example: t[5][3,0,1] gives hopping parameter
                from orbital 3 of the cell *above*, to orbital 5.

            Constraints:
                t[β][α,δ] = t[α][β,-δ]

        """
        return NotImplemented

    # def _construct_doubles_(self, **parameters):
    #     """
    #     Returns
    #     -------
    #     t: list with len(M) of dict[tuple(int) with len(3*(1+d))]: float
    #
    #         Two-body fermionic interaction parameters.
    #
    #         t[α][(β,δ1,γ,δ2,δ,δ3)] is the coefficient for
    #             a†[(α,0)] a†[(β,δ1)] a[(γ,δ2)] a[(δ,δ3)]
    #
    #         Constraints:
    #             lol i dunno.
    #             TODO: haha this isn't gonna work... so much double counting
    #
    #     """
    #     return NotImplemented

    def Hk(self, b):
        """ Hamiltonian of system restricted to reciprocal coordinate b.

        Parameters
        ----------
        b: ndarray of length d
            reciprocal coordinate, usually on high-symmetry path

        Returns
        -------
        H: ndarray with shape (M, M)
            H[α,β] gives hopping parameter from orbital β to orbital α

        """
        H = numpy.zeros((self.M,self.M), dtype=complex)

        for α, hops in enumerate(self.t):
            for hop in hops:
                β = hop[0]                  # THE SOURCE ORBITAL
                δ = numpy.array(hop[1:])    # THE SOURCE UNIT CELL
                t = hops[hop]               # THE ACTUAL HOPPING PARAMETER

                Δ = self.a[β] + δ - self.a[α]               # SEPARATION
                H[α,β] += t * numpy.exp( 2*pi*1j * b @ Δ )  # MATRIX TERM

        return H

    def E0(self, b):
        """ Analytical energy of each band at reciprocal coordinate b.

        Parameters
        ----------
        b: ndarray of length d
            reciprocal coordinate, usually on high-symmetry path

        Returns
        -------
        E: ndarray of length M
            E[l] gives the energy of level l

        """
        H = self.Hk(b)                      # RESTRICTED HAMILTONIAN
        E, U = numpy.linalg.eigh(H)         # CLASSICAL DIAGONALIZATION
        return E                            # SORTED EIGENVALUES

    def to_json(self):
        """ Construct serializable dict from Model object. """
        return {
            "id": self.id,
            "parameters": self.parameters,
        }

    def from_json(json):
        """ Construct Model object from serializable dict `json`. """
        return REGISTRY[json["id"]](**json["parameters"])

REGISTRY = {}
def register(cls):
    REGISTRY[cls.id] = cls
