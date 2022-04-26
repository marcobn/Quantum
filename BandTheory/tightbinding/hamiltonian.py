""" Represent specific physical systems in Hamiltonian objects. """

import numpy

class Hamiltonian:
    """ Encapsulate physics of a tight-binding system.

    Attributes
    ----------
    id: str
        unique identifier of the Hamiltonian, for serialization
    D: int
        dimension of the system
    M: int
        number of orbitals per unit cell
    R: D×D array
        Lattice vectors of system
        R[i] is the ith lattice vector in Cartesian coordinates
    r: M×D array
        Crystal coordinates of each orbital.
        r[α] is the crystal coordinate of orbital α

            NOTE: Cartesian coordinates can be gotten by (r @ R)

    t: list with len(M) of dict { tuple with len(1+D) of int : float }
        Real-space hopping parameters of system.
        t[β][(α,Δ)] gives hopping parameter from orbital β in unit cell
            to orbital α in adjacent unit cell specified by Δ=[i,j...]
            Each i,j,... indexes a different dimension and is -1, 0, or 1.

        Example: t[5][3,0,1] gives hopping parameter from orbital 5,
            to orbital 3 of the cell *above*.

        Constraints:
            t[β][α,0..0] = t[α][β,0..0]
            t[β][α,Δ] = t[α][β,-Δ]

    Subclasses will add additional attributes in their constructors.

    """
    id = ""
    D  = 0
    M  = 0
    # NOTE: Subclasses should overwrite these class attributes.

    def __init__(self, R, r, t, path):
        """ Default constructor.

        Parameters
        ----------
        R: D×D array
            Lattice vectors of system
            R[i] is the ith lattice vector in Cartesian coordinates
        r: M×D array
            Crystal coordinates of each orbital.
            r[α] is the crystal coordinate of orbital α

                NOTE: Cartesian coordinates can be gotten by (r @ R)

        t: list with len(M) of dict { tuple with len(1+D) of int : float }
            Real-space hopping parameters of system.
            Each dict maps (1+D)-tuple: float
            t[β][(α,Δ)] gives hopping parameter from orbital β in unit cell
                to orbital α in adjacent unit cell specified by Δ=[i,j...]
                Each i,j,... indexes a different dimension and is -1, 0, or 1.

            Example: t[5][3,0,1] gives hopping parameter from orbital 5,
                to orbital 3 of the cell *above*.

            Constraints:
                t[β][α,0..0] = t[α][β,0..0]
                t[β][α,Δ] = t[α][β,-Δ]
        path: ?xD array
            high-symmetry k-points to path through for a typical band structure
            each point normalized by 2π

        """
        self.R  = R
        self.iR = numpy.linalg.inv(R)
        self.r  = r
        self.t  = t
        self.path = path

    def Hk(self, ka):
        """ Reciprocal-space hopping parameters of system at a specific kpt.

        Subclasses may override this method to provide specific known values.
        Otherwise, calculates sum ∑_Δ t[β][α,Δ] exp(𝑖k·δ[Δ])

        Parameters
        ----------
        ka: ndarray of length D
            point in Brillouin zone

        Returns
        -------
        H: ndarray with shape (M, M)
            H[α,β] gives hopping parameter from orbital β to orbital α

        """
        H = numpy.zeros((self.M,self.M), dtype=complex)

        for β, hops in enumerate(self.t):
            for hop in hops:
                t = hops[hop]       # THE ACTUAL HOPPING PARAMETER
                α = hop[0]          # THE TARGET ORBITAL
                Δ = hop[1:]         # THE TARGET UNIT CELL

                # CALCULATE CARTESIAN SEPARATION VECTOR
                δ = (self.r[α] - self.r[β]) @ self.R
                for i in range(self.D):
                    δ += Δ[i] * self.R[i]

                # ADD TERM TO HAMILTONIAN
                H[α,β] += t * numpy.exp( 1j * ka @ δ )

        return H

    def E0(self, ka):
        """ Analytical energy of each band at a specific kpt.

        Subclasses may override this method to provide specific known values.
        Otherwise, diagonalizes Hk with numpy.

        Parameters
        ----------
        ka: ndarray length D
            point in Brillouin zone

        Returns
        -------
        E: ndarray of length M
            E[l] gives the energy of level l

        """
        H = self.Hk(ka)
        E, U = numpy.linalg.eigh(H)
        return E

    def kpt(self, pN):
        """ Convert quantized vector into kpt.

        A Cartesian lattice simply multiplies each component of pN by 2π.
        Non-Cartesian lattices will have linear combinations of pN,
            and must override this method.

        Parameters
        ----------
        pN: ndarray of length D
            discretized vector p/N representing kpt in finite lattice

        Returns
        -------
        ka: ndarray of length D
            actual kpt, in Cartesian basis

        """
        return 2*numpy.pi * self.iR @ pN




    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        return {
            "id":               self.id,
            "D":                self.D,
            "M":                self.M,
        }
        # NOTE: Subclasses should extend this method to append parameters.

    # NOTE: fromJSON is found in __init__.py of this package

    def template(self, kpt, l):
        """ Generate template string for individual energy evaluations.
        
        Tab-delimited entries for:
            - each attribute in H's JSON, sorted (data only, no key)
            - each element of kpt
            - each parameter (as sorted by ansatz)
            - energy level `l`, energy `E`, and scaling parameter `scale`

        Parameters
        ----------
        kpt: list of float
            the self.D coordinates of the kpt being explored
        l: int
            the energy level (0 -> ground state, etc.)

        Returns
        -------
        template: str
            template string suitable for formatting with `x0`, `x1`...
            and then again with `E` and `scale`
        
        """
        jsonified = self.toJSON()
        
        templatestrings = []
        templatestrings.extend(
            [str(jsonified[key]) for key in sorted(jsonified)]  # H PARAMETERS
            + [str(kpti) for kpti in kpt]                       # KPTS
            + ["{x"+str(i)+"}" for i in range(2*(self.M-1))]    # x PARAMETERS
            + ["{{E}}","{{scale}}", str(l)]     # ENERGY, ZNE SCALING, AND LEVEL
        )

        return "\t".join(templatestrings)
