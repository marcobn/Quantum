""" Represent simple one-dimensional systems. """

import numpy

from tightbinding.hamiltonian import Hamiltonian


class Polonium(Hamiltonian):
    """ 3d fcc-lattice of identical atoms, each with s, px, py, pz orbitals.

    orbitals: [ s, px, py, pz ]


    Additional Attributes
    ---------------------
    Es: float
        self-interaction energy on each s-orbital
    Ep: float
        self-interaction energy on each p-orbital
    Vss: float
        hopping energy between adjacent s orbitals
    Vsp: float
        hopping energy between adjacent s and p orbitals
    Vxx: float
        hopping energy between parallel p orbitals
    Vxy: float
        hopping energy between perpendicular p orbitals

    """
    id = "Polonium"
    D = 3
    M = 4

    def __init__(self,
        Es  = -14,
        Ep  =  0,
        Vss =  0,
        Vsp =  1,
        Vxx =  2,
    ):
        """ Constructor.

        Parameters
        ----------
        Es: float
            self-interaction energy on each s-orbital
        Ep: float
            self-interaction energy on each p-orbital
        Vss: float
            hopping energy between adjacent s orbitals
        Vsp: float
            hopping energy between adjacent s and p orbitals
        Vxx: float
            hopping energy between colinear p orbitals

        """
        self.Es = Es
        self.Ep = Ep
        self.Vss = Vss
        self.Vsp = Vsp
        self.Vxx = Vxx

        # ALIAS ORBITAL LABELS
        s, x, y, z = 0, 1, 2, 3

        # ALIAS SYMMETRY POINTS (normalized by 2π)
        Γ = [  0,   0,   0]
        R = [1/2, 1/2, 1/2]
        X = [  0, 1/2,   0]
        M = [1/2, 1/2,   0]

        super().__init__(
            R = numpy.array([       # Lattice vectors of system.
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]),
            r = numpy.array([       # Crystal coordinates of each orbital.
                [  0,   0,   0],
                [  0,   0,   0],
                [  0,   0,   0],
                [  0,   0,   0],
            ]),
            t = [                   # Real-space hopping parameters of system.
                {
                    (s,  0, 0, 0): Es,
                    (s,  1, 0, 0): Vss, # s-orbital interactions
                    (s, -1, 0, 0): Vss,
                    (s,  0, 1, 0): Vss,
                    (s,  0,-1, 0): Vss,
                    (s,  0, 0, 1): Vss,
                    (s,  0, 0,-1): Vss,
                    (x,  1, 0, 0): Vsp, # p-orbital interactions
                    (x, -1, 0, 0):-Vsp,
                    (y,  0, 1, 0): Vsp,
                    (y,  0,-1, 0):-Vsp,
                    (z,  0, 0, 1): Vsp,
                    (z,  0, 0,-1):-Vsp,
                },
                {
                    (x,  0, 0, 0): Ep,
                    (s,  1, 0, 0):-Vsp, # s-orbital interactions
                    (s, -1, 0, 0): Vsp,
                    (x,  1, 0, 0): Vxx, # p-orbital interactions
                    (x, -1, 0, 0): Vxx,
                },
                {
                    (y,  0, 0, 0): Ep,
                    (s,  0, 1, 0):-Vsp, # s-orbital interactions
                    (s,  0,-1, 0): Vsp,
                    (y,  0, 1, 0): Vxx, # p-orbital interactions
                    (y,  0,-1, 0): Vxx,
                },
                {
                    (z,  0, 0, 0): Ep,
                    (s,  0, 0, 1):-Vsp, # s-orbital interactions
                    (s,  0, 0,-1): Vsp,
                    (z,  0, 0, 1): Vxx, # p-orbital interactions
                    (z,  0, 0,-1): Vxx,
                },
            ],
            # path = 2*numpy.pi * numpy.array([ R, Γ, X, M, Γ ]),
            # path = 2*numpy.pi * numpy.array([ R, Γ, X ]),
            path = 2*numpy.pi * numpy.array([ X, M, Γ ]),
        )
    
    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        json = super().toJSON()
        json['Es']  = self.Es
        json['Ep']  = self.Ep
        json['Vss'] = self.Vss
        json['Vsp'] = self.Vsp
        json['Vxx'] = self.Vxx
        return json
