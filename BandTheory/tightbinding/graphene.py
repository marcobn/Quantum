""" Represent simple one-dimensional systems. """

import numpy

from tightbinding.hamiltonian import Hamiltonian


class Graphene(Hamiltonian):
    """ A 2d honeycomb of identical atoms, each with one orbital.

                            α -- β        α -- β
                                  \      /
                      Unit Cell: [ α -- β ]
                                  /      \
                            α -- β        α -- β

    Additional Attributes
    ---------------------
    V0: float
        self-interaction energy on each atom
    t0: float
        hopping energy between adjacent atoms

    """
    id = "Graphene"
    D = 2
    M = 2

    def __init__(self, V0=0, t0=1):
        """ Constructor.

        Parameters
        ----------
        V0: float
            self-interaction energy on each atom
        t0: float
            hopping energy between adjacent atoms

        """
        # ALIAS SYMMETRY POINTS (normalized by 2π)
        Γ = [              0,   0]
        M = [1/numpy.sqrt(3),   0]
        K = [1/numpy.sqrt(3), 1/3]

        super().__init__(
            R = numpy.array([       # Lattice vectors of system.
                [numpy.sqrt(3), -1],
                [numpy.sqrt(3),  1],
            ]) / 2,
            r = numpy.array([       # Crystal coordinates of each orbital.
                [  0,   0],
                [1/3, 1/3],
            ]),
            t = [                   # Real-space hopping parameters of system.
                {  # α  Δ
                    (0, 0, 0):  V0,         # INTRA-CELL INTERACTIONS
                    (1, 0, 0): -t0,
                    (1,-1, 0): -t0,         # LEFT INTERACTIONS
                    (1, 0,-1): -t0,
                },
                {
                    (0, 0, 0): -t0,         # INTRA-CELL INTERACTIONS
                    (1, 0, 0):  V0,
                    (0, 1, 0): -t0,         # RIGHT INTERACTIONS
                    (0, 0, 1): -t0,
                }
            ],
            path = 2*numpy.pi * numpy.array([ Γ, M, K, Γ ]),
        )
        self.V0 = V0
        self.t0 = t0

    def Hk(self, ka):
        """ Reciprocal-space hopping parameters of system at a specific kpt. """
        kax, kay = ka   # UNPACK TWO COMPONENTS
        f = -self.t0 * (
            numpy.exp(-1j*kax/numpy.sqrt(3))
                +   2*numpy.exp(1j*kax/2/numpy.sqrt(3)) * numpy.cos(kay/2)
        )
        return numpy.array([
            [       self.V0,             f ],
            [ f.conjugate(),       self.V0 ],
        ])

    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        json = super().toJSON()
        json['V0']  = self.V0
        json['t0']  = self.t0
        return json
