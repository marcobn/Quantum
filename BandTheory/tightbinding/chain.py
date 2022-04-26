""" Represent simple one-dimensional systems. """

import numpy

from tightbinding.hamiltonian import Hamiltonian

class Chain(Hamiltonian):
    """ A 1d chain of uniform atoms, with nearest-neighbor interaction.

       -- α -- α -- α -- α -- α -- α
                  ⌊___⌋
                Unit cell

    Additional Attributes
    ---------------------
    V0: float
        self-interaction energy on each atom
    t0: float
        hopping energy between adjacent atoms

    """
    id = "Chain"
    D = 1
    M = 1

    def __init__(self, V0=0, t0=1):
        """ Constructor.

        Parameters
        ----------
        V0: float
            self-interaction energy on each atom
        t0: float
            hopping energy between α and β

        """
        super().__init__(
            R = numpy.array([[1]]), # Lattice vectors of system.
            r = numpy.array([[0]]), # Crystal coordinates of each orbital.
            t = [                   # Real-space hopping parameters of system.
                {  # α  Δ
                    (0, 0): V0,        #  SELF INTERACTION
                    (0,-1):-t0,        #  LEFT INTERACTION
                    (0, 1):-t0,        # RIGHT INTERACTION
                }
            ],
            path = 2*numpy.pi * numpy.array([[0],[1/2],[1]])
        )
        self.V0 = V0
        self.t0 = t0

    def Hk(self, ka):
        """ Reciprocal-space hopping parameters of system at a specific kpt. """
        ka = ka[0]  # UNPACK SINGLE-ITEM LIST
        return numpy.array([
            [ self.V0 - self.t0 * 2*numpy.cos(ka) ],
        ])

    def E0(self, ka):
        """ Analytical energy of each band at a specific kpt. """
        ka = ka[0]  # UNPACK SINGLE-ITEM LIST
        return numpy.array([
            self.V0 - self.t0 * 2*numpy.cos(ka),
        ])

    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        json = super().toJSON()
        json['V0']  = self.V0
        json['t0']  = self.t0
        return json

class AlternatingChain(Hamiltonian):
    """ A 1d chain of alternating atoms, with nearest-neighbor interaction.

       -- β -- α -- β -- α -- β -- α -- β -- α -- β -- α -- β -- α -- β
                                 ⌊________⌋
                                  Unit cell

    Additional Attributes
    ---------------------
    V0: float
        self-interaction energy on atom α
    V1: float
        self-interaction energy on atom β
    t0: float
        hopping energy between α and β

    """
    id = "Alternating Chain"
    D = 1
    M = 2

    def __init__(self, V0=-.5, V1=.5, t0=1):
        """ Constructor.

        Parameters
        ----------
        V0: float
            self-interaction energy on atom α
        V1: float
            self-interaction energy on atom β
        t0: float
            hopping energy between α and β

        """
        super().__init__(
            R = numpy.array([[1]]), # Lattice vectors of system.
            r = numpy.array([       # Crystal coordinates of each orbital.
                [0],
                [0.5],
            ]),
            t = [                   # Real-space hopping parameters of system.
                {  # α  Δ
                    (0, 0):  V0,        #  SELF INTERACTION
                    (1,-1): -t0,        #  LEFT INTERACTION
                    (1, 0): -t0,        # RIGHT INTERACTION
                },
                {
                    (1, 0):  V1,        #  SELF INTERACTION
                    (0, 0): -t0,        #  LEFT INTERACTION
                    (0, 1): -t0,        # RIGHT INTERACTION
                }
            ],
            path = 2*numpy.pi * numpy.array([[0],[1/2],[1]])
        )
        self.V0 = V0
        self.V1 = V1
        self.t0 = t0

    def Hk(self, ka):
        """ Reciprocal-space hopping parameters of system at a specific kpt. """
        ka = ka[0]  # UNPACK SINGLE-ITEM LIST
        return numpy.array([
            [                     self.V0, -self.t0 * 2*numpy.cos(ka/2) ],
            [ -self.t0 * 2*numpy.cos(ka/2),                     self.V1 ],
        ])

    def E0(self, ka):
        """ Analytical energy of each band at a specific kpt. """
        ka = ka[0]  # UNPACK SINGLE-ITEM LIST
        Eµ = (self.V1 + self.V0)/2
        Er = (self.V1 - self.V0)/2
        f = 2*numpy.cos(ka/2)
        return numpy.array([
            Eµ - numpy.sqrt( Er**2 + (self.t0*f)**2 ),
            Eµ + numpy.sqrt( Er**2 + (self.t0*f)**2 ),
        ])

    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        json = super().toJSON()
        json['V0']  = self.V0
        json['V1']  = self.V1
        json['t0']  = self.t0
        return json
