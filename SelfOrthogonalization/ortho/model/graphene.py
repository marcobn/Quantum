import numpy

import ortho.lattice
from ortho.model.model import Model, register

class Graphene_s(ortho.model.model.Model):
    id = "Graphene_s"

    def __init__(self, V0=0, t0=1, a=1):
        """ A 2d honeycomb of identical atoms, each with one orbital.

                                α -- β        α -- β
                                      \      /
                          Unit Cell: [ α -- β ]
                                      /      \
                                α -- β        α -- β

        Parameters
        ----------
        V0: float
            self-interaction energy on each atom
        t0: float
            hopping energy between adjacent atoms
        a: float
            lattice constant

        """
        super().__init__(
            self.id,
            {
                "V0":   V0,
                "t0":   t0,
                "a":    a,
            },
        )

    def _construct_lattice_(self, V0, t0, a):
        return ortho.lattice.Hexagonal(a)

    def _construct_orbitals(self, V0, t0, a):
        return numpy.array([
            [  0,   0],
            [1/3, 1/3],
        ])

    def _construct_singles_(self, V0, t0, a):
        return [
            {  # β  δ
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
        ]

        return lattice, a, t
register(Graphene_s)
