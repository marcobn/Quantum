import numpy
import qctb.lattice
from qctb.model.model import Model, register


class SimpleChain(Model):
    id = "SimpleChain"

    def __init__(self, V0=0, t0=1, a=1):
        """ A 1d chain of atoms, with nearest-neighbor interaction.

           -- α -- α -- α -- α -- α -- α
                      ⌊___⌋
                    Unit cell

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
        return qctb.lattice.Linear(a)

    def _construct_orbitals(self, V0, t0, a):
        return numpy.array([
            [0],
        ])

    def _construct_singles_(self, V0, t0, a):
        return [
            {  # β  δ
                (0, 0):  V0,        #  SELF INTERACTION
                (0,-1): -t0,        #  LEFT INTERACTION
                (0, 1): -t0,        # RIGHT INTERACTION
            },
        ]
register(SimpleChain)








class AlternatingChain(qctb.model.model.Model):
    id = "AlternatingChain"

    def __init__(self, V0=-.5, V1=.5, t0=1, a=1):
        """ A 1d chain of alternating atoms, with nearest-neighbor interaction.

           -- β -- α -- β -- α -- β -- α -- β -- α -- β -- α -- β -- α -- β
                                     ⌊________⌋
                                      Unit cell

        Parameters
        ----------
        V0: float
            self-interaction energy on atom α
        V1: float
            self-interaction energy on atom β
        t0: float
            hopping energy between α and β
        a: float
            lattice constant

        """
        super().__init__(
            self.id,
            {
                "V0":   V0,
                "V1":   V1,
                "t0":   t0,
                "a":    a,
            },
        )

    def _construct_lattice_(self, V0, V1, t0, a):
        return qctb.lattice.Linear(a)

    def _construct_orbitals(self, V0, V1, t0, a):
        return numpy.array([
            [0],
            [0.5],
        ])

    def _construct_singles_(self, V0, V1, t0, a):
        return [
            {  # β  δ
                (0, 0):  V0,        #  SELF INTERACTION
                (1,-1): -t0,        #  LEFT INTERACTION
                (1, 0): -t0,        # RIGHT INTERACTION
            },
            {
                (1, 0):  V1,        #  SELF INTERACTION
                (0, 0): -t0,        #  LEFT INTERACTION
                (0, 1): -t0,        # RIGHT INTERACTION
            }
        ]
register(AlternatingChain)









class HubbardChain(Model):
    id = "HubbardChain"

    def __init__(self, V0=0, t0=1, u=0, a=1):
        """ A 1d spin-chain of atoms with basic on-site correlations.

                      ⌈‾‾‾⌉
           -- α -- α -- α -- α -- α -- α
              |    |    |    |    |    |
           -- β -- β -- β -- β -- β -- β
                      ⌊___⌋
                    Unit cell

        Parameters
        ----------
        V0: float
            self-interaction energy on each atom
        t0: float
            hopping energy between adjacent atoms
        u: float
            on-site correlation energy
        a: float
            lattice constant

        """
        super().__init__(
            self.id,
            {
                "V0":   V0,
                "t0":   t0,
                "u":    u,
                "a":    a,
            },
        )

    def _construct_lattice_(self, V0, t0, u, a):
        return qctb.lattice.Linear(a)

    def _construct_orbitals(self, V0, t0, u, a):
        return numpy.array([
            [0],
            [0],
        ])

    def _construct_singles_(self, V0, t0, u, a):
        return [
            {  # β  δ
                (0, 0):  V0,        #  SELF INTERACTION
                (0,-1): -t0,        #  LEFT INTERACTION
                (0, 1): -t0,        # RIGHT INTERACTION
            },
            {  # β  δ
                (1, 0):  V0,        #  SELF INTERACTION
                (1,-1): -t0,        #  LEFT INTERACTION
                (1, 1): -t0,        # RIGHT INTERACTION
            },
        ]

    def _construct_doubles_(self, V0, t0, u, a):
        return [
            {
                (1,0,  0,0,  1,0): u,
            },
            {
                (0,0,  1,0,  0,0): u,
            },
        ]

register(HubbardChain)
