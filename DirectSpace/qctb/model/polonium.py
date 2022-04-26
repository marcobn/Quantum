import numpy
import qctb.lattice
from qctb.model.model import Model, register


class Polonium_sp(Model):
    id = "Polonium_sp"

    def __init__(self, Es=0, Ep=0, Vss=0, Vsp=1, Vpσ=1, Vpπ=0, a=1):
        """ 3d simple-cubic lattice of atoms with s, px, py, pz orbitals.

                                     O
                                     |  O
                         Unit Cell:  | /
                               O --[ O ]-- O
                                    /|
                                  O  |
                                     O


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
        Vpσ: float
            hopping energy between parallel p orbitals
        Vpπ: float
            hopping energy between perpendicular p orbitals
        a: float
            lattice constant

        """
        super().__init__(
            self.id,
            {
                "Es"    : Es,
                "Ep"    : Ep,
                "Vss"   : Vss,
                "Vsp"   : Vsp,
                "Vpσ"   : Vpσ,
                "Vpπ"   : Vpπ,
                "a"     : a,
            },
        )

    def _construct_lattice_(self, Es, Ep, Vss, Vsp, Vpσ, Vpπ, a):
        return qctb.lattice.SimpleCubic(a)

    def _construct_orbitals(self, Es, Ep, Vss, Vsp, Vpσ, Vpπ, a):
        return numpy.array([
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
        ])

    def _construct_singles_(self, Es, Ep, Vss, Vsp, Vpσ, Vpπ, a):
        s, x, y, z = 0, 1, 2, 3         # ALIAS ORBITAL INDICES

        return [
            {  # β   δ
                (s,  0, 0, 0): Es,
                (s,  1, 0, 0): Vss, # s-orbital interactions
                (s, -1, 0, 0): Vss,
                (s,  0, 1, 0): Vss,
                (s,  0,-1, 0): Vss,
                (s,  0, 0, 1): Vss,
                (s,  0, 0,-1): Vss,
                (x,  1, 0, 0):-Vsp, # p-orbital interactions
                (x, -1, 0, 0): Vsp,
                (y,  0, 1, 0):-Vsp,
                (y,  0,-1, 0): Vsp,
                (z,  0, 0, 1):-Vsp,
                (z,  0, 0,-1): Vsp,
            },
            {
                (x,  0, 0, 0): Ep,
                (s,  1, 0, 0): Vsp, # s-orbital interactions
                (s, -1, 0, 0):-Vsp,
                (x,  1, 0, 0): Vpσ, # colinear p-orbital interactions
                (x, -1, 0, 0): Vpσ,
                (x,  0, 1, 0): Vpπ, # parallel p-orbital interactions
                (x,  0,-1, 0): Vpπ,
                (x,  0, 0, 1): Vpπ,
                (x,  0, 0,-1): Vpπ,
            },
            {
                (y,  0, 0, 0): Ep,
                (s,  0, 1, 0): Vsp, # s-orbital interactions
                (s,  0,-1, 0):-Vsp,
                (y,  0, 1, 0): Vpσ, # colinear p-orbital interactions
                (y,  0,-1, 0): Vpσ,
                (y,  1, 0, 0): Vpπ, # parallel p-orbital interactions
                (y, -1, 0, 0): Vpπ,
                (y,  0, 0, 1): Vpπ,
                (y,  0, 0,-1): Vpπ,
            },
            {
                (z,  0, 0, 0): Ep,
                (s,  0, 0, 1): Vsp, # s-orbital interactions
                (s,  0, 0,-1):-Vsp,
                (z,  0, 0, 1): Vpσ, # colinear p-orbital interactions
                (z,  0, 0,-1): Vpσ,
                (z,  1, 0, 0): Vpπ, # parallel p-orbital interactions
                (z, -1, 0, 0): Vpπ,
                (z,  0, 1, 0): Vpπ,
                (z,  0,-1, 0): Vpπ,
            },
        ]

        return lattice, a, t
register(Polonium_sp)
