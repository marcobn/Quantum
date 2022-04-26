""" Represent simple one-dimensional systems. """

import numpy

from tightbinding.hamiltonian import Hamiltonian


class Silicon(Hamiltonian):
    """ 3d fcc-lattice of identical atoms, each with s, px, py, pz orbitals.

    orbitals: [ s1, px1, py1, pz1,
                s2, px2, py2, pz2 ]


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
    id = "Silicon"
    D = 3
    M = 8

    def __init__(self,
        Es  = -4.03,
        Ep  =  3.17,
        Vss = -8.13,
        Vsp =  5.88,
        Vxx =  3.17,
        Vxy =  7.51,
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
            hopping energy between parallel p orbitals
        Vxy: float
            hopping energy between perpendicular p orbitals

        """
        self.Es = Es
        self.Ep = Ep
        self.Vss = Vss
        self.Vsp = Vsp
        self.Vxx = Vxx
        self.Vxy = Vxy

        # ALIAS ORBITAL LABELS
        s1, x1, y1, z1 = 0, 1, 2, 3
        s2, x2, y2, z2 = 4, 5, 6, 7

        # Corrections
        Vss = Vss/4
        Vsp = Vsp/4
        Vxx = Vxx/4
        Vxy = Vxy/4

        # ALIAS SYMMETRY POINTS (normalized by 2π)
        Γ = [  0,   0,   0]
        L = [1/2, 1/2, 1/2]
        K = [3/4, 3/4,   0]
        X = [  1,   0,   0]
        Y = [  0,   1,   0]
        Z = [  0,   0,   1]
        W = [  1, 1/2,   0]
        U = [  1, 1/4, 1/4]

        super().__init__(
            R = numpy.array([       # Lattice vectors of system.
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ]) / 2,
            r = numpy.array([       # Crystal coordinates of each orbital.
                [  0,   0,   0],
                [  0,   0,   0],
                [  0,   0,   0],
                [  0,   0,   0],
                [1/4, 1/4, 1/4],
                [1/4, 1/4, 1/4],
                [1/4, 1/4, 1/4],
                [1/4, 1/4, 1/4],
            ]),
            t = [                   # Real-space hopping parameters of system.
                {  # s1
                    (s1, 0, 0, 0):  Es,         # SELF INTERACTION
                    (s2, 0, 0, 0): Vss,         # SAME CELL
                    (x2, 0, 0, 0): Vsp,
                    (y2, 0, 0, 0): Vsp,
                    (z2, 0, 0, 0): Vsp,
                    (s2,-1, 0, 0): Vss,         # ADJACENT CELLS
                    (x2,-1, 0, 0): Vsp,
                    (y2,-1, 0, 0):-Vsp,
                    (z2,-1, 0, 0):-Vsp,
                    (s2, 0,-1, 0): Vss,         # 
                    (x2, 0,-1, 0):-Vsp,
                    (y2, 0,-1, 0): Vsp,
                    (z2, 0,-1, 0):-Vsp,
                    (s2, 0, 0,-1): Vss,         # 
                    (x2, 0, 0,-1):-Vsp,
                    (y2, 0, 0,-1):-Vsp,
                    (z2, 0, 0,-1): Vsp,
                },
                {  # x1
                    (x1, 0, 0, 0):  Ep,         # SELF INTERACTION
                    (s2, 0, 0, 0):-Vsp,         # SAME CELL
                    (x2, 0, 0, 0): Vxx,
                    (y2, 0, 0, 0): Vxy,
                    (z2, 0, 0, 0): Vxy,
                    (s2,-1, 0, 0):-Vsp,         # ADJACENT CELLS
                    (x2,-1, 0, 0): Vxx,
                    (y2,-1, 0, 0):-Vxy,
                    (z2,-1, 0, 0): Vxy,
                    (s2, 0,-1, 0): Vsp,         # 
                    (x2, 0,-1, 0): Vxx,
                    (y2, 0,-1, 0):-Vxy,
                    (z2, 0,-1, 0):-Vxy,
                    (s2, 0, 0,-1): Vsp,         # 
                    (x2, 0, 0,-1): Vxx,
                    (y2, 0, 0,-1): Vxy,
                    (z2, 0, 0,-1):-Vxy,
                },
                {  # y1
                    (y1, 0, 0, 0):  Ep,         # SELF INTERACTION
                    (s2, 0, 0, 0):-Vsp,         # SAME CELL
                    (x2, 0, 0, 0): Vxy,
                    (y2, 0, 0, 0): Vxx,
                    (z2, 0, 0, 0): Vxy,
                    (s2,-1, 0, 0): Vsp,         # ADJACENT CELLS
                    (x2,-1, 0, 0):-Vxy,
                    (y2,-1, 0, 0): Vxx,
                    (z2,-1, 0, 0): Vxy,
                    (s2, 0,-1, 0):-Vsp,         # 
                    (x2, 0,-1, 0):-Vxy,
                    (y2, 0,-1, 0): Vxx,
                    (z2, 0,-1, 0):-Vxy,
                    (s2, 0, 0,-1): Vsp,         # 
                    (x2, 0, 0,-1): Vxy,
                    (y2, 0, 0,-1): Vxx,
                    (z2, 0, 0,-1):-Vxy,
                },
                {  # z1
                    (z1, 0, 0, 0):  Ep,         # SELF INTERACTION
                    (s2, 0, 0, 0):-Vsp,         # SAME CELL
                    (x2, 0, 0, 0): Vxy,
                    (y2, 0, 0, 0): Vxy,
                    (z2, 0, 0, 0): Vxx,
                    (s2,-1, 0, 0): Vsp,         # ADJACENT CELLS
                    (x2,-1, 0, 0): Vxy,
                    (y2,-1, 0, 0): Vxy,
                    (z2,-1, 0, 0): Vxx,
                    (s2, 0,-1, 0): Vsp,         # 
                    (x2, 0,-1, 0):-Vxy,
                    (y2, 0,-1, 0):-Vxy,
                    (z2, 0,-1, 0): Vxx,
                    (s2, 0, 0,-1):-Vsp,         # 
                    (x2, 0, 0,-1):-Vxy,
                    (y2, 0, 0,-1):-Vxy,
                    (z2, 0, 0,-1): Vxx,
                },
                {  # s2
                    (s2, 0, 0, 0):  Es,         # SELF INTERACTION
                    (s1, 0, 0, 0): Vss,         # SAME CELL
                    (x1, 0, 0, 0):-Vsp,
                    (y1, 0, 0, 0):-Vsp,
                    (z1, 0, 0, 0):-Vsp,
                    (s1,+1, 0, 0): Vss,         # ADJACENT CELLS
                    (x1,+1, 0, 0):-Vsp,
                    (y1,+1, 0, 0): Vsp,
                    (z1,+1, 0, 0): Vsp,
                    (s1, 0,+1, 0): Vss,         # 
                    (x1, 0,+1, 0): Vsp,
                    (y1, 0,+1, 0):-Vsp,
                    (z1, 0,+1, 0): Vsp,
                    (s1, 0, 0,+1): Vss,         # 
                    (y1, 0, 0,+1): Vsp,
                    (x1, 0, 0,+1): Vsp,
                    (z1, 0, 0,+1):-Vsp,
                },
                {  # x2
                    (x2, 0, 0, 0):  Ep,         # SELF INTERACTION
                    (s1, 0, 0, 0): Vsp,         # SAME CELL
                    (x1, 0, 0, 0): Vxx,
                    (y1, 0, 0, 0): Vxy,
                    (z1, 0, 0, 0): Vxy,
                    (s1,+1, 0, 0): Vsp,         # ADJACENT CELLS
                    (x1,+1, 0, 0): Vxx,
                    (y1,+1, 0, 0):-Vxy,
                    (z1,+1, 0, 0): Vxy,
                    (s1, 0,+1, 0):-Vsp,         # 
                    (x1, 0,+1, 0): Vxx,
                    (y1, 0,+1, 0):-Vxy,
                    (z1, 0,+1, 0):-Vxy,
                    (s1, 0, 0,+1):-Vsp,         # 
                    (x1, 0, 0,+1): Vxx,
                    (y1, 0, 0,+1): Vxy,
                    (z1, 0, 0,+1):-Vxy,
                },
                {  # y2
                    (y2, 0, 0, 0):  Ep,         # SELF INTERACTION
                    (s1, 0, 0, 0): Vsp,         # SAME CELL
                    (x1, 0, 0, 0): Vxy,
                    (y1, 0, 0, 0): Vxx,
                    (z1, 0, 0, 0): Vxy,
                    (s1,+1, 0, 0):-Vsp,         # ADJACENT CELLS
                    (x1,+1, 0, 0):-Vxy,
                    (y1,+1, 0, 0): Vxx,
                    (z1,+1, 0, 0): Vxy,
                    (s1, 0,+1, 0): Vsp,         # 
                    (x1, 0,+1, 0):-Vxy,
                    (y1, 0,+1, 0): Vxx,
                    (z1, 0,+1, 0):-Vxy,
                    (s1, 0, 0,+1):-Vsp,         # 
                    (x1, 0, 0,+1): Vxy,
                    (y1, 0, 0,+1): Vxx,
                    (z1, 0, 0,+1):-Vxy,
                },
                {  # z2
                    (z2, 0, 0, 0):  Ep,         # SELF INTERACTION
                    (s1, 0, 0, 0): Vsp,         # SAME CELL
                    (x1, 0, 0, 0): Vxy,
                    (y1, 0, 0, 0): Vxy,
                    (z1, 0, 0, 0): Vxx,
                    (s1,+1, 0, 0):-Vsp,         # ADJACENT CELLS
                    (x1,+1, 0, 0): Vxy,
                    (y1,+1, 0, 0): Vxy,
                    (z1,+1, 0, 0): Vxx,
                    (s1, 0,+1, 0):-Vsp,         # 
                    (x1, 0,+1, 0):-Vxy,
                    (y1, 0,+1, 0):-Vxy,
                    (z1, 0,+1, 0): Vxx,
                    (s1, 0, 0,+1): Vsp,         # 
                    (x1, 0, 0,+1):-Vxy,
                    (y1, 0, 0,+1):-Vxy,
                    (z1, 0, 0,+1): Vxx,
                },
            ],
            path = 2*numpy.pi * numpy.array([ Γ, K, L, Γ, X, U, L, W, X ]),
            # path = 2*numpy.pi * numpy.array([ L, Γ, X, W, K, Γ ]),
            # path = 2*numpy.pi * numpy.array([ L, Γ, X]),
        )
    
    def Hk(self, k):
        import numpy as np
        Es, Ep, Vss, Vsp, Vxx, Vxy = (
            self.Es,
            self.Ep,
            self.Vss,
            self.Vsp,
            self.Vxx,
            self.Vxy,
        )
        
        d1 = np.array([ 1, 1, 1])/4
        d2 = np.array([ 1,-1,-1])/4
        d3 = np.array([-1, 1,-1])/4
        d4 = np.array([-1,-1, 1])/4

        g1 = (np.exp(1j*k@d1)+np.exp(1j*k@d2)+np.exp(1j*k@d3)+np.exp(1j*k@d4))/4
        g2 = (np.exp(1j*k@d1)+np.exp(1j*k@d2)-np.exp(1j*k@d3)-np.exp(1j*k@d4))/4
        g3 = (np.exp(1j*k@d1)-np.exp(1j*k@d2)+np.exp(1j*k@d3)-np.exp(1j*k@d4))/4
        g4 = (np.exp(1j*k@d1)-np.exp(1j*k@d2)-np.exp(1j*k@d3)+np.exp(1j*k@d4))/4

        g1t = g1.conjugate()
        g2t = g2.conjugate()
        g3t = g3.conjugate()
        g4t = g4.conjugate()

        return np.array([
            [    Es,      0,      0,      0, Vss*g1t, Vsp*g2t,Vsp*g3t,Vsp*g4t],
            [     0,     Ep,      0,      0,-Vsp*g2t, Vxx*g1t,Vxy*g4t,Vxy*g2t],
            [     0,      0,     Ep,      0,-Vsp*g3t, Vxy*g4t,Vxx*g1t,Vxy*g2t],
            [     0,      0,      0,     Ep,-Vsp*g4t, Vxy*g2t,Vxy*g2t,Vxx*g1t],
            [Vss*g1,-Vsp*g2,-Vsp*g3,-Vsp*g4,      Es,       0,      0,      0],
            [Vsp*g2, Vxx*g1, Vxy*g4, Vxy*g2,       0,      Ep,      0,      0],
            [Vsp*g3, Vxy*g4, Vxx*g1, Vxy*g2,       0,       0,     Ep,      0],
            [Vsp*g4, Vxy*g2, Vxy*g2, Vxx*g1,       0,       0,      0,     Ep],
        ])

    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        json = super().toJSON()
        json['Es']  = self.Es
        json['Ep']  = self.Ep
        json['Vss'] = self.Vss
        json['Vsp'] = self.Vsp
        json['Vxx'] = self.Vxx
        json['Vxy'] = self.Vxy
        return json
