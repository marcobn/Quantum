""" Encapsulate crystallographic information common to point groups. """

import numpy
from numpy import array, sqrt, pi

class Lattice:
    def __init__(self, R, SYM=None):
        """ Encapsulate lattice geometry.

        Parameters
        ----------
        R: square ndarray
            lattice vectors in Cartesian vectors
        SYM: dict[str]: ndarray
            high-symmetry k-points as reciprocal coordinates

        """
        self.SYM = SYM
        self.R = R
        self.d = len(R)
        self.iR = numpy.linalg.inv(R)

    def vertices(self, sym):
        """ Construct vertices of a path traversing the First Brillouin Zone.

        Parameters
        ----------
        sym: list[str]
            labels of each high-symmetry vertex in the desired path

        Returns
        -------
        x: ndarray[float] with length len(sym)
            plot-points in [0,1] for each sym point
        b: ndarray[float] with shape ( len(sym), d )
            reciprocal coordinates for each sym point
        """
        b = array([self.SYM[label] for label in sym])
        norms = numpy.linalg.norm( numpy.diff(b, axis=0), axis=1 )
        x = numpy.cumsum(norms) / numpy.sum(norms)
        x = numpy.insert(x,0,0)

        return x, b

    def onpath(self, sym, b):
        """ Find where b appears along path traversing the First Brillouin Zone.

        Parameters
        ----------
        sym: list[str]
            labels of each high-symmetry vertex in the desired path
        b: list[float] with length d
            reciprocal coordinate to locate

        Returns
        -------
        x: list[float]
            x∈[0,1] where b appears in path defined by sym

        """
        xv, bv = self.vertices(sym)
        x = []
        for i in range(len(nodes)-1):
            A, B = bv[i], bv[i+1]                   # VERTICES OF ACTIVE SEGMENT

            # FIND FIRST INDEX OF PATH THAT CHANGES
            Δ = B - A                               # DISPLACEMENT VECTOR
            ix = 0
            while (abs(Δ[ix]) < numpy.finfo(float).eps) and (ix < self.d):
                ix += 1                             # ADVANCE WHILE 0
            if ix == self.d: continue               # Δ = 0, SKIP THIS SEGMENT

            # CHECK IF b IS CONSISTENT WITH SEGMENT
            λ = (b[ix] - A[ix]) / Δ[ix]             # GUESS SEGMENT FRACTION
            b_ = λ*Δ + A                            # CALCULATE SEGMENT POINT
            if numpy.allclose(b, b_):               # CHECK FOR MATCH
                x.append(λ*(xv[i+1]-xv[i]) + xv[i]) # CALCULATE PATH COORDINATE

        return x

    def traverse(self, sym, N=100):
        """ Generate a path traversing the First Brillouin Zone.

        Parameters
        ----------
        sym: list[str]
            labels of each high-symmetry vertex in the desired path
        N: int
            number of points for each edge in path

        Returns
        -------
        x: ndarray[float] with length len(sym)
            path coordinate in [0,1] for each point along trajectory
        b: ndarray[float] with shape ( len(sym), d )
            reciprocal coordinates for each point along trajectory

        """
        xv, bv = self.vertices(sym)
        x = numpy.zeros( N*(len(sym)-1)+1 )
        b = numpy.zeros( (N*(len(sym)-1)+1, self.d) )
        linspace = numpy.linspace(0,1,N,endpoint=False)
        for i in range(len(xv)-1):
            for j, λ in enumerate(linspace):        # SET SEGMENT FRACTION
                b[N*i+j] = λ*(bv[i+1]-bv[i]) + bv[i]# CALCULATE SEGMENT
                x[N*i+j] = λ*(xv[i+1]-xv[i]) + xv[i]# CALCULATE PATH COORDINATE
        x[-1] = xv[-1]                              # ADD IN LAST VERTEX
        b[-1] = bv[-1]

        return x, b

    def traverse_discrete(self, sym, N):
        """ Generate a discrete path traversing the First Brillouin Zone.

        Parameters
        ----------
        sym: list[str]
            labels of each high-symmetry vertex in the desired path
        N: list[int] with length d
            resolution in each dimension

        Returns
        -------
        x: ndarray[float]
            path coordinate in [0,1] for each discrete point along trajectory
        p: ndarray[int] with shape ( len(x), d )
            reciprocal coordinates b=p/N are points on the specified path

        """
        xv, bv = self.vertices(sym)
        x, p = [], []
        for i in range(len(xv)-1):
            A, B = bv[i], bv[i+1]                   # VERTICES OF ACTIVE SEGMENT

            # FIND FIRST INDEX OF PATH THAT CHANGES
            Δ = B - A                               # DISPLACEMENT VECTOR
            ix = 0
            while (abs(Δ[ix]) < numpy.finfo(float).eps) and (ix < self.d):
                ix += 1                             # ADVANCE WHILE 0
            if ix == self.d: continue               # Δ = 0, SKIP THIS SEGMENT

            # REGISTER ALL INTEGER COORDINATES
            AN, BN, ΔN = A*N, B*N, Δ*N              # INFLATE COORDINATES BY N
            for pix in _intrange_( AN[ix], BN[ix] ):
                λ  = (pix - AN[ix]) / ΔN[ix]        # GUESS SEGMENT FRACTION
                bN = λ*ΔN + AN                      # INFLATE SEGMENT COORDINATE
                if numpy.allclose( bN, bN.round() ):    # CHECK FOR INTEGER-NESS
                    x.append(λ*(xv[i+1]-xv[i]) + xv[i])
                    p.append( bN.round().astype(int) )

        return array(x), array(p)

    def a2r(self, a):
        """ Convert crystal coordinate ν into Cartestian vector. """
        return self.iR @ a

    def r2a(self, r):
        """ Convert Cartesian coordinate r into crystal coordinate. """
        return self.R  @ r

    def b2k(self, b):
        """ Convert reciprocal coordinate b into Brillouin vector. """
        return self.iR @ b * (2*pi)

    def k2b(self, k):
        """ Convert Brillouin vector k into reciprocal coordinate. """
        return self.R  @ k / (2*pi)

    def to_json(self):
        """ Construct serializable dict `json` from Lattice object. """
        return {
            "R": self.R.tolist(),
            "SYM": {label: list(self.SYM[label]) for label in self.SYM},
        }

    def from_json(json):
        """ Construct Lattice object from serializable dict `json`. """
        return Lattice(
            array(json["R"]),
            {label: array(json["SYM"][label]) for label in json["SYM"]},
        )


""" Common lattice templates.

Lattice vectors and high-symmetry points taken from:
    http://lamp.tu-graz.ac.at/~hadley/ss1/bzones/
"""

def Linear(a=1):
    return Lattice(
        a * array([[1]]),
        {
            "Γ": array([0]),
            "π": array([.5]),
        },
    )


def Hexagonal(a=1):
    return Lattice(
        a * array([
            [1, 0],
            [1/2, sqrt(3)/2],
        ]),
        {
            "Γ": array([0, 0]),
            "K": array([2/3, 1/3]),
            "M": array([1/2, 0]),
        },
    )

def FaceCenteredCubic(a=1):
    return Lattice(
        a * array([
            [1/2, 0, 1/2],
            [1/2, 1/2, 0],
            [0, 1/2, 1/2],
        ]),
        {
            "Γ": array([0, 0, 0]),
            "X": array([0, 1/2, 1/2]),
            "L": array([1/2, 1/2, 1/2]),
            "W": array([1/4, 3/4, 1/2]),
            "U": array([1/4, 5/8, 5/8]),
            "K": array([3/8, 3/4, 3/8]),
        },
    )

def SimpleCubic(a=1):
    return Lattice(
        a * array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        {
            "Γ": array([0, 0, 0]),
            "R": array([1/2, 1/2, 1/2]),
            "X": array([0, 1/2, 0]),
            "M": array([1/2, 1/2, 0]),
        },
    )










def _intrange_(a, b):
    """ Generate integers from a to b. """
    if a < b:
        n = int(numpy.ceil(a))
        while n < b:
            yield n
            n += 1
    else:
        n = int(a)
        while n > b:
            yield n
            n -= 1
