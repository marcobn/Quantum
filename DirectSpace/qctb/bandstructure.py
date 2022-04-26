import numpy
import matplotlib.pyplot

import qctb.circuit
from qctb.hamiltonian.hamiltonian import Hamiltonian
from qctb.computer.computer import Computer
from qctb.optimization.optimization import Optimizer
from qctb.hamiltonian.singlebody import eigenangles

class BandStructure:
    """
    TODO
    - to/from_json protocol
    - Note that state variables can be saved, ensuring easy resume-computations.
    """
    def __init__(self, H, sym,
        computer,
        optimizer=None,
        N=None,
        discrete=False
    ):
        # DEFAULT VALUE: N CAN'T BE None IF PATH IS DISCRETE
        if discrete and N is None:  N = H.N

        self.H = H
        self.M = H.M
        self.d = H.d
        self.sym = sym
        self.computer = computer
        self.optimizer = optimizer
        self.N = N
        self.discrete = discrete

        # GENERATE PATH
        self.xv, self.bv = H.model.lattice.vertices(sym)
        if discrete:
            self.x, self.p = H.model.lattice.traverse_discrete(sym, N)
            self.b = self.p / N
        else:
            self.x, self.b = H.model.lattice.traverse(sym, N)
        self.num_points = len(self.x)

        # STATE VARIABLES
        self.E      = numpy.zeros((self.num_points, self.M))
        self.nfev   = numpy.zeros((self.num_points, self.M), dtype=int)
        self.angles = [None]*self.num_points # LIST OF DICTS FOR EACH PATH INDEX
        self.i = 0                           # NEXT UNKNOWN PATH INDEX
        self.l = 0                           # NEXT UNKNOWN ENERGY LEVEL

    def solve(self):
        while self.i < self.num_points:
            b = self.b[self.i]
            l = self.l

            # INITIALIZE ANGLES
            if self.angles[self.i] is None:
                if self.optimizer is None:
                    angles = eigenangles(self.H.model, b)
                elif self.i > 0 and self.angles[self.i-1] is not None:
                    angles = self.angles[self.i-1].copy()
                else:
                    angles = {
                        qctb.circuit.angle(axis,l,q): 0
                            for axis in range(2)
                            for l in range(self.M-1)
                            for q in range(l, self.M-1)
                    }

            if self.optimizer is None:
                # USE CLASSICALLY OBTAINED EIGEN-ANGLES
                E = self.H.energy(b, l, angles, self.computer)
                nfev = 1
            else:
                # DEFINE OPTIMIZATION-COMPATIBLE COST FUNCTION
                def costfn(x):
                    # UPDATE angles MAP
                    i = 0
                    for q in range(l, self.M-1):
                        for axis in range(2):
                            angles[qctb.circuit.angle(axis,l,q)] = x[i]
                            i += 1
                    # DELEGATE TO HAMILTONIAN ENERGY FUNCTION
                    return self.H.energy(b, l, angles, self.computer)

                # PERFORM THE OPTIMIZATION
                optimization = self.optimizer.optimize(
                    costfn,
                    x0=[
                        angles[qctb.circuit.angle(axis,l,q)]
                            for q in range(l, self.M-1)
                            for axis in range(2)
                    ],
                )

                # EXTRACT OPTIMIZED ENERGY AND PARAMETERS
                E    = optimization.E           # MINIMIZED ENERGY
                nfev = optimization.nfev        # NUMBER OF FUNCTION EVALUATIONS
                i    = 0                        # OPTIMIZED angles MAP
                for q in range(l, self.M-1):
                    for axis in range(2):
                        angles[qctb.circuit.angle(axis,l,q)] = optimization.x[i]
                        i += 1

            # UPDATE STATE VARIABLES
            self.   E[self.i, self.l] = E
            self.nfev[self.i, self.l] = nfev
            self.angles[self.i] = angles
            self.l += 1
            if self.l == self.M:
                self.i += 1
                self.l = 0

            yield E, angles, nfev

    def plot(self, ax=None):
        # SET UP FIGURE WITH ANALYTICAL CURVES
        if ax is None:

            # CALCULATE ANALYTICAL ENERGIES
            x0, b0 = self.H.model.lattice.traverse(self.sym)
            E0 = numpy.array([self.H.model.E0(b) for b in b0])

            # INITIALIZE FIGURE
            fig, ax = matplotlib.pyplot.subplots(
                dpi=100,
                figsize=[7,4.75],
            )

            # CALCULATE y-LIMITS
            Δ = 1
            ylim = [
                numpy.min(E0) - Δ,
                numpy.max(E0) + Δ
            ]

            # CONFIGURE y-AXIS
            ax.set_ylim(ylim)
            ax.set_ylabel("Energy (eV)")
            for lim in ylim:
                ax.axhline(lim, color='black', lw=1, clip_on=False)

            # CONFIGURE x-AXIS AND MARK HIGH-SYMMETRY POINTS
            ax.set_xlim([0,1])
            ax.set_xticks(self.xv)
            ax.set_xticklabels(self.sym)
            for x in self.xv:
                ax.axvline(x, color='black', lw=1, clip_on=False)

            # PLOT ANALYTICAL ENERGIES
            for l in range(E0.shape[1]):
                ax.plot(
                    x0, E0[:,l],
                    ls='-', lw=1.5, c='black',
                )
        else:
            fig = None

        # PLOT MEASURED ENERGIES
        for l in range(self.E.shape[1]):
            ax.plot(
                self.x, self.E[:,l],
                ls='', marker='x', lw=1.5, c='black',
            )

        return fig, ax

    def to_json(self):
        """ Construct serializable dict from BandStructure object. """
        # MANAGE CAREFULLY THE TYPE OF N
        N = self.N
        if isinstance(N, numpy.ndarray): N = N.tolist()

        # MANAGE CAREFULLY THE TYPE OF optimizer
        optimizer = self.optimizer
        if optimizer is not None:   optimizer = self.optimizer.to_json()

        return {
            # CONSTRUCTION PARAMETERS
            "H": self.H.to_json(),
            "sym": self.sym,
            "N": N,
            "discrete": self.discrete,
            "computer": self.computer.to_json(),
            "optimizer": optimizer,
            # REDUNDANT METADATA
            "num_points": self.num_points,
            # STATE VARIABLES
            "i": self.i,
            "l": self.l,
            "E": self.E.tolist(),
            "nfev": self.nfev.tolist(),
            "angles": self.angles,
        }


    def from_json(json):
        """ Construct BandStructure from serializable dict `json`. """
        # MANAGE CAREFULLY THE TYPE OF N
        N = json["N"]
        if isinstance(N, list): N = numpy.array(N)

        # MANAGE CAREFULLY THE TYPE OF optimizer
        optimizer = json["optimizer"]
        if optimizer is not None:   optimizer = Optimizer.from_json(optimizer)

        # CONSTRUCT OBJECT
        band = BandStructure(
            Hamiltonian.from_json(json["H"]),
            json["sym"],
            Computer.from_json(json["computer"]),
            optimizer = optimizer,
            N = N,
            discrete = json["discrete"],
        )

        # UPDATE STATE VARIABLES
        band.E      = numpy.array(json["E"])
        band.nfev   = numpy.array(json["E"])
        band.angles = json["angles"]
        band.i      = json["i"]
        band.l      = json["l"]

        return band
