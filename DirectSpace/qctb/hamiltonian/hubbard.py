import itertools

import numpy
import sympy
import cirq

import qctb.circuit
import qctb.binary
import qctb.pauli
from qctb.model.chain import SimpleChain
from qctb.hamiltonian.hamiltonian import Hamiltonian, register

# ALIAS INSERTION STRATEGIES
EARLIEST        = cirq.circuits.InsertStrategy.EARLIEST
NEW             = cirq.circuits.InsertStrategy.NEW
INLINE          = cirq.circuits.InsertStrategy.INLINE
NEW_THEN_INLINE = cirq.circuits.InsertStrategy.NEW_THEN_INLINE





##############################################################################
#                          SPECIALIZED CIRCUITS

class CompactSpinState(cirq.TwoQubitGate):
    # This one has just 1 ISWAPPowGate (and 2 CNOTs); it may be more compact.
    def __init__(self, η, μ, S):
        super().__init__()
        self.η = η
        self.μ = μ
        self.S = S

        try:                # CALCULATE ROTATION PARAMETERS
            from numpy import arcsin, arccos, sqrt, pi
            self.t  = arcsin( sqrt(S) ) * 2 / pi
            self.tN = arccos( -(1 - 2*η) * (1/S) )  / pi
            self.tM = arccos( -2*η*(1-2*μ)/(1-2*η) * (S/(1-S)) )  / pi
        except TypeError:   # PARAMETERS ARE SYMBOLIC, SO MAKE t SYMBOLS
            self.t  = sympy.Symbol( "t(η,μ,S)")
            self.tN = sympy.Symbol("tN(η,μ,S)")
            self.tM = sympy.Symbol("tM(η,μ,S)")

    def _decompose_(self, qubits):
        q0, q1 = qubits

        yield (     # ROTATE SPECIFIED AMOUNT INTO |11⟩ SPACE
            (cirq.Y**self.t).on(q0),
             cirq.CNOT.on(q0,q1),
        )

        yield (     # ROTATE CONSTRAINED AMOUNT INTO {|01⟩, |10⟩} SPACE
            (cirq.Y**self.tN).on(q0),
             cirq.X.on(q0),
             cirq.CNOT.on(q0,q1),
             cirq.X.on(q0),
        )

        yield (     # ROTATE CONSTRAINED AMOUNT BETWEEN {|01⟩, |10⟩} SPACE
            (cirq.ISWAP**self.tM).on(q0,q1),
        )

    def _resolve_parameters_(self, param_resolver, recursive):
        return SpinState(
            param_resolver.value_of(self.η, recursive),
            param_resolver.value_of(self.μ, recursive),
            param_resolver.value_of(self.S, recursive),
        )

    def _circuit_diagram_info_(self, args):
        return [f"V({self.η},{self.μ},{self.S})"]*self.num_qubits()

class SpinState(cirq.TwoQubitGate):
    def __init__(self, η, μ, ξ):
        super().__init__()
        self.η = η
        self.μ = μ
        self.ξ = ξ

        try:                # CALCULATE ROTATION PARAMETERS
            from numpy import arccos, sqrt
            from numpy import pi as π
            self.t11 = 2/π * arccos(sqrt( (2-η*μ*ξ) / (2) ))
            self.t10 = 2/π * arccos(sqrt( (2-η*μ) / (2-η*μ*ξ) ))
            self.t01 = 2/π * arccos(sqrt( (2-η*(2-μ*ξ)) / (2-η*μ) ))
        except TypeError:   # PARAMETERS ARE SYMBOLIC, SO MAKE t SYMBOLS
            self.t11 = sympy.Symbol("t11(η,μ,ξ)")
            self.t10 = sympy.Symbol("t10(η,μ,ξ)")
            self.t01 = sympy.Symbol("t01(η,μ,ξ)")

    def _decompose_(self, qubits):
        q0, q1 = qubits

        yield (     # ROTATE INTO |11⟩ SPACE
             cirq.X.on(q1),
            (cirq.ISWAP**self.t11).on(q0,q1),
             cirq.X.on(q1),
        )

        yield (     # ROTATE INTO |10⟩ SPACE
             cirq.X.on(q0),
             cirq.CNOT.on(q0,q1),
             cirq.X.on(q0),
            (cirq.ISWAP**self.t10).on(q0,q1),
             cirq.X.on(q0),
             cirq.CNOT.on(q0,q1),
             cirq.X.on(q0),
        )

        yield (     # ROTATE INTO |01⟩ SPACE
             cirq.X.on(q1),
             cirq.CNOT.on(q1,q0),
             cirq.X.on(q1),
            (cirq.ISWAP**self.t01).on(q0,q1),
             cirq.X.on(q1),
             cirq.CNOT.on(q1,q0),
             cirq.X.on(q1),
        )

    def _resolve_parameters_(self, param_resolver, recursive):
        return SpinState(
            param_resolver.value_of(self.η, recursive),
            param_resolver.value_of(self.μ, recursive),
            param_resolver.value_of(self.ξ, recursive),
        )

    def _circuit_diagram_info_(self, args):
        return [f"V({self.η},{self.μ},{self.ξ})"]*self.num_qubits()




##############################################################################
#                              HAMILTONIANS

class PrototypeHubbardHamiltonian(Hamiltonian):
    id = "PrototypeHubbard"

    def __init__(self, L, u):
        # FOLLOWING ESSLER, u = U/4t, t TAKEN AS UNIT ENERGY
        self.L = L
        self.n = qctb.binary.length(L-1)    # NUMBER OF QUBITS IN SITE REGISTER
        self.u = u
        model = SimpleChain()
        super().__init__(model, self.id, {
            "L":    L,
            "u":    u,
        })


    def _construct_ansatz(self):
        n = self.n
        mreg = cirq.LineQubit.range(2)                          # SPIN REGISTER
        nreg = cirq.LineQubit.range(2,2+n)                      # SITE REGISTER
        qreg = mreg + nreg

        ansatz = cirq.Circuit()

        # PREPARE SPIN STATE
        ansatz.append( SpinState(
            sympy.Symbol("η"),
            sympy.Symbol("μ"),
            sympy.Symbol("ξ"),
        ).on(*mreg) )

        # INTRODUCE ARBITRARY PHASES INTO SPIN STATE
        ansatz.append( qctb.circuit.PhaseRotation([
            0,                      # ARBITRARILY ASSIGN 00 STATE TO REAL VALUE
            *(sympy.Symbol(power(σ)) for σ in range(1,4))
        ]).on(*mreg) )

        # BALANCE ALL SITE REGISTER STATES
        ansatz.append( cirq.H.on_each(*nreg) )

        # APPLY CONTROLLED PHASE ROTATIONS ON SITE REGISTER
        for σ in range(4):
            # APPLY X's TO FLIP CONTROL BASIS
            for q, qbit in enumerate(reversed(mreg)):
                if not ((1<<q)-1) & σ:      ansatz.append( cirq.X.on(qbit) )
            # APPLY CONTROLLED PHASE ROTATION
            ansatz.append( qctb.circuit.PhaseRotation([
                *(sympy.Symbol(power(σ,z)) for z in range(self.L))
            ]).controlled(2).on(*(*mreg, *nreg)) )

        self.mreg = mreg
        self.nreg = nreg

        return ansatz, qreg

    def _construct_groups(self):
        groups = []

        # SINGLE-BODY TERMS: (ZI + IZ - 2*II) x A_N
        AN = qctb.pauli.AdjacentSets(self.n, 0)
        AN = qctb.pauli.XY2IZ(AN, 0)

        for r, ANr in enumerate(AN):
            gp = qctb.pauli.Group( cirq.Circuit(
                self.ansatz,
                qctb.circuit.XYRotation(self.n,r,0).on(*self.nreg),
                cirq.measure(*self.qreg),
            ) )

            for term in ANr:
                gp.append( qctb.pauli.Term((term.z<<2)+0, -2*term.c) )  # -2*II
                gp.append( qctb.pauli.Term((term.z<<2)+1,    term.c) )  #    ZI
                gp.append( qctb.pauli.Term((term.z<<2)+2,    term.c) )  #    IZ

            groups.append(gp)

        # CORRELATION TERMS: u*(II + ZZ - ZI -IZ) x I_N
        # THESE CAN BE ADDED TO ANY MEASUREMENT GROUP, BUT WE'll STILL...
        if not len(AN):     # ...PROTECT AGAINST EDGE CASE L=1
            groups.append( qctb.pauli.Group( cirq.Circuit(
                self.ansatz,
                cirq.measure(*self.qreg)
            ) ) )
        gp = groups[-1]

        gp.append( qctb.pauli.Term(0, self.u) )                         #    II
        gp.append( qctb.pauli.Term(1,-self.u) )                         #   -ZI
        gp.append( qctb.pauli.Term(2,-self.u) )                         #   -IZ
        gp.append( qctb.pauli.Term(3, self.u) )                         #    ZZ

        return groups

    def energy(self, M, N, ξ, mpowers, npowers, computer):
        # mpowers is a list of 3 t in [-1,1]
        # npowers is a 4-list of L t in [-1,1]
        L = self.L

        parameters = {
            "η":   N/L,
            "μ": 2*M/L,
            "ξ":     ξ,
            **{ power(σ): mpowers[i] for i, σ in enumerate(range(1,4)) },
            **{ power(σ,z): npowers[σ][z] for σ in range(4) for z in range(L) },
        }

        return super().energy(parameters, computer)



register(PrototypeHubbardHamiltonian)

def E0(L,M,N,u):
    """
    Dallaire-Demers 2020 (Eq. 2) considers an infinite chain at half occupancy.
    I guess we have to implement solution to Wieb-Lu equations:
        Essler Eqs. 3.95-3.97.
    """
    pass
    # TOOD: Bethe ansatz, Lieb-Wu equations.





def power(σ, z=None):
    label = "t"
    # if σ & 1:   label += "↑"
    # if σ & 2:   label += "↓"
    label += "↑" if σ & 1 else "_"
    label += "↓" if σ & 2 else "_"
    if z is not None:   label += f".{z}"
    return label
