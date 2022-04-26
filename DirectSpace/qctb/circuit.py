
import numpy
import sympy
import cirq

import qctb.binary

""" ALIASES """
π = numpy.pi

class A(cirq.TwoQubitGate):
    def __init__(self, θ, φ):
        super().__init__()
        self.θ = θ
        self.φ = φ

    def _decompose_(self, qubits):
        q0, q1 = qubits

        yield cirq.CNOT.on(q0, q1)
        yield (cirq.rz(self.φ + π  )**-1).on(q0)
        yield (cirq.ry(self.θ + π/2)**-1).on(q0)
        yield cirq.CNOT.on(q1, q0)
        yield (cirq.ry(self.θ + π/2)    ).on(q0)
        yield (cirq.rz(self.φ + π  )    ).on(q0)
        yield cirq.CNOT.on(q0, q1)
        # NOTE: This seems to result in a unitary which is the transpose of that
        #   written in Gard et al.. I'm guessing it's a notational relic...?

    def _resolve_parameters_(self, param_resolver, recursive):
        return A(
            param_resolver.value_of(self.θ, recursive),
            param_resolver.value_of(self.φ, recursive),
        )

    def __pow__(self, exp):
        # TODO: Actually A is self-adjoint...
        if exp == -1:
            return A(-self.θ, -self.φ)
        else:
            return NotImplemented

    def _circuit_diagram_info_(self, args):
        return [f"A({self.θ},{self.φ})"]*self.num_qubits()

class PauliRotation(cirq.SingleQubitGate):
    def __init__(self, fm, to):
        super(PauliRotation)
        # TODO: this is NOT how super works...
        self.fm = fm
        self.to = to

    def _num_qubits_(self):
        return 1

    def _decompose_(self, qubits):
        q, = qubits
        return cirq.SingleQubitCliffordGate.from_single_map(
            { self.fm: [self.to, False] }
        ).on(q)

    def __pow__(self, exp):
        if exp == -1:
            return PauliRotation(fm=self.to, to=self.fm)
        else:
            return NotImplemented

    def _circuit_diagram_info_(self, args):
        return f"{str(self.fm)}2{str(self.to)}"

class LinearQFT(cirq.QuantumFourierTransformGate):
    def __init__(self, num_qubits):
        super().__init__(num_qubits, without_reverse=False)

    def _decompose_(self, qubits):
        n = self._num_qubits

        for i in range(n-1):
            yield cirq.H.on(qubits[0])
            for q in range(n-1-i):
                yield (cirq.CZ**(2**-(q+1))).on(qubits[q],qubits[q+1])
                yield cirq.SWAP.on(qubits[q],qubits[q+1])
                # if q == 0:  yield cirq.I.on(qubits[1])
        yield cirq.H.on(qubits[0])

class PhaseRotation(cirq.Gate):
    def __init__(self, powers):
        self.powers = powers
        self.n = qctb.binary.length(len(powers)-1)

    def _num_qubits_(self):
        return self.n

    def _decompose_(self, qubits):
        n = self.n
        for z, t in enumerate(self.powers):
            # APPLY X's TO FLIP CONTROL BASIS
            for q, qbit in enumerate(reversed(qubits)):
                if not ((1<<q)-1) & z:      yield cirq.X.on(qbit)
            # APPLY C..CZ
            yield (cirq.Z**t).controlled(n-1).on(*qubits)

    def _resolve_parameters_(self, param_resolver, recursive):
        return PhaseRotation(
            [param_resolver.value_of(t, recursive) for t in self.powers],
        )

    def _circuit_diagram_info_(self, args):
        return ["Ω"] + ["#"]*(self.n-1)



class XYRotation(cirq.Gate):
    """ Rotates Z basis suitably to measure {XY} words. """
    def __init__(self, n, r, parity):
        self.n = n
        self.r = r
        self.parity = parity
        super().__init__()

    def _num_qubits_(self):
        return self.n

    def _circuit_diagram_info_(self, args):
        label = f"{{IZ}}2{{XY}}.{self.parity}"
        return [""]*(self.n-self.r-1) + [label] + ["#"]*self.r

    def _decompose_(self, qubits):
        n, r, parity = self.n, self.r, self.parity
        α = n - r - 1
        αbase = cirq.Y if parity else cirq.X

        # APPLY exp(-𝑖π/4 Z0) exp(-𝑖π/4 ?X..X) exp(-𝑖π/4 Z0)
        yield cirq.rz(π/2).on(qubits[α])
        ###################################################
        yield PauliRotation(cirq.Z, αbase).on(qubits[α])
        yield PauliRotation(cirq.X,cirq.Z).on_each(qubits[α+1:])
        ###################################################
        yield (cirq.CNOT.on(qubits[q],qubits[q-1]) for q in range(n-1,α,-1))
        yield cirq.rz(π/2).on(qubits[α])
        yield (cirq.CNOT.on(qubits[q+1],qubits[q]) for q in range(α,n-1))
        ###################################################
        yield PauliRotation(αbase, cirq.Z).on(qubits[α])
        yield PauliRotation(cirq.Z,cirq.X).on_each(qubits[α+1:])
        ###################################################
        yield cirq.rz(π/2).on(qubits[α])

        # APPLY EACH exp(-𝑖π/4 X) exp(-𝑖π/4 ZI..IZ) exp(-𝑖π/4 X)
        for q in range(α,n-1):
            yield           cirq.rx(π/2).on(qubits[q+1])
            if q > α: yield cirq.SWAP(qubits[q-1],qubits[q])
            yield           cirq.CNOT.on(qubits[q+1],qubits[q])
            yield           cirq.rz(π/2).on(qubits[q])
            yield           cirq.CNOT.on(qubits[q+1],qubits[q])
            yield           cirq.rx(π/2).on(qubits[q+1])

        # RE-PERMUTE QUBIT 0 DOWN
        yield ( cirq.SWAP(qubits[q],qubits[q-1]) for q in range(n-2,α,-1) )

        # TRANSFORM TO X BASIS ON NON-PARITY QUBITS
        yield PauliRotation(cirq.Z, cirq.X).on_each(*qubits[α+1:])
