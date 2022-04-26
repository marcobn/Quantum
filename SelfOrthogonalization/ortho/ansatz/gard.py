""" Implement the chain of A-gates useful for spanning the 1e- space. """

import numpy
from numpy import pi as π

import sympy
import cirq

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
