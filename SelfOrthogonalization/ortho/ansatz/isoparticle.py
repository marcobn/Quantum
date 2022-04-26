""" Implement the UCC-like circuit preserving particle number. """

import numpy as np
np.set_printoptions(3, suppress=True)
from numpy import pi as π

import sympy

import scipy.stats
import scipy.linalg
import scipy.special

import cirq
import openfermion

import ortho.binary as binary

##############################################################################
#               SOME BASIC NUMBER THEORY

def dimension(n,η):
    """ The choose function.

    Calculates the number of complex variables in a statevector
    spanning the restricted Hilbert subspace of n modes and η electrons.
    """
    return int(scipy.special.comb(n,η))

_BASIS_STATE = {}
def basisstates(n,η):
    """ Iterate through the basis states with n modes and η fermions. """
    # FILL IN TUPLES FOR n MODES, IF NOT ALREADY DONE
    if (n,η) not in _BASIS_STATE:
        states = {η_: [] for η_ in range(n+1)}
        for z in range(1<<n):
            states[binary.weight(z)].append(z)
        for η_ in states:
            _BASIS_STATE[n,η_] = states[η_]

    return _BASIS_STATE[n,η]



##############################################################################
#               BACK TO QUANTUM

def projector(n,η,µ,ν):
    """ Pauli sum for the projector |µ⟩⟨ν|. """
    from openfermion import QubitOperator as Pauli

    # DECOMPOSE INDEXES INTO BINARY ARRAYS
    µ = binary.binarize(µ,n)
    ν = binary.binarize(ν,n)

    # CONSTRUCT FERMION OPERATOR
    fermiop_list = []
    for q in reversed(range(n)):
        if µ[q]: fermiop_list.append((q,1))
    for q in reversed(range(n)):
        if ν[q]: fermiop_list.append((q,0))
    fermiop = openfermion.FermionOperator(fermiop_list)

    # APPLY JORDAN-WIGNER TO CONVERT TO QUBIT OPERATOR
    qubitop = openfermion.jordan_wigner(fermiop)
    return qubitop

def prepare(qreg, η, l):
    """ Prepare the basis state |l⟩ from |0⟩, where |l⟩ depends on n and η. """
    n = len(qreg)
    z = binary.binarize(basisstates(n,η)[l], n)

    # APPLY X's AS NEEDED
    ct = cirq.Circuit()
    for q, qbit in enumerate(qreg):
        if z[q]:    ct += cirq.X(qbit)
    return ct

def hamiltonian(n, η, t, α, l):
    """ Pauli sum for the operator (T + T†)/2 """
    N = dimension(n,η)

    # FILL IN MODULI
    r = np.empty(N)
    Π_cos = 1
    for n in range(N-1):
        r[n]   = np.sin(π/2 * t[n]) * Π_cos
        Π_cos *= np.cos(π/2 * t[n])
    r[N-1] = Π_cos

    # FILL IN PHASE ANGLES
    γ = np.zeros(N)
    γ[1:] = 2*π * α

    # FIX ORDERING OF BASIS STATES
    basis = basisstates(n,η)
    zl = basis[l]

    H = 0
    for i, zi in enumerate(basis):
        if i < l:
            H += projector(n,η,zi,zi)
        else:
            H += r[i]/2 * np.exp( 1j*γ[i]) * projector(n,η,zi,zl)
            H += r[i]/2 * np.exp(-1j*γ[i]) * projector(n,η,zl,zi)
    return H

def operator(n, η, t, α, l):
    """ Return the operator Ω[l] = exp[-𝑖πH] as a matrix. """
    # CONSTRUCT PAULI SUM
    H = hamiltonian(n,η,t,α,l)  # openfermion's QubitOperator
    H = H - H.constant          # LOP OFF I..I TERM (standardizes global phase)
    H = openfermion.qubit_operator_to_pauli_sum(H)      # cirq's PauliSum

    # CONVERT TO MATRIX
    qubits = cirq.LineQubit.range(n)    # We have to fix the number of qubits.
    Hm = H.matrix(qubits)

    # EXPONENTIATE MATRIX
    return scipy.linalg.expm(-1j*π * Hm)

def trottercircuit(qreg, H, r, τ):
    """ Construct Trotterized circuit for exp(-𝑖Hτ). """
    p = 2   # Suzuki "order". I see no good reason not to hard-code 2 here.

    # LOP OFF CONSTANT TERM (only different by a global phase)
    H = H - H.constant

    # CONSTRUCT THE CIRCUIT
    ct = cirq.Circuit()

    # THE IDENTITY OPERATOR HAS AN IDENTITY CIRCUIT
    if H.isclose(openfermion.QubitOperator()):  return ct

    # SPLIT H INTO TROTTER TERMS, CONVERT TO cirq OBJECT, AND EXPONENTIATE
    for qubitop in openfermion.trotter_operator_grouping(H, r, p, k_exp=-τ):
        paulisum = openfermion.qubit_operator_to_pauli_sum(qubitop, qreg)
        op = cirq.ops.PauliSumExponential(paulisum)
        ct += op
    return ct

def random_parameters(n, η, l, rng):
    """ Generate random set of parameters. """
    N = dimension(n,η)
    t = np.concatenate(([0]*l, rng.random(N-1-l)))
    α = np.concatenate(([0]*l, rng.random(N-1-l)))
    return t, α


def statevector(t, α):
    """ Construct statevector from parameterization. """
    N = 1 + len(t)                              # NUMBER OF COMPLEX VALUES

    # FILL IN MODULI
    r = np.empty(N)
    Π_cos = 1
    for n in range(N-1):
        r[n]   = np.sin(π/2 * t[n]) * Π_cos
        Π_cos *= np.cos(π/2 * t[n])
    r[N-1] = Π_cos

    # FILL IN PHASE ANGLES
    γ = np.zeros(N)
    γ[1:] = 2*π * α

    # CONSTRUCT STATEVECTOR
    return r * np.exp(1j*γ)

def fidelity(ϕ,ψ):
    """ Calculate fidelity ⟨ϕ|ψ⟩⟨ψ|ϕ⟩ """
    return abs(np.vdot(ϕ,ψ))**2
