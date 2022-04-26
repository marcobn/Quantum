""" Implement the UCC-like circuit exploring all basis states. """

import numpy as np
np.set_printoptions(3, suppress=True)
from numpy import pi as π

import scipy.stats
import scipy.linalg

import cirq
import openfermion

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

def projector(n,µ,ν):
    """ Pauli sum for the projector |µ⟩⟨ν|. """
    from openfermion import QubitOperator as Pauli

    µ = np.array(list(np.binary_repr(µ, n)), dtype=int)
    ν = np.array(list(np.binary_repr(ν, n)), dtype=int)

    Π = Pauli(f'')
    for q in range(n):
        s = 0.5 * Pauli(f'')
        if ν[q]:        s *= Pauli(f'X{q}')
        if µ[q]!=ν[q]:  s *= Pauli(f'X{q}')
        s *= Pauli(f'') + Pauli(f'Z{q}')
        if ν[q]:        s *= Pauli(f'X{q}')

        Π *= s
    return Π

def projectormatrix(N,i,j):
    """ Return the compact basis matrix operator for a†[i] a[j]. """
    Π = np.zeros((N,N), dtype=complex)
    Π[i,j] = 1
    return Π

def generator(ϕ,l):
    """ Return the generator T[l] = ∑ ϕ[i] a†[i] a[l] as a matrix. """
    N = len(ϕ)
    T = np.zeros((N,N), dtype=complex)
    for i in range(l):
        T += projectormatrix(N,i,i)
    for i in range(l,N):
        T += ϕ[i] * projectormatrix(N,i,l)
    return T

def operator(t,α,l):
    """ Return the operator Ω[l] = exp[-𝑖π (T[l]+T[l]†)/2] as a matrix. """
    T = generator(statevector(t,α), l)
    H = (T + T.T.conjugate())/2

    # We need to subtract off constant from H to match quantum implementation,
    #   to eliminate an awkward global phase term we can't actually implement.
    N = len(t) + 1
    I = np.eye(N)
    H -= np.trace(H)/N * I

    return scipy.linalg.expm(-1j*π * H)

def hamiltonian(n, t, α, l):
    """ Pauli sum for the operator (T + T†)/2 """
    N = 2**n

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

    H = 0
    for i in range(l):
        H += projector(n,i,i)
    for i in range(l,N):
        H += r[i]/2 * np.exp( 1j*γ[i]) * projector(n,i,l)
        H += r[i]/2 * np.exp(-1j*γ[i]) * projector(n,l,i)
    return H

def trottercircuit(qreg, H, r, τ):
    """ Construct Trotterized circuit for exp(-𝑖Hτ). """
    p = 2   # Suzuki "order". I see no good reason not to hard-code 2 here.

    # LOP OFF CONSTANT TERM (only different by a global phase)
    H = H - H.constant

    # CONSTRUCT THE CIRCUIT
    ct = cirq.Circuit()

    if H.isclose(openfermion.QubitOperator()):  return ct   # IDENTITY CIRCUIT

    for qubitop in openfermion.trotter_operator_grouping(H, r, p, k_exp=-τ):
        paulisum = openfermion.qubit_operator_to_pauli_sum(qubitop, qreg)
        op = cirq.ops.PauliSumExponential(paulisum)
        ct += op
    return ct


def pauliword(n, i):
    """ Construct the pauli word corresponding to integer i.

    Ordering is made by representing I->0, X->1, etc. and writing i in base 4.
    """
    # CONVERT i TO BASE 4 REPRESENTATION
    i4 = np.base_repr(i, 4)         # DECOMPOSE INTO BASE 4
    i4 = "0"*(n-len(i4)) + i4       # PAD 0's ON LEFT

    # CONSTRUCT PAULI WORD
    qreg = cirq.LineQubit.range(n)
    pauli = 1
    for q in range(n):
        if i4[q] == "0":  pauli *= cirq.PauliString( cirq.I(qreg[q]) )
        if i4[q] == "1":  pauli *= cirq.PauliString( cirq.X(qreg[q]) )
        if i4[q] == "2":  pauli *= cirq.PauliString( cirq.Y(qreg[q]) )
        if i4[q] == "3":  pauli *= cirq.PauliString( cirq.Z(qreg[q]) )
    return pauli
