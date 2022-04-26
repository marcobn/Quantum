""" Test out circuit to span compact basis.

To illustrate our self-orthogonalizing circuitry,
    we need a circuit construction which can span a full subset of Hilbert space
    but also be conveniently restricted to act as identity on the rest of it.

Our best idea right now is one we developed aeons ago,

    Ω[l] = exp[-𝑖π (T[l]+T[l]†)/2]

    T[l] = ∑[i<l] a†[i] a[i] + ∑[i≥l] ϕ[i] a†[i] a[l]

Note the extra term from attempt 1. :P

Constructing each term a†a in the compact basis is straightforward:

    a†[i] a[l] = |i⟩⟨l| = |i[0]⟩⟨l[0]| ⊗ ... ⊗ |i[n]⟩⟨l[n]|

    |0⟩⟨0| = (I + Z) / 2
    |0⟩⟨1| = (X + 𝑖Y) / 2
    |1⟩⟨0| = (X - 𝑖Y) / 2
    |1⟩⟨1| = (I - Z) / 2

In this file, we must check that constructing the full analytic matrix gives
    the identity matrix in the top-left l×l block,
    and the desired ϕ in the lth column.

"""

import numpy as np
np.set_printoptions(3, suppress=True)
from numpy import pi as π
import scipy.linalg


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

def projector(N,i,j):
    """ Return the compact basis matrix operator for a†[i] a[j]. """
    Π = np.zeros((N,N), dtype=complex)
    Π[i,j] = 1
    return Π

def generator(ϕ,l):
    """ Return the generator T[l] = ∑ ϕ[i] a†[i] a[l] as a matrix. """
    N = len(ϕ)
    T = np.zeros((N,N), dtype=complex)
    for i in range(l):
        T += projector(N,i,i)
    for i in range(l,N):
        T += ϕ[i] * projector(N,i,l)
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

def fidelity(ϕ,ψ):
    """ Calculate fidelity ⟨ϕ|ψ⟩⟨ψ|ϕ⟩ """
    return abs(np.vdot(ϕ,ψ))**2

def random_parameters(N, l, rng):
    """ Generate random set of parameters. """
    t = np.concatenate(([0]*l, rng.random(N-1-l)))
    α = np.concatenate(([0]*l, rng.random(N-1-l)))
    return t, α



##############################################################################
#                   CHECK VALIDITY OF PROPOSED OPERATOR

N = 4
l = 3
rng = np.random.default_rng()

# GENERATE PARAMETERS
t, α = random_parameters(N, l, rng)

# CONSTRUCT OPERATOR
Ω = operator(t,α,l)
Ω

# CHECK THAT TOP-LEFT BLOCK IS IDENTITY
if l > 0:
    print(f"Identity block: {abs(np.trace(Ω[:l,:l])/l)}" )

# CHECK FIDELITY OF lth COLUMN
F = fidelity(statevector(t, α), Ω[:,l])
print (f"Target vector: {F:.3f}")
