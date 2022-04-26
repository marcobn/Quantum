""" Test out self-orthogonalization protocol.

We have established a convenient parameterization,
    which *should* let us slice off basis vectors very conveniently.

Given n qubits, we have N=2**n sized statevectors, spanned by 2N-2 parameters.
    We split that up into two N-1 sized lists of parameters in [0,1).
    We zero out the first element by...zeroing the first element in each list!
        And so forth. It actually makes perfect sense.

The only hiccup is that our parameterization isn't exactly uniform,
    but that really shouldn't affect the orthogonalization.
It may influence optimization,
    but I doubt it matters for noise-free simulations.

All we need to do here is confirm that the alleged operator works as intended.
So, we need to generate |Ψ[l]⟩ = ∏[i≤l] Ω[i] |l⟩,
    and check that they form an orthogonal basis.

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
#       CHECK THE ACTUAL ORTHOGONALITY WHEN DOING CIRCUITS IN SEQUENCE

N = 4
rng = np.random.default_rng()

# PLACEHOLDERS
ϕ = np.empty((N,N), dtype=complex)      # |ϕ[l]⟩ = Ω[l] |l⟩
ψ = np.empty((N,N), dtype=complex)      # |Ψ[l]⟩ = ∏[i<l] Ω[i] |ϕ[l]⟩

ΠΩ = np.eye(N)
for l in range(N):
    t, α = random_parameters(N, l, rng)     # RANDOMLY SELECT PARAMETERS

    Ω  = operator(t, α, l)                  # LEVELED OPERATOR
    ΠΩ = ΠΩ @ Ω                             # PREPEND LATEST OPERATOR

    ϕ[l] =  Ω[:,l]                          #  Ω |l⟩
    ψ[l] = ΠΩ[:,l]                          # ΠΩ |l⟩


# CHECK THAT EACH ψ IS ORTHOGONAL TO ALL OTHERS
#   IN OTHER WORDS, CHECK ψ FORMS AN ORTHOGONAL BASIS, ie. IS UNITARY
ϕ

np.abs(ψ @ ψ.T.conjugate())
""" Beautiful. Just beautiful. """
