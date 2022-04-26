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

All we need to do here is confirm that the alleged operator works with zeros.
To that end, check that
    the parameters derived from the statevector produced by the operator
    always match the original parameters, within error.

"""

import numpy as np
np.set_printoptions(3, suppress=True)
from numpy import pi as π

import scipy.stats

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

def parameters(ϕ):
    """ Construct parameters from statevector. """
    N = len(ϕ)

    # POLAR PARAMETERIZATION
    r = np.abs(ϕ)
    γ = np.angle(ϕ)

    # CHECK THAT ϕ IS STANDARDIZED
    for i in range(N):
        if abs(r[i]) <= np.finfo(float).eps:
            continue    # SCAN TO FIRST NON-ZERO ELEMENT
        if abs(γ[i]) > 4 * np.finfo(float).eps:
            raise ValueError(f"|γ[0]|={abs(γ[0])} > 4*{np.finfo(float).eps}")
        break           # IF ERROR WAS NOT RAISED, ϕ IS GOOD

    # FILL IN t
    t = np.empty(N-1)
    Π_cos = 1
    for n in range(N-1):
        t[n] = 2/π * np.arcsin( r[n] / Π_cos )
        Π_cos *= np.cos(π/2 * t[n])

    # FILL IN α
    α = γ[1:] / (2*π)

    return t, α

def standardize(ϕ):
    """ Factor out the angle of first non-zero element. """
    for i in range(N):
        if abs(ϕ[i]) <= np.finfo(float).eps:
            continue    # SCAN TO FIRST NON-ZERO ELEMENT
        γ = np.angle(ϕ[i])
        break
    return ϕ * np.exp(-1j*γ)

def fidelity(ϕ,ψ):
    """ Calculate fidelity ⟨ϕ|ψ⟩⟨ψ|ϕ⟩ """
    return abs(np.vdot(ϕ,ψ))**2

def random_statevector(N, seed):
    """ Construct a statevector, randomized according to the seed. """
    rng = np.random.default_rng(seed)
    return scipy.stats.unitary_group.rvs(N, random_state=rng)[:,0]

def random_parameters(N, seed):
    """ Generate N+2 random numbers in [0, 1), and split into two lists. """
    rng = np.random.default_rng(seed)
    rnd = rng.random(N+2)
    return rnd[:(N+2)//2], rnd[(N+2)//2:]

def atial(N,i,l):
    """ Return the compact basis matrix operator for a†[i] a[l].

    We should implement this with Pauli matrices as prescribed above,
        but first let's just use, you know, the obvious one...
    """
    Π = np.zeros((N,N), dtype=complex)
    Π[i,l] = 1
    return Π

def generator(ϕ):
    """ Return the generator T = ∑ ϕ[i] a†[i] a[0] as a matrix. """
    N = len(ϕ)
    T = np.zeros((N,N), dtype=complex)
    for i in range(N):
        T += ϕ[i] * atial(N,i,0)
    return T

def operator(t, α):
    """ Return the operator Ω = exp[-𝑖π (T+T†)/2] as a matrix. """
    ϕ = statevector(t,α)
    T = generator(ϕ)
    H = (T + T.T.conjugate())/2
    return scipy.linalg.expm(-1j*π * H)

##############################################################################
#      GENERATE RANDOM PARAMETERS, PASS THROUGH OPERATOR, AND PASS BACK

N = 4
t, α = random_parameters(N,342)
t
α

Ω = operator(t,α)
ϕx = standardize(Ω[:,0])
ϕx

tx, αx = parameters(ϕx)
tx
αx

""" Good. """


##############################################################################
#           AGAIN, BUT ZEROING OUT THE FIRST t AND α

N = 4
t, α = random_parameters(N,248)
t[0] = α[0] = 0
t
α

Ω = operator(t,α)
ϕx = standardize(Ω[:,0])
ϕx

tx, αx = parameters(ϕx)
tx
αx

""" Good. """




##############################################################################
#       CHECK THE ACTUAL ORTHOGONALITY WHEN DOING CIRCUITS IN SEQUENCE!

N = 4

# PLACEHOLDERS
ϕ = np.empty((N,N), dtype=complex)
ψ = np.empty((N,N), dtype=complex)

seed = 10
Ω_whole = np.eye(N)
for l in range(N):
    t, α = random_parameters(N,seed+l)      # RANDOMLY SELECTED PARAMETERS
    t[:l] = α[:l] = 0                       # LOP OFF FIRST l BASIS VECTORS

    Ω = operator(t, α)                      # PAULI SUM, AS A QubitOperator
    Ω_whole = Ω_whole @ Ω                   # PREPEND LATEST STATE PREP

    ϕ[l] = Ω[:,0]                           # FINAL STATE, PRE -ORTHO
    ψ[l] = Ω_whole[:,0]                     # FINAL STATE, POST-ORTHO


# CHECK THAT EACH ψ IS ORTHOGONAL TO ALL OTHERS
#   IN OTHER WORDS, CHECK ψ FORMS AN ORTHOGONAL BASIS, ie. IS UNITARY
ϕ

ϕ @ ϕ.T.conjugate()

ψ @ ψ.T.conjugate()
abs(ψ @ ψ.T.conjugate())

""" Ai, this is what we were supposed to figure out *before* we moved on..!

But, I think I know how to fix it...
"""
