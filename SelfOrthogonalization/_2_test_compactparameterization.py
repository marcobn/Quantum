""" Test out circuit to span compact basis.

To illustrate our self-orthogonalizing circuitry,
    we need a circuit construction which can span the full Hilbert space
    but can also be conveniently restricted to cut out individual basis states.

Our best idea right now is one we developed aeons ago,
    (probably related to the "Variational Hamiltonian Ansatz" maybe?)

    Ω[l] = exp[-𝑖π (T[l]+T[l]†)/2]

    T[l] = ∑ ϕ[i] a†[i] a[l]

Constructing each term a†a in the compact basis is straightforward:

    a†[i] a[l] = |i⟩⟨l| = |i[0]⟩⟨l[0]| ⊗ ... ⊗ |i[n]⟩⟨l[n]|

    |0⟩⟨0| = (I + Z) / 2
    |0⟩⟨1| = (X + 𝑖Y) / 2
    |1⟩⟨0| = (X - 𝑖Y) / 2
    |1⟩⟨1| = (I - Z) / 2

We have confirmed this circuit produces the desired result,
    provided the phase on first element of ϕ is 0.

Now we've developed a parameterization for ϕ which feels right:

    |ϕ⟩ = r0 |0⟩ + r1 exp(2π𝑖 α1) |1⟩ + ... rn exp(2π𝑖 αn) |n⟩

        r0 = sin(π/2 t0)
        r1 = cos(π/2 t0) * sin(π/2 t1)
        ...
        r[n-1] = cos(π/2 t0) ... cos(π/2 t[n-2]) sin(π/2 t[n-1])
        r[n]   = cos(π/2 t0) ... cos(π/2 t[n-2]) cos(π/2 t[n-1])

    Thus both t and α are taken from [0,1],
        and we can eliminate |m<l⟩ by setting t[m<l] = α[m≤l] = 0.

So we just want to make sure that, for any ϕ,
    we can reliably find parameters t and α which produce that ϕ.
    α is trivially scaled by γ, the phase angle for each element of ϕ.
    A little harder to show, but solving for t yields only t∈[0,1].

But I'd like to convince myself that these are, like, good parameters.
    If we randomly generate ϕ, we should see a random distribution of α, t.
    It NEEDS to fill the whole space. Ideally it does so uniformally.

So we should generate lots of ϕ (standardized), calculate α and t,
    then histogram each variable. 6 plots; should all look uniform.
    Next plot each variable against the other (6c2)=15 plots.
    They should all appear uncorrelated.

We can also generate lots of t, α uniform,
    and check that fidelities distribute suitably random.

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

    # CHECK THAT ϕ IS NORMALIZED
    if abs(γ[0]) > 4 * np.finfo(float).eps:
        raise ValueError(f"|γ[0]|={abs(γ[0])} > 4*{np.finfo(float).eps}")

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
    """ Factor out the angle of element 0. """
    γ = np.angle(ϕ[0])
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


##############################################################################
#           GET A FEEL FOR HOW FIDELITIES OF RANDOM STATEVECTORS LOOK
import matplotlib.pyplot as plt

T = 1000
N = 4
ϕs = [standardize(random_statevector(N,i)) for i in range(T)]

F_matrix = np.empty((T,T))
for i in range(T):
    for j in range(T):
        F_matrix[i,j] = fidelity(ϕs[i], ϕs[j])

plt.matshow(F_matrix)

_ = plt.hist(F_matrix.flatten(), 50)

##############################################################################
#           SEE IF IT STILL LOOKS RIGHT FROM RANDOM PARAMETERS
import matplotlib.pyplot as plt

T = 1000
N = 4
ϕs = [statevector(*random_parameters(N,i)) for i in range(T)]

F_matrix = np.empty((T,T))
for i in range(T):
    for j in range(T):
        F_matrix[i,j] = fidelity(ϕs[i], ϕs[j])

plt.matshow(F_matrix)

_ = plt.hist(F_matrix.flatten(), 50)

""" No, it does not look right.

This does not strictly mean it's BAD, but
1) it must be born in mind when discussing optimization.
2) a transformation to make it uniform would be good.
3) we need another check to validate we're spanning the whole space properly.

            ...eh, oh well.

"""

##############################################################################
#               CHECK RANDOMNESS OF PARAMETERIZATION
import matplotlib.pyplot as plt

T = 10000
N = 4
params = np.empty((T,N+2))
for i in range(T):
    ϕ = standardize(random_statevector(N,i))
    t, α = parameters(ϕ)
    params[i] = np.concatenate((t,α))

# PLOT HISTOGRAMS OF EACH PARAMETER
for p in range(N+2):
    plt.hist(params[:,p], 50)
    plt.title(f"p={p}")
    plt.figure()

# PLOT SCATTER PLOTS CORRELATING EACH PARAMETER
for p in range(N+2):
    for q in range(p):
        plt.plot(params[:,p], params[:,q], '.k')
        plt.title(f"(p,q)=({p},{q})")
        plt.figure()

""" t is clearly not exactly uniform.

However I do judge it good enough - onward!

"""
