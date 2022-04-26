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

We have previously confirmed this circuit produces the desired result.
    But I think that was for a second-quantized basis?
    And I don't know where to find that code anymore...
    ...and I think I only confirmed it via Suzuki-Trotter approximation.

So, in this file, we're going to construct the full analytic matrix,
    and just make sure the first column is equal to ϕ.
    Fingers crossed!

"""

import numpy as np
np.set_printoptions(3, suppress=True)
from numpy import pi as π
import scipy.linalg




def atial(N,i,l):
    """ Return the compact basis matrix operator for a†[i] a[l].

    We should implement this with Pauli matrices as prescribed above,
        but first let's just use, you know, the obvious one...
    """
    Π = np.zeros((N,N), dtype=complex)
    Π[i,l] = 1
    return Π

def generator(ϕ,l):
    """ Return the generator T[l] = ∑ ϕ[i] a†[i] a[l] as a matrix. """
    N = len(ϕ)
    T = np.zeros((N,N), dtype=complex)
    for i in range(N):
        T += ϕ[i] * atial(N,i,l)
    return T

def operator(ϕ,l):
    """ Return the operator Ω[l] = exp[-𝑖π (T[l]+T[l]†)/2] as a matrix. """
    T = generator(ϕ,l)
    H = (T + T.T.conjugate())/2
    return scipy.linalg.expm(-1j*π * H)

def standardize(ϕ):
    # return ϕ * np.exp(-1j*γ)
    """ Factor out the angle of element 0. """
    γ = np.angle(ϕ[0])
    return ϕ * np.exp(-1j*γ)

def fidelity(ϕ,ψ):
    """ Calculate fidelity ⟨ϕ|ψ⟩⟨ψ|ϕ⟩ """
    return abs(np.vdot(ϕ,ψ))**2




##############################################################################
#                   CHECK VALIDITY OF PROPOSED OPERATOR

N = 4

# GENERATE A RANDOM STATEVECTOR
import scipy.stats
U = scipy.stats.unitary_group.rvs(N)
ϕ = standardize(U[:,0])
ϕ = U[:,0]

# CONSTRUCT OPERATOR AND APPLY IT TO |0⟩
Ω = operator(ϕ,0)
ϕx = Ω[:,0]

# CHECK FIDELITY
F = fidelity(ϕ,ϕx)
print (f"Fidelity: {F:.3f}")

##############################################################################
#                       MORE THOROUGH TESTING
""" Strangely enough, it seems like it's...USUALLY very close,
    but sometimes it's ...not. Let's see if we can't do some analytics
    to find a property of ϕ which makes the operator fail.
"""
import scipy.stats
import matplotlib.pyplot as plt

# COLLECT STATISTICS
DATA = []
for seed in range(1000):
    rng = np.random.default_rng(seed)
    ϕ = scipy.stats.unitary_group.rvs(N, random_state=rng)[:,0]
    F = fidelity(ϕ, operator(ϕ,0)[:,0])

    DATA.append({ "rng": rng, "ϕ": ϕ, "F": F })
# DATA

# DISTRIBUTION OF FIDELITIES
_ = plt.hist([datum["F"] for datum in DATA], 50)

# CORRELATION WITH VARIOUS SCALAR QUANTITIES OF ϕ
def plot_correlation(fn):
    x = [fn(datum["ϕ"]) for datum in DATA]
    F = [datum["F"] for datum in DATA]
    _ = plt.plot(x,F,'.k')

# plot_correlation(lambda ϕ: np.abs(ϕ).sum())     # L1 NORM
plot_correlation(lambda ϕ: np.abs(ϕ[0]))        # 0 MODULUS
plot_correlation(lambda ϕ: np.angle(ϕ[0]))      # 0 ANGLE


plot_correlation(lambda ϕ: np.abs(ϕ[0])**2)        # 0 COMPONENT FRACTION
plot_correlation(lambda ϕ: ϕ[0].real)        # 0 COMPONENT REAL
plot_correlation(lambda ϕ: ϕ[0].imag)        # 0 COMPONENT IMAG
plot_correlation(lambda ϕ: np.sin(np.angle(ϕ[0])))        # 0 ANGLE sin
plot_correlation(lambda ϕ: np.cos(np.angle(ϕ[0])))        # 0 ANGLE sin

""" Turns out it works fine if we restrict γ0=0. Interesting, but not now. """
