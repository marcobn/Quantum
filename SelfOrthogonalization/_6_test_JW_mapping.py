""" Construct orthogonalization circuits for multi-electron system.

We're going to just extend the system developed for the compact basis:

    Ω[l] = exp[-𝑖π (T[l]+T[l]†)/2]

    T[l] = ∑[i<l] A†[i] A[i] + ∑[i≥l] ϕ[i] A†[i] A[l]

But now i is not every integer up to 2**n, but only those with Hamming weight η.
    The operator A†[i] is the product of all a†[q] where q is "on" in i.

We will use JW to map H=(T+T†)/2 onto qubit operators.
    openfermion pretty much does this part for us.

Note that η=2 would permit standard UCCSD implementation for Ω,
    but not in general, so don't worry about trying to optimize it.
    Just present this as a concept.

In this file we need to standardize basis state and qubit ordering,
    and make sure we know which way the durned Z strings go,
    and make sure the exact matrix exp(-𝑖πH) does what it's meant to.

Then we will construct the actual Trotterized circuit and validate.


"""

import numpy as np
np.set_printoptions(3, suppress=True)
from numpy import pi as π

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

def binarize(z,n):
    """ Construct an array of the binary decomposition of z, with width n. """
    return np.array(list(np.binary_repr(z, n)), dtype=int)


##############################################################################
#               BACK TO QUANTUM

def projector(n,η,µ,ν):
    """ Pauli sum for the projector |µ⟩⟨ν|. """
    from openfermion import QubitOperator as Pauli

    # DECOMPOSE INDEXES INTO BINARY ARRAYS
    µ = binarize(µ,n)
    ν = binarize(ν,n)

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
    z = binarize(basisstates(n,η)[l], n)

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

    # EXPONENTIATE MATRIX
    return scipy.linalg.expm(-1j*π * H.matrix())

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



##############################################################################
#       CHECK THE ACTUAL ORTHOGONALITY WHEN DOING MATRIX CALCULATIONS

n, η = 4, 2
N = dimension(n,η)
rng = np.random.default_rng()

# PLACEHOLDERS
ϕ = np.empty((N,2**n), dtype=complex)      # |ϕ[l]⟩ = Ω[l] |l⟩
ψ = np.empty((N,2**n), dtype=complex)      # |Ψ[l]⟩ = ∏[i<l] Ω[i] |ϕ[l]⟩

ΠΩ = np.eye(2**n)
for l, zl in enumerate(basisstates(n,η)):
    t, α = random_parameters(n,η,l,rng)     # RANDOMLY SELECT PARAMETERS

    Ω  = operator(n,η,t,α,l)                # LEVELED OPERATOR
    ΠΩ = ΠΩ @ Ω                             # PREPEND LATEST OPERATOR

    ϕ[l] =  Ω[:,zl]                         #  Ω |l⟩
    ψ[l] = ΠΩ[:,zl]                         # ΠΩ |l⟩


# ONLY THE INDICES SPECIFIED BY THE BASIS ARE IN THE ACTIVE SPACE
basis = basisstates(n,η)
basis

# CHECK THAT ϕ IS UPPER DIAGONAL IN THE ACTIVE SPACE
ϕ[:,basis]

# CHECK THAT BOTH ϕ AND ψ ARE ZERO IN THE INACTIVE SPACE
notbasis = np.array([i for i in range(2**n) if i not in basis])
np.allclose(ϕ[:,notbasis], np.zeros((N,len(notbasis))))
np.allclose(ψ[:,notbasis], np.zeros((N,len(notbasis))))

# CHECK THAT EACH ψ IS ORTHOGONAL TO ALL OTHERS
#   IN OTHER WORDS, CHECK ψ FORMS AN ORTHOGONAL BASIS, ie. IS UNITARY
np.abs(ψ @ ψ.T.conjugate())
""" Beautiful. Just beautiful. """




##############################################################################
#            CHECK THE ACTUAL ORTHOGONALITY WHEN DOING CIRCUITS
import time

n, η = 4, 2
N = dimension(n,η)
rng = np.random.default_rng(0)

# CIRQ PAPERWORK
sim = cirq.Simulator()
qreg = cirq.LineQubit.range(n)

# PLACEHOLDERS
ϕ = np.empty((N,2**n), dtype=complex)      # |ϕ[l]⟩ = Ω[l] |l⟩
ψ = np.empty((N,2**n), dtype=complex)      # |Ψ[l]⟩ = ∏[i<l] Ω[i] |ϕ[l]⟩

r = 4
wholect = cirq.Circuit()
for l in range(N):
    start = time.time()

    t, α = random_parameters(n,η,l,rng)     # RANDOMLY SELECT PARAMETERS

    H = hamiltonian(n,η,t,α,l)              # PAULI SUM, AS A QubitOperator
    ct = trottercircuit(qreg, H, r, π)      # TROTTERIZED exp(-𝑖Hπ)
    wholect = ct + wholect                  # PREPEND LATEST STATE PREP

    # SIMULATE CIRCUIT ACTIONS
    ϕ[l] = sim.simulate(prepare(qreg,η,l)+     ct).final_state_vector
    ψ[l] = sim.simulate(prepare(qreg,η,l)+wholect).final_state_vector

    end = time.time()
    print (f"Finished l={l} after {end-start:.2f}s")



# ONLY THE INDICES SPECIFIED BY THE BASIS ARE IN THE ACTIVE SPACE
basis = basisstates(n,η)
basis

# CHECK THAT ϕ IS UPPER DIAGONAL IN THE ACTIVE SPACE
ϕ[:,basis]

# CHECK THAT BOTH ϕ AND ψ ARE ZERO IN THE INACTIVE SPACE
notbasis = np.array([i for i in range(2**n) if i not in basis])

ϕ[:,notbasis]
np.linalg.norm(ϕ[:,notbasis])

ψ[:,notbasis]
np.linalg.norm(ψ[:,notbasis])


# CHECK THAT EACH ψ IS ORTHOGONAL TO ALL OTHERS
#   IN OTHER WORDS, CHECK ψ FORMS AN ORTHOGONAL BASIS, ie. IS UNITARY
np.abs(ψ @ ψ.T.conjugate())
""" Beautiful. Nothing but beautiful. """
