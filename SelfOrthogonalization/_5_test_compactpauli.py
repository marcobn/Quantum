""" Construct compact orthogonalization operator circuit.

Our orthogonalization operator works as a matrix but now we need a circuit.
    So, we need to construct the Hermitian operator H=(T+T†)/2 as a Pauli sum.
    Then we need to ask cirq to make the exp(-𝑖π H) into a circuit.
        Probably this will require Trotterization.
        Remember to use Suzuki order 2.

We should measure how many Trotter steps are needed to match desired ϕ,
    but more importantly how well our quantum protocol orthogonalizes.

"""

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

def prepare(qreg, l):
    """ Prepare the basis state |l⟩ from |0⟩. """
    n = len(qreg)
    l = np.array(list(np.binary_repr(l, n)), dtype=int)

    # APPLY X's AS NEEDED
    ct = cirq.Circuit()
    for q, qbit in enumerate(qreg):
        if l[q]:    ct += cirq.X(qbit)
    return ct


def random_parameters(n, l, rng):
    """ Generate random set of parameters. """
    N = 2**n
    t = np.concatenate(([0]*l, rng.random(N-1-l)))
    α = np.concatenate(([0]*l, rng.random(N-1-l)))
    return t, α


def fidelity(ϕ,ψ):
    """ Calculate fidelity ⟨ϕ|ψ⟩⟨ψ|ϕ⟩ """
    return abs(np.vdot(ϕ,ψ))**2


##############################################################################
#                   CHECK OPERATOR STILL MATCHES
rng = np.random.default_rng(324)
n = 2
l = 1

t, α = random_parameters(n,l,rng)
t, α
H = hamiltonian(n, t, α, l)
Ω = scipy.linalg.expm(-1j*π*openfermion.qubit_operator_to_pauli_sum(H).matrix())
Ω
ψ = Ω[:,l]
ϕ = statevector(t,α)
fidelity(ψ,ϕ)

##############################################################################
#                   SIMULATE H WITH openfermion CIRCUIT

rng = np.random.default_rng(324)
n = 2
l = 1

t, α = random_parameters(n,l,rng)
t, α
H = hamiltonian(n, t, α, l)
H
r = 10
qreg = cirq.LineQubit.range(n)
ct = trottercircuit(qreg, H, r, π)
sim = cirq.Simulator()
res = sim.simulate(prepare(qreg,l) + ct)
ψ = res.final_state_vector
ϕ = statevector(t,α)

fidelity(ψ,ϕ)

##############################################################################
#            MEASURE CIRCUIT FIDELITIES FOR INCREASING r, l
rng = np.random.default_rng(0)

# SYSTEM SELECTION
n = 2
N = 2**n

# CIRQ PAPERWORK
sim = cirq.Simulator()
qreg = cirq.LineQubit.range(n)

# THINGS TO PLOT
r_    = 1+(np.arange(10))
F_    = np.empty((N,len(r_)))
F_err = np.empty((N,len(r_)))


# QUALITY OF LIFE FUNCTIONS
def singletrial(rng, r, l):
    """ Compute fidelity for random trial using level l and r Trotter steps. """
    t, α = random_parameters(n,l,rng)       # RANDOMLY SELECTED PARAMETERS
    t[:l] = α[:l] = 0                       # LOP OFF FIRST l BASIS VECTORS
    ϕ = statevector(t, α)                   # TARGET STATE

    H = hamiltonian(n, t, α, l)             # PAULI SUM, AS A QubitOperator
    ct = trottercircuit(qreg, H, r, π)      # TROTTERIZED exp(-𝑖Hπ)
    ψ = sim.simulate(prepare(qreg,l)+ct).final_state_vector # FINAL STATE

    return fidelity(ψ,ϕ)                    # CALCULATE FIDELITY

def stats(r, l, T):
    """ Get mean and standard deviation of fidelity from T trials. """
    Fs = [singletrial(rng, r, l) for _ in range(T)]
    return np.mean(Fs), np.std(Fs)

# PERFORMING THE COMPUTATION
import time
for l in range(3,N):
    for i, r in enumerate(r_):
        start = time.time()
        F_[l,i], F_err[l,i] = stats(r,l,30)
        end = time.time()
        print (f"Finished l={l}, r={r} after {end-start}s")

# PLOT RESULTS
import matplotlib.pyplot as plt
for l in range(N):
    plt.errorbar(r_, F_[l], F_err[l], label=f"l={l}")
plt.legend()

for l in range(N):
    plt.plot(r_, np.log(1-F_[l]), label=f"l={l}")
plt.legend()



##############################################################################
#       CHECK THE ACTUAL ORTHOGONALITY WHEN DOING CIRCUITS IN SEQUENCE!
import time

rng = np.random.default_rng(0)

n = 2
N = 2**n

# CIRQ PAPERWORK
sim = cirq.Simulator()
qreg = cirq.LineQubit.range(n)

# PLACEHOLDERS
ϕ = np.empty((N,N), dtype=complex)
ψ = np.empty((N,N), dtype=complex)

r = 2
wholect = cirq.Circuit()
for l in range(N):
    start = time.time()

    t, α = random_parameters(n,l,rng)       # RANDOMLY SELECTED PARAMETERS
    t[:l] = α[:l] = 0                       # LOP OFF FIRST l BASIS VECTORS

    H = hamiltonian(n, t, α, l)             # PAULI SUM, AS A QubitOperator
    ct = trottercircuit(qreg, H, r, π)      # TROTTERIZED exp(-𝑖Hπ)
    wholect = ct + wholect                  # PREPEND LATEST STATE PREP

    # SIMULATE CIRCUIT ACTIONS
    ϕ[l] = sim.simulate(prepare(qreg,l)+     ct).final_state_vector # PRE -ORTHO
    ψ[l] = sim.simulate(prepare(qreg,l)+wholect).final_state_vector # POST-ORTHO

    end = time.time()
    print (f"Finished l={l} after {end-start}s")


# CHECK THAT EACH ψ IS ORTHOGONAL TO ALL OTHERS
#   IN OTHER WORDS, CHECK ψ FORMS AN ORTHOGONAL BASIS, ie. IS UNITARY
print (ϕ)

print (abs(ψ @ ψ.T.conjugate()))

""" Beautiful. Nothing but beautiful. """
