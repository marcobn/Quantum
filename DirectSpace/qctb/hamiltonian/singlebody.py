import itertools

import numpy
import sympy
import cirq

import qctb.circuit
import qctb.binary
import qctb.pauli
from qctb.model.model import Model
from qctb.hamiltonian.hamiltonian import Hamiltonian, register

# ALIAS INSERTION STRATEGIES
EARLIEST        = cirq.circuits.InsertStrategy.EARLIEST
NEW             = cirq.circuits.InsertStrategy.NEW
INLINE          = cirq.circuits.InsertStrategy.INLINE
NEW_THEN_INLINE = cirq.circuits.InsertStrategy.NEW_THEN_INLINE


##############################################################################
#                              HAMILTONIANS

class OrbitalHamiltonian(Hamiltonian):
    id = "Direct_M"

    def __init__(self, model):
        super().__init__(model, self.id, {})

    def _construct_ansatz(self):
        qreg = cirq.LineQubit.range(self.M)
        ansatz = cirq.Circuit()

        # PREPARE BASIS STATE δ_lq
        for q in range(self.M):
            δ_lq = sympy.Symbol(level(q))
            ansatz.append((cirq.X**δ_lq).on(qreg[q]), EARLIEST)

        # INTRODUCE GIVENS ROTATIONS
        for l in reversed(range(self.M-1)):
            for q in range(l,self.M-1):
                θ = sympy.Symbol(angle(0,l,q))
                φ = sympy.Symbol(angle(1,l,q))
                ansatz.append( qctb.circuit.A(θ,φ).on(qreg[q], qreg[q+1]) )

        # MODIFY PHASE ACCORDING TO ATOM LOCATION
        for q in range(self.M):
            ab2_α = sympy.Symbol(phase(q))
            ansatz.append((cirq.Z**ab2_α).on(qreg[q]))

        return ansatz, qreg

    def _construct_groups(self, b=None):
        # THIS HAMILTONIAN REQUIRES b TO KNOW COEFFICIENTS
        if b is None:   b = numpy.zeros(self.d)

        # δ == 0 CONTRIBUTION
        δ = (0,)*self.d
        AM, BM = _orbital_groups(self.model, δ, self.ansatz, self.qreg)

        # δ != 0 INTERACTIONS
        δs = []
        for δ in itertools.product((-1,0,1), repeat=self.d):
            if δ == (0,)*self.d:                    continue
            if not tuple(-δ_ for δ_ in δ) in δs:    δs.append(δ)

        for δ in δs:
            # CALCULATE SITE REGISTER CONTRIBUTIONS
            PHASE = complex(numpy.exp( 2*numpy.pi*1j * (b @ δ)))

            # FOLD ADJACENT INTERACTIONS INTO ORIGINAL MEASUREMENT GROUPS
            AMδ, BMδ = _orbital_groups(self.model, δ, self.ansatz, self.qreg)
            for g, gp in enumerate(AM):
                for term in AMδ[g]:
                    c =  2 * PHASE.real * term.c
                    gp.append(qctb.pauli.Term(term.z, c))
            for g, gp in enumerate(BM):
                for term in BMδ[g]:
                    c = -2 * PHASE.imag * term.c
                    gp.append(qctb.pauli.Term(term.z, c))

        groups = [*AM, *BM]

        # APPEND MEASUREMENTS
        for group in groups:
            group.circuit.append(cirq.measure(*self.qreg))

        return groups

    def energy(self, b, l, angles, computer):
        self.groups = self._construct_groups(b)

        a = self.model.a

        levels = { level(q):  int(l == q) for q in range(self.M) }
        phases = { phase(α): 2*(a[α] @ b) for α in range(self.M) }

        parameters = {**angles, **levels, **phases}
        return super().energy(parameters, computer)

register(OrbitalHamiltonian)

class SiteHamiltonian(Hamiltonian):
    id = "Direct_MN"

    def __init__(self, model, n):
        self.n = n
        self.N = numpy.array([2**n_ for n_ in n])
        super().__init__(model, self.id, {"n": n})

    def _construct_ansatz(self):
        n = self.n
        mreg = cirq.LineQubit.range(self.M)                 # ORBITAL REGISTER
        nreg = []
        qreg = mreg.copy()

        q0 = self.M
        for i in range(self.d):
            nreg.append(cirq.LineQubit.range(q0,q0+n[i]))   # SITE REGISTERS
            qreg += nreg[i]
            q0 += n[i]

        ansatz = cirq.Circuit()

        ######################################################################
        #                          ORBITAL REGISTER
        # PREPARE BASIS STATE δ_lq
        for q in range(self.M):
            δ_lq = sympy.Symbol(level(q))
            ansatz.append((cirq.X**δ_lq).on(qreg[q]), EARLIEST)
        # INTRODUCE GIVENS ROTATIONS
        for l in reversed(range(self.M-1)):
            for q in range(l,self.M-1):
                θ = sympy.Symbol(angle(0,l,q))
                φ = sympy.Symbol(angle(1,l,q))
                ansatz.append( qctb.circuit.A(θ,φ).on(mreg[q], mreg[q+1]) )
        # MODIFY PHASE ACCORDING TO ATOM LOCATION
        for q in range(self.M):
            ab2_α = sympy.Symbol(phase(q))
            ansatz.append((cirq.Z**ab2_α).on(mreg[q]))

        ######################################################################
        #                          SITE REGISTERS
        for i in range(self.d):
            # PREPARE BASIS STATE p_iq
            for q in range(n[i]):
                p_ie = sympy.Symbol(mmntm(i, n[i]-1-q))
                ansatz.append((cirq.X**p_ie).on(nreg[i][q]), EARLIEST)
            # APPLY QUANTUM FOURIER TRANSFORM
            ansatz.append(qctb.circuit.LinearQFT(n[i]).on(*nreg[i]))

        self.mreg = mreg
        self.nreg = nreg

        return ansatz, qreg

    def _construct_groups(self):
        n = self.n

        # δ == 0 CONTRIBUTION
        δ = (0,)*self.d
        groups, _ = _orbital_groups(self.model, δ, self.ansatz, self.mreg)

        # δ != 0 INTERACTIONS
        δs = []
        for δ in itertools.product((-1,0,1), repeat=self.d):
            if δ == (0,)*self.d:                    continue
            if not tuple(-δ_ for δ_ in δ) in δs:    δs.append(δ)

        for δ in δs:
            # CALCULATE ORBITAL REGISTER CONTRIBUTIONS
            AM, BM = _orbital_groups(self.model, δ, self.ansatz, self.mreg)

            # CALCULATE SITE REGISTER CONTRIBUTIONS
            AN, BN = [], []
            for i in range(self.d):
                if δ[i] == 0:
                    AN.append([[qctb.pauli.Term(0,1)]])
                    BN.append([[]])
                else:
                    AN_ = qctb.pauli.AdjacentSets(n[i], 0)
                    AN_ = qctb.pauli.XY2IZ( AN_, 0)
                    AN.append(AN_)

                    BN_ = qctb.pauli.AdjacentSets(n[i], 1)
                    BN_ = qctb.pauli.XY2IZ( BN_, 1)
                    BN_ = [
                        [qctb.pauli.Term(term.z, term.c*δ[i]) for term in group]
                            for group in BN_
                    ]
                    BN.append(BN_)

            # MERGE ORBITAL AND SITE REGISTERS
            for f in range( 1<<(1+self.d) ):    # ENCODE PARITY STRINGS (A OR B)
                if qctb.binary.parity(f):    continue    # SKIP IMAGINARY TERM
                phase = (-1)**((qctb.binary.weight(f)>>1)&1) # 2 B INTRODUCE -1

                for orbital_gp in (BM if f & 1 else AM):
                    for ix in itertools.product(
                        *(range(len(BN[i])) for i in range(self.d))
                    ):
                        merged_gp = qctb.pauli.Group(cirq.Circuit(
                            orbital_gp.circuit,
                            ( qctb.circuit.XYRotation(
                                n[i],
                                ix[i],
                                (f>>(1+i)) & 1,     # PARITY OF ITERATOR
                            ).on(*self.nreg[i]) for i in range(self.d) ),
                        ))

                        # NOW WE NEED TO MULTIPLY THE GROUPS
                        # ie. for each (1+d)-tuple of Terms, multiply c and combine z
                        for terms in itertools.product(*[
                            orbital_gp,
                            *(
                                BN[i][ix[i]] if (f>>(1+i)) & 1 else AN[i][ix[i]]
                                    for i in range(self.d)
                            ),
                        ]):
                            z = terms[0].z
                            c = terms[0].c * phase * 2

                            shift = self.M
                            for i in range(self.d):
                                z += terms[1+i].z << shift
                                c *= terms[1+i].c
                                shift += n[i]

                            if not numpy.isclose(c, 0):
                                merged_gp.append(qctb.pauli.Term(z,c))

                        groups.append(merged_gp)

        # APPEND MEASUREMENTS
        for group in groups:
            group.circuit.append(cirq.measure(*self.qreg))

        return groups

    def energy(self, b, l, angles, computer):
        a = self.model.a

        p = (b * self.N).astype(int)

        levels = { level(q):  int(l == q) for q in range(self.M) }
        phases = { phase(α): 2*(a[α] @ b) for α in range(self.M) }

        mmntms = {
            mmntm(i,e): (p[i]>>e) & 1
                for i in range(self.d)
                for e in range(self.n[i])
        }

        parameters = {**angles, **levels, **phases, **mmntms}
        return super().energy(parameters, computer)

register(SiteHamiltonian)




##############################################################################
#                              HELPER METHODS


def _orbital_groups(model, δ, ansatz, qreg):
    M = model.M
    t = model.t

    # PREPARE GROUPS
    Z = qctb.pauli.Group(cirq.Circuit(
        ansatz,
    ))

    XX = qctb.pauli.Group(cirq.Circuit(
        ansatz,
        qctb.circuit.PauliRotation(cirq.Z, cirq.X).on_each(*qreg),
    ))

    YY = qctb.pauli.Group(cirq.Circuit(
        ansatz,
        qctb.circuit.PauliRotation(cirq.Z, cirq.Y).on_each(*qreg),
    ))

    XY = [
        qctb.pauli.Group(cirq.Circuit(
            ansatz,
            qctb.circuit.PauliRotation(cirq.Z, cirq.X).on(qreg[α]),
            qctb.circuit.PauliRotation(cirq.Z, cirq.Y).on_each(*qreg[α+1:]),
        )) for α in range(M-1)
    ]

    YX = [
        qctb.pauli.Group(cirq.Circuit(
            ansatz,
            qctb.circuit.PauliRotation(cirq.Z, cirq.Y).on(qreg[α]),
            qctb.circuit.PauliRotation(cirq.Z, cirq.X).on_each(*qreg[α+1:]),
        )) for α in range(M-1)
    ]

    # CALCULATE TERMS
    for α in range(M):
        z = (1 << α)
        c  = t[α].get((α,*δ),0) / 2
        if not numpy.isclose(c, 0):
            Z.append(qctb.pauli.Term(0,  c))
            Z.append(qctb.pauli.Term(z, -c))

        for β in range(α+1, M):
            z = (1 << α) + (1 << β)

            c  = (t[β].get((α,*δ),0) + t[α].get((β,*δ),0)) / 4
            if not numpy.isclose(c, 0):
                XX.append(qctb.pauli.Term(z, c))
                YY.append(qctb.pauli.Term(z, c))

            c  = (t[β].get((α,*δ),0) - t[α].get((β,*δ),0)) / 4
            if not numpy.isclose(c, 0):
                XY[α].append(qctb.pauli.Term(z, -c))
                YX[α].append(qctb.pauli.Term(z,  c))

    AM = [Z, XX, YY]
    BM = [  *XY,*YX]
    return AM, BM











##############################################################################
#                          PARAMETER SYMBOL NAMES

def angle(axis,l,q):
    letter = "θ" if not axis else "φ"
    return f"{letter}.{l}.{q}"

def level(q):
    return f"δl.{q}"

def phase(α):
    return f"2ab.{α}"

def mmntm(i,e):
    return f"p.{i}.{e}"






def eigenangles(model, b):
    """ Find the angles which produce the eigenvectors,
            as determined by the eigenvectors of Hk.
        This applies to any single-electron Hamiltonian
            using the Givens rotation matrix.
    """
    M = model.M
    E, U = numpy.linalg.eigh( model.Hk(b) )

    # DEFINE GIVENS ROTATION
    def A(θ,φ, q=0, M=2):
        if q == 0 and M == 2:
            return numpy.array([
                [ numpy.cos(θ),    numpy.exp( 1j*φ)*numpy.sin(θ)],
                [ numpy.exp(-1j*φ) * numpy.sin(θ), -numpy.cos(θ)],
            ])
        A_ = numpy.eye(M, dtype=complex)
        A_[q:q+2,q:q+2] = A(θ,φ)
        return A_

    # INITIALIZE TO ALL ZEROS
    angles = {
        angle(axis,l,q): 0
            for axis in range(2)
            for l in range(M-1)
            for q in range(l, M-1)
    }

    # CALCULATE ANGLES ONE BY ONE
    for l in range(M-1):
        for q in reversed(range(l, M-1)):
            φ = numpy.angle(U[q,l]) - numpy.angle(U[q+1,l])
            θ = numpy.arctan2(abs(U[q+1,l]), abs(U[q,l]))
            U = A(θ,φ,q,M) @ U                  # "UN-TWIRL" THE ROTATIONS

            angles[angle(0,l,q)] = θ
            angles[angle(1,l,q)] = φ

    return angles
