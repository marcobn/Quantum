import itertools
import numpy

import qctb.binary

class Term:
    def __init__(self, z, c):
        self.z = z
        self.c = c

    def __str__(self):
        return f"{self.c}*{paulistring(self.z)}"

class Group(list):
    def __init__(self, circuit):
        self.circuit = circuit

    def __str__(self):
        return str(self.circuit) + "\n" + "\n".join(str(term) for term in self)

def paulistring(z, n=None, ZERO="I", ONE="Z"):
    if not n: n = qctb.binary.length(z)
    return "".join(ONE if (z >> i) & 1 else ZERO for i in range(n))

def groupstring(terms, n=None, ZERO="I", ONE="Z"):
    if not n:   n = max(qctb.binary.length(term.z) for term in terms)
    return " + ".join(
        f"{term.c}*{paulistring(term.z,n,ZERO,ONE)}" for term in terms
    )



##############################################################################
#                            BASIC CALCULATIONS


def estimate(z, counts):
    # SAY z REPRESENTS THE PAULI STRING I0 Z1 I2 Z3.
    #   THEN z = 2**1 + 2**3
    # SAY ix IN counts REPRESNTS THE MEASUREMENT 1_0 1_1 0_2 1_3.
    #   THEN ix = 2**0 + 2**1 + 2**3
    if z == 0:  return 1        # THE VERY SPECIAL EXPECTATION VALUE ⟨I⟩
    # n = max(length(term.z) for term in terms)
    E = 0
    for ix in counts:
        E += counts[ix] * (-1)**qctb.binary.parity(z & ix)
    return E / sum(counts.values())



##############################################################################
#                   UTILITIES FOR NEAREST-NEIGHBOR INTERACTION

def XYSet(n,k):
    """ Create a group where "pauli" integer represents locations of Y. """
    terms = []
    for ix in itertools.combinations(range(n),k):
        z = sum((1<<i) for i in ix)
        terms.append( Term(z, 1) )
    return terms

def ΠrSet(r, parity):
    """ Generate XYSets and edit them a bit. """
    terms = []
    for k in range((r-parity)//2 + 1):
        xyset = XYSet(r, 2*k+parity)
        for term in xyset:
            term.c *= (-1)**(k+parity) / 2**r
        terms.extend(xyset)
    return terms

def AdjacentSets(n, parity):
    groups = []
    for r in range(n):
        xset = ΠrSet(r, parity)
        for term in xset:
            term.z = (term.z << 1)      # PREPEND X
            term.c /= 2 if (r < n-1) else 1

        # ADD IN |1⟩⟨0|, WHICH HAS EFFECT OF DOUBLING xset AND CANCELING yset
        if r == n-1:
            groups.append(xset)
            continue

        yset = ΠrSet(r, parity^1)
        for term in yset:
            term.c /= 2 * (-1)**(1^parity)
            term.z = (term.z << 1) + 1  # PREPEND Y

        groups.append(xset+yset)

    # PADDING ON THE LEFT WITH I's IS UNDERSTOOD

    return groups


def XY2IZ(groups, parity):
    # CONVERTS XY WORD INTO IZ WORD WITH {τ}
    n = len(groups)
    izgroups = []
    for r, group in enumerate(groups):
        mask = (1<<(r+1)) - 2   # 1..10

        izterms = []
        for xyterm in group:
            wt = qctb.binary.weight( xyterm.z & mask )  # WEIGHT OF Y's (PAST 0)
            p = wt & 1                                  # PARITY OF Y's (PAST 0)
            k = (wt >> 1) & 1                           # PARITY OF PAIRS

            sgn = (-1) ** (1+k^p) if parity else (-1) ** ( 1+k )

            z = xyterm.z | 1
            z <<= (n-1-r)       # THIS IS WHERE YOU PAD ON THE LEFT WITH I's

            izterms.append( Term(z, xyterm.c * sgn) )
        izgroups.append(izterms)
    return izgroups
