""" Implement common circuit elements. """

import numpy
import qiskit

def R(θ,ϕ):
    """ Rotation gate from from Gard et al. 2020, sub-circuit of A.

                -- [ Ry(θ + π/2) ] -- [ Rz(ϕ + π) ] --

    Parameters
    ----------
    θ, ϕ: float or qiskit.circuit.Parameter
        rotation angles, generally parameterized

    Returns
    -------
    ct: qiskit.QuantumCircuit

    """
    ct = qiskit.QuantumCircuit(1)
    ct.ry(θ + numpy.pi/2, 0)
    ct.rz(ϕ + numpy.pi,   0)
    return ct

def A(θ,ϕ):
    """ Entangling gate from Gard et al. 2020 preserving particle number.

                    --o-- [ R† ] --⊕-- [ R ] --o--
                      |            |           |
                    --⊕------------o-----------⊕--

    The R gate rotates through a qubit's Bloch sphere.

    Parameters
    ----------
    θ, ϕ: float or qiskit.circuit.Parameter
        rotation angles, generally parameterized

    Returns
    -------
    ct: qiskit.QuantumCircuit

    """
    ct = qiskit.QuantumCircuit(2)
    ct.cnot(0,1)
    ct.compose( R(θ,ϕ).inverse(), [0], inplace=True)
    ct.cnot(1,0)
    ct.compose( R(θ,ϕ),           [0], inplace=True)
    ct.cnot(0,1)
    return ct

def ansatz(M, linear=True):
    """ Ansatz spanning 2**M-dimensional Hilbert space with occupation number 1.

                    -- ⌈‾‾‾⌉ ------------------------
                       | A |
                    -- ⌊___⌋ ---- ⌈‾‾‾⌉ -------------
                                  | A |
                    ------------- ⌊___⌋ ---- ⌈‾‾‾⌉ --
                                             | A |
                    ------------------------ ⌊___⌋ -- ...

    The A gate is drawn from Gard et al. 2020; each contributes two parameters.

    Parameters
    ----------
    M: int
        the number of qubits
    linear: bool
        use successive A gates on each pair. Otherwise, uses recursive strategy.

    Returns
    -------
    ct: qiskit.QuantumCircuit
        a circuit with 2(M-1) parameters spanning singly-occupied space
    """
    # LINEAR STRATEGY: SUCCESSIVE A GATES
    if linear:
        ct = qiskit.QuantumCircuit(M)
        ct.x(0)
        for α in range(0,M-1):
            θ = qiskit.circuit.Parameter(f"θ{α}")
            ϕ = qiskit.circuit.Parameter(f"ϕ{α}")
            ct.compose (A(θ,ϕ), [α,α+1], inplace=True)
        return ct

    # RECURSIVE STRATEGY: ROUNDS OF "EVERY SPANNED WIRE" A's WITH ANOTHER
    ct = qiskit.QuantumCircuit(M)
    ct.x(0)

    ctr = 1 # TRACK FIRST INDEX REQUIRING TREATMENT

    while ctr < M:
        for α in range(0,ctr):
            if ctr+α == M: break  # FINISHED
            # ADD A GATE
            θ = qiskit.circuit.Parameter(f"θ{ctr+α-1}")
            ϕ = qiskit.circuit.Parameter(f"ϕ{ctr+α-1}")
            ct.compose (A(θ,ϕ), [α,ctr+α], inplace=True)

        ctr *= 2
    return ct

def bind(ct, x):
    """ Bind a list of parameters to a circuit or operator.

    Parameters
    ----------
    ct: qiskit.QuantumCircuit
        parameterized quantum circuit
    x: list of float
        list of parameters, of same length as number of parameters in ct

    Returns
    -------
    bound_ct: qiskit.QuantumCircuit
        copy of ct with its parameters (sorted lexicographically) to x

    """
    # SORT PARAMETER LIST
    parameters = sorted(
        ct.parameters,
        key=lambda parameter: parameter.name,
    )
    # NOTE: This puts θ10 before θ2. Be wary when reading results.

    # BIND PARAMETERS
    return ct.bind_parameters({
        parameters[i]: x[i] for i in range(len(x))
    })

def linearQFT(n):
    """ Implement QFT circuit optimized for linear qubit architecture.

    Parameters
    ----------
    n: int
        the number of qubits

    Returns
    -------
    ct: qiskit.QuantumCircuit
        performs QFT using only nearest-neighbor entanglements

    """
    ct = qiskit.QuantumCircuit(n)
    for q in range(1,n):
        ct.h(0)
        for r in range(n-q):
            ct.cu1(numpy.pi/2**(r+1), r+1, r)
            ct.swap(r, r+1)
    ct.h(0)
    return ct



def blochansatz(M, n, p, linear=True):
    """ Combine ansatz with QFT at a specific kpt on site register.

    Parameters
    ----------
    M: int
        number of qubits in orbital register
    n: list(int) of length D
        list of number of qubits in each site register
    p: list(int)
        list of quantized momentum vector: ka=2π/N p
    linear: bool
        use linearQFT. Otherwise, use qiskit.circuit.library.QFT

    Returns
    -------
    ct: qiskit.QuantumCircuit
        a circuit with 2(M-1) parameters spanning singly-occupied orbital space
        and applying QFT to each site register

    """
    # INITIALIZE CIRCUIT
    ct = qiskit.QuantumCircuit(M+sum(n))

    # APPLY ANSATZ TO ORBITAL REGISTER, QUBITS [0,m)
    ct.compose(
        ansatz(M, linear),
        qubits=range(M),
        inplace=True,
    )

    q1 = M                      # INITIALIZE QUBIT RANGE
    # FOR EACH SITE REGISTER
    for i in reversed(range(len(n))):

        # UPDATE QUBIT RANGE
        q0 = q1                 # FIRST QUBIT: WHERE WE ENDED BEFORE
        q1 += n[i]              #  LAST QUBIT: n QUBITS LATER

        # PREPARE STATE |p⟩ IN SITE REGISTER
        mask = p[i]
        for r in range(q0,q1):
            if mask & 1:    ct.x(r)
            mask >>= 1

        # APPLY QUANTUM FOURIER TRANSFORM CIRCUIT
        qft = linearQFT(n[i]) if linear else qiskit.circuit.library.QFT(n[i])
        ct.compose(
            qft,
            qubits=range(q1-1,q0-1,-1), # QFT MADE FOR OPPOSITE ENDIAN, SO FLIP
            inplace=True,
        )

    return ct


def sitecircuit(n,ni):
    """ Prepare site register with QFT.

    Parameters
    ----------
    n: int
        size of specific site register
    ni: int
        dimension index (required for distinguishing parameters)

    Returns
    -------
    ct: qiskit.QuantumCircuit
        a circuit with n qubits and parameters controlling X gates on each qubit
    """
    # INITIALIZE CIRCUIT
    ct = qiskit.QuantumCircuit(n)

    # PREPARE STATE |p⟩ IN SITE REGISTER
    for r in range(n):
        p = qiskit.circuit.Parameter(f"p{ni}.{r}")  # rth BIT, ni-th DIMENSION
        ct.rx( p * numpy.pi, r )
        ct.s(r)

    # APPLY QUANTUM FOURIER TRANSFORM CIRCUIT
    qft = linearQFT(n)
    ct.compose(
        qft,
        qubits=range(n),
        inplace=True,
    )

    return ct

def bind_p(ct, p):
    """ Bind discretized kpts to a circuit or operator.

    Parameters
    ----------
    ct: qiskit.QuantumCircuit
        parameterized circuit on site register
        each parameter p[i,r] named by "p{i}.{r}" will be bound
    x: list of int
        number whose binary representation determines parameters
                        p[i] = ∑ p[i,r] * 2**r

    Returns
    -------
    bound_ct: qiskit.QuantumCircuit
        copy of ct with parameters fixed by binary representations of p

    """
    # SORT PARAMETERS INTO CATEGORIES BASED ON SUBSTRING UP TO FIRST .
    p_params = {}
    for param in ct.parameters:
        if not param.name.startswith("p"): continue     # SKIP non-p PARAMETERS
        cls = int(param.name[1:param.name.index(".")])  # DIMENSION INDEX
        if cls in p_params:
            p_params[cls].add(param)
        else:
            p_params[cls] = {param}

    # ASSIGN PARAMETERS BASED ON p LIST
    parametermap = {}
    for i in p_params:

        # SORT PARAMETER LIST
        parameters = sorted(
            p_params[i],
            key=lambda param: int(param.name[param.name.index(".")+1:]),
        )

        # CONVERT ith p TO INTEGER ARRAY OF BINARY REPRESENTATION
        pi = numpy.binary_repr(p[i], width=len(parameters))

        # REGISTER PARAMETER BINDINGS
        parametermap.update({
            parameter: int(pi[j]) for j, parameter in enumerate(parameters)
        })
        # NOTE: p{i}.0 IS MOST SIGNIFCANT BIT OF p[i]

    return ct.bind_parameters(parametermap)

def measure_xy(n, parity):
    """ Circuit for the unitary operator diagonalizing commuting XY strings.

    Parameters
    ----------
    n: int
        length of strings
    parity: int
        0 if commuting strings have even number of Y, or 1 if odd

    Returns
    -------
    ct: qiskit.QuantumCircuit

    """
    # INITIALIZE CIRCUIT
    ct = qiskit.QuantumCircuit(n)

    # HANDLE EDGE CASE
    if n == 1:                      # JUST MEASURE "X" OR "Y"
        if parity:
            ct.z(0); ct.s(0); ct.h(0)
        else:
            ct.h(0)
        return ct

    # APPLY exp(-𝑖π/4 Z0) exp(-𝑖π/4 ?X..X) exp(-𝑖π/4 Z0)
    ct.rz( numpy.pi/2, 0)       # APPLY exp(-𝑖π/4 Z0)
    ###################################################
    if parity:                  # CHANGE QUBIT 0 TO X OR Y BASIS
        ct.z(0); ct.s(0); ct.h(0)
    else:
        ct.h(0)
    for q in range(1,n):        # CHANGE HIGHER QUBITS TO X BASIS
        ct.h(q)
    ###################################################
    for q in range(n-1, 0, -1): # COMPUTE PARITY
        ct.cnot(q,q-1)
    ct.rz( numpy.pi/2, 0 )      # PERFORM ROTATION
    for q in range(n-1):        # UNCOMPUTE PARITY
        ct.cnot(q+1,q)
    ###################################################
    for q in range(1,n):        # UNDO HIGHER QUBITS BASIS ROTATION
        ct.h(q)
    if parity:                  # UNDO QUBIT 0 BASIS ROTATION
        ct.h(0); ct.sdg(0); ct.z(0)
    else:
        ct.h(0)
    ###################################################
    ct.rz( numpy.pi/2, 0)          # APPLY exp(-𝑖π/4 Z0)

    # APPLY EACH exp(-𝑖π/4 X) exp(-𝑖π/4 Z..Z) exp(-𝑖π/4 X)
    for q in range(n-1):
        if q > 0: ct.swap(q-1,q)    # PERMUTE QUBIT 0 UP TO PRIME THIS ROUND
        ct.rx( numpy.pi/2, q+1) # APPLY exp(-𝑖π/4 X)
        ct.cnot(q+1,q)          # APPLY exp(-𝑖π/4 ZZ)
        ct.rz( numpy.pi/2, q )
        ct.cnot(q+1,q)
        ct.rx( numpy.pi/2, q+1) # APPLY exp(-𝑖π/4 X)

    # RE-PERMUTE QUBIT 0 DOWN
    for q in range(n-2, 0, -1):
        ct.swap(q,q-1)

    return ct
