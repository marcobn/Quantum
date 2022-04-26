import numpy
import cirq

from qctb.computer.computer import Computer, register
import qctb.pauli


class BasicShotSimulator(Computer):
    id = "ShotSimulator"

    def __init__(self, shots=1024):
        super().__init__(
            self.id,
            {
                'shots': shots,
            },
        )

        self.simulator = cirq.Simulator()

    def estimate(self, group, parameters=None):
        if len(group) == 0:     return 0

        qreg = sorted(group.circuit.all_qubits())
        n = len(qreg)
        resolver = cirq.ParamResolver(parameters) if parameters else None
        result = self.simulator.run(
            group.circuit,
            param_resolver=resolver,
            repetitions=self.parameters['shots'],
        )
        counts = result.histogram(key=qreg)

        # SWAP TO LITTLE ENDIAN FOR CONSISTENCY
        #   (ie. MEASUREMENT ON QUBIT i CORRESPONDS TO `ix & 2**i`)
        counts = { qctb.pauli.reverse(ix, n): counts[ix] for ix in counts }

        E = 0
        for term in group:
            E += term.c * qctb.pauli.estimate(term.z, counts)
        return E
register(BasicShotSimulator)



class StatevectorSimulator(Computer):
    id = "StatevectorSimulator"

    def __init__(self):
        super().__init__(
            self.id,
            {},
        )

        self.simulator = cirq.Simulator()
        self.mstripper = RemoveMeasurements()

    def estimate(self, group, parameters=None):
        if len(group) == 0:     return 0

        # STRIP MEASUREMENTS FROM CIRCUIT
        circuit = group.circuit.copy()
        self.mstripper.optimize_circuit(circuit)

        # CONSTRUCT PAULI SUM OBJECT
        qreg = sorted(group.circuit.all_qubits())
        paulisum = 0
        for term in group:
            string = cirq.PauliString(
                cirq.Z(qbit) if ((term.z >> q) & 1) else cirq.I(qbit)
                    for q, qbit in enumerate(qreg)
            )
            paulisum += term.c * string


        # CALCULATE EXPECTATION VALUES
        resolver = cirq.ParamResolver(parameters) if parameters else None
        result = self.simulator.simulate_expectation_values(
            circuit, paulisum,
            param_resolver=resolver,
        )[0].real

        return result
register(StatevectorSimulator)


class RemoveMeasurements(cirq.PointOptimizer):
    """ CODE ADAPTED DIRECTLY FROM cirq DOCUMENTATION:
        https://quantumai.google/cirq/transform
    """
    def optimization_at(self, circuit, index, op):
        return cirq.PointOptimizationSummary(
            clear_span=1,
            new_operations=[],
            clear_qubits=op.qubits
        ) if isinstance(op.gate, cirq.MeasurementGate) else None
