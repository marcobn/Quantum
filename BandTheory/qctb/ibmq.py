""" Interface with IBMQ Experience. """

import warnings

import qiskit
import mitiq

# DISABLE MULTIPROCESSING
import os
os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'

class QuantumComputer:
    """ Manage connection to a quantum computer with specific settings.

    Attributes
    ----------
    qpu: str
        name of IBMQ computer to use, or None to simulate
    qvm: bool
        True simulates specified qpu. set to False to run real trial
    shots: int
        number of shots per basis measurement, or None for statevector sim
    zne_scale: list of float
        list of scales to run zero noise extrapolation, or None to skip
    autocalibrate: bool
        set to True to apply readout error correction
    pretranspile: bool
        whether to transpile circuit onto architecture of qpu
    gate_error: bool
        whether to apply noise model for gate error
    readout_error: bool
        whether to apply noise model for readout error
    thermal_relaxation: bool
        whether to apply noise model for thermal relaxation

    """

    def __init__(self,
        qpu=None,
        qvm=True,
        shots=2**13,
        zne_scale=[1.0,3.0,5.0],
        autocalibrate=True,
        pretranspile=True,
        gate_error=True,
        readout_error=True,
        thermal_relaxation=True,
    ):
        """ Initialize the QuantumComputer.

        Parameters
        ----------
        qpu: str
            name of IBMQ computer to use, or None to simulate
        qvm: bool
            True simulates specified qpu. set to False to run real trial
        shots: int
            # of shots per basis measurement, or None for statevector sim
        zne_scale: list of float
            list of scales to run zero noise extrapolation, or None to skip
        autocalibrate: bool
            set to True to apply readout error correction
        pretranspile: bool
            whether to transpile circuit onto architecture of qpu
        gate_error: bool
            whether to apply noise model for gate error
        readout_error: bool
            whether to apply noise model for readout error
        thermal_relaxation: bool
            whether to apply noise model for thermal relaxation

        """
        self.qpu                = qpu
        self.qvm                = qvm
        self.shots              = shots
        self.autocalibrate      = autocalibrate
        self.zne_scale          = zne_scale
        self.pretranspile       = pretranspile
        self.gate_error         = gate_error
        self.readout_error      = readout_error
        self.thermal_relaxation = thermal_relaxation

        # GET BACKEND
        if qpu:
            try:
                tokenfile = "token.txt"
                with open(tokenfile) as tokenfile:
                    with warnings.catch_warnings(): # SUPPRESS TIMESTAMP WARNING
                        warnings.simplefilter("ignore")
                        qiskit.IBMQ.enable_account(tokenfile.read().strip())
            except FileNotFoundError:
                raise FileNotFoundError(
                    "token.txt is missing. Get it from IBMQ Experience website."
                )
            except:
                pass
            provider = qiskit.IBMQ.get_provider(group='open')
            backend = provider.get_backend(qpu)
        else:
            backend = qiskit.Aer.get_backend('qasm_simulator')

        # GET NOISE MODEL
        if qpu and qvm:
            model = qiskit.providers.aer.noise.NoiseModel.from_backend(backend,
                gate_error=gate_error,
                readout_error=readout_error,
                thermal_relaxation=thermal_relaxation,
            )
            backend = qiskit.Aer.get_backend('qasm_simulator')
        else:
            model = None

        # GET NOISE MITIGATION
        from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
        fitter = CompleteMeasFitter if autocalibrate else None

        # READY STATEVECTOR SIMULATOR
        if not (qpu or qvm) or not shots:
            self._svm = True    # FLAG FOR STATEVECTOR SIMULATOR
            backend = qiskit.Aer.get_backend('statevector_simulator')
            model = None
            fitter = None
        else:
            self._svm = False

        # CREATE THE INSTANCE
        self.qi = qiskit.aqua.QuantumInstance(
            backend,
            shots=shots,
            noise_model=model,
            measurement_error_mitigation_cls=fitter,
        )

    def transpile(self, ct):
        """ Tranpsile circuit into form consistent with qubit architecture.

        If `qpu` is None, or if `pretranspile` is False,
            gates are decomposed into 'u3' and 'cx'.

        Parameters
        ----------
        ct: qiskit.QuantumCircuit
            the circuit to transpile

        Returns
        -------
        ctᵀ: qiskit.QuantumCircuit
            the transpiled circuit
        """
        if self.qpu and self.pretranspile:  # TRANSPILE CIRCUIT FOR BACKEND
            provider = qiskit.IBMQ.get_provider(group='open')
            backend = provider.get_backend(self.qpu)
            return qiskit.transpile(ct, backend)
        return qiskit.transpile(ct, basis_gates=['u3','cx'])

    def expectation(self, op, template=None, filename=None):
        """ Calculate expectation value of qiskit aqua Operator.

        Parameters
        ----------
        op: qiskit.aqua.operators.OperatorBase
            One of the following:
                1) A ComposedOp of StateFn(measurement=True) and CircuitStateFn
                2) A SummedOp of such ComposedOps
        template: string
            string containing patterns for `E` and `scale`, like "{E} {scale}"
        filename: string
            filename to append rows of data

        Returns
        -------
        E: float
            the expectation value

        """
        # USE STATEVECTOR
        if self._svm:
            matrixop = qiskit.aqua.operators.MatrixExpectation().convert(op)
            E = matrixop.eval().real

            # WRITE TO FILE
            if template and filename:
                with open(filename, 'a') as file:
                    print (template.format(E=E, scale=1.0), file=file)

            return E

        # RUN ZPE
        if self.zne_scale:
            # IF ONLY ONE MEASUREMENT GROUP, WRAP INTO SummedOp
            if isinstance(op, qiskit.aqua.operators.ComposedOp):
                op = qiskit.aqua.operators.SummedOp([op])

            # EXTRACT INITIAL CIRCUITS FOR EACH MEASUREMENT GROUP, AND TRANSPILE
            cts = [
                self.transpile(braket[1].to_circuit())
                for braket in op
            ]
            # NOTE: TRANPSILATION REQUIRED TO ENSURE mitiq CAN FOLD CIRCUIT

            # RUN EACH SCALED CIRCUIT
            expvals = []
            for scale in self.zne_scale:
                # CONSTRUCT SCALED OPERATOR
                composedops = []
                for i, braket in enumerate(op):
                    bra = op.coeff * braket.coeff * braket[0]
                    scaledct = mitiq.zne.scaling.fold_global(cts[i], scale)
                    ket = qiskit.aqua.operators.CircuitStateFn(scaledct)
                    composedops.append( bra @ ket )
                scaledop = qiskit.aqua.operators.SummedOp(composedops)

                # MEASURE SCALED OPERATOR
                sampled = qiskit.aqua.operators.CircuitSampler(
                    self.qi,
                ).convert(scaledop)
                E = sampled.eval().real

                # WRITE TO FILE
                if template and filename:
                    with open(filename, 'a') as file:
                        print (template.format(E=E, scale=scale), file=file)

                expvals.append(E)

            # EXTRAPOLATE TO ZERO NOISE
            E = mitiq.zne.inference.RichardsonFactory.extrapolate(
                self.zne_scale,
                expvals,
            )

            # WRITE TO FILE
            if template and filename:
                with open(filename, 'a') as file:
                    print (template.format(E=E, scale=0.0), file=file)

            return E

        # RUN ON BACKEND
        sampled = qiskit.aqua.operators.CircuitSampler(
            self.qi,
        ).convert(op)

        # EXTRACT RESULTS
        E = sampled.eval().real

        # WRITE TO FILE
        if template and filename:
            with open(filename, 'a') as file:
                print (template.format(E=E, scale=1.0), file=file)

        return E

    def toJSON(self):
        """ Serialize this object into a JSON-compatible dict object. """
        return {
            "qpu":                  self.qpu,
            "qvm":                  self.qvm,
            "shots":                self.shots,
            "zne_scale":            self.zne_scale,
            "autocalibrate":        self.autocalibrate,
            "pretranspile":         self.pretranspile,
            "gate_error":           self.gate_error,
            "readout_error":        self.readout_error,
            "thermal_relaxation":   self.thermal_relaxation,
        }

    @staticmethod
    def fromJSON(json):
        """ Reconstruct an object from a JSON-compatible dict object. """
        return QuantumComputer(
            qpu                     = json['qpu'],
            qvm                     = json['qvm'],
            shots                   = json['shots'],
            zne_scale               = json['zne_scale'],
            autocalibrate           = json['autocalibrate'],
            pretranspile            = json['pretranspile'],
            gate_error              = json['gate_error'],
            readout_error           = json['readout_error'],
            thermal_relaxation      = json['thermal_relaxation'],
        )

SVM = QuantumComputer(
    qpu=None,
    qvm=True,
    shots=None,
    zne_scale=None,
    autocalibrate=False,
    pretranspile=False,
    gate_error=False,
    readout_error=False,
    thermal_relaxation=False,
)

QVM = QuantumComputer(
    qpu=None,
    qvm=True,
    shots=2**13,
    zne_scale=None,
    autocalibrate=False,
    pretranspile=False,
    gate_error=False,
    readout_error=False,
    thermal_relaxation=False,
)

RAW = lambda qpu, qvm=True: QuantumComputer(
    qpu=qpu,
    qvm=qvm,
    shots=2**13,
    zne_scale=None,
    autocalibrate=False,
    pretranspile=False,
    gate_error=True,
    readout_error=True,
    thermal_relaxation=True,
)

CAL = lambda qpu, qvm=True: QuantumComputer(
    qpu=qpu,
    qvm=qvm,
    shots=2**13,
    zne_scale=[1.0,3.0,5.0],
    autocalibrate=True,
    pretranspile=False,
    gate_error=True,
    readout_error=True,
    thermal_relaxation=True,
)
