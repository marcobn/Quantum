from qctb.model.model import Model

class Hamiltonian:
    def __init__(self, model, id, parameters):
        self.model             = model
        self.id                = id
        self.parameters        = parameters
        self.M                 = model.M
        self.d                 = model.lattice.d
        self.ansatz, self.qreg = self._construct_ansatz()
        self.groups            = self._construct_groups()

    def _construct_ansatz(self):
        return NotImplemented

    def _construct_groups(self):
        return NotImplemented

    def energy(self, parameters, computer):
        return sum(
            computer.estimate(group, parameters)
                for group in self.groups
        )

    def to_json(self):
        """ Construct serializable dict from Model object. """
        return {
            "id": self.id,
            "parameters": self.parameters,
            "model": self.model.to_json(),
        }

    def from_json(json):
        """ Construct Hamiltonian object from serializable dict `json`. """
        return REGISTRY[json["id"]](
            Model.from_json(json["model"]),
            **json["parameters"],
        )

REGISTRY = {}
def register(cls):
    REGISTRY[cls.id] = cls
