class Computer:
    def __init__(self, id, parameters):
        """ Encapsulate orbitals and hopping parameters.

        Parameters
        ----------
        id: str
            specify which model to use
        parameters: dict
            model-specific parameters

        """
        self.id         = id
        self.parameters = parameters

    def estimate(self, group, parameters=None):
        return NotImplemented

    def to_json(self):
        """ Construct serializable dict from Model object. """
        return {
            "id": self.id,
            "parameters": self.parameters,
        }

    def from_json(json):
        """ Construct Model object from serializable dict `json`. """
        return REGISTRY[json["id"]](**json["parameters"])

REGISTRY = {}
def register(cls):
    REGISTRY[cls.id] = cls
