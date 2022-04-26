import scipy
import numpy

class Result:
    def __init__(self, E, x, nfev, converged=True):
        self.E    = E
        self.x    = x
        self.nfev = nfev
        self.converged = converged

    def to_json(self):
        """ Construct serializable dict from Model object. """
        return {
            "E":            self.E,
            "x":            list(self.x),
            "nfev":         self.nfev,
            "converged":    bool(self.converged),
        }

    def from_json(json):
        """ Construct Model object from serializable dict `json`. """
        return Result(*[json[key] for key in json])


class Optimizer:
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

    def optimize(self, costfn, x0):
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
