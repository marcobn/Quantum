""" Represent specific physical systems in Hamiltonian objects. """

import tightbinding.hamiltonian     as hamiltonian
import tightbinding.chain           as chain
import tightbinding.graphene        as graphene
import tightbinding.silicon         as silicon
import tightbinding.polonium        as polonium

# MAP LABELS TO HAMILTONIAN SUBCLASSES, FOR DESERIALIZATION
subclass = {H.id: H for H in [
    chain.Chain,
    chain.AlternatingChain,
    silicon.Silicon,
    graphene.Graphene,
    polonium.Polonium,
]}


def fromJSON(json):
    """ Reconstruct an object from a JSON-compatible dict object. """
    # VALIDATE LABEL
    id = json['id']
    if not id in subclass: raise ValueError(f"Unsupported Hamiltonian: {id}")

    # EXTRACT PARAMETERS
    parameters = json.copy()
    del parameters['id']
    del parameters['D']
    del parameters['M']

    # CONSTRUCT OBJECT
    return subclass[id](**parameters)
