# Direct Space

This project is the code base behind doi:10.21203/rs.3.rs-1318951.
Starting from a tight-binding Hamiltonian describing the hopping parameters between each atomic orbital, we provide a strategy for directly preparing the Bloch function for a particular electron momentum (ie. k-point) in a real-space basis, using a hybrid first/second-quantized qubit mapping.
This library also re-implements the reciprocal space approach from `BandTheory` in a much cleaner way, to the point where you should really probably ignore `BandTheory` as much as you can.


### Dependencies

- `python 3.8`
- `numpy 1.21.2`
- `scipy 1.7.1`
- `sympy 1.8`
- `cirq 0.13.1`
- `matplotlib 3.3.1`

### Code Description

- `generate_bandstructure.py` - script to generate band structures
- `load_bandstructure.py` - script to generate a plot of a band structure from previously calculated data
- `qctb` - all library code; I appear to have *not* written documenation for some reason, but it should be relatively easy to follow the import chains
- `validate_project.py` - script to import all library code and ensure there are no syntax errors
- `bethe.py`, `hubbardsandbox.py`, `test_hubbard.py`, `wieblusandbox.py`, - scripts exploring the (failed) attempt to adapt the hybrid qubit mapping for a 1D Hubbard chain



### Data Description

- `json/*` - Serializations of the `qctb.bandstructure.BandStructure` object generated with different physical models, reciprocal-space paths, and computational frameworks. The file names and the json attributes should be fairly self-explanatory.
