# Band Theory

This project is the code base behind arXiv:2104.03409.
Starting from a tight-binding Hamiltonian describing the hopping parameters between each atomic orbital, we propose the following steps for band theory on a quantum computer:
1. Construct the reciprocal-space single-body Hamiltonian for a specific k-point.
2. Perform VQE/D optimization to obtain a first estimate of each band energy at that k-point.
3. Perform QPE on optimal state to obtain a refined estimate of each band energy at that k-point.

### Dependencies

- `python 3.8` (NOTE: Should work in any Python 3 compatible with f-strings.)
- `numpy 1.21.2` (NOTE: Should work with any modern version of numpy.)
- `qiskit 0.23.1` (NOTE: later versions will absolutely not work.)
- `mitiq 0.10.0` (NOTE: untested on other versions)
- `matplotlib 3.3.1`

### Code Description

- `bandstructure.py` - script to generate band structures using VQE/D
- `plot_bandstructure.py` - script to generate a plot of simple cubic band structure, from previously calculated data
- `plot_graphene.py` - script to generate a plot of hexagonal lattice band structure, from previously calculated data
- `redoband.py` - script to recalculate and refine all the energies from one band structure on another computational framework (eg. obtain QPE refinements, or run operator estimation on a real machine)
- `tools.py` - module facilitating QPE refinements
- `tightbinding` - package which implements specific tightbinding models for various lattices
- `qctb` - the library code implementing all of the quantum computations not explicitly performed by `qiskit`

### Data Description

- `json/*.recap_QPU.json` - First, we generated band structures in a simulated noise model, with ZNE calibration. Then we ran the `redoband.py` script to re-calculate energies (for the already-optimized parameters) on the real quantum machines hosted by IBM. Those re-calculated energies are what you see in these recap files.
