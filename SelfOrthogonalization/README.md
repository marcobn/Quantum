# Self Orthogonalization

This project is the code base behind arXiv:2204.04361.
We present an alternative to VQD (a method to iteratively locate excited states) which uses an ansatz enforcing orthogonality to previously-located states, rather than adding an overlap term to the variational cost-function.

### Dependencies

- `python 3.8`
- `numpy 1.21.2`
- `scipy 1.7.1`
- `sympy 1.8`
- `cirq 0.13.1`
- `openfermion 1.3.0`
- `matplotlib 3.3.1`

### Code Description

- `generate_bandstructure.py` - script to generate band structures (either reciprocal basis or compact basis)
- `generate_dissociation.py` - script to generate hydrogen dissociation curve
- `plot_bandstructure.py` - script to generate a plot of a band structure from previously calculated data
- `plot_dissociation.py` - script to generate a plot of a hydrogen dissociation curve from previously calculated data
- `ortho` - all library code; I appear to have *not* written documenation for some reason, but it should be relatively easy to follow the import chains
- All the other scripts are "sandbox" files in which I validated and confirmed the method. They are reasonably well-documented and may be of tutorial interest to the patient student.

### Data Description

- `json/*` - Serializations of the `ortho.experiment.Experiment` object generated with different physical models, dependent variables, and computational frameworks. The `r` attribute in each file name refers to the number of Trotter steps used to implement the UCC-like ansatz. Other parts of the file name and the json attributes should be fairly self-explanatory.
