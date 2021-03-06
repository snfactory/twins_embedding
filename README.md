# The Twins Embedding of Type Ia Supernovae

This repository includes all of the code used to perform the Twins Embedding analysis in
Boone et al. 2021a and 2021b. This analysis systematically decomposes the
spectra of Type Ia supernovae into their different components. We use manifold learning
to parametrize the intrinsic diversity of Type Ia supernovae, and show how this can be
used to standardize Type Ia supernovae.

This package depends on the kboone/idrtools package to work with data from the Nearby
Supernova Factory. All of the code used for the main analysis is contained within the
`twins_embedding.py` file. The `embedding_generation_*.ipynb` notebooks contains all
of the code used to generate plots and numbers for Article I, and the
`standardization_plots.ipynb` notebook was used to produce all of the results shown in
Article II.

# Usage

The following code can be used to evaluate a pretrained Twins Embedding model:

```
from twins_embedding import TwinsEmbeddingModel

model = TwinsEmbeddingModel()
flux, flux_error = model.evaluate(phase=2., magnitude=0.1, color=0.1, coordinates=[0., 1., 2.])
wave = model.wave
```

# Acknowledgements

The code used to calculate spectral indicators comes from Sam Dixon
(https://github.com/sam-dixon).
