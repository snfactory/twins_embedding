# The Twins Embedding of Type Ia Supernovae

This repository includes all of the code used to perform the Twins Embedding analysis in
Boone et al. 2021a and 2021b. This analysis systematically decomposes the
spectra of Type Ia supernovae into their different components. We use manifold learning
to parametrize the intrinsic diversity of Type Ia supernovae, and show how this can be
used to standardize Type Ia supernovae.

This package depends on the kboone/idrtools package to work with data from the Nearby
Supernova Factory. All of the code used for the main analysis is contained within the
`twins_embedding.py` file. The `embedding_generation_*.ipynb` notebooks contains all
of the code used to generate plots and numbers for Boone et al. 2021a (Paper I), and the
`standardization_plots.ipynb` notebook was used to produce all of the results shown in
Boone et al. 2021b (Paper II).

# Usage

The following code can be used to evaluate a pretrained Twins Embedding model:

```
from twins_embedding import TwinsEmbeddingModel

model = TwinsEmbeddingModel()
flux, flux_error = model.evaluate(phase=2., magnitude=0.1, color=0.1, coordinates=[0., 1., 2.])
wave = model.wave
```

In this package, we provide all of the code that was used in the analyses in Boone et al. 2021a
and 2021b. Rerunning the preprocessing steps and estimating the spectra at maximum light requires
the SNfactory internal data release which is not currently publicly available, although this package
does contain the code that was used for these steps. The estimated spectra at maximum light for each supernova were
released with Boone et al. 2021a and can be found [here](https://snfactory.lbl.gov/snf/data/).
These spectra can be used to reproduce all of the results beyond the step of estimating spectra
at maximum light including building the Twins Embedding latent space,
constructing the Twins Embedding model, and performing all
of the standardization analyses in Boone et al. 2021b.

Some of the indicators discussed in Boone et al. 2021a and host properties discussed in
Boone et al. 2021b were extracted from other publicly-available papers, and we do not
have permission to reproduce them in this repository. Contact us if you need help
accessing these data.

# Acknowledgements

The code used to calculate spectral indicators comes from Sam Dixon
(https://github.com/sam-dixon).
