# `chemDM`

Code for analyzing properties of molecular chemicals in the context of searches for the cosmological dark matter.

## Installation 

We provide an environment specification file for `conda` or `mamba` users at `environment.yml`. With `conda`, an environment is created by `conda env create -f environment.yml`. With `micromamba` the `env` is omitted and a new environment is instead created with `micromamba create -f environment.yml`.

From the top-level directory, you can do `pip install .`

## Usage

The usage of this code is documented in `notebooks/demo_simulation.ipynb`. A detailed walkthrough of the functions available in this code is in `notebooks/demo_full_pipeline.ipynb`.

A full list of potential inputs is documented in `settings/config.yaml` and you can edit `settings/inputdata.yaml` to reflect your desired simulation settings.

## Citation

If you use this code in your research, please cite this GitHub repo.

## Contributing

If you would like to contribute, please open a new [issue](https://github.com/profjuri/chemDM/issues), and/or be in touch with the [authors](#contact)

## Contact

The code was developed by Juri Smirnov, Carlos Blanco, and Cameron Cook. [Samuel D. McDermott](https://samueldmcdermott.github.io) provided help with packaging.