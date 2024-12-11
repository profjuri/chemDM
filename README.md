# `chemDM`

Code for analyzing properties of molecular chemicals in the context of searches for the cosmological dark matter.

## Installation 

We provide an environment specification file for `conda` or `mamba` users at `environment.yml`. With `conda`, an environment is created by `conda env create -f environment.yml`. With `micromamba` the `env` is omitted and a new environment is instead created with `micromamba create -f environment.yml`. This will create a new virtual environemnt named `chemdm`, as specified in the first line of `environment.yml`.

From the top-level directory, you can do `pip install .`

## Usage

This github contains 4 main files: 
  1. ./Models/vae.py-file containing our VAE
  2. ./Models/mlp.py -file containing our MLP
  3. ./Models/bf.py -file containing the brute force algorithm
  4. ./Clustering/clustering.py -file containing the clustering algorithm

Notebooks to run the files are located at:
  1. ./Models/Main.ipynb -notebook to run vae.py
  2. ./Models/Main.ipynb -notebook to run mlp.py
  3. ./Models/Main.ipynb -notebook to run bf.py
  4. ./Clustering/Main.ipynb -notebook to run clustering.py

Settings.yml files to configure the 4 main files are located at:
  1. ./Models/vae_settings.yml
  2. ./Models/mlp_settings.yml
  3. ./Model/bf_settings.yml
  4. ./Clustering/clustering.yml

Input data required to run your own simulations is available at datasets/.DS_Store. Input files should be .csv/.txt and the column containing the SMILES representations should be labelled 'smiles'.

## Citation

If you use this code in your research, please cite this GitHub repo.

## Contributing

If you would like to contribute, please open a new [issue](https://github.com/profjuri/chemDM/issues), and/or be in touch with the [authors](#contact)

## Contact

The code was developed by Juri Smirnov, Carlos Blanco, and Cameron Cook. [Samuel D. McDermott](https://samueldmcdermott.github.io) provided help with packaging.
