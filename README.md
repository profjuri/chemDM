This is a machine learning tool collection to encode molecules in the SMILES and SELFIES language, as well as uising the trained models for property prediction 

The function files are in the .py files, while the ipynb are used to evaluate, call, and train the models

chemestry_vae_selfies.py contains a VAE model and training loops calling settings.yml and trains a VAE with a RNN as decoder

chemestry_vae_symmetric.py contains a VAE model that is a symmetric layer model for the encoder and decoder

chemestry_perceptron contains a perceptron model and auxilary funcitons used to train a perceptron on the latent space and property vectors 

auxiliary_funcitons contains useful tensor and vae functions that can be used in the jupiter notebooks

use the TrainingVAE to train the VAE models

ValidationVAE reads in a trained VAE model and demonstrates how to use it, it calculates the similarity score

use PropertyReconstruction to train the perceptron and validate the trained model by property reconstruction

use the LatentSpacePlots notebook to visulaize the latent space, and its connection to property data

use the xyzFileReader to read in .xyz files and convert to csv for faster evaluation

Note that the .xyz files for the QM9 dataset show only an example file and the full dataset can be downloaded from: https://springernature.figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904/4

The homo-lumo gap and the dipole moment data are converted and provided as a csv file for the applied studies 