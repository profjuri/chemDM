import selfies
import torch
import torch.distributions as dist
import data_loader

# useful functions

def translate_selfie(sequence,selfies_alphabet):  

        SELFIESGenerated = ""

        for i in range(len(sequence)):
                SELFIESGenerated = SELFIESGenerated + selfies_alphabet[sequence[i]]
        return SELFIESGenerated

def translate_smile(sequence,selfies_alphabet):  

        SELFIESGenerated = ""

        for i in range(len(sequence)):
                SELFIESGenerated = SELFIESGenerated + smiles_alphabet[sequence[i]]
        return SELFIESGenerated
        
def create_onehot_instance(selfie_input,largest_selfies_len,selfies_alphabet_in):

    inttest_hot, arraytest_hot = data_loader.selfies_to_hot(selfie_input,largest_selfies_len, selfies_alphabet_in)
    x = torch.from_numpy(arraytest_hot).flatten().float().unsqueeze(0)
    return x

def create_onehot_instance_smiles(smile_input,largest_smiles_len,smiles_alphabet_in):

    inttest_hot, arraytest_hot = data_loader.smile_to_hot(smile_input,largest_smiles_len, smiles_alphabet_in)
    x = torch.from_numpy(arraytest_hot).flatten().float().unsqueeze(0)
    return x

def create_latent_space_vector(vae_encoder, vae_decoder, selfie_input,largest_selfies_len,selfies_alphabet_in):

    x = create_onehot_instance(selfie_input,largest_selfies_len,selfies_alphabet_in)

    z =set()
    vae_encoder.eval()
    vae_decoder.eval()
    z, mu, log_var = vae_encoder(x)

    return z.unsqueeze(0)

def create_latent_space_vector_smiles(vae_encoder, vae_decoder, smile_input,largest_smiles_len,smiles_alphabet_in):

    x = create_onehot_instance_smiles(smile_input,largest_smiles_len,smiles_alphabet_in)

    z =set()
    vae_encoder.eval()
    vae_decoder.eval()
    z, mu, log_var = vae_encoder(x)

    return z.unsqueeze(0)

def create_random_latent_space_vector(vae_encoder, vae_decoder, largest_selfies_len,selfies_alphabet_in):

    # Random input tensor for tests
    in_dimension_input = largest_selfies_len*len(selfies_alphabet_in)
    x = torch.randn(in_dimension_input).unsqueeze(0)

    z =set()
    vae_encoder.eval()
    vae_decoder.eval()
    z, mu, log_var = vae_encoder(x)

    return z.unsqueeze(0)


def decode_from_latentspace(vae_encoder, vae_decoder, latent_point_in, largest_selfies_len_in, selfies_alphabet_len, method):

        #one_hot_dimension = torch.zeros(selfies_alphabet_len,largest_selfies_len_in)
        #out_one_hot = torch.zeros_like(one_hot_dimension, device=device)

        vae_decoder.eval()
        vae_encoder.eval()

        sequence = []

        hidden = vae_decoder.init_hidden(batch_size=1)

         
        for seq_index in range(largest_selfies_len_in):
                out_one_hot_line, hidden = vae_decoder(latent_point_in, hidden)

                if method == 0:
                        sequence.append(out_one_hot_line.argmax())

                elif method ==1:
                        # Apply softmax and sample from the distribution to get the next token
                        softmax = torch.nn.Softmax(dim=2)
                        probabilities = softmax(out_one_hot_line)
                        categorical_dist = dist.Categorical(probabilities)
                        sample = categorical_dist.sample()
                        sequence.append(sample)

                else:
                        print("method is 0 for argmax or 1 for stat sampling")
        
        

        return sequence