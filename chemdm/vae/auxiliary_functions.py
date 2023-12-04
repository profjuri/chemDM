import selfies as sf
import torch
import torch.distributions as dist
import data_loader
import numpy as np

def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # Pad with [nop] to reach the desired length
    padding_length = largest_selfie_len - sf.len_selfies(selfie)
    selfie += '[nop]' * padding_length

    # Integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # One-hot encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    # Add approximately 5 blank spaces
    blank_space = [0] * len(alphabet)
    for _ in range(15):
        onehot_encoded.append(blank_space)

    return integer_encoded, np.array(onehot_encoded)



def multiple_selfies_to_onehot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

def gen_latent_vectors(vae_encoder, data):

    vae_encoder.eval()

    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)

    z, mus, log_vars = vae_encoder(inp_flat_one_hot)
    return mus

def decode_lpoints(vae_decoder, data_shape1, lpoints, selfies_alphabet, device):

    final_smiles_list = []

    vae_decoder.eval()

    seq_tensor = torch.empty(0, dtype=torch.float32).to(device)
    hidden = vae_decoder.init_hidden(batch_size=lpoints.shape[0])

    print(lpoints.size())


    for seq_index in range(data_shape1):

        out_one_hot_line, hidden = vae_decoder(lpoints.unsqueeze(0), hidden)
        out_one_hot_line = out_one_hot_line.squeeze()
        out_one_hot_line_arg = torch.argmax(out_one_hot_line, dim = 1).unsqueeze(0)
        seq_tensor = torch.cat((seq_tensor, out_one_hot_line_arg), dim = 0)

    sequences = seq_tensor.t()
    list_of_continuous_strings = [sf.decoder(''.join([selfies_alphabet[int(i)] for i in row])) for row in sequences]
    final_smiles_list = final_smiles_list + list_of_continuous_strings

    return final_smiles_list

def decode_single_lpoint(vae_decoder, data_shape1, lpoints, selfies_alphabet, device):

    final_smiles_list = []

    vae_decoder.eval()

    seq_tensor = torch.empty(0, dtype=torch.float32).to(device)
    hidden = vae_decoder.init_hidden(batch_size=lpoints.shape[0])


    for seq_index in range(data_shape1):

        out_one_hot_line, hidden = vae_decoder(lpoints.unsqueeze(0), hidden)
        out_one_hot_line = out_one_hot_line.squeeze(0)
        out_one_hot_line_arg = torch.argmax(out_one_hot_line, dim = 1).unsqueeze(0)
        seq_tensor = torch.cat((seq_tensor, out_one_hot_line_arg), dim = 0)

    sequences = seq_tensor.t()
    list_of_continuous_strings = [sf.decoder(''.join([selfies_alphabet[int(i)] for i in row])) for row in sequences]
    final_smiles_list = final_smiles_list + list_of_continuous_strings

    return final_smiles_list
