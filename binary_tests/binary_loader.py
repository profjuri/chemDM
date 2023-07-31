import numpy as np
import math
import itertools

def count_unique_characters(strings_list):
    unique_characters = set()

    for string in strings_list:
        for char in string:
            unique_characters.add(char)

    unique_characters = list(unique_characters)
    return len(unique_characters), unique_characters



def print_bit_permutations(length):

    binary_string_list = []


    total_permutations = 2 ** length

    for i in range(total_permutations):
        binary_string = format(i, f'0{length}b')
        binary_string_list.append(binary_string)

    return binary_string_list



def replace_letters_with_numbers(word, char_list):
    letter_to_number = dict(char_list)


    return ''.join(letter_to_number.get(letter, letter) for letter in word)



def smiles_to_binary(smiles_list):


    N_unique, characters = count_unique_characters(smiles_list)

    bit_sq = math.ceil(np.log2(N_unique))

    bin_list = print_bit_permutations(bit_sq)


    character_list = []

    for i in range(N_unique):
        character_list.append([characters[i], bin_list[i]])


    letter_to_number = dict(character_list)


    binary_rep = np.array([replace_letters_with_numbers(word, character_list) for word in smiles_list])

    return binary_rep, character_list



def binary_to_smiles(binary_rep, character_list):

# Your list with binary-letter components
    binary_letter_list = character_list

# Create a dictionary from the list for easy lookup
    binary_to_letter = {binary: letter for letter, binary in binary_letter_list}

# Function to replace binary strings with letters
    def replace_binary_with_letters(binary_string):
        result = ''
        i = 0
        while i < len(binary_string):
            found = False
            for binary, letter in binary_to_letter.items():
                if binary_string.startswith(binary, i):
                    result += letter
                    i += len(binary)
                    found = True
                    break
            if not found:
                # If no match is found, add the original substring to the result
                result += binary_string[i]
                i += 1
        return result

    # Example list of binary strings
    binary_list = binary_rep

    # Replace binary strings with letters in each entry
    result_letters = [replace_binary_with_letters(binary_string) for binary_string in binary_list]

    return result_letters





