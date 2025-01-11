####Generate Protein Sequences####

##Utitlity Class for Generator and Discriminator Function####

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Amino acids (20 common ones)
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
idx_to_aa = {i: aa for i, aa in enumerate(amino_acids)}

# Function to encode a sequence
def encode_sequence(seq):
      encoding = np.zeros((len(seq), len(amino_acids)), dtype=np.float32)
      for i, aa in enumerate(seq):
        if aa in aa_to_idx:
          encoding[i, aa_to_idx[aa]] = 1.0
      return encoding

# Function to decode a sequence
def decode_sequence(encoding):
    indices = np.argmax(encoding, axis=1)
    #print(indices)
    return ''.join([idx_to_aa[idx] for idx in indices])


sequence_length = 1000  # Example protein sequence length
latent_dim =  20    # Dimensionality of the noise vector
hidden_dim = 20      # Hidden layer size

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length * len(amino_acids)),
            nn.Softmax(dim=1)  # Output probabilities for each amino acid position
        )

    def forward(self, z):
        generated_seq = self.fc(z)
        return generated_seq.view(-1, sequence_length, len(amino_acids))


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(sequence_length * len(amino_acids), hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, seq):
        flat_seq = seq.view(seq.size(0), -1)  # Flatten the sequence
        validity = self.fc(flat_seq)
        return validity
