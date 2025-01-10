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

####FID Score####

def calculate_fid_for_seq(real_seq, generated_seq, model_name='bert-base-uncased'):
    """
    Calculate the Fréchet Inception Distance (FID) between two sets of text data.

    Args:
        real_seq (list of str): List of real samples.
        generated_seq (list of str): List of generated samples.
        model_name (str): Name of the Protein Sequencing model to use for embeddings.

    Returns:
        float: FID score.
    """
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(texts):
        """Generate embeddings for a list of texts."""
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = model(**inputs)
            # Use the last hidden state and average across tokens
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    # Compute embeddings
    real_embeddings = get_embeddings(real_seq)
    generated_embeddings = get_embeddings(generated_seq)

    # Calculate mean and covariance for real and generated embeddings
    mu_real, sigma_real = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu_gen, sigma_gen = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)

    # Compute FID
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real * sigma_gen)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fiddef calculate_fid_for_seq(real_seq, generated_seq, model_name='bert-base-uncased'):
    """
    Calculate the Fréchet Inception Distance (FID) between two sets of text data.

    Args:
        real_seq (list of str): List of real samples.
        generated_seq (list of str): List of generated samples.
        model_name (str): Name of the Protein Sequencing model to use for embeddings.

    Returns:
        float: FID score.
    """
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embeddings(texts):
        """Generate embeddings for a list of texts."""
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = model(**inputs)
            # Use the last hidden state and average across tokens
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    # Compute embeddings
    real_embeddings = get_embeddings(real_seq)
    generated_embeddings = get_embeddings(generated_seq)

    # Calculate mean and covariance for real and generated embeddings
    mu_real, sigma_real = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu_gen, sigma_gen = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)

    # Compute FID
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real * sigma_gen)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

####GAN Model####

# Training loop
num_epochs = 50
batch_size = 195   #total number of sequences
dloss = []
gloss = []
rloss = []
floss = []
fid  = []
real_texts_list = [] # To store a batch of real sequences
generated_texts_list = [] # To store a batch of generated sequences

for epoch in range(num_epochs):
    # Train Discriminator
    optimizer_D.zero_grad()

    # Real sequences
    real_labels = torch.ones((batch_size, 1))
    fake_labels = torch.zeros((batch_size, 1))

    real_preds = discriminator(real_data)
    real_loss = adversarial_loss(real_preds, real_labels)

    # Fake sequences
    z = torch.bernoulli(torch.rand(batch_size, latent_dim))
    fake_data = generator(z).detach()  # Generate fake sequences
    fake_preds = discriminator(fake_data)
    fake_loss = adversarial_loss(fake_preds, fake_labels)

    # Backprop and optimize Discriminator
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()

    # Generate sequences and classify them as real
    z = torch.bernoulli(torch.rand(batch_size, latent_dim))
    generated_seqs = generator(z)
    gen_preds = discriminator(generated_seqs)
    g_loss = adversarial_loss(gen_preds, real_labels)

    # Backprop and optimize Generator
    g_loss.backward()
    optimizer_G.step()

    decoded_sequence1 = decode_sequence(generated_seqs.detach().numpy()[0]) # Convert encoding back to sequence for logging purposes

    # Store the generated and real sequences for this batch
    x=random.randint(1,900)
    real_texts_list.extend(modified_sequences)
    generated_texts_list.extend([decoded_sequence1[:(x+195)]])


    if epoch % 10 == 0: #Calculate FID only every 10 epochs.
        # Calculate and log the FID for this batch
        plt.title('Number of epochs = %i'%epoch)
        fid_score = calculate_fid_for_seq(real_texts_list, generated_texts_list)
        print(f"Epoch [{epoch}/{num_epochs}] - D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
        print(f"FID Score for Text: {fid_score}")
        print(f"Real Loss: {real_loss.item():.4f}, Fake Loss: {fake_loss.item():.4f}")

        plt.plot(dloss,'.',label='Discriminator Loss',color='firebrick')
        plt.plot(gloss,'.',label = 'Generator Loss',color='navy')
        plt.legend(fontsize=10)
        plt.grid(False)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
    dloss.append(d_loss.item())
    gloss.append(g_loss.item())
    rloss.append(real_loss.item())
    floss.append(fake_loss.item())
    fid.append(fid_score)
