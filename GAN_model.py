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
