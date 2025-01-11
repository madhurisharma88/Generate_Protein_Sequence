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
