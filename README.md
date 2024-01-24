# ES-MDA-FCVAE for Seismic Facies Inversion
Ensemble Smoother with Multiple Data Assimilation integrated with Fully Variational Autoencoder for Seismic Facies Inversion

# Create pre-salt synthetic dataset
```bash
python create_dataset.py
```

# Train Model

### Train Fully Variational Autoencoder
```bash
python train.py --train_dataset_path data_48x48_2D.h5 --latent_dim 3 --epochs 100 --model fcvae
```

### Train Variational Autoencoder
```bash
python train.py --train_dataset_path data_48x48_2D.h5 --latent_dim 432 --epochs 100 --model vae
```

# Run Seismic Inversion
```bash
See ES-MDA-DL_2D.ipynb
```