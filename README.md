# ES-MDA-FCVAE for Seismic Facies Inversion
Ensemble Smoother with Multiple Data Assimilation integrated with Fully Variational Autoencoder for Seismic Facies Inversion

# Train Model

### Train Fully Variational Autoencoder
```bash
python train.py --train_dataset_path data_48x48_2D.h5 --latent_dim 1 --epochs 100 --model fcvae
```

### Train Variational Autoencoder
```bash
python train.py --train_dataset_path data_48x48_2D.h5 --latent_dim 1 --epochs 100 --model vae
```