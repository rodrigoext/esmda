import argparse
import numpy as np
import tensorflow as tf
from vae_models import *
import h5py

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_args():
    parser = argparse.ArgumentParser(description="train Geofacies Class",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="train dataset path (tfrecorfs)")
    parser.add_argument("--test_dataset_path", type=str, default=None,
                        help="test dataset path (tfrecorfs)")
    parser.add_argument("--filters", type=str, default='32-32-32',
                        help="Filters number")
    parser.add_argument("--kernel_dim", type=str, default='3-3-3',
                        help="Dimension of the Kernel")
    parser.add_argument("--strides_values", type=str, default='2-2-2',
                        help="Strides values")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Dimension of the hidden layer")
    parser.add_argument("--latent_dim", type=int, default=500,
                        help="Dimension of the latent vector")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate")                        
    parser.add_argument("--steps", type=int, default=500,
                        help="steps per epoch")
    parser.add_argument("--save_path_weights", type=str, default=None,
                        help="path to save the weights")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--activation", type=str, default="sigmoid",
                        help="activation in the last layer")
    parser.add_argument("--optimizer", type=str, default="RMSprop",
                        help="optimizer ('RMSprop','Adam' or other)")
    parser.add_argument("--model", type=str, default="fcvae",
                        help="model architecture ('vae','fcvae')")
    parser.add_argument("--patience", type=int, default=20,
                        help="step patience to stop train")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="epsilon to compute PCA")
    args = parser.parse_args()
    return args

def load_data_set(path, isArray=False, batch=4, nclasses=4):
    hf = h5py.File(path, 'r')
    x_train = np.array(hf['X'])
    return tf.keras.utils.to_categorical(x_train, num_classes=nclasses)

def main():
    args = get_args()
    
    x_train = load_data_set(args.train_dataset_path)
    print(x_train.shape)

    opt = tf.keras.optimizers.Adam(lr=args.lr)

    if args.model == 'vae':
        vae = VAE(lat_size=args.latent_dim, input_dim=x_train.shape[1:])
        csv_logger = tf.keras.callbacks.CSVLogger("vae_history.csv", append=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="vae_f/",
                                                 save_best_only=True,
                                                 verbose=1)
        initial_learning_rate = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
        vae.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule))
        history = vae.fit(x_train, epochs=args.epochs, batch_size=8, callbacks=[csv_logger, cp_callback])
        tf.keras.models.save_model(vae.encoder, "vae_f/encoder")
        tf.keras.models.save_model(vae.decoder, "vae_f/decoder")
    
    if args.model == 'fcvae':
        fcvae = FCVAE(lat_size=args.latent_dim, input_dim=x_train.shape[1:])
        csv_logger = tf.keras.callbacks.CSVLogger("fcvae_history.csv", append=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="fcvae_f/",
                                                 save_best_only=True,
                                                 verbose=1)
        fcvae.compile(optimizer=opt)
        history = fcvae.fit(x_train, epochs=args.epochs, batch_size=8, callbacks=[csv_logger, cp_callback])
        tf.keras.models.save_model(fcvae.encoder, "fcvae_f/encoder")
        tf.keras.models.save_model(fcvae.decoder, "fcvae_f/decoder")

if __name__ == '__main__':
    main()