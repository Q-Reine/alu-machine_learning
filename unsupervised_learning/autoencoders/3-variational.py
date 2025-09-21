#!/usr/bin/env python3
""" Variational Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    # ----- Encoder -----
    X_input = keras.Input(shape=(input_dims,))
    Y_prev = X_input

    for nodes in hidden_layers:
        Y_prev = keras.layers.Dense(units=nodes,
                                    activation="relu")(Y_prev)

    # Separate Dense layers for mean and log variance
    z_mean = keras.layers.Dense(units=latent_dims,
                                activation=None)(Y_prev)
    z_log_sigma = keras.layers.Dense(units=latent_dims,
                                     activation=None)(Y_prev)

    # Reparameterization trick
    def sampling(args):
        z_m, z_log_var = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,)
                            )([z_mean, z_log_sigma])

    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma])

    # ----- Decoder -----
    X_decode = keras.Input(shape=(latent_dims,))
    Y_prev = X_decode

    for nodes in reversed(hidden_layers):
        Y_prev = keras.layers.Dense(units=nodes,
                                    activation="relu")(Y_prev)

    output = keras.layers.Dense(units=input_dims,
                                activation="sigmoid")(Y_prev)
    decoder = keras.Model(X_decode, output)

    # ----- Autoencoder -----
    e_z, e_mean, e_log_sigma = encoder(X_input)
    d_output = decoder(e_z)
    auto = keras.Model(X_input, d_output)

    # Custom loss = reconstruction + KL divergence
    def vae_loss(x, x_decoded):
        xent_loss = keras.backend.binary_crossentropy(x, x_decoded)
        xent_loss = keras.backend.sum(xent_loss, axis=1)
        kl_loss = -0.5 * keras.backend.sum(
            1 + e_log_sigma
            - keras.backend.square(e_mean)
            - keras.backend.exp(e_log_sigma),
            axis=1,
        )
        return xent_loss + kl_loss

    auto.compile(optimizer="adam", loss=vae_loss)

    return encoder, decoder, auto
