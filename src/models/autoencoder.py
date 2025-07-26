from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Reshape, UpSampling1D, Activation, LeakyReLU,
    BatchNormalization, Dropout, Concatenate, Cropping1D, Lambda
)

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from src.config import CONFIG
import numpy as np

def build_layer(x, layer_cfg, skip_connections=None, layer_type=None):
    """Helper function to build individual layers"""
    cfg = CONFIG['autoencoder']
    
    if layer_type == 'conv1d':
        x = Conv1D(
            filters=layer_cfg['filters'],
            kernel_size=layer_cfg['kernel_size'],
            strides=layer_cfg.get('strides', 1),
            padding=layer_cfg.get('padding', 'same'),
            kernel_regularizer=l2(cfg.get('l2_regularization', 0))
        )(x)
        
        # Activation
        if layer_cfg.get('activation', '') == 'leaky_relu':
            x = LeakyReLU(alpha=0.1)(x)
        elif 'activation' in layer_cfg:
            x = Activation(layer_cfg['activation'])(x)
            
        # Batch normalization
        if layer_cfg.get('batch_norm', False):
            x = BatchNormalization()(x)
            
        # Dropout
        if layer_cfg.get('dropout', 0) > 0:
            x = Dropout(layer_cfg['dropout'])(x)
            
    elif layer_type == 'maxpool1d':
        x = MaxPooling1D(
            pool_size=layer_cfg['pool_size'],
            strides=layer_cfg.get('strides', None),
            padding=layer_cfg.get('padding', 'valid')
        )(x)
        
    elif layer_type == 'global_avg_pool':
        x = GlobalAveragePooling1D()(x)
        
    elif layer_type == 'dense':
        x = Dense(
            layer_cfg['units'],
            activation=layer_cfg['activation'],
            kernel_regularizer=l2(cfg.get('l2_regularization', 0))
        )(x)
        
    elif layer_type == 'reshape':
        x = Reshape(layer_cfg['target_shape'])(x)
        
    elif layer_type == 'upsampling1d':
        x = UpSampling1D(size=layer_cfg['size'])(x)
        
        # Add skip connection if available
        if (cfg.get('use_skip_connections', False) and
            skip_connections and
            len(skip_connections) > 0):
            skip = skip_connections.pop()
            try:
                x = Concatenate()([x, skip])
            except ValueError:
                # Handle dimension mismatch
                x = Lambda(lambda y: y[:, :skip.shape[1], :])(x)
                x = Concatenate()([x, skip])
    
    return x

def build_dcae():
    """Build Deep Convolutional Autoencoder with guaranteed dimension matching"""
    cfg = CONFIG['autoencoder']
    input_shape = cfg['input_shape']
    
    # Input layer
    input_layer = Input(shape=input_shape)
    x = input_layer
    skip_connections = []
    
    # Encoder
    for layer_cfg in cfg['encoder_layers']:
        x = build_layer(x, layer_cfg, skip_connections, layer_cfg['type'])
        
        # Store for skip connections if needed
        if (cfg.get('use_skip_connections', False) and 
            layer_cfg['type'] == 'conv1d'):
            skip_connections.append(x)
    
    # Bottleneck with explicit name
    encoded = Dense(
        cfg['bottleneck_size'],
        activation='tanh',
        kernel_regularizer=l2(cfg.get('l2_regularization', 0)),
        name='bottleneck'
    )(x)
    
    # Decoder
    x = encoded
    for layer_cfg in cfg['decoder_layers']:
        x = build_layer(x, layer_cfg, skip_connections, layer_cfg['type'])
    
    # Final dimension adjustment
    if x.shape[1] < input_shape[0]:
        # Calculate required upsampling factor
        upsample_factor = int(np.ceil(input_shape[0] / x.shape[1]))
        x = UpSampling1D(size=upsample_factor)(x)
    
    if x.shape[1] > input_shape[0]:
        # Apply cropping if needed
        crop_amount = x.shape[1] - input_shape[0]
        x = Cropping1D((0, crop_amount))(x)
    
    # Final output layer
    decoded = Conv1D(
        filters=input_shape[-1],
        kernel_size=15,  # Matches initial kernel size
        activation='sigmoid',
        padding='same'
    )(x)
    
    # Create models
    autoencoder = Model(input_layer, decoded, name='DCAE')
    encoder = Model(input_layer, encoded, name='Encoder')
    
    # Compile with configurable optimizer
    optimizer = Adam(
        learning_rate=cfg.get('learning_rate', 1e-4),
        clipnorm=cfg.get('clipnorm', 1.0)
    )
    autoencoder.compile(
        optimizer=optimizer,
        loss=cfg['loss'],
        metrics=cfg.get('metrics', ['mae'])
    )
    
    return autoencoder, encoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D

def build_autoencoder(input_shape=(3750, 1)):
    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same', name='bottleneck')(x)  # Named layer
    
    # Decoder
    x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(inputs, decoded)
    return autoencoder