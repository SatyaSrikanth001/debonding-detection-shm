CONFIG = {
    "data": {
        "sample_rate": 25e6,
        "fundamental_freq": 140e3,
        "harmonics": [1, 2, 3],
        "bandwidth": 20e3,
        "segment_length": 3750,
        "num_segments": 4,
        "noise_level": 0.05,
        "defect_variability": 0.3,
        "num_defects": 1
    },
    "paths": {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "augmented_data": "data/augmented",
        "models": "saved_models",
        "logs": "logs"
    },
    "augmentation": {
        "snr_levels": [50, 55, 60],
        "subsample_factors": [2, 3, 4],
        "num_augmentations": 36,
        "time_warping": True,
        "random_noise": True,
        "time_warp_sigma": 0.2,
        "noise_level": 0.05
    },
    "autoencoder": {
        "input_shape": (3750, 1),
        "encoder_layers": [
            # Initial conv block (stride 2 for downsampling)
            {'type': 'conv1d', 'filters': 64, 'kernel_size': 15, 'strides': 2,
             'activation': 'leaky_relu', 'padding': 'same', 'batch_norm': True, 'dropout': 0.1},

            # Second conv block
         
            {'type': 'conv1d', 'filters': 128, 'kernel_size': 7, 'strides': 2,
             'activation': 'leaky_relu', 'padding': 'same', 'batch_norm': True, 'dropout': 0.1},

            # Third conv block
            {'type': 'conv1d', 'filters': 256, 'kernel_size': 5, 'strides': 2,
             'activation': 'leaky_relu', 'padding': 'same', 'batch_norm': True, 'dropout': 0.2},

            # Final conv block
            {'type': 'conv1d', 'filters': 512, 'kernel_size': 3, 'strides': 2,
             'activation': 'leaky_relu', 'padding': 'same', 'batch_norm': True, 'dropout': 0.2},

            # Global pooling
            {'type': 'global_avg_pool'}
        ],
        "bottleneck_size": 256,
        "decoder_layers": [
            # Expand from bottleneck
            {'type': 'dense', 'units': 512, 'activation': 'leaky_relu', 'batch_norm': True},
            {'type': 'reshape', 'target_shape': (512, 1)},

            # First upsampling block
            {'type': 'upsampling1d', 'size': 2},
            {'type': 'conv1d', 'filters': 512, 'kernel_size': 3, 'activation': 'leaky_relu',
             'padding': 'same', 'batch_norm': True, 'dropout': 0.2},

            # Second upsampling block
            {'type': 'upsampling1d', 'size': 2},
            {'type': 'conv1d', 'filters': 256, 'kernel_size': 5, 'activation': 'leaky_relu',
             'padding': 'same', 'batch_norm': True, 'dropout': 0.2},

            # Third upsampling block
            {'type': 'upsampling1d', 'size': 2},
            {'type': 'conv1d', 'filters': 128, 'kernel_size': 7, 'activation': 'leaky_relu',
             'padding': 'same', 'batch_norm': True, 'dropout': 0.1},

            # Fourth upsampling block (critical addition)
            {'type': 'upsampling1d', 'size': 2},
            {'type': 'conv1d', 'filters': 64, 'kernel_size': 15, 'activation': 'leaky_relu',
             'padding': 'same', 'batch_norm': True, 'dropout': 0.1},

            # Output layer with cropping if needed
            {'type': 'conv1d', 'filters': 1, 'kernel_size': 15, 'activation': 'sigmoid', 'padding': 'same'},

            # Explicit cropping layer to ensure exact 3750 output
            {'type': 'cropping1d', 'cropping': (0, 0)}  # Adjust if needed
        ],
        "use_skip_connections": True,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "loss": "mse",
        "metrics": ["mae"],
        "epochs": 150,
        "batch_size": 32,
        "validation_split": 0.2,
        "early_stopping": True,
        "patience": 20,
        "reduce_lr": True,
        "reduce_lr_patience": 8,
        "min_lr": 1e-6,
        "l2_regularization": 1e-4,
        "clipnorm": 1.0
    },
    "classifier": {
        "input_dim": 256,
        "layers": [
            {"units": 256, "activation": "leaky_relu", "dropout": 0.3, "batch_norm": True},
            {"units": 128, "activation": "leaky_relu", "dropout": 0.2, "batch_norm": True},
            {"units": 64, "activation": "leaky_relu", "batch_norm": True},
        ],
        "output_units": 5,
        "output_activation": "softmax",
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "epochs": 100,
        "batch_size": 64,
        "validation_split": 0.2,
        "class_weight": True,
        "early_stopping": True,
        "patience": 15
    },
    "models": {
        "svm": {"kernel": "rbf", "probability": True, "class_weight": "balanced"},
        "rf": {"n_estimators": 200, "class_weight": "balanced"},
        "ada": {"n_estimators": 100, "learning_rate": 0.01},
        "gb": {"n_estimators": 150, "subsample": 0.8},
        "knn": {"n_neighbors": 7, "weights": "distance"}
    },
    "training": {
        "memory_optimization": True,
        "visualization": True,
        "tensorboard": True,
        "checkpoint": True,
        "mixed_precision": False
    },
    "evaluation": {
        "confusion_matrix": True,
        "classification_report": True,
        "roc_curves": True,
        "feature_visualization": True,
        "latent_space_analysis": True
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
