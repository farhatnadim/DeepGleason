#==============================================================================#
#  Author:       Claude Code (Migration Assistant)                            #
#  Copyright:    2025 DeepGleason Project                                     #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
"""
Keras 3.x Compatibility Layer for Legacy Models

This module provides utilities to load legacy Keras 2.x models (TensorFlow 2.13 and earlier)
with Keras 3.x (TensorFlow 2.16+) by sanitizing layer names that contain invalid characters.

Keras 3.x enforces stricter layer naming conventions and does not allow '/' characters in
layer names, which were common in older model architectures like DenseNet.

Usage:
    from model_loader_keras3_compat import load_model_keras3_compat

    model = load_model_keras3_compat('models/model.DenseNet121.hdf5')
"""

import h5py
import json
import tempfile
import shutil
import os
from tensorflow import keras


def sanitize_layer_name(name):
    """
    Sanitize layer names to be Keras 3.x compatible.

    Replaces invalid characters (like '/') with underscores.

    Args:
        name (str): Original layer name

    Returns:
        str: Sanitized layer name
    """
    return name.replace('/', '_').replace(':', '_')


def sanitize_model_config(config):
    """
    Recursively sanitize all layer names in a model configuration.

    Args:
        config (dict): Model configuration dictionary

    Returns:
        dict: Sanitized configuration
    """
    if isinstance(config, dict):
        # Sanitize layer names
        if 'name' in config:
            config['name'] = sanitize_layer_name(config['name'])

        # Handle layer connections (inbound_nodes)
        if 'inbound_nodes' in config:
            for node in config['inbound_nodes']:
                if isinstance(node, list):
                    for connection in node:
                        if isinstance(connection, list) and len(connection) > 0:
                            # First element is typically the layer name
                            if isinstance(connection[0], str):
                                connection[0] = sanitize_layer_name(connection[0])

        # Recursively process nested configs
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                config[key] = sanitize_model_config(value)

    elif isinstance(config, list):
        config = [sanitize_model_config(item) for item in config]

    return config


def load_model_keras3_compat(filepath, custom_objects=None, compile=True):
    """
    Load a legacy Keras model with Keras 3.x compatibility fixes.

    This function creates a temporary copy of the model file, sanitizes layer names
    in the model configuration, and loads the modified model.

    Args:
        filepath (str): Path to the HDF5 model file
        custom_objects (dict): Optional dictionary of custom objects
        compile (bool): Whether to compile the model after loading

    Returns:
        keras.Model: Loaded model with sanitized layer names

    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the model format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Create a temporary file for the sanitized model
    temp_fd, temp_path = tempfile.mkstemp(suffix='.h5')
    os.close(temp_fd)

    try:
        # Copy the original model file
        shutil.copy2(filepath, temp_path)

        # Open and modify the model configuration
        with h5py.File(temp_path, 'r+') as f:
            # Check if model_config exists
            if 'model_config' not in f.attrs:
                # Try loading without sanitization (might be SavedModel format)
                os.remove(temp_path)
                return keras.models.load_model(filepath, custom_objects=custom_objects, compile=compile)

            # Load the model configuration
            config_json = f.attrs['model_config']
            if isinstance(config_json, bytes):
                config_json = config_json.decode('utf-8')

            config = json.loads(config_json)

            # Sanitize the configuration
            sanitized_config = sanitize_model_config(config)

            # Save the sanitized configuration back
            sanitized_json = json.dumps(sanitized_config)
            del f.attrs['model_config']
            f.attrs['model_config'] = sanitized_json

            # Also sanitize layer_names if present
            if 'layer_names' in f.attrs:
                layer_names = f.attrs['layer_names']
                if isinstance(layer_names, bytes):
                    layer_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                                  for name in layer_names]
                sanitized_names = [sanitize_layer_name(name) for name in layer_names]
                del f.attrs['layer_names']
                f.attrs['layer_names'] = [name.encode('utf-8') for name in sanitized_names]

        # Load the model with sanitized configuration
        model = keras.models.load_model(temp_path, custom_objects=custom_objects, compile=compile)

        return model

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_model(model_path):
    """
    Test function to verify model loading works correctly.

    Args:
        model_path (str): Path to model file

    Returns:
        bool: True if loading succeeded, False otherwise
    """
    try:
        print(f"Attempting to load: {model_path}")
        model = load_model_keras3_compat(model_path, compile=False)
        print(f"✓ Successfully loaded model: {model.name}")
        print(f"  Total parameters: {model.count_params():,}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


if __name__ == "__main__":
    # Test with DenseNet121 model
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/model.DenseNet121.hdf5"

    success = test_load_model(model_path)
    sys.exit(0 if success else 1)
