#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.version.VERSION if hasattr(tf.keras, 'version') else "Built-in with TensorFlow")
print()

# Check GPU availability
print("=== GPU Information ===")
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("CUDA Version:", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN Version:", tf.sysconfig.get_build_info()['cudnn_version'])
print()

# Test GPU computation
if tf.config.list_physical_devices('GPU'):
    print("=== Testing GPU Computation ===")
    try:
        # Create tensors
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Time GPU computation
            import time
            start = time.time()
            c = tf.matmul(a, b)
            gpu_time = time.time() - start
            
        print(f"✅ GPU computation successful!")
        print(f"GPU computation time: {gpu_time:.4f} seconds")
        print(f"Result shape: {c.shape}")
        
        # Compare with CPU
        with tf.device('/CPU:0'):
            start = time.time()
            c_cpu = tf.matmul(a, b)
            cpu_time = time.time() - start
            
        print(f"CPU computation time: {cpu_time:.4f} seconds")
        print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        
else:
    print("❌ No GPU devices found")
    print("\nPossible solutions:")
    print("1. Check if NVIDIA drivers are installed: nvidia-smi")
    print("2. Verify GPU is detected by system: lspci | grep -i nvidia")
    print("3. Install NVIDIA drivers if missing")

print("\n=== Memory Information ===")
if tf.config.list_physical_devices('GPU'):
    try:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
    except Exception as e:
        print(f"Could not set memory growth: {e}")

print("\n=== Ready for Training! ===")
if tf.config.list_physical_devices('GPU'):
    print("🚀 GPU is available and working - your training will be accelerated!")
else:
    print("⚠️  Training will run on CPU - still functional but slower")