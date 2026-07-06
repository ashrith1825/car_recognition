"""
Convert PyTorch model (.pth) to TensorFlow.js format for web deployment.
This script converts model.pth -> model.onnx -> web_model/ (model.json + .bin files)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json

# ----------------------
# Configuration
# ----------------------
MODEL_PATH = "model.pth"
ONNX_PATH = "model.onnx"
WEB_MODEL_DIR = "../client/public/web_model"
DEVICE = torch.device("cpu")  # Use CPU for conversion

print("🚀 Starting model conversion process...")
print("=" * 60)

# ----------------------
# Step 1: Load PyTorch Model
# ----------------------
print(f"\n📦 Step 1: Loading PyTorch model from {MODEL_PATH}...")

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Recreate the model architecture
    model = models.resnet50(weights=None)
    num_classes = len(checkpoint["class_to_idx"])
    
    print(f"   ✓ Number of classes: {num_classes}")
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    # Save class mappings for the web app
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"   ✅ PyTorch model loaded successfully!")
    print(f"   ✓ Model architecture: ResNet50")
    print(f"   ✓ Output classes: {num_classes}")
    
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

# ----------------------
# Step 2: Convert to ONNX
# ----------------------
print(f"\n🔄 Step 2: Converting PyTorch model to ONNX...")

try:
    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    
    # Export to ONNX format
    torch.onnx.export(
        model,                      # Model to export
        dummy_input,                # Sample input
        ONNX_PATH,                  # Output file
        export_params=True,         # Store trained weights
        opset_version=11,           # ONNX version
        do_constant_folding=True,   # Optimize constant operations
        input_names=['input'],      # Input tensor name
        output_names=['output'],    # Output tensor name
        dynamic_axes={              # Support dynamic batch sizes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"   ✅ ONNX model saved to {ONNX_PATH}")
    
except Exception as e:
    print(f"   ❌ Error converting to ONNX: {e}")
    exit(1)

# ----------------------
# Step 3: Convert ONNX to TensorFlow.js
# ----------------------
print(f"\n🌐 Step 3: Converting ONNX to TensorFlow.js format...")
print(f"   Output directory: {WEB_MODEL_DIR}")

import subprocess
import os

try:
    # Create output directory if it doesn't exist
    Path(WEB_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run tensorflowjs_converter
    cmd = [
        "tensorflowjs_converter",
        "--input_format=onnx",
        "--output_format=tfjs_graph_model",
        ONNX_PATH,
        WEB_MODEL_DIR
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"   ✅ TensorFlow.js model created successfully!")
    
except subprocess.CalledProcessError as e:
    print(f"   ❌ Error converting to TensorFlow.js: {e}")
    print(f"   stderr: {e.stderr}")
    exit(1)
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")
    exit(1)

# ----------------------
# Step 4: Save Class Mappings
# ----------------------
print(f"\n📝 Step 4: Saving class mappings...")

try:
    classes_file = Path(WEB_MODEL_DIR) / "classes.json"
    
    class_data = {
        "classes": list(class_to_idx.keys()),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": num_classes
    }
    
    with open(classes_file, 'w') as f:
        json.dump(class_data, f, indent=2)
    
    print(f"   ✅ Class mappings saved to {classes_file}")
    
except Exception as e:
    print(f"   ❌ Error saving class mappings: {e}")
    exit(1)

# ----------------------
# Summary
# ----------------------
print("\n" + "=" * 60)
print("🎉 CONVERSION COMPLETE!")
print("=" * 60)
print(f"\n📂 Generated files:")
print(f"   • {ONNX_PATH} (intermediate ONNX format)")
print(f"   • {WEB_MODEL_DIR}/model.json (TensorFlow.js model architecture)")
print(f"   • {WEB_MODEL_DIR}/group*.bin (model weights)")
print(f"   • {WEB_MODEL_DIR}/classes.json (class mappings)")

print(f"\n📋 Next steps:")
print(f"   1. The web model is now in: {WEB_MODEL_DIR}")
print(f"   2. You can use TensorFlow.js in your React app to load the model")
print(f"   3. Install @tensorflow/tfjs in your client:")
print(f"      npm install @tensorflow/tfjs")
print(f"   4. Load the model in your React component:")
print(f"      const model = await tf.loadGraphModel('/web_model/model.json');")

print("\n✨ Your model is now ready for client-side inference!")
