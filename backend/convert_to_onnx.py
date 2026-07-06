"""
Convert PyTorch model (.pth) to ONNX format for web deployment.
ONNX can be directly used in the browser with ONNX Runtime Web.
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
ONNX_PATH = "../client/public/model.onnx"
CLASSES_PATH = "../client/public/classes.json"
DEVICE = torch.device("cpu")  # Use CPU for conversion

print("🚀 Starting model conversion to ONNX...")
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
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    
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
    # Create output directory if needed
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    
    # Export to ONNX format
    torch.onnx.export(
        model,                      # Model to export
        dummy_input,                # Sample input
        ONNX_PATH,                  # Output file
        export_params=True,         # Store trained weights
        opset_version=13,           # ONNX version (13 is well supported)
        do_constant_folding=True,   # Optimize constant operations
        input_names=['input'],      # Input tensor name
        output_names=['output'],    # Output tensor name
        dynamic_axes={              # Support dynamic batch sizes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"   ✅ ONNX model saved to {ONNX_PATH}")
    
    # Get file size
    file_size_mb = Path(ONNX_PATH).stat().st_size / (1024 * 1024)
    print(f"   ✓ Model size: {file_size_mb:.2f} MB")
    
except Exception as e:
    print(f"   ❌ Error converting to ONNX: {e}")
    exit(1)

# ----------------------
# Step 3: Save Class Mappings
# ----------------------
print(f"\n📝 Step 3: Saving class mappings...")

try:
    Path(CLASSES_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    class_data = {
        "classes": list(class_to_idx.keys()),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": num_classes
    }
    
    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_data, f, indent=2)
    
    print(f"   ✅ Class mappings saved to {CLASSES_PATH}")
    print(f"   ✓ Total classes: {num_classes}")
    
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
print(f"   • {ONNX_PATH} (ONNX model - {file_size_mb:.2f} MB)")
print(f"   • {CLASSES_PATH} (class mappings)")

print(f"\n📋 Next steps:")
print(f"   1. Install ONNX Runtime Web in your React app:")
print(f"      cd client")
print(f"      npm install onnxruntime-web")
print(f"   2. The model is ready to use in your client app")
print(f"   3. Load the model in your React component:")
print(f"      import * as ort from 'onnxruntime-web';")
print(f"      const session = await ort.InferenceSession.create('/model.onnx');")
print(f"   4. No backend server needed! The model runs entirely in the browser")

print("\n✨ Your model is now ready for client-side inference!")
print("💡 ONNX Runtime Web provides fast, efficient inference in the browser")
