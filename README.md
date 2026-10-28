# Car Recognition

A Streamlit application that classifies an uploaded car image as one of 196 car
models. It runs inference with the project's trained ResNet50 checkpoint; no
separate web server or JavaScript frontend is needed.

## Project files

```text
.
├── app.py
├── inference.py
├── requirements.txt
├── train_model.py
├── test.py
└── backend/
    ├── data/
    │   ├── train/
    │   └── val/  (optional)
    └── model.pth
```

`app.py` provides the upload interface and displays the prediction.
`inference.py` loads the checkpoint, prepares the image, and runs the model.
Images are resized to 224 × 224 pixels and normalized with ImageNet mean and
standard deviation values before inference.

## Requirements

- Python 3.10 or newer
- The trained model file at `backend/model.pth`

The project uses Streamlit, PyTorch, TorchVision, Pillow, NumPy, and tqdm. The
app runs predictions on the CPU. Training uses a CUDA GPU when available.

## Training

Organize the labeled images into class-named directories. For example:

```text
backend/data/
├── train/
│   ├── Audi R8 Coupe 2012/
│   │   └── image1.jpg
│   └── BMW 3 Series Sedan 2012/
│       └── image2.jpg
└── val/  (optional; use the same class directories as train)
```

If `backend/data/val` is not present, the training script creates a
class-stratified validation split from the training images. The test script
accepts an individual image path for prediction.

Train the classifier from the project directory:

```shell
python train_model.py
```

By default, training uses ResNet50 with 224 × 224 image inputs, runs for 40
epochs, and saves the best validation checkpoint to `backend/model.pth`. This
is the checkpoint used by `app.py`. ImageNet weights are used to initialize
training and may be downloaded the first time. Use `--no-pretrained` to train
without downloading them.

To classify an image with the saved checkpoint, run:

```shell
python test.py path/to/car-image.jpg
```

Use `python train_model.py --help` or `python test.py --help` to see available
options.

## Run locally

From the project directory, install the dependencies and start Streamlit:

```shell
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit. Upload a JPG, JPEG, PNG, or WebP
image to view the predicted car model and confidence percentage.

## Deploy

For Streamlit Community Cloud, push the project to GitHub with `app.py`,
`inference.py`, `requirements.txt`, and `backend/model.pth` included. Create an
app for the repository and set its main file to `app.py`.

The model checkpoint must be present at the same relative path in the deployed
project. The app loads it on the first prediction and reuses it for subsequent
predictions.
