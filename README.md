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
└── backend/
    └── model.pth
```

`app.py` provides the upload interface and displays the prediction.
`inference.py` loads the checkpoint, prepares the image, and runs the model.
Images are resized to 224 × 224 pixels and normalized with ImageNet mean and
standard deviation values before inference.

## Requirements

- Python 3.10 or newer
- The trained model file at `backend/model.pth`

The application uses Streamlit, PyTorch, TorchVision, and Pillow. Predictions
run on the CPU.

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
