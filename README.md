# Car Recognition

A Streamlit app that classifies an uploaded image as one of 196 car models. It
uses the existing ResNet50 checkpoint and runs inference on the CPU.

## Project structure

```text
.
├── .streamlit/
│   └── config.toml
├── app.py
├── inference.py
├── models/
│   └── resnet50_car.pth
├── requirements.txt
├── test.py
├── train_model.py
└── data/                 # Local training images; not committed
```

`app.py` is the Streamlit entry point. `inference.py` contains the shared model
loading and image preprocessing used by the app and `test.py`. The checkpoint
expects 224 × 224 RGB images normalized with ImageNet mean and standard
deviation values.

## Run locally

Use Python 3.13, then install the pinned CPU dependencies and launch the app:

```shell
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit and upload a JPG, JPEG, PNG, or WebP
image. Uploads are limited to 10 MB by `.streamlit/config.toml`.

To run a prediction from the command line:

```shell
python test.py path/to/car-image.jpg
```

## Training

Place images in class-named directories under `data/train`. An optional
validation set can be placed under `data/val`, using the same class directories.
If there is no validation folder, `train_model.py` creates a class-stratified
split from the training set.

```text
data/
├── train/
│   ├── Audi R8 Coupe 2012/
│   └── BMW 3 Series Sedan 2012/
└── val/                  # Optional
    ├── Audi R8 Coupe 2012/
    └── BMW 3 Series Sedan 2012/
```

Train with `python train_model.py`. The script saves the best validation
checkpoint to `models/resnet50_car.pth`, which is the checkpoint used by the
app. Training uses pretrained ResNet50 weights by default and may download them
on the first run; pass `--no-pretrained` to skip that download.

## Deploy on Streamlit Community Cloud

Push the repository to GitHub with `app.py`, `inference.py`, `requirements.txt`,
and `models/resnet50_car.pth`. In Community Cloud, select the repository, set
the app file to `app.py`, and choose Python 3.13 in the deployment settings.

The checkpoint is about 91.5 MiB, below GitHub's 100 MiB per-file limit, so Git
LFS is not required. The CPU-only PyTorch dependencies avoid installing CUDA
packages. The app maps checkpoint weights while loading them and caches the
model for reuse. The training dataset is excluded from Git by `.gitignore`.
