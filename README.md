# FintamathImageRecognition

Handwritten Mathematical Expression Recognition for Fintamath.

Based on [CoMER](https://github.com/Green-Wood/CoMER).

The dataset consists of images from the [Aida Calculus Math Handwriting Recognition Dataset](https://www.kaggle.com/datasets/aidapearson/ocr-data/data), self-drawn images, and images from the web.

## Install

```bash
# clone project   
git clone https://github.com/fintarin/FintamathImageRecognition

# install project
cd FintamathImageRecognition
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

## Training

Then start a new training by running `train.py`.

```bash
python train.py fit --config config.yaml
```

Or continue the training after the checkpoint by running `train_continue.py`.

```bash
python train_continue.py --config config.yaml
```

## Evaluation

Run `test.py`.

```bash
python test.py
```

## Export to ONNX

Run `export.py`.

```bash
python export.py
```
