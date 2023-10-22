# Denoise Images Research
Denoise Images Research

## Setup virtual environment
- Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- Install dependencies:
```bash
pip install -r requirements.txt
```

- Deactivate virtual environment:
```bash
deactivate
```

## Run scripts

### Model
This create a U-Net model with default parameters and prints out the sumary of it. This model is created using the _Tensorflow_ library.
```bash
python3 model.py
```