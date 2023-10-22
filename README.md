# Denoise Images Research
Denoise Images Research

## Setup local virtual environment
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

### Using Tensorflow Docker image with NVidia GPU support
- Build the Docker image:
```bash
./build_docker.sh
```
- Start the Docker container:
```bash
./start_docker.sh
```
- Execute the train.py script inside Docker container:
```bash
cd /tmp
python3 train.py
```
- For expermiments, please use the command line arguments:
```bash
cd /tmp
python3 train.py --learning-rate=1e-6 --model-name='denoise_unet.h5'
```