# DeepEchoes: Marine Mammal Spectrogram Synthesis


## Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/deepechoes.git
cd deepechoes
```
2. **Install Dependencies**
```bash
pip install -r requirements.txt
```


# VAE

A PyTorch implementation for training Variational Autoencoders (VAEs) on marine mammal spectrogram data, with advanced features including mixed precision training, perceptual loss, and adversarial discrimination.


## Usage

### Example Training Command

```bash
python -m deepechoes.diffusion.latent.vae_train \
  --dataset_path /path/to/spectrograms \
  --model_config vae_config.json \
  --output_dir ./results \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 4.5e-6 \
  --mixed_precision fp16
```

### Dataset Preparation

The training script uses torchvision.datasets.ImageFolder to load spectrogram data. Your dataset should follow this structure:

```
marine_mammal_sounds/
├── humpback/
│   ├── spec_001.png
│   ├── spec_002.png
├── orca/
│   ├── killerwhale_001.png
│   ├── killerwhale_002.png
└── dolphin/
    ├── bottlenose_001.png
    ├── spinner_001.png
```

#### Class-Agnostic Training:

If you don’t need class-specific training (e.g., all spectrograms are treated as one class), you can use a single directory:

```
dataset_root/
└── all_specs/
    ├── spec_001.png
    ├── spec_002.png
    └── ...
```

#### Validation Images:

The `--validation_image` argument can accept either:

1. **Explicit paths:** Provide specific image paths for validation.

```bash
--validation_image spec1.png spec2.png
```

2. **Class-containing subdirectories:** If using a class-based structure, validation images can be sampled from these subdirectories.

3. **None**: In which case the script will sample 8 random images from the training set to reconstruct. 

### Optimizations

We recommend exploring these optimization options to balance speed, memory usage, and training stability: 

- `--mixed_precision`: Reduces memory usage and speeds up training. Options: `no` (FP32), `fp16` (FP16). Recommendation: Use `fp16` if using modern NVIDIA GPUs (Volta+)
```bash
--mixed_precision fp16  # Requires CUDA-compatible GPU
```
- `--enable_xformers`: Enables memory-efficient attention
```bash
--enable_xformers
```
- `--use_8bit_adam`: 8-bit Adam optimizer for reduced memory usage. Best for GPUs with less than 16GB VRAM.
```bash
--use_8bit_adam
```
- `--allow_tf32`: Enables TF-32 math on Ampere+ GPUs
```bash
--allow_tf32
```
- `--gradient_accumulation_steps`: Simulates larger batches
```bash
--gradient_accumulation_steps 4  # Effective batch size = batch_size * steps 
```

### Monitoring

The script supports multiple logging backends:

```bash
# TensorBoard (default)
--report_to tensorboard

# Weights & Biases
--report_to wandb
```

### Output Structure

```
results/
├── logs/                  # Logs
├── samples/               # Validation reconstructions
├── checkpoint-1000/       # Training state snapshots
├── pytorch_model.bin      # Final discriminator weights
└── config.json            # Final VAE configuration
```


# Database Creation

The `create_db` script converts audio files into spectrograms and organizes them into a database suitable for training. It supports various audio formats and provides options for spectrogram generation parameters.

## Usage

### Example Command

```bash
python -m deepechoes.create_db 
/path/to/audio/files \ 
spec_config.json \
--mode img \
--annotations annotations_csv.csv \
--labels SRKW=1 \
--output output_root_dir \
--seed 40
```

### Audio File Organization

The script supports two input directory structures:

#### Class-Based Structure:
```
audio_files/
├── humpback/
│   ├── recording_001.wav
│   ├── recording_002.mp3
├── orca/
│   ├── pod_recording_001.wav
│   ├── pod_recording_002.wav
└── dolphin/
    ├── whistles_001.wav
    ├── clicks_001.wav
```

#### Flat Structure:
```
audio_files/
├── recording_001.wav
├── recording_002.mp3
└── recording_003.wav
```

## Key Components

1. Annotations CSV Format:
```csv
filename,start,end,label
recording_001.wav,12.5,15.2,SRKW
recording_002.wav,45.1,47.8,humpback
```
- Required columns: filename, start, end, label
- Labels are mapped to integers using --labels argument

2. Spectrogram Configuration (spec_config.json):

```json
{
  "sr": 24000,
  "window": 0.05,
  "step": 0.0125,
  "num_filters": 128,
  "duration": 3,
  "fmin": 0,
  "fmax": 12000
}
```
- sr: Target sample rate
- duration: Clip length in seconds
- window: Spectrogram window size (seconds)
- step: Spectrogram hop length (seconds)

## Output Modes

### HDF5 Mode (--mode hdf5) ToDo

### Image Mode (--mode img)
Generates PNG spectrograms organized by class:
```
output_root_dir/
├── 0/  
│   ├── file1_0.png
│   └── file2_0.png
└── 1/  
    ├── file1_1.png
    └── file2_1.png
```
