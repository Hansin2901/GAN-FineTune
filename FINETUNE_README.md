# GAN Fine-tuning for Music Generation

This repository now includes comprehensive tools for fine-tuning your pre-trained GAN on new musical data.

## 🎯 Overview

The fine-tuning system allows you to:
- **Fine-tune** your existing GAN models on new MIDI data
- **Combine** new data with original training data
- **Generate samples** during training to monitor progress
- **Save checkpoints** at regular intervals
- **Visualize training** progress with detailed plots
- **Evaluate model quality** with multiple samples

## 📁 Files Added

### Core Fine-tuning Scripts
- **`finetune_gan.py`** - Basic fine-tuning implementation
- **`advanced_finetune.py`** - Advanced fine-tuning with sophisticated features
- **`finetune_config.py`** - Configuration file for all parameters
- **`run_finetune.py`** - Simple interface to run fine-tuning

### Configuration
- **`finetune_config.py`** contains all configurable parameters:
  - Learning rates
  - Training epochs
  - Data paths
  - Advanced features toggles

## 🚀 Quick Start

### 1. Prepare Your Data
```bash
# The script will create a 'data' folder for you
python run_finetune.py
```

Add your new MIDI files (`.mid` or `.midi`) to the `data/` folder.

### 2. Run Fine-tuning
```bash
python run_finetune.py
```

Choose between:
- **Basic fine-tuning**: Simple and fast
- **Advanced fine-tuning**: More features and monitoring

## ⚙️ Configuration Options

Edit `finetune_config.py` to customize:

### Basic Parameters
```python
FINETUNE_EPOCHS = 50        # Number of training epochs
BATCH_SIZE = 16             # Training batch size
GENERATOR_LR = 0.0001       # Generator learning rate
DISCRIMINATOR_LR = 0.0001   # Discriminator learning rate
```

### Data Options
```python
NEW_DATA_FOLDER = "data"           # Your new MIDI files
USE_ORIGINAL_DATA = True           # Mix with original data
```

### Advanced Features
```python
LEARNING_RATE_DECAY = False        # Apply LR decay
PROGRESSIVE_UNFREEZING = False     # Gradual layer unfreezing
WARMUP_EPOCHS = 5                  # LR warmup period
```

## 🎵 Usage Examples

### Basic Fine-tuning
```python
from finetune_gan import GANFineTuner

# Initialize
finetuner = GANFineTuner()

# Fine-tune
finetuner.finetune(
    new_data_folder="data",
    epochs=30,
    batch_size=16,
    use_original_data=True
)
```

### Advanced Fine-tuning
```python
from advanced_finetune import AdvancedGANFineTuner

# Initialize with advanced features
finetuner = AdvancedGANFineTuner()

# Run with all features
output_dir = finetuner.finetune("data")

# Evaluate quality
notes = finetuner.get_notes_from_folder("data")
finetuner.evaluate_model_quality(notes, num_samples=5)
```

## 📊 Monitoring Progress

### During Training
- **Loss values** printed every few epochs
- **Sample MIDI files** generated periodically
- **Model checkpoints** saved at intervals

### After Training
- **Training plots** showing loss curves and accuracy
- **Multiple samples** for quality evaluation
- **Training history** saved as JSON

## 🔧 Advanced Features

### 1. Learning Rate Scheduling
- **Warmup**: Gradual LR increase at start
- **Decay**: Exponential LR decrease over time
- **Adaptive**: Different rates for generator/discriminator

### 2. Progressive Training
- **Layer freezing**: Freeze early layers initially
- **Progressive unfreezing**: Gradually unfreeze during training
- **Transfer learning**: Leverage pre-trained features

### 3. Data Augmentation
- **Pitch shifting**: Random pitch variations
- **Tempo variations**: Speed modifications
- **Data mixing**: Combine original and new data

### 4. Quality Evaluation
- **Multiple samples**: Generate diverse outputs
- **Training metrics**: Track loss and accuracy
- **Visual monitoring**: Real-time progress plots

## 📂 Output Structure

After fine-tuning, you'll find:

```
finetuned_models/
├── finetune_YYYYMMDD_HHMMSS/
│   ├── final_generator.h5          # Fine-tuned generator
│   ├── final_discriminator.h5      # Fine-tuned discriminator
│   ├── sample_epoch_000.mid        # Training samples
│   ├── sample_epoch_010.mid
│   ├── final_sample.mid            # Final sample
│   ├── training_history.png        # Training plots
│   ├── training_history.json       # Detailed metrics
│   └── config.json                 # Training configuration
```

## 🎼 Tips for Best Results

### Data Preparation
1. **Quality over quantity**: Better to have fewer high-quality MIDI files
2. **Consistent style**: Similar musical styles work better
3. **Proper format**: Ensure MIDI files are valid and load correctly

### Training Strategy
1. **Start small**: Begin with fewer epochs (20-30)
2. **Monitor closely**: Watch loss curves and listen to samples
3. **Gradual approach**: Fine-tune in stages rather than long sessions

### Parameter Tuning
1. **Lower learning rates**: Start with 0.0001 or lower
2. **Smaller batches**: May work better for small datasets
3. **Patience**: Fine-tuning takes time to show results

## 🐛 Troubleshooting

### Common Issues

**"No notes found in data folder"**
- Ensure MIDI files are in the correct format (.mid or .midi)
- Check that files are not corrupted
- Try with different MIDI files

**"Models not found"**
- Make sure `generator_model.h5` and `discriminator_model.h5` exist
- Run the original training script first if needed

**"Memory errors"**
- Reduce batch size in config
- Use fewer training epochs
- Close other applications

**"Poor quality results"**
- Try longer training (more epochs)
- Adjust learning rates (lower values)
- Use more/better quality training data

## 📈 Monitoring Training

### Key Metrics to Watch
- **Discriminator Loss**: Should stabilize around 0.5-0.7
- **Generator Loss**: Should decrease gradually
- **Discriminator Accuracy**: Should stay around 50-70%

### Warning Signs
- **Loss oscillating wildly**: Learning rate too high
- **Generator loss not decreasing**: May need more epochs
- **Discriminator too strong**: Try lower discriminator learning rate

## 🎯 Next Steps

After successful fine-tuning:

1. **Test the models**: Generate multiple samples
2. **Compare quality**: Listen to before/after results
3. **Iterate**: Adjust parameters and re-train if needed
4. **Deploy**: Use fine-tuned models for music generation

## 🤝 Need Help?

If you encounter issues:
1. Check the configuration in `finetune_config.py`
2. Start with basic fine-tuning before advanced
3. Monitor the console output for error messages
4. Try with smaller datasets first

---

Happy fine-tuning! 🎵✨
