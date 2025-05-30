import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from music21 import converter, instrument, note, chord, stream
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import finetune_config as config
from create_generator_model import prepare_sequences, create_midi


class AdvancedGANFineTuner:
    def __init__(self, generator_path=None, discriminator_path=None):
        """
        Advanced GAN fine-tuner with sophisticated training techniques
        """
        self.generator_path = generator_path or config.GENERATOR_MODEL_PATH
        self.discriminator_path = discriminator_path or config.DISCRIMINATOR_MODEL_PATH
        
        self.seq_length = config.SEQUENCE_LENGTH
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = config.LATENT_DIMENSION
        
        # Training history
        self.training_history = {
            'disc_loss': [],
            'gen_loss': [],
            'disc_acc': [],
            'epoch_times': [],
            'lr_history': []
        }
        
        # Load models
        self._load_models()
        self._setup_optimizers()
        
        print("Advanced GAN Fine-tuner initialized successfully!")

    def _load_models(self):
        """Load pre-trained models with error handling"""
        try:
            print("Loading pre-trained models...")
            self.generator = load_model(self.generator_path)
            self.discriminator = load_model(self.discriminator_path)
            print("✓ Models loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load models: {e}")

    def _setup_optimizers(self):
        """Setup optimizers with configurable learning rates"""
        self.gen_lr = config.GENERATOR_LR
        self.disc_lr = config.DISCRIMINATOR_LR
        
        self.generator_optimizer = Adam(learning_rate=self.gen_lr, beta_1=0.5)
        self.discriminator_optimizer = Adam(learning_rate=self.disc_lr, beta_1=0.5)
        
        # Recompile models
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.discriminator_optimizer,
            metrics=['accuracy']
        )
        
        # Combined model for generator training
        self.discriminator.trainable = False
        z = tf.keras.Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        validity = self.discriminator(generated_seq)
        self.combined = tf.keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

    def freeze_layers(self, model, num_layers_to_freeze):
        """Freeze early layers of a model"""
        for i, layer in enumerate(model.layers):
            if i < num_layers_to_freeze:
                layer.trainable = False
            else:
                layer.trainable = True
        return model

    def get_notes_from_folder(self, folder_path):
        """Extract notes with improved error handling and progress tracking"""
        notes = []
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"⚠️  Folder {folder_path} does not exist!")
            return notes
        
        midi_files = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))
        
        if not midi_files:
            print(f"⚠️  No MIDI files found in {folder_path}")
            return notes
        
        print(f"📁 Processing {len(midi_files)} MIDI files from {folder_path}")
        
        successful_files = 0
        for i, file in enumerate(midi_files):
            try:
                midi = converter.parse(file)
                print(f"  📝 [{i+1}/{len(midi_files)}] {file.name}")
                
                notes_to_parse = midi.flat.notes
                file_notes = 0
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                        file_notes += 1
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                        file_notes += 1
                
                successful_files += 1
                print(f"     ✓ Extracted {file_notes} notes")
                        
            except Exception as e:
                print(f"     ❌ Error parsing {file}: {e}")
                continue
        
        print(f"✓ Successfully processed {successful_files}/{len(midi_files)} files")
        print(f"✓ Total notes extracted: {len(notes)}")
        return notes

    def prepare_finetuning_data(self, new_data_folder, use_original_data=True):
        """Prepare data with optional augmentation"""
        print("🔄 Preparing fine-tuning data...")
        
        # Get new data
        new_notes = self.get_notes_from_folder(new_data_folder)
        
        if not new_notes:
            raise ValueError("❌ No notes found in the new data folder!")
        
        # Combine with original data if requested
        if use_original_data and Path(config.ORIGINAL_DATA_FOLDER).exists():
            print("📚 Including original training data...")
            original_notes = self.get_notes_from_folder(config.ORIGINAL_DATA_FOLDER)
            all_notes = original_notes + new_notes
            print(f"📊 Combined dataset: {len(original_notes):,} original + {len(new_notes):,} new = {len(all_notes):,} total")
        else:
            all_notes = new_notes
            print(f"📊 Using only new data: {len(all_notes):,} notes")
        
        # Calculate vocabulary
        unique_notes = set(all_notes)
        n_vocab = len(unique_notes)
        print(f"🎵 Vocabulary size: {n_vocab} unique notes/chords")
        
        # Apply data augmentation if configured
        if config.PITCH_SHIFT_RANGE > 0:
            all_notes = self._apply_pitch_augmentation(all_notes)
            print(f"🎨 Applied pitch augmentation, new size: {len(all_notes):,}")
        
        # Prepare sequences
        X_train, y_train = prepare_sequences(all_notes, n_vocab)
        
        print(f"✓ Training data prepared:")
        print(f"  📐 Input shape: {X_train.shape}")
        print(f"  📐 Output shape: {y_train.shape}")
        
        return X_train, y_train, all_notes, n_vocab

    def _apply_pitch_augmentation(self, notes):
        """Apply pitch shifting for data augmentation"""
        # This is a simplified augmentation - you can expand this
        augmented_notes = notes.copy()
        
        # Add some randomly pitch-shifted versions
        for _ in range(len(notes) // 4):  # Add 25% more data
            # Simple pitch shifting logic here
            # This is a placeholder - implement proper pitch shifting
            augmented_notes.extend(notes[:100])  # Simplified example
        
        return augmented_notes

    def adaptive_learning_rate(self, epoch, base_lr, warmup_epochs=5):
        """Implement adaptive learning rate with warmup"""
        if epoch < warmup_epochs:
            # Warmup phase
            return base_lr * (epoch + 1) / warmup_epochs
        elif config.LEARNING_RATE_DECAY:
            # Exponential decay after warmup
            return base_lr * (0.95 ** (epoch - warmup_epochs))
        else:
            return base_lr

    def progressive_unfreezing(self, epoch, total_epochs):
        """Gradually unfreeze layers during training"""
        if not config.PROGRESSIVE_UNFREEZING:
            return
        
        # Calculate how many layers to unfreeze based on epoch
        unfreeze_schedule = epoch / total_epochs
        
        # Unfreeze generator layers progressively
        gen_layers_to_unfreeze = int(len(self.generator.layers) * unfreeze_schedule)
        for i, layer in enumerate(self.generator.layers):
            layer.trainable = i < gen_layers_to_unfreeze

    def finetune(self, new_data_folder, epochs=None, batch_size=None, 
                sample_interval=None, use_original_data=None, save_interval=None):
        """
        Advanced fine-tuning with sophisticated techniques
        """
        # Use config defaults if not specified
        epochs = epochs or config.FINETUNE_EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        sample_interval = sample_interval or config.SAMPLE_INTERVAL
        use_original_data = use_original_data if use_original_data is not None else config.USE_ORIGINAL_DATA
        save_interval = save_interval or config.SAVE_INTERVAL
        
        print("🚀 Starting advanced fine-tuning...")
        print(f"⚙️  Configuration:")
        print(f"   📅 Epochs: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Sample interval: {sample_interval}")
        print(f"   💾 Save interval: {save_interval}")
        print(f"   📚 Use original data: {use_original_data}")
        
        # Prepare data
        X_train, y_train, all_notes, n_vocab = self.prepare_finetuning_data(
            new_data_folder, use_original_data
        )
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config.OUTPUT_DIR}/finetune_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        self._save_config(output_dir, {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rates': {'generator': self.gen_lr, 'discriminator': self.disc_lr},
            'data_folder': new_data_folder,
            'use_original_data': use_original_data,
            'vocab_size': n_vocab,
            'training_samples': X_train.shape[0]
        })
        
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        print(f"🎯 Training on {X_train.shape[0]:,} sequences...")
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # Apply adaptive learning rate
            if config.LEARNING_RATE_DECAY or epoch < config.WARMUP_EPOCHS:
                new_gen_lr = self.adaptive_learning_rate(epoch, config.GENERATOR_LR, config.WARMUP_EPOCHS)
                new_disc_lr = self.adaptive_learning_rate(epoch, config.DISCRIMINATOR_LR, config.WARMUP_EPOCHS)
                
                tf.keras.backend.set_value(self.generator_optimizer.learning_rate, new_gen_lr)
                tf.keras.backend.set_value(self.discriminator_optimizer.learning_rate, new_disc_lr)
                
                self.training_history['lr_history'].append({'epoch': epoch, 'gen_lr': new_gen_lr, 'disc_lr': new_disc_lr})
            
            # Progressive unfreezing
            self.progressive_unfreezing(epoch, epochs)
            
            # Training step
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]
            
            # Generate fake sequences
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_seqs = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, real)
            
            # Store training history
            self.training_history['disc_loss'].append(d_loss[0])
            self.training_history['gen_loss'].append(g_loss)
            self.training_history['disc_acc'].append(d_loss[1])
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.training_history['epoch_times'].append(epoch_time)
            
            # Progress reporting
            if epoch % sample_interval == 0:
                elapsed = datetime.now() - start_time
                eta = elapsed * (epochs - epoch - 1) / (epoch + 1) if epoch > 0 else "Unknown"
                
                print(f"🔄 Epoch {epoch:3d}/{epochs} | "
                      f"D_loss: {d_loss[0]:.4f} | D_acc: {100*d_loss[1]:5.1f}% | "
                      f"G_loss: {g_loss:.4f} | Time: {epoch_time:.1f}s | ETA: {str(eta).split('.')[0]}")
                
                if config.GENERATE_SAMPLES:
                    self.generate_sample(all_notes, f"{output_dir}/sample_epoch_{epoch:03d}")
            
            # Save intermediate models
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"{output_dir}/checkpoint_epoch_{epoch:03d}")
        
        # Final saves
        self.save_models(f"{output_dir}/final")
        self.generate_sample(all_notes, f"{output_dir}/final_sample")
        self.plot_training_history(f"{output_dir}/training_history.png")
        self._save_training_history(f"{output_dir}/training_history.json")
        
        total_time = datetime.now() - start_time
        print(f"✅ Fine-tuning completed in {str(total_time).split('.')[0]}")
        print(f"📁 Models and samples saved in: {output_dir}")
        
        return output_dir

    def _save_config(self, output_dir, config_dict):
        """Save training configuration"""
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _save_training_history(self, filepath):
        """Save detailed training history"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def generate_sample(self, input_notes, filename):
        """Generate sample with improved error handling"""
        try:
            pitchnames = sorted(set(item for item in input_notes))
            int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
            n_vocab = len(pitchnames)
            
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            predictions = self.generator.predict(noise, verbose=0)
            
            pred_notes = [x * (n_vocab / 2) + (n_vocab / 2) for x in predictions[0]]
            
            pred_notes_mapped = []
            for x in pred_notes:
                index = int(np.clip(x, 0, len(pitchnames) - 1))
                if index in int_to_note:
                    pred_notes_mapped.append(int_to_note[index])
                else:
                    pred_notes_mapped.append('C5')
            
            create_midi([[note] for note in pred_notes_mapped], filename)
            
        except Exception as e:
            print(f"❌ Error generating sample: {e}")

    def plot_training_history(self, filename):
        """Create comprehensive training plots"""
        if not self.training_history['disc_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['disc_loss'], label='Discriminator', alpha=0.7)
        axes[0, 0].plot(self.training_history['gen_loss'], label='Generator', alpha=0.7)
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['disc_acc'], label='Discriminator Accuracy', color='green', alpha=0.7)
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epoch times
        axes[1, 0].plot(self.training_history['epoch_times'], color='orange', alpha=0.7)
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate history (if available)
        if self.training_history['lr_history']:
            epochs = [item['epoch'] for item in self.training_history['lr_history']]
            gen_lrs = [item['gen_lr'] for item in self.training_history['lr_history']]
            disc_lrs = [item['disc_lr'] for item in self.training_history['lr_history']]
            
            axes[1, 1].plot(epochs, gen_lrs, label='Generator LR', alpha=0.7)
            axes[1, 1].plot(epochs, disc_lrs, label='Discriminator LR', alpha=0.7)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No LR history available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def save_models(self, prefix):
        """Save models with error handling"""
        try:
            self.generator.save(f"{prefix}_generator.h5")
            self.discriminator.save(f"{prefix}_discriminator.h5")
        except Exception as e:
            print(f"❌ Error saving models: {e}")

    def evaluate_model_quality(self, input_notes, num_samples=5):
        """Generate multiple samples for quality evaluation"""
        print(f"🎼 Generating {num_samples} samples for quality evaluation...")
        
        samples_dir = "evaluation_samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        for i in range(num_samples):
            self.generate_sample(input_notes, f"{samples_dir}/evaluation_sample_{i+1}")
        
        print(f"✅ Generated {num_samples} evaluation samples in {samples_dir}/")


def main():
    """Example usage with advanced features"""
    print("🎵 Advanced GAN Fine-tuning Tool")
    print("=" * 50)
    
    # Check for pre-trained models
    if not (Path(config.GENERATOR_MODEL_PATH).exists() and Path(config.DISCRIMINATOR_MODEL_PATH).exists()):
        print("❌ Pre-trained models not found!")
        print(f"   Expected: {config.GENERATOR_MODEL_PATH}")
        print(f"   Expected: {config.DISCRIMINATOR_MODEL_PATH}")
        return
    
    # Check/create data folder
    if not Path(config.NEW_DATA_FOLDER).exists():
        print(f"📁 Creating data folder: {config.NEW_DATA_FOLDER}")
        print("⚠️  Please add your new MIDI files to this folder before running fine-tuning.")
        os.makedirs(config.NEW_DATA_FOLDER, exist_ok=True)
        return
    
    # Initialize fine-tuner
    finetuner = AdvancedGANFineTuner()
    
    # Run fine-tuning
    output_dir = finetuner.finetune(config.NEW_DATA_FOLDER)
    
    # Evaluate results
    notes = finetuner.get_notes_from_folder(config.NEW_DATA_FOLDER)
    if config.USE_ORIGINAL_DATA and Path(config.ORIGINAL_DATA_FOLDER).exists():
        original_notes = finetuner.get_notes_from_folder(config.ORIGINAL_DATA_FOLDER)
        notes.extend(original_notes)
    
    finetuner.evaluate_model_quality(notes, config.NUM_DIVERSITY_SAMPLES)
    
    print("🎉 Fine-tuning process completed successfully!")


if __name__ == "__main__":
    main()
