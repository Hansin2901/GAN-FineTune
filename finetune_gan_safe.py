import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from music21 import converter, instrument, note, chord, stream
from pathlib import Path
import matplotlib.pyplot as plt
import os
from create_generator_model import prepare_sequences, create_midi


def load_model_safe(model_path):
    """Load model with compatibility fixes for different TensorFlow versions"""
    try:
        # Try loading normally first
        return load_model(model_path)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Trying compatibility mode...")
        
        # Try with compile=False to avoid optimizer issues
        try:
            return load_model(model_path, compile=False)
        except Exception as e2:
            print(f"Compatibility loading also failed: {e2}")
            raise Exception(f"Could not load model {model_path}. This might be due to version incompatibility.")


class GANFineTuner:
    def __init__(self, generator_path="generator_model.h5", discriminator_path="discriminator_model.h5"):
        """
        Initialize the GAN fine-tuner with pre-trained models
        """
        self.seq_length = 100
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss = []
        
        # Load pre-trained models with safety checks
        print("Loading pre-trained models...")
        try:
            self.generator = load_model_safe(generator_path)
            self.discriminator = load_model_safe(discriminator_path)
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            raise
        
        # Set up optimizers with lower learning rates for fine-tuning
        self.generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
        self.discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
        
        # Recompile models with new optimizers
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.discriminator_optimizer,
            metrics=['accuracy']
        )
        
        # Create combined model for generator training
        self.discriminator.trainable = False
        z = tf.keras.Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        validity = self.discriminator(generated_seq)
        self.combined = tf.keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

    def get_notes_from_folder(self, folder_path):
        """Extract notes and chords from MIDI files in a specific folder"""
        notes = []
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Warning: Folder {folder_path} does not exist!")
            return notes
        
        midi_files = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))
        
        if not midi_files:
            print(f"Warning: No MIDI files found in {folder_path}")
            return notes
        
        print(f"Processing {len(midi_files)} MIDI files from {folder_path}")
        
        successful_files = 0
        for i, file in enumerate(midi_files):
            try:
                midi = converter.parse(file)
                if (i + 1) % 50 == 0:  # Progress update every 50 files
                    print(f"  Processed {i+1}/{len(midi_files)} files...")
                
                notes_to_parse = midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                        
                successful_files += 1
                        
            except Exception as e:
                print(f"Error parsing {file}: {e}")
                continue
        
        print(f"Successfully processed {successful_files}/{len(midi_files)} files")
        print(f"Extracted {len(notes)} notes/chords from data")
        return notes

    def prepare_finetuning_data(self, new_data_folder, use_original_data=True):
        """Prepare training data for fine-tuning"""
        print("Preparing fine-tuning data...")
        
        # Get notes from new data
        new_notes = self.get_notes_from_folder(new_data_folder)
        
        if not new_notes:
            raise ValueError("No notes found in the new data folder!")
        
        # Optionally combine with original data
        if use_original_data and Path("archive").exists():
            print("Including original training data...")
            original_notes = self.get_notes_from_folder("archive")
            all_notes = original_notes + new_notes
            print(f"Combined dataset: {len(original_notes):,} original + {len(new_notes):,} new = {len(all_notes):,} total")
        else:
            all_notes = new_notes
            print(f"Using only new data: {len(all_notes):,} notes")
        
        # Calculate vocabulary size
        n_vocab = len(set(all_notes))
        print(f"Vocabulary size: {n_vocab}")
        
        # Prepare sequences
        X_train, y_train = prepare_sequences(all_notes, n_vocab)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Output data shape: {y_train.shape}")
        
        return X_train, y_train, all_notes, n_vocab

    def finetune(self, new_data_folder, epochs=30, batch_size=16, sample_interval=5, 
                use_original_data=True, save_interval=10):
        """Fine-tune the GAN on new data"""
        print(f"Starting fine-tuning for {epochs} epochs...")
        
        # Prepare data
        X_train, y_train, all_notes, n_vocab = self.prepare_finetuning_data(
            new_data_folder, use_original_data
        )
        
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Create output directory
        output_dir = "finetuned_models"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Training on {X_train.shape[0]:,} sequences...")
        
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]
            
            # Generate fake sequences
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_seqs = self.generator.predict(noise, verbose=0)
            
            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, real)
            
            # Store losses
            self.disc_loss.append(d_loss[0])
            self.gen_loss.append(g_loss)
            
            # Print progress
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"D_loss: {d_loss[0]:.4f} | D_acc: {100*d_loss[1]:5.1f}% | "
                      f"G_loss: {g_loss:.4f}")
                
                # Generate sample
                self.generate_sample(all_notes, f"{output_dir}/sample_epoch_{epoch:03d}")
            
            # Save intermediate models
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"{output_dir}/epoch_{epoch:03d}")
        
        # Final saves
        self.save_models(f"{output_dir}/final")
        self.generate_sample(all_notes, f"{output_dir}/final_sample")
        self.plot_losses(f"{output_dir}/training_losses.png")
        
        print("Fine-tuning completed!")
        return output_dir

    def generate_sample(self, input_notes, filename):
        """Generate a sample MIDI file using the current generator"""
        try:
            # Get pitch names and create mapping
            pitchnames = sorted(set(item for item in input_notes))
            int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
            n_vocab = len(pitchnames)
            
            # Generate sequence
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            predictions = self.generator.predict(noise, verbose=0)
            
            # Convert predictions back to note indices (fixed scaling)
            pred_notes = []
            for x in predictions[0]:
                # Scale back from [-1, 1] to [0, n_vocab-1]
                scaled_value = (x + 1) * (n_vocab - 1) / 2
                index = int(np.clip(scaled_value, 0, n_vocab - 1))
                
                if index in int_to_note:
                    pred_notes.append(int_to_note[index])
                else:
                    pred_notes.append('C5')  # Default fallback
            
            # Create MIDI file
            self.create_midi_from_notes(pred_notes, filename)
            
        except Exception as e:
            print(f"Error generating sample: {e}")

    def create_midi_from_notes(self, notes, filename):
        """Create MIDI file from note list"""
        offset = 0
        output_notes = []

        for note_name in notes:
            try:
                # Check if it's a chord (contains '.')
                if '.' in note_name and not note_name.replace('.', '').isdigit():
                    # It's a chord
                    notes_in_chord = note_name.split('.')
                    chord_notes = []
                    for current_note in notes_in_chord:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        chord_notes.append(new_note)
                    new_chord = chord.Chord(chord_notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
                else:
                    # It's a single note
                    new_note = note.Note(note_name)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                    
                offset += 0.5
                
            except Exception as e:
                # Skip invalid notes
                continue

        if output_notes:
            midi_stream = stream.Stream(output_notes)
            midi_stream.write('midi', fp=f'{filename}.mid')
            print(f"Sample generated: {filename}.mid")
        else:
            print(f"Warning: No valid notes to create MIDI file {filename}")

    def plot_losses(self, filename="training_losses.png"):
        """Plot and save training losses"""
        if not self.disc_loss or not self.gen_loss:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.disc_loss, label='Discriminator Loss', color='red', alpha=0.7)
        plt.plot(self.gen_loss, label='Generator Loss', color='blue', alpha=0.7)
        plt.title('GAN Fine-tuning Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved: {filename}")

    def save_models(self, prefix="finetuned"):
        """Save the fine-tuned models"""
        try:
            self.generator.save(f"{prefix}_generator.h5")
            self.discriminator.save(f"{prefix}_discriminator.h5")
            print(f"Models saved with prefix: {prefix}")
        except Exception as e:
            print(f"Error saving models: {e}")


def main():
    """Test the fine-tuner"""
    if not (Path("generator_model.h5").exists() and Path("discriminator_model.h5").exists()):
        print("Error: Pre-trained models not found!")
        return
    
    new_data_folder = "data"
    if not Path(new_data_folder).exists():
        print(f"Error: Data folder '{new_data_folder}' not found!")
        return
    
    # Initialize and run fine-tuning
    finetuner = GANFineTuner()
    finetuner.finetune(
        new_data_folder=new_data_folder,
        epochs=20,  # Start with fewer epochs for testing
        batch_size=16,
        sample_interval=5,
        use_original_data=True,
        save_interval=10
    )


if __name__ == "__main__":
    main()
