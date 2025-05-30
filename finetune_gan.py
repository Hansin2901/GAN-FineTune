import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from music21 import converter, instrument, note, chord, stream
from pathlib import Path
import matplotlib.pyplot as plt
from create_generator_model import prepare_sequences, create_midi
import os


class GANFineTuner:
    def __init__(self, generator_path="generator_model.h5", discriminator_path="discriminator_model.h5"):
        """
        Initialize the GAN fine-tuner with pre-trained models
        
        Args:
            generator_path: Path to the saved generator model
            discriminator_path: Path to the saved discriminator model
        """
        self.seq_length = 100
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss = []
        
        # Load pre-trained models
        print("Loading pre-trained models...")
        self.generator = load_model(generator_path)
        self.discriminator = load_model(discriminator_path)
        
        # Set up optimizers with potentially lower learning rates for fine-tuning
        self.generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)  # Lower LR for fine-tuning
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
        
        print("Models loaded and compiled successfully!")

    def get_notes_from_folder(self, folder_path):
        """
        Extract notes and chords from MIDI files in a specific folder
        
        Args:
            folder_path: Path to folder containing MIDI files
            
        Returns:
            List of notes and chords as strings
        """
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
        
        for file in midi_files:
            try:
                midi = converter.parse(file)
                print(f"Parsing {file.name}")
                
                notes_to_parse = midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                        
            except Exception as e:
                print(f"Error parsing {file}: {e}")
                continue
        
        print(f"Extracted {len(notes)} notes/chords from new data")
        return notes

    def prepare_finetuning_data(self, new_data_folder, use_original_data=True):
        """
        Prepare training data for fine-tuning
        
        Args:
            new_data_folder: Path to folder with new MIDI files
            use_original_data: Whether to combine with original training data
            
        Returns:
            X_train, y_train: Prepared training data
        """
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
            print(f"Combined dataset: {len(original_notes)} original + {len(new_notes)} new = {len(all_notes)} total notes")
        else:
            all_notes = new_notes
            print(f"Using only new data: {len(all_notes)} notes")
        
        # Calculate vocabulary size
        n_vocab = len(set(all_notes))
        print(f"Vocabulary size: {n_vocab}")
        
        # Prepare sequences
        X_train, y_train = prepare_sequences(all_notes, n_vocab)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Output data shape: {y_train.shape}")
        
        return X_train, y_train, all_notes, n_vocab

    def finetune(self, new_data_folder, epochs=50, batch_size=16, sample_interval=10, 
                use_original_data=True, save_interval=20):
        """
        Fine-tune the GAN on new data
        
        Args:
            new_data_folder: Path to folder containing new MIDI files
            epochs: Number of fine-tuning epochs
            batch_size: Batch size for training
            sample_interval: Interval for printing progress
            use_original_data: Whether to include original training data
            save_interval: Interval for saving intermediate models
        """
        print(f"Starting fine-tuning for {epochs} epochs...")
        
        # Prepare data
        X_train, y_train, all_notes, n_vocab = self.prepare_finetuning_data(
            new_data_folder, use_original_data
        )
        
        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Create output directory for fine-tuned models
        output_dir = "finetuned_models"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Fine-tuning on {X_train.shape[0]} sequences...")
        
        for epoch in range(epochs):
            # Train Discriminator
            # Select random batch of real sequences
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
                print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
                
                # Generate sample music
                self.generate_sample(all_notes, f"finetuned_sample_epoch_{epoch}")
            
            # Save intermediate models
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"{output_dir}/epoch_{epoch}")
                print(f"Models saved at epoch {epoch}")
        
        # Final save
        self.save_models(f"{output_dir}/final")
        
        # Generate final sample and plot losses
        self.generate_sample(all_notes, "finetuned_final")
        self.plot_losses("finetuning_losses.png")
        
        print("Fine-tuning completed!")

    def generate_sample(self, input_notes, filename):
        """
        Generate a sample MIDI file using the current state of the generator
        
        Args:
            input_notes: List of notes for vocabulary mapping
            filename: Output filename (without extension)
        """
        try:
            # Get pitch names and create mapping
            pitchnames = sorted(set(item for item in input_notes))
            int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
            n_vocab = len(pitchnames)
            
            # Generate sequence
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            predictions = self.generator.predict(noise, verbose=0)
            
            # Convert predictions back to note indices
            pred_notes = [x * (n_vocab / 2) + (n_vocab / 2) for x in predictions[0]]
            
            # Map to actual notes with error handling
            pred_notes_mapped = []
            for x in pred_notes:
                index = int(np.clip(x, 0, len(pitchnames) - 1))
                if index in int_to_note:
                    pred_notes_mapped.append(int_to_note[index])
                else:
                    pred_notes_mapped.append('C5')  # Default fallback
            
            # Create MIDI file
            create_midi([[note] for note in pred_notes_mapped], filename)
            print(f"Sample generated: {filename}.mid")
            
        except Exception as e:
            print(f"Error generating sample: {e}")

    def plot_losses(self, filename="finetuning_losses.png"):
        """
        Plot and save the training losses
        
        Args:
            filename: Output filename for the plot
        """
        if not self.disc_loss or not self.gen_loss:
            print("No loss data to plot")
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
        """
        Save the fine-tuned models
        
        Args:
            prefix: Prefix for the saved model files
        """
        try:
            self.generator.save(f"{prefix}_generator.h5")
            self.discriminator.save(f"{prefix}_discriminator.h5")
            print(f"Models saved with prefix: {prefix}")
        except Exception as e:
            print(f"Error saving models: {e}")

    def evaluate_diversity(self, input_notes, num_samples=5):
        """
        Generate multiple samples to evaluate diversity
        
        Args:
            input_notes: List of notes for vocabulary mapping
            num_samples: Number of samples to generate
        """
        print(f"Generating {num_samples} samples for diversity evaluation...")
        
        for i in range(num_samples):
            self.generate_sample(input_notes, f"diversity_sample_{i+1}")
        
        print(f"Generated {num_samples} samples for comparison")


def main():
    """
    Example usage of the GANFineTuner
    """
    print("GAN Fine-tuning Script")
    print("=" * 50)
    
    # Check if pre-trained models exist
    if not (Path("generator_model.h5").exists() and Path("discriminator_model.h5").exists()):
        print("Error: Pre-trained models not found!")
        print("Please ensure 'generator_model.h5' and 'discriminator_model.h5' exist.")
        return
    
    # Initialize fine-tuner
    finetuner = GANFineTuner()
    
    # Set the path to your new data folder
    # You'll need to create this folder and add your new MIDI files
    new_data_folder = "data"  # Change this to your data folder path
    
    if not Path(new_data_folder).exists():
        print(f"Creating data folder: {new_data_folder}")
        print("Please add your new MIDI files to this folder before running fine-tuning.")
        os.makedirs(new_data_folder, exist_ok=True)
        return
    
    # Fine-tune the model
    finetuner.finetune(
        new_data_folder=new_data_folder,
        epochs=50,  # Adjust as needed
        batch_size=16,
        sample_interval=5,
        use_original_data=True,  # Set to False to train only on new data
        save_interval=10
    )
    
    # Generate diversity samples
    notes = finetuner.get_notes_from_folder(new_data_folder)
    if Path("archive").exists():
        original_notes = finetuner.get_notes_from_folder("archive")
        notes.extend(original_notes)
    
    finetuner.evaluate_diversity(notes, num_samples=3)


if __name__ == "__main__":
    main()
