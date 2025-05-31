"""
Simple and safe script to run GAN fine-tuning
"""

import os
import sys
from pathlib import Path

def main():
    print("🎵 GAN Fine-tuning (Safe Mode)")
    print("=" * 40)
    
    # Check if models exist
    if not (Path("generator_model.h5").exists() and Path("discriminator_model.h5").exists()):
        print("❌ Pre-trained models not found!")
        print("   Expected: generator_model.h5")
        print("   Expected: discriminator_model.h5")
        return
    
    # Check data folder
    data_folder = "data"
    if not Path(data_folder).exists():
        print(f"❌ Data folder '{data_folder}' not found!")
        print("Please create the data folder and add your MIDI files.")
        return
    
    # Check for MIDI files
    midi_files = list(Path(data_folder).glob("*.mid")) + list(Path(data_folder).glob("*.midi"))
    if not midi_files:
        print(f"❌ No MIDI files found in '{data_folder}' folder!")
        print("Please add your MIDI files (.mid or .midi) to the data folder.")
        return
    
    print(f"✅ Found {len(midi_files)} MIDI files in data folder")
    
    # Run fine-tuning
    print("\n🚀 Starting fine-tuning...")
    
    try:
        from finetune_gan_safe import GANFineTuner
        
        finetuner = GANFineTuner()
        output_dir = finetuner.finetune(
            new_data_folder="data",
            epochs=20,  # Start with fewer epochs
            batch_size=16,
            sample_interval=5,
            use_original_data=True,
            save_interval=10
        )
        
        print("\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        print("\n📋 Next steps:")
        print("1. Check the generated MIDI samples")
        print("2. Listen to the music quality")
        print("3. Adjust parameters if needed")
        
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
