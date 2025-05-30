"""
Simple script to run GAN fine-tuning
This script provides an easy interface to fine-tune your GAN
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    print("🔍 Checking requirements...")
    
    required_files = [
        "generator_model.h5",
        "discriminator_model.h5", 
        "create_generator_model.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def setup_data_folder():
    """Create data folder if it doesn't exist"""
    data_folder = "data"
    
    if not Path(data_folder).exists():
        os.makedirs(data_folder)
        print(f"📁 Created data folder: {data_folder}")
        print("📝 Please add your new MIDI files (.mid or .midi) to this folder")
        return False
    
    # Check if folder has MIDI files
    midi_files = list(Path(data_folder).glob("*.mid")) + list(Path(data_folder).glob("*.midi"))
    
    if not midi_files:
        print(f"⚠️  Data folder exists but contains no MIDI files")
        print(f"📝 Please add your new MIDI files (.mid or .midi) to: {data_folder}")
        return False
    
    print(f"✅ Found {len(midi_files)} MIDI files in data folder")
    return True

def run_basic_finetune():
    """Run basic fine-tuning"""
    print("🚀 Starting basic fine-tuning...")
    
    try:
        from finetune_gan import GANFineTuner
        
        # Initialize and run
        finetuner = GANFineTuner()
        finetuner.finetune(
            new_data_folder="data",
            epochs=30,  # Start with fewer epochs
            batch_size=16,
            sample_interval=5,
            use_original_data=True,
            save_interval=10
        )
        
        print("✅ Basic fine-tuning completed!")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        return False
    
    return True

def run_advanced_finetune():
    """Run advanced fine-tuning with more features"""
    print("🚀 Starting advanced fine-tuning...")
    
    try:
        from advanced_finetune import AdvancedGANFineTuner
        
        # Initialize and run
        finetuner = AdvancedGANFineTuner()
        output_dir = finetuner.finetune("data")
        
        # Generate evaluation samples
        notes = finetuner.get_notes_from_folder("data")
        if Path("archive").exists():
            original_notes = finetuner.get_notes_from_folder("archive")
            notes.extend(original_notes)
        
        finetuner.evaluate_model_quality(notes, 3)
        
        print("✅ Advanced fine-tuning completed!")
        print(f"📁 Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during advanced fine-tuning: {e}")
        return False
    
    return True

def main():
    print("🎵 GAN Fine-tuning Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please ensure you have the required files.")
        return
    
    # Setup data folder
    if not setup_data_folder():
        print("\n⏸️  Setup incomplete. Add MIDI files to the data folder and run again.")
        return
    
    # Ask user which version to run
    print("\n🎯 Choose fine-tuning method:")
    print("1. Basic fine-tuning (simple, faster)")
    print("2. Advanced fine-tuning (more features, sophisticated)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            success = run_basic_finetune()
            break
        elif choice == "2":
            success = run_advanced_finetune()
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    if success:
        print("\n🎉 Fine-tuning process completed!")
        print("\n📋 Next steps:")
        print("1. Check the generated MIDI files")
        print("2. Use the fine-tuned models for music generation")
        print("3. Adjust parameters in finetune_config.py if needed")
    else:
        print("\n❌ Fine-tuning failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
