"""
Debug script to test GAN fine-tuning with better error handling
"""

import os
import sys
from pathlib import Path
import traceback

def test_basic_finetune():
    """Test basic fine-tuning with detailed error reporting"""
    print("🔧 Testing basic fine-tuning with debug info...")
    
    try:
        from finetune_gan import GANFineTuner
        
        # Check data folder
        data_folder = "data"
        if not Path(data_folder).exists():
            print(f"❌ Data folder '{data_folder}' does not exist")
            return False
        
        midi_files = list(Path(data_folder).glob("*.mid")) + list(Path(data_folder).glob("*.midi"))
        print(f"📁 Found {len(midi_files)} MIDI files in {data_folder}")
        
        if len(midi_files) == 0:
            print("❌ No MIDI files found in data folder")
            return False
        
        # Initialize fine-tuner
        print("🔄 Initializing GANFineTuner...")
        finetuner = GANFineTuner()
        print("✅ GANFineTuner initialized successfully")
        
        # Test data loading
        print("🔄 Testing data loading...")
        notes = finetuner.get_notes_from_folder(data_folder)
        print(f"✅ Loaded {len(notes)} notes from data folder")
        
        if len(notes) == 0:
            print("❌ No notes extracted from MIDI files")
            return False
        
        # Test data preparation
        print("🔄 Testing data preparation...")
        X_train, y_train, all_notes, n_vocab = finetuner.prepare_finetuning_data(
            data_folder, use_original_data=False  # Start with just new data
        )
        print(f"✅ Data prepared: X_train={X_train.shape}, vocab_size={n_vocab}")
        
        # Test sample generation
        print("🔄 Testing sample generation...")
        finetuner.generate_sample(all_notes, "test_sample")
        print("✅ Sample generation test passed")
        
        # Run short fine-tuning
        print("🔄 Running short fine-tuning test (3 epochs)...")
        finetuner.finetune(
            new_data_folder=data_folder,
            epochs=3,  # Very short test
            batch_size=8,  # Smaller batch
            sample_interval=1,
            use_original_data=False,  # Just new data for now
            save_interval=5
        )
        
        print("✅ Basic fine-tuning test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during fine-tuning test:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def main():
    print("🧪 GAN Fine-tuning Debug Tool")
    print("=" * 50)
    
    # Check requirements
    required_files = ["generator_model.h5", "discriminator_model.h5", "create_generator_model.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        return
    
    print("✅ All required files found")
    
    # Run test
    success = test_basic_finetune()
    
    if success:
        print("\n🎉 Debug test completed successfully!")
        print("✅ You can now run the full fine-tuning process")
    else:
        print("\n❌ Debug test failed")
        print("🔧 Please check the error messages above")

if __name__ == "__main__":
    main()
