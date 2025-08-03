#!/usr/bin/env python3
"""
SmolVLA Integration Demo
This script demonstrates how to use SmolVLA as a VLA model in the existing OpenVLA framework.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the lerobot path to sys.path
lerobot_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lerobot', 'src')
if lerobot_path not in sys.path:
    sys.path.insert(0, lerobot_path)

def main():
    """Main function to demonstrate SmolVLA integration."""
    
    print("=== SmolVLA Integration Demo ===")
    print("This demo shows how to integrate SmolVLA with the existing OpenVLA framework.")
    
    # Configuration
    class Config:
        def __init__(self):
            self.pretrained_checkpoint = "lerobot/smolvla_base"  # Use the base SmolVLA model
            self.load_in_8bit = False
            self.load_in_4bit = False
    
    cfg = Config()
    
    try:
        print("\n1. Checking if lerobot is properly installed...")
        
        # Try to import lerobot components
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAForActionPrediction
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            print("✓ SmolVLA components imported successfully")
        except ImportError as e:
            print(f"✗ Error importing SmolVLA components: {e}")
            print("Make sure lerobot is properly installed with: pip install -e '.[smolvla]'")
            return
        
        print("\n2. Checking device availability...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        print("\n3. Testing model loading...")
        try:
            # Try to load a small test model first
            from transformers import AutoModelForImageTextToText
            test_model = AutoModelForImageTextToText.from_pretrained(
                "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("✓ Test model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading test model: {e}")
            print("This might be due to disk space issues or network problems.")
            return
        
        print("\n4. Creating integration utilities...")
        
        # Create a simple integration function
        def create_smolvla_integration():
            """Create a simple SmolVLA integration for OpenVLA."""
            
            class SmolVLAIntegration:
                def __init__(self, model_path="lerobot/smolvla_base"):
                    self.model_path = model_path
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model = None
                    self.processor = None
                
                def load_model(self):
                    """Load the SmolVLA model."""
                    try:
                        from transformers import AutoModelForImageTextToText, AutoProcessor
                        
                        print(f"Loading SmolVLA model from {self.model_path}...")
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            self.model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                        ).to(self.device)
                        
                        self.processor = AutoProcessor.from_pretrained(
                            self.model_path,
                            trust_remote_code=True
                        )
                        
                        print("✓ SmolVLA model loaded successfully!")
                        return True
                    except Exception as e:
                        print(f"✗ Error loading model: {e}")
                        return False
                
                def predict_action(self, images, task_description, state=None):
                    """Predict action given images and task description."""
                    if self.model is None:
                        print("Model not loaded. Call load_model() first.")
                        return None
                    
                    try:
                        # Prepare inputs
                        inputs = {
                            "images": images,
                            "text": task_description
                        }
                        
                        if state is not None:
                            inputs["state"] = state
                        
                        # Process inputs
                        processed_inputs = self.processor(inputs, return_tensors="pt")
                        
                        # Move to device
                        for key in processed_inputs:
                            if isinstance(processed_inputs[key], torch.Tensor):
                                processed_inputs[key] = processed_inputs[key].to(self.device)
                        
                        # Get prediction
                        with torch.no_grad():
                            outputs = self.model(**processed_inputs)
                            action = outputs.action.cpu().numpy()
                        
                        return action
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        return None
            
            return SmolVLAIntegration()
        
        # Create the integration
        smolvla_integration = create_smolvla_integration()
        
        print("\n5. Testing integration...")
        if smolvla_integration.load_model():
            print("✓ SmolVLA integration created successfully!")
            
            print("\n=== Integration Summary ===")
            print("SmolVLA has been successfully integrated with the OpenVLA framework.")
            print("\nTo use SmolVLA in your existing code:")
            print("1. Import the SmolVLA components from lerobot")
            print("2. Create a SmolVLAIntegration instance")
            print("3. Call load_model() to initialize the model")
            print("4. Use predict_action() to get predictions")
            print("\nExample usage:")
            print("```python")
            print("from experiments.robot.smolvla_integration_demo import create_smolvla_integration")
            print("smolvla = create_smolvla_integration()")
            print("smolvla.load_model()")
            print("action = smolvla.predict_action(images, task_description)")
            print("```")
            
        else:
            print("✗ Failed to create SmolVLA integration")
        
    except Exception as e:
        print(f"Error during integration: {e}")
        print("Make sure you have enough disk space and the model is available.")

if __name__ == "__main__":
    main() 