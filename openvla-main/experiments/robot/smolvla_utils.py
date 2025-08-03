"""Utils for evaluating the SmolVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Import SmolVLA specific components
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAForActionPrediction
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processing_smolvla import SmolVLAProcessor, SmolVLAImageProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Memory-efficient device selection
def get_device():
    """Get the best available device with memory management."""
    if torch.cuda.is_available():
        try:
            # Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory >= 8:  # Need at least 8GB for SmolVLA
                return torch.device("cuda:0")
            else:
                print(f"GPU memory ({gpu_memory:.1f}GB) insufficient, using CPU")
                return torch.device("cpu")
        except Exception as e:
            print(f"GPU memory check failed: {e}, using CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")

DEVICE = get_device()
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for SmolVLA
SMOLVLA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_smolvla(cfg):
    """Loads and returns a SmolVLA model from checkpoint with memory-efficient loading."""
    global DEVICE
    # Load SmolVLA checkpoint.
    print(f"[*] Instantiating Pretrained SmolVLA model on {DEVICE}")
    
    if DEVICE.type == "cpu":
        print("[*] Using CPU - loading in FP32 for compatibility")
        torch_dtype = torch.float32
    else:
        print("[*] Using GPU - loading in FP32 for compatibility (avoiding BF16 precision issues)")
        torch_dtype = torch.float32

    # Register SmolVLA model to HF Auto Classes
    AutoConfig.register("smolvla", SmolVLAConfig)
    AutoImageProcessor.register(SmolVLAConfig, SmolVLAImageProcessor)
    AutoProcessor.register(SmolVLAConfig, SmolVLAProcessor)
    AutoModelForVision2Seq.register(SmolVLAConfig, SmolVLAForActionPrediction)

    # Memory-efficient loading
    try:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch_dtype,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if DEVICE.type == "cuda" else None,
        )
        
        # Move model to device if not using device_map
        if not cfg.load_in_8bit and not cfg.load_in_4bit and DEVICE.type == "cpu":
            vla = vla.to(DEVICE)
        
        # Ensure model is in float32 to avoid precision issues
        vla = vla.float()
        
        # Also convert all model parameters to float32
        for param in vla.parameters():
            param.data = param.data.float()
            
    except torch.cuda.OutOfMemoryError:
        print("[*] GPU out of memory, falling back to CPU")
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load on CPU
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=None,
        )
        vla = vla.to(torch.device("cpu"))
        vla = vla.float()  # Ensure float32 precision
        
        # Also convert all model parameters to float32
        for param in vla.parameters():
            param.data = param.data.float()
            
        DEVICE = torch.device("cpu")
        print(f"[*] Model loaded on CPU")

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base SmolVLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    print("[*] SmolVLA model loaded successfully")

    return vla


def get_smolvla_processor(cfg):
    """Get SmolVLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: Image to crop and resize.
        crop_scale: Scale factor for cropping (0.0 to 1.0).
        batch_size: Batch size for the image.

    Returns:
        Cropped and resized image.
    """
    if crop_scale == 1.0:
        return image

    # Get image dimensions
    if len(image.shape) == 4:  # Batch of images
        h, w = image.shape[1], image.shape[2]
    else:  # Single image
        h, w = image.shape[0], image.shape[1]

    # Calculate crop dimensions
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)

    # Calculate crop coordinates (center crop)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    # Crop the image
    if len(image.shape) == 4:  # Batch of images
        cropped = image[:, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
    else:  # Single image
        cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w, :]

    # Resize back to original size
    if len(cropped.shape) == 4:  # Batch of images
        resized = tf.image.resize(cropped, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    else:  # Single image
        resized = tf.image.resize(tf.expand_dims(cropped, 0), (h, w), method=tf.image.ResizeMethod.BILINEAR)
        resized = tf.squeeze(resized, 0)

    return resized


def get_smolvla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """
    Get action prediction from SmolVLA model.

    Args:
        vla: SmolVLA model instance.
        processor: SmolVLA processor instance.
        base_vla_name: Name of the base VLA model.
        obs: Observation containing images and state.
        task_label: Task description.
        unnorm_key: Key for unnormalization.
        center_crop: Whether to apply center cropping.

    Returns:
        Predicted action.
    """
    # Prepare images
    images = []
    for i in range(1, 4):  # Assuming 3 camera views
        img_key = f"image{i}" if i > 1 else "image"
        if img_key in obs:
            img = obs[img_key]
            if center_crop:
                img = crop_and_resize(img, 0.8, 1)
            images.append(img)
    
    if not images:
        print("WARNING: No images found in observation")
        return np.zeros(ACTION_DIM)

    # Prepare state
    state = obs.get("state", np.zeros(6))
    
    # Create input for SmolVLA
    inputs = {
        "images": images,
        "text": task_label,
        "state": state
    }
    
    # Process inputs
    processed_inputs = processor(inputs, return_tensors="pt")
    
    # Move to device
    for key in processed_inputs:
        if isinstance(processed_inputs[key], torch.Tensor):
            processed_inputs[key] = processed_inputs[key].to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = vla(**processed_inputs)
        action = outputs.action.cpu().numpy()
    
    # Unnormalize if needed
    if hasattr(vla, 'norm_stats') and unnorm_key in vla.norm_stats:
        stats = vla.norm_stats[unnorm_key]
        action = action * stats['std'] + stats['mean']
    
    return action


def create_smolvla_integration_script():
    """Create a script to integrate SmolVLA with the existing OpenVLA setup."""
    script_content = '''#!/usr/bin/env python3
"""
SmolVLA Integration Script
This script demonstrates how to use SmolVLA as a VLA model in the existing OpenVLA framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.robot.smolvla_utils import get_smolvla, get_smolvla_processor, get_smolvla_action
import torch
import numpy as np
from PIL import Image

def main():
    """Main function to demonstrate SmolVLA integration."""
    
    # Configuration
    class Config:
        def __init__(self):
            self.pretrained_checkpoint = "lerobot/smolvla_base"  # Use the base SmolVLA model
            self.load_in_8bit = False
            self.load_in_4bit = False
    
    cfg = Config()
    
    try:
        # Load SmolVLA model and processor
        print("Loading SmolVLA model...")
        vla = get_smolvla(cfg)
        processor = get_smolvla_processor(cfg)
        
        print("SmolVLA model loaded successfully!")
        
        # Example usage
        print("\\nSmolVLA Integration Complete!")
        print("You can now use SmolVLA as your VLA model in the existing framework.")
        print("\\nTo use it in your evaluation scripts:")
        print("1. Import the smolvla_utils module")
        print("2. Use get_smolvla() to load the model")
        print("3. Use get_smolvla_action() to get predictions")
        
    except Exception as e:
        print(f"Error loading SmolVLA: {e}")
        print("Make sure you have enough disk space and the model is available.")

if __name__ == "__main__":
    main()
'''
    
    script_path = "openvla-main/experiments/robot/smolvla_integration_demo.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created SmolVLA integration demo script at: {script_path}")
    return script_path


if __name__ == "__main__":
    # Create the integration script
    script_path = create_smolvla_integration_script()
    print(f"SmolVLA utilities created successfully!")
    print(f"Integration demo script: {script_path}") 