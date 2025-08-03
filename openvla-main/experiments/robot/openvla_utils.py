"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.vlas.sparse_autoencoder import SparseAutoencoderTorch

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
            if gpu_memory >= 12:  # Need at least 12GB for the model
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

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint with memory-efficient loading."""
    global DEVICE
    # Load VLA checkpoint.
    print(f"[*] Instantiating Pretrained VLA model on {DEVICE}")
    
    if DEVICE.type == "cpu":
        print("[*] Using CPU - loading in FP32 for compatibility")
        torch_dtype = torch.float32
    else:
        print("[*] Using GPU - loading in FP32 for compatibility (avoiding BF16 precision issues)")
        torch_dtype = torch.float32

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

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
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    # === SPARSE AUTOENCODER WILL BE CREATED AND ATTACHED LATER ===
    # The sparse autoencoder will be created and trained in the main script
    # to ensure it's properly trained on the environment's action distribution
    print("[*] Sparse autoencoder will be created and attached later in the main script")

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    print("get_vla_action: Starting...")
    print('get_vla_action: vla repr:', vla)
    print('get_vla_action: vla class:', vla.__class__)
    print('get_vla_action: vla id:', id(vla))
    print('get_vla_action: vla dir:', dir(vla))
    print('get_vla_action: vla predict_action:', getattr(vla, 'predict_action', None))
    print('get_vla_action: vla sparse_autoencoder:', getattr(vla, 'sparse_autoencoder', None))
    for attr in ['model', 'base_model', 'module']:
        if hasattr(vla, attr):
            sub = getattr(vla, attr)
            print(f'get_vla_action: vla.{attr} repr:', sub)
            print(f'get_vla_action: vla.{attr} class:', sub.__class__)
            print(f'get_vla_action: vla.{attr} id:', id(sub))
            print(f'get_vla_action: vla.{attr} dir:', dir(sub))
            print(f'get_vla_action: vla.{attr} predict_action:', getattr(sub, 'predict_action', None))
            print(f'get_vla_action: vla.{attr} sparse_autoencoder:', getattr(sub, 'sparse_autoencoder', None))
    if hasattr(vla, "sparse_autoencoder"):
        print(f'get_vla_action: vla.sparse_autoencoder is None? {getattr(vla, "sparse_autoencoder", None) is None}')
    # FORCE: Always overwrite vla.sparse_autoencoder with the trained one if available
    import builtins
    if hasattr(builtins, 'GLOBAL_TRAINED_SAE'):
        print('get_vla_action: Forcibly overwriting vla.sparse_autoencoder with GLOBAL_TRAINED_SAE')
        vla.sparse_autoencoder = builtins.GLOBAL_TRAINED_SAE
    else:
        print('get_vla_action: No GLOBAL_TRAINED_SAE found!')
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")
    print("get_vla_action: Image converted to PIL")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    print(f"get_vla_action: Prompt created: {prompt}")

    # Process inputs.
    print("get_vla_action: Processing inputs with processor...")
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.float32)
    print("get_vla_action: Inputs processed and moved to device")

    # Get action.
    print("get_vla_action: Calling vla.predict_action...")
    result = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    print("get_vla_action: predict_action result:", result)
    print("get_vla_action: predict_action result type:", type(result))
    if isinstance(result, dict) and "actions" in result and "sparse_code" in result:
        print(f"get_vla_action: Action received: {result['actions']}")
        print(f"get_vla_action: Sparse code: {result['sparse_code']}")
        return result['actions'], result['sparse_code']
    else:
        print(f"get_vla_action: Action received: {result}")
        return result



