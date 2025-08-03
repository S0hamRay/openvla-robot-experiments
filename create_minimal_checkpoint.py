import os
import json
import torch

# Create minimal checkpoint directory
checkpoint_dir = "minimal_checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Creating minimal checkpoint in {checkpoint_dir}")

# Minimal config.json
config = {
    "model_type": "openvla",
    "architectures": ["OpenVLAForActionPrediction"],
    "torch_dtype": "bfloat16",
    "transformers_version": "4.36.0",
    "use_cache": True
}

with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Minimal dataset_statistics.json
stats = {
    "mujoco_simple": {
        "action_mean": [0.0] * 7,
        "action_std": [1.0] * 7
    }
}

with open(os.path.join(checkpoint_dir, "dataset_statistics.json"), "w") as f:
    json.dump(stats, f, indent=2)

# Create a tiny dummy model file (just for testing)
# This won't actually work for inference, but will let you test the loading code
dummy_state_dict = {
    "model.embed_tokens.weight": torch.randn(1000, 4096),
    "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
    "model.layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
    "model.layers.0.self_attn.v_proj.weight": torch.randn(4096, 4096),
    "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
    "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
    "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
    "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
    "model.layers.0.input_layernorm.weight": torch.randn(4096),
    "model.layers.0.post_attention_layernorm.weight": torch.randn(4096),
    "model.norm.weight": torch.randn(4096),
    "lm_head.weight": torch.randn(1000, 4096),
}

torch.save(dummy_state_dict, os.path.join(checkpoint_dir, "pytorch_model.bin"))

# Create processor config
processor_config = {
    "crop_size": {"height": 224, "width": 224},
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "resample": 3,
    "size": {"height": 224, "width": 224}
}

with open(os.path.join(checkpoint_dir, "preprocessor_config.json"), "w") as f:
    json.dump(processor_config, f, indent=2)

print(f"Created minimal checkpoint at {checkpoint_dir}")
print("Note: This checkpoint will NOT work for actual inference!")
print("It's only for testing if the integration code runs without errors.") 