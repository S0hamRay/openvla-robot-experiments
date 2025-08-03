# download_smallest_checkpoint.py
from huggingface_hub import snapshot_download

print("Downloading smallest OpenVLA checkpoint (Phi-2 3B)...")
snapshot_download(repo_id="openvla/openvla-v01-7b", local_dir="openvla-v01-7b")
print("Download complete!")
