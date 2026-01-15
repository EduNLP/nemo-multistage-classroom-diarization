import json
import os
from tqdm import tqdm
from vad_inference import load_vad_model, vad

import argparse

parser = argparse.ArgumentParser(description="Run VAD script with specified arguments.")

parser.add_argument(
    "--manifest_file", 
    type=str, 
    default="/nlp/scr/askhan1/umd_dataset/UMD_test_manifest_2s_denoised.json", 
    help="Path to the input manifest file."
)
parser.add_argument(
    "--checkpoint_path", 
    type=str, 
    default="/nlp/scr/askhan1/nemo_w2v_ahmed_umd/checkpoints/umd.pt", 
    help="Path to the VAD model checkpoint."
)

parser.add_argument(
    "--vad_manifest_path", 
    type=str, 
    default="./test-denoised.json", 
    help="Path to the output VAD manifest file."
)

parser.add_argument(
    "--frames_output_path", 
    type=str, 
    default="./test-denoised.json", 
    help="Path to the frame output DIR"
)


args = parser.parse_args()

# print("INFERENCE ON BOTH ---------")
# /nlp/scr/askhan1/CLASSBANK/CLASSBANK_test_manifest_2s_noisy.json
manifest_file=args.manifest_file
checkpoint_path=args.checkpoint_path
vad_manifest_path=args.vad_manifest_path
frames_output=args.frames_output_path

print(f"Using:")
print(f"Manifest: {manifest_file}")
print(f"CKPT: {checkpoint_path}")
print(f"vad_manifest_path: {vad_manifest_path}\n\n")

if not os.path.exists(manifest_file):
    print(f"Manifest file not found at: {manifest_file}")
    exit()

print(f"Manifest file found at: {manifest_file}")

with open(manifest_file, "r") as file:
    manifest_lines = file.readlines()

# print("Loading VAD model...")
model = load_vad_model(checkpoint_path)
# print("VAD Model loaded successfully")

for line in tqdm(manifest_lines):
    try:
        entry = json.loads(line.strip())
        audio_path = entry.get("audio_filepath")
        rttm_path = entry.get('rttm_filepath')
        print("Trying", audio_path)
        if not audio_path:
            print("Missing audio_filepath in entry:", entry)
            continue

        vad(audio_path, model, frames_output_path=frames_output, vad_manifest_path=vad_manifest_path)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON line: {line.strip()}\nError: {e}")

