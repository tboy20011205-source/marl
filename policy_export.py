"""
Export policy models for self-play inference.
Loads trained models from custom PPO checkpoints and exports them.
"""

import os
import torch
from models.torch_models_hetero import Fight1, Fight2, Esc1, Esc2

# Define experiment folder name
LEVEL = 3
MODE = 'fight'
EXP_DIR = f'L{LEVEL}_{MODE}_2-vs-2'

# Define policy folder name
POL_DIR = 'policies'

# Map mode to model classes
if MODE == "fight":
    model_classes = {1: Fight1, 2: Fight2}
else:
    model_classes = {1: Esc1, 2: Esc2}

os.makedirs(POL_DIR, exist_ok=True)

for i in range(1, 3):
    # Load from custom PPO checkpoint
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), 'results', EXP_DIR,
        'checkpoint', f'ac{i}_policy.pt')

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        continue

    # Create model and load weights
    model = model_classes[i]()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # Export for self-play (save the full model)
    policy_name = f'L{LEVEL}_AC{i}_{MODE}'
    save_path = os.path.join(POL_DIR, f'{policy_name}.pt')
    torch.save(model, save_path)
    print(f"Exported {policy_name} to {save_path}")

print(f"{MODE} policies exported to folder {POL_DIR}")
