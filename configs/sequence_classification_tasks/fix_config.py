import os
import glob

# Find all yaml files
yaml_files = glob.glob("*.yaml")

for yaml_file in yaml_files:
    with open(yaml_file, 'r') as f:
        content = f.read()
    
    # Fix max_steps from 10000 to 1000
    content = content.replace(
        "max_steps: 10000  # Total number of training steps (overrides num_epochs)",
        "max_steps: 1000  # Total number of training steps (overrides num_epochs) - following paper protocol"
    )
    
    # Add a comment about the learning rate grid search requirement
    content = content.replace(
        "learning_rate: 0.00002  # Learning rate for optimizer",
        "learning_rate: 0.00002  # Learning rate for optimizer - NOTE: Paper recommends grid search over [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]"
    )
    
    with open(yaml_file, 'w') as f:
        f.write(content)
    
    print(f"Updated {yaml_file}")

print("\nTo fully comply with the paper's protocol, you should:")
print("1. Run experiments with different learning rates: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]")
print("2. Select the best learning rate based on validation performance")
print("3. The configs now use max_steps=1000 as recommended")
