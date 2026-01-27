"""
Shared utilities for training configuration and interactive CLI.
"""

import sys
from typing import Dict, Any

def set_training_defaults(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update training configuration from defaults and CLI arguments.

    defaults: Dictionary of default configuration values.

    Returns an updated configuration dictionary.
    """
    config = defaults.copy()
    
    # Arg validation
    if len(sys.argv) > 3:
        print("Error: Too many arguments.")
        print("Usage: python -m train.train_deep [asset] [-n]")
        sys.exit(1)
        
    skip_modify = False
    
    # Handle asset and -n flag
    if len(sys.argv) >= 2:
        # First arg is asset
        valid_assets = ["crypto", "equities", "forex", "comm", "interest"]
        if sys.argv[1].lower() in valid_assets:
            config["asset"] = sys.argv[1].lower()
        else:
            print(f"Warning: '{sys.argv[1]}' is not a recognized asset type.")
            
    if len(sys.argv) == 3:
        if sys.argv[2].lower() == "-n":
            skip_modify = True
        else:
            print(f"Error: Unrecognized argument '{sys.argv[2]}'. Use '-n' to skip modification.")
            sys.exit(1)

    print(f"Training Config - Defaults: {', '.join([f'{k}={v}' for k, v in config.items()])}")
    
    if skip_modify:
        modify = "n"
    else:
        modify = input("Modify defaults? (y/n): ").strip().lower()
    
    if modify in ["y", "yes", "ya", "yeah", "yep", "yea"]:
        for key, value in config.items():
            prompt = f"{key.replace('_', ' ').capitalize()} ({value}): "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
                
            # Type casting based on default value type
            if isinstance(value, bool):
                config[key] = user_input.lower() in ["y", "yes", "true", "1"]
            elif isinstance(value, int):
                if user_input.isdigit():
                    config[key] = int(user_input)
            elif isinstance(value, float):
                try:
                    config[key] = float(user_input)
                except ValueError:
                    pass
            else:
                config[key] = user_input
                
    print(f"\nFinal configuration: {', '.join([f'{k}={v}' for k, v in config.items()])}\n")
    return config
