import json
import os
import argparse

def rewrite_paths(input_path, output_path, new_base_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    for sample in data:
        if 'chosen_path' in sample:
            filename = os.path.basename(sample['chosen_path'])
            sample['chosen_path'] = os.path.join(new_base_path, filename)

        if 'reject_path' in sample:
            filename = os.path.basename(sample['reject_path'])
            sample['reject_path'] = os.path.join(new_base_path, filename)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite image paths in JSON.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--base_path", type=str, required=True, help="New base path for image files")

    args = parser.parse_args()
    rewrite_paths(args.input, args.output, args.base_path)