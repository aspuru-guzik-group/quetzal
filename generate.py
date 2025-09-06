from dataclasses import dataclass, asdict, MISSING
from train import Config, LitQuetzal
import argparse
import os
import torch
import time
import json

SAVE_LOCATION = "samples/gen"

@dataclass
class GenConfig:
    ckpt: str
    name: str
    device: str = "cuda"

    num_samples: int = 10000
    num_chunks: int = 1
    diff_steps: int = 100
    truncate_w: float = 1.0
    truncate_t: float = 0.0
    max_len: int = 32

def gen(config):
    # create the output directory
    output_dir = os.path.join(SAVE_LOCATION, config.name)
    os.makedirs(output_dir, exist_ok=False)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)

    lit = LitQuetzal.load_from_checkpoint(config.ckpt, map_location=config.device)
    model = lit.ema.module
    model.eval()

    assert config.num_samples % config.num_chunks == 0, "num_samples must be divisible by num_chunks"

    total_start_time = time.time()  # Start timing the entire process

    timings = []  # List to store timing information
    all_out = []
    for i in range(config.num_chunks):
        print(f"Generating chunk {i + 1}/{config.num_chunks}...")
        start_time = time.time()  # Start timing
        out = model.generate(
            config.num_samples // config.num_chunks,
            device=config.device,
            num_steps=config.diff_steps,
            truncate_w=config.truncate_w,
            truncate_t=config.truncate_t,
            max_len=config.max_len,
            pbar=True
        )
        end_time = time.time()  # End timing
        duration = end_time - start_time
        print(f"Chunk {i + 1} generated in {duration:.2f} seconds.")
        timings.append(duration)  # Save timing for this chunk
        all_out.append(out)

    total_end_time = time.time()  # End timing the entire process
    total_duration = total_end_time - total_start_time
    print(f"Total generation time: {total_duration:.2f} seconds.")

    # Save the generated outputs
    output_path = os.path.join(output_dir, "gen.pt")
    torch.save(all_out, output_path)
    print(f"Generated samples saved to {output_path}")

    # Save timing information
    timing_data = {
        "chunk_timings": timings,
        "total_time": total_duration
    }
    timing_path = os.path.join(output_dir, "timings.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=4)
    print(f"Timing information saved to {timing_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    for field in GenConfig.__dataclass_fields__.values():
        if isinstance(field.default, bool):
            if field.default is False:
                parser.add_argument(f"--{field.name}", dest=field.name, action="store_true", help=f"Set {field.name} to True (default: False)")
            else:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action="store_false", help=f"Set {field.name} to False (default: True)")
            parser.set_defaults(**{field.name: field.default})
        elif field.default is MISSING:  # Check if the field has no default value
            parser.add_argument(f"--{field.name}", type=field.type, required=True)
        else:
            parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    return GenConfig(**vars(parser.parse_args()))

if __name__ == "__main__":
    config = parse_args()
    print(config)

    gen(config)
