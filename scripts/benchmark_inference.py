import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from PIL import Image
from src.dataset import load_config, get_transforms
from src.model import build_model


def benchmark(num_runs=100, max_latency_ms=100):
    config = load_config()
    device = torch.device("cpu")

    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    model = build_model(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = get_transforms(config, split="test")
    dummy_img = Image.fromarray(np.uint8(np.random.rand(48, 48) * 255), mode="L")
    input_tensor = transform(dummy_img).unsqueeze(0).to(device)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)

    # benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(input_tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)

    print(f"Average inference time: {avg_ms:.2f}ms")
    print(f"P95 inference time: {p95_ms:.2f}ms")

    if avg_ms > max_latency_ms:
        print(f"FAILED: average latency {avg_ms:.2f}ms exceeds {max_latency_ms}ms")
        sys.exit(1)
    else:
        print(f"PASSED: average latency {avg_ms:.2f}ms is within {max_latency_ms}ms")
        sys.exit(0)


if __name__ == "__main__":
    benchmark()