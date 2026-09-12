"""
Verify model parameter counts match Table 1 from the paper.

Expected counts:
    MLP/raw:         116,096
    MLP/angle:       105,088
    MLP/raw_angle:   121,216
    Transformer/raw: 282,240
    Transformer/angle: 284,416
    Transformer/raw_angle: 292,480
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model, count_parameters
from data.representation import get_input_dim


def main():
    print("=" * 60)
    print("Model Parameter Count Verification (Table 1)")
    print("=" * 60)
    print()

    expected = {
        ("mlp", "raw"): 116_096,
        ("mlp", "angle"): 105_088,
        ("mlp", "raw_angle"): 121_216,
        ("transformer", "raw"): 282_240,
        ("transformer", "angle"): 284_416,
        ("transformer", "raw_angle"): 292_480,
    }

    results = []

    for (encoder_name, repr_name), expected_count in expected.items():
        input_dim = get_input_dim(repr_name)

        model = create_model(
            encoder_name=encoder_name,
            input_dim=input_dim,
            embedding_dim=128,
            hidden_dim=256,
            dropout=0.3,
            representation=repr_name,
        )

        actual_count = count_parameters(model)
        match = "OK" if actual_count == expected_count else "FAIL"
        diff = actual_count - expected_count

        results.append((encoder_name, repr_name, input_dim, actual_count, expected_count, match, diff))

        print(f"  {encoder_name:12s} / {repr_name:10s} (dim={input_dim:2d}) | "
              f"Actual: {actual_count:>8,} | Expected: {expected_count:>8,} | "
              f"{match} (diff: {diff:+d})")

    print()

    all_match = all(r[5] == "OK" for r in results)
    if all_match:
        print("ALL PARAMETER COUNTS MATCH TABLE 1 [OK]")
    else:
        print("SOME COUNTS DIFFER -- see details above")
        print("Note: Minor differences may arise from LayerNorm/bias configurations")


if __name__ == "__main__":
    main()
