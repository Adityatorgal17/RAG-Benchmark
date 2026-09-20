"""Calibrate the post-retrieval relevance threshold for the project.

The plan specifies a fixed calibration policy: compute the 10th percentile over positive
chunk scores, clip it to [0.02, 0.30], and freeze it in config.yaml as `post.tau`.
This is a simple, deterministic helper used during T17 dev calibration.
"""

from __future__ import annotations

import argparse


def calibrate_tau(scores: list[float]) -> float:
    if not scores:
        return 0.02
    ordered = sorted(float(s) for s in scores)
    if len(ordered) == 1:
        value = ordered[0]
    else:
        idx = max(0, min(len(ordered) - 1, int(0.10 * len(ordered))))
        value = ordered[idx]
    return max(0.02, min(0.30, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the post-retrieval tau threshold.")
    parser.add_argument("scores", nargs="*", type=float, help="Positive chunk scores used for the 10th-percentile calibration.")
    args = parser.parse_args()
    tau = calibrate_tau(args.scores)
    print(f"tau={tau:.4f}")
    return None


if __name__ == "__main__":
    main()
