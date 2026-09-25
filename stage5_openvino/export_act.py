#!/usr/bin/env python3
"""Export the ACT policy core to OpenVINO IR and check it against PyTorch.

    python -m stage5_openvino.export_act --checkpoint <pretrained_model dir>

What is exported: ACTPolicy.predict_action_chunk, i.e. the neural network that maps
one observation to a (1, chunk_size, 12) action chunk. That is the real inference
cost - select_action() only pops one action per call from a queue and runs the
network once per n_action_steps.

Preprocessing and postprocessing are PRESERVED, not re-implemented: the checkpoint's
own lerobot processors (normalizer / unnormalizer, MEAN_STD) run in Python around
both the PyTorch and the OpenVINO path, so the comparison is end to end.

This is a compatibility and baseline tool, not a benchmark report. It prints only
measured numbers; it never substitutes placeholders.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage3_policy.learned import schema


class ActionChunkModule(torch.nn.Module):
    """Tensor-in / tensor-out view of the policy core, for tracing and export."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.policy.predict_action_chunk(
            {schema.STATE_KEY: state, schema.IMAGE_KEY: image}
        )


def _timed(fn, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "repeats": repeats,
        "warmup": warmup,
        "mean_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
        "stdev_ms": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs" / "openvino" / "act_open_drawer")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--dataset", type=Path,
        help="LeRobotDataset root. A REAL frame from it is used as the benchmark input. "
             "Strongly recommended: synthetic inputs (e.g. zeros) are far from the training "
             "mean, and dividing by a near-zero per-joint std produces values that overflow "
             "fp16 on the NPU. Without this flag a synthetic zero observation is used.",
    )
    ap.add_argument("--repo-id", default="local/butler_open_drawer")
    ap.add_argument("--frame-index", type=int, default=0)
    args = ap.parse_args(argv)

    import openvino as ov
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    core = ov.Core()
    report: dict = {
        "openvino_version": ov.__version__,
        "torch_version": torch.__version__,
        "available_devices": list(core.available_devices),
        "device_names": {},
    }
    for dev in core.available_devices:
        try:
            report["device_names"][dev] = core.get_property(dev, "FULL_DEVICE_NAME")
        except Exception as exc:  # a device can be listed but not queryable
            report["device_names"][dev] = f"(unavailable: {type(exc).__name__})"

    ckpt = str(args.checkpoint)
    policy = ACTPolicy.from_pretrained(ckpt).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    report["checkpoint"] = ckpt
    report["chunk_size"] = int(policy.config.chunk_size)
    report["n_action_steps"] = int(policy.config.n_action_steps)
    report["parameters"] = int(sum(p.numel() for p in policy.parameters()))

    if args.dataset is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(args.repo_id, root=str(args.dataset))
        frame = ds[args.frame_index]
        img = np.asarray(frame[schema.IMAGE_KEY])          # (3,H,W) float 0..1
        raw = {
            schema.STATE_KEY: np.asarray(frame[schema.STATE_KEY], dtype=np.float32),
            schema.IMAGE_KEY: np.ascontiguousarray(
                (img.transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
            ),
        }
        report["input_source"] = {
            "kind": "real dataset frame",
            "dataset": str(args.dataset),
            "frame_index": args.frame_index,
            "dataset_frames": len(ds),
        }
    else:
        raw = {
            schema.STATE_KEY: np.zeros(len(schema.MOTOR_NAMES), dtype=np.float32),
            schema.IMAGE_KEY: np.zeros(schema.IMAGE_SHAPE, dtype=np.uint8),
        }
        report["input_source"] = {
            "kind": "SYNTHETIC zeros",
            "warning": "not representative: far from the training mean, so near-zero-std "
                       "joints normalize to values that overflow fp16 on the NPU",
        }
    from lerobot.policies.utils import prepare_observation_for_inference

    batch = prepare_observation_for_inference(
        raw, torch.device("cpu"), task="Open the top drawer with arm A.",
        robot_type=schema.ROBOT_TYPE,
    )
    normalized = preprocessor(batch)          # PRESERVED preprocessing
    state = normalized[schema.STATE_KEY].contiguous()
    image = normalized[schema.IMAGE_KEY].contiguous()
    report["input_shapes"] = {"state": list(state.shape), "image": list(image.shape)}

    # MEAN_STD normalization divides by the dataset's per-joint std. A joint that never
    # moves in the training data has std ~= 0, which blows the normalized value up. That
    # matters for fp16-only devices (the NPU), where anything above 65504 becomes inf/NaN.
    state_abs_max = float(torch.max(torch.abs(state)))
    report["normalized_state"] = {
        "abs_max": state_abs_max,
        "fp16_max_representable": 65504,
        "values_exceeding_fp16_range": int(torch.sum(torch.abs(state) > 65504)),
        "note": ("Large values here come from near-zero per-joint std in the training data "
                 "(a joint that never moves), not from the exporter."),
    }

    module = ActionChunkModule(policy).eval()
    with torch.no_grad():
        torch_chunk = module(state, image)
    report["chunk_shape"] = list(torch_chunk.shape)

    print("devices :", report["available_devices"], report["device_names"])
    print("params  : {:,}".format(report["parameters"]))
    print("chunk   :", report["chunk_shape"])

    # --- measured PyTorch CPU baseline -------------------------------------
    with torch.no_grad():
        report["baseline_pytorch_cpu"] = _timed(
            lambda: module(state, image), args.repeats, args.warmup
        )
    print("\nPyTorch CPU baseline (per action-chunk forward):")
    print("  ", report["baseline_pytorch_cpu"])

    # --- export ------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    xml_path = args.out / "act_policy.xml"
    t0 = time.perf_counter()
    with torch.no_grad():
        ov_model = ov.convert_model(module, example_input=(state, image))

    # Tracing leaves the batch and image dims dynamic (state [?,?], image [?,3,?,?]).
    # The policy always sees one observation of a fixed size, and the NPU plugin
    # requires fully static shapes, so pin them to the real shapes.
    report["shapes_before_reshape"] = [str(i.get_partial_shape()) for i in ov_model.inputs]
    ov_model.reshape({
        ov_model.inputs[0]: ov.PartialShape(list(state.shape)),
        ov_model.inputs[1]: ov.PartialShape(list(image.shape)),
    })
    report["shapes_after_reshape"] = [str(i.get_partial_shape()) for i in ov_model.inputs]
    report["all_inputs_static"] = not any(i.get_partial_shape().is_dynamic for i in ov_model.inputs)

    ov.save_model(ov_model, xml_path, compress_to_fp16=False)
    report["export"] = {
        "ok": True,
        "seconds": round(time.perf_counter() - t0, 2),
        "xml": str(xml_path),
        "bin_mb": round((xml_path.with_suffix(".bin")).stat().st_size / 1e6, 2),
        "precision": "FP32 (compress_to_fp16=False)",
    }
    print("\nexport  :", report["export"])

    # --- per-device compile + correctness ----------------------------------
    report["devices_tested"] = {}
    for dev in core.available_devices:
        entry: dict = {}
        try:
            compiled = core.compile_model(ov_model, dev)
            out = compiled([state.numpy(), image.numpy()])[compiled.output(0)]
            ov_chunk = torch.from_numpy(np.asarray(out))
            diff = float(torch.max(torch.abs(ov_chunk - torch_chunk)))

            # PRESERVED postprocessing, applied to both paths, compared end to end.
            # The processor takes the policy's action tensor directly, exactly as
            # stage3_policy/learned/inference.py calls it.
            first_torch = torch.as_tensor(postprocessor(torch_chunk[:, 0]))
            first_ov = torch.as_tensor(postprocessor(ov_chunk[:, 0]))
            finite = bool(torch.isfinite(ov_chunk).all())
            entry = {
                "compiled": True,
                "output_all_finite": finite,
                "inference_precision_hint": str(
                    core.get_property(dev, "INFERENCE_PRECISION_HINT")
                ) if dev != "CPU" else "f32 (default)",
                "numerics_note": None if finite else (
                    "output contains non-finite values; this device runs fp16 and the "
                    "normalized input exceeds the fp16 range (see normalized_state)"
                ),
                "chunk_max_abs_diff_vs_pytorch": diff,
                "final_action_max_abs_diff_vs_pytorch": float(
                    torch.max(torch.abs(first_ov - first_torch))
                ),
                "final_action_shape": list(first_ov.shape),
                "latency": _timed(
                    lambda: compiled([state.numpy(), image.numpy()]), args.repeats, args.warmup
                ),
            }
        except Exception as exc:
            entry = {"compiled": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        report["devices_tested"][dev] = entry
        print(f"\n{dev}:")
        print(" ", {k: v for k, v in entry.items() if k != "latency"})
        if "latency" in entry:
            print("  latency:", entry["latency"])

    report_path = args.out / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nreport ->", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
