# SPDX-License-Identifier: Apache-2.0
"""Check whether an NPU consumer stream needs an explicit event dependency."""

import argparse
import json

import torch
import torch_npu  # noqa: F401


def _run_case(mode: str, rounds: int, matmul_repeats: int) -> int:
    producer = torch.npu.Stream()
    event = torch.npu.Event()
    source = torch.randn((512, 512), dtype=torch.float16, device="npu")
    marker = torch.zeros(1, dtype=torch.int32, device="npu")
    observed = torch.zeros_like(marker)
    stale = 0

    for expected in range(1, rounds + 1):
        torch.npu.synchronize()
        marker.zero_()
        observed.zero_()
        with torch.npu.stream(producer):
            value = source
            for _ in range(matmul_repeats):
                value = torch.mm(value, source)
            marker.fill_(expected)
            event.record()

        consumer = torch.npu.current_stream()
        if mode == "wait_event":
            consumer.wait_event(event)
        elif mode == "host_sync":
            event.synchronize()
        observed.copy_(marker)
        torch.npu.synchronize()
        stale += int(observed.item() != expected)

    return stale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--matmul-repeats", type=int, default=4)
    args = parser.parse_args()

    torch.npu.set_device(0)
    results = {
        "rounds": args.rounds,
        "no_wait_stale": _run_case("none", args.rounds, args.matmul_repeats),
        "wait_event_stale": _run_case("wait_event", args.rounds, args.matmul_repeats),
        "host_sync_stale": _run_case("host_sync", args.rounds, args.matmul_repeats),
    }
    print(json.dumps(results, sort_keys=True))
    if results["wait_event_stale"] or results["host_sync_stale"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
