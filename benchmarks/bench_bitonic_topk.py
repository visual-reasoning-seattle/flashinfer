"""
Benchmark comparing bitonic sort-based top-k sampling vs baseline implementation.

The bitonic implementation is automatically used for vocab_size 64 and 128.
For comparison, we benchmark against larger vocab sizes that use the baseline.

Key safety features:
- Per-iteration timeout to prevent hangs
- Warmup with correctness validation
- Graceful error handling
"""

import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import flashinfer


@dataclass
class BenchmarkResult:
    vocab_size: int
    batch_size: int
    k: int
    median_us: float
    std_us: float
    bandwidth_gb_s: float
    implementation: str
    success: bool
    error: Optional[str] = None


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timeout using SIGALRM."""

    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_topk_result(probs: torch.Tensor, samples: torch.Tensor, k: int) -> bool:
    """Validate that samples are within the top-k elements."""
    batch_size, vocab_size = probs.shape

    # Get top-k indices for each batch
    _, topk_indices = torch.topk(probs, min(k, vocab_size), dim=-1)

    # Check each sample is in top-k
    return all(samples[b].item() in topk_indices[b].tolist() for b in range(batch_size))


def bench_single_config(
    vocab_size: int,
    batch_size: int,
    k: int,
    num_warmup: int = 5,
    num_iterations: int = 100,
    timeout_seconds: int = 10,
) -> BenchmarkResult:
    """Benchmark a single configuration with timeout protection."""

    impl = "bitonic" if vocab_size in [64, 128] else "baseline"

    try:
        # Create test data
        torch.manual_seed(42)
        logits = torch.randn(batch_size, vocab_size, device="cuda")
        probs = torch.softmax(logits, dim=-1)

        # Warmup and validate
        with timeout(timeout_seconds):
            for _ in range(num_warmup):
                torch.manual_seed(42)
                samples = flashinfer.sampling.top_k_sampling_from_probs(probs, k)
                torch.cuda.synchronize()

        # Validate correctness on last warmup
        if not validate_topk_result(probs, samples, k):
            return BenchmarkResult(
                vocab_size=vocab_size,
                batch_size=batch_size,
                k=k,
                median_us=0,
                std_us=0,
                bandwidth_gb_s=0,
                implementation=impl,
                success=False,
                error="Validation failed: sample not in top-k",
            )

        # Benchmark
        times_ms = []
        with timeout(
            timeout_seconds * num_iterations // 10
        ):  # Allow more time for full benchmark
            for i in range(num_iterations):
                torch.manual_seed(i)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                _ = flashinfer.sampling.top_k_sampling_from_probs(probs, k)
                end.record()

                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))

        times_us = np.array(times_ms) * 1000
        median_us = np.median(times_us)
        std_us = np.std(times_us)

        # Calculate effective bandwidth
        io_bytes = (
            probs.numel() * probs.element_size() + batch_size * 4
        )  # probs + output indices
        bandwidth_gb_s = io_bytes / (median_us * 1e-6) / 1e9

        return BenchmarkResult(
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=k,
            median_us=median_us,
            std_us=std_us,
            bandwidth_gb_s=bandwidth_gb_s,
            implementation=impl,
            success=True,
        )

    except TimeoutError as e:
        return BenchmarkResult(
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=k,
            median_us=0,
            std_us=0,
            bandwidth_gb_s=0,
            implementation=impl,
            success=False,
            error=str(e),
        )
    except Exception as e:
        return BenchmarkResult(
            vocab_size=vocab_size,
            batch_size=batch_size,
            k=k,
            median_us=0,
            std_us=0,
            bandwidth_gb_s=0,
            implementation=impl,
            success=False,
            error=str(e),
        )


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    if result.success:
        print(
            f"vocab={result.vocab_size:>6}, batch={result.batch_size:>4}, k={result.k:>4}, "
            f"impl={result.implementation:>8}: {result.median_us:>8.2f} us (±{result.std_us:>6.2f}), "
            f"BW={result.bandwidth_gb_s:>6.2f} GB/s"
        )
    else:
        print(
            f"vocab={result.vocab_size:>6}, batch={result.batch_size:>4}, k={result.k:>4}, "
            f"impl={result.implementation:>8}: FAILED - {result.error}"
        )


def run_comparison_benchmark():
    """Run comparison between bitonic and baseline implementations."""

    print("=" * 80)
    print("Bitonic vs Baseline Top-K Sampling Benchmark")
    print("=" * 80)
    print()

    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    results = []

    # Test configurations
    # Bitonic: vocab_size 64, 128
    # Baseline: vocab_size 256, 512, 1024 (for comparison)

    batch_sizes = [1, 8, 32, 128, 512]
    k_values = [1, 5, 10, 32, 64]

    print("-" * 80)
    print("Section 1: Bitonic Implementation (vocab_size 64, 128)")
    print("-" * 80)

    for vocab_size in [64, 128]:
        print(f"\n--- Vocab Size: {vocab_size} ---")
        for batch_size in batch_sizes:
            for k in k_values:
                if k > vocab_size:
                    continue
                result = bench_single_config(vocab_size, batch_size, k)
                results.append(result)
                print_result(result)

    print()
    print("-" * 80)
    print("Section 2: Baseline Implementation (vocab_size 256, 512, 1024)")
    print("-" * 80)

    for vocab_size in [256, 512, 1024]:
        print(f"\n--- Vocab Size: {vocab_size} ---")
        for batch_size in batch_sizes:
            for k in [1, 5, 10, 32, 64, 128]:
                if k > vocab_size:
                    continue
                result = bench_single_config(vocab_size, batch_size, k)
                results.append(result)
                print_result(result)

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    bitonic_results = [
        r for r in results if r.implementation == "bitonic" and r.success
    ]
    baseline_results = [
        r for r in results if r.implementation == "baseline" and r.success
    ]
    failed_results = [r for r in results if not r.success]

    print(f"Total tests: {len(results)}")
    print(
        f"Bitonic tests: {len([r for r in results if r.implementation == 'bitonic'])} "
        f"(passed: {len(bitonic_results)})"
    )
    print(
        f"Baseline tests: {len([r for r in results if r.implementation == 'baseline'])} "
        f"(passed: {len(baseline_results)})"
    )

    if failed_results:
        print(f"\nFailed tests: {len(failed_results)}")
        for r in failed_results:
            print(f"  - vocab={r.vocab_size}, batch={r.batch_size}, k={r.k}: {r.error}")

    if bitonic_results:
        print(
            f"\nBitonic median latency range: {min(r.median_us for r in bitonic_results):.2f} - "
            f"{max(r.median_us for r in bitonic_results):.2f} us"
        )

    if baseline_results:
        print(
            f"Baseline median latency range: {min(r.median_us for r in baseline_results):.2f} - "
            f"{max(r.median_us for r in baseline_results):.2f} us"
        )

    # Compare similar batch sizes (bitonic vocab=128 vs baseline vocab=256)
    print()
    print("-" * 80)
    print("Direct Comparison: Bitonic (vocab=128) vs Baseline (vocab=256)")
    print("-" * 80)

    for batch_size in batch_sizes:
        for k in [1, 5, 10, 32, 64]:
            bitonic = next(
                (
                    r
                    for r in bitonic_results
                    if r.vocab_size == 128 and r.batch_size == batch_size and r.k == k
                ),
                None,
            )
            baseline = next(
                (
                    r
                    for r in baseline_results
                    if r.vocab_size == 256 and r.batch_size == batch_size and r.k == k
                ),
                None,
            )

            if bitonic and baseline:
                speedup = baseline.median_us / bitonic.median_us
                print(
                    f"batch={batch_size:>4}, k={k:>4}: "
                    f"bitonic={bitonic.median_us:>7.2f}us, baseline={baseline.median_us:>7.2f}us, "
                    f"speedup={speedup:>5.2f}x"
                )

    return results


def run_stress_test():
    """Run stress test to ensure no hangs."""
    print("=" * 80)
    print("Stress Test: Verify No Hangs")
    print("=" * 80)
    print()

    test_configs = [
        # (vocab_size, batch_size, k)
        (64, 1, 1),
        (64, 1, 5),
        (64, 1, 32),
        (64, 1, 64),
        (64, 100, 1),
        (64, 100, 5),
        (64, 100, 32),
        (64, 100, 64),
        (128, 1, 1),
        (128, 1, 5),
        (128, 1, 32),
        (128, 1, 100),
        (128, 1, 128),
        (128, 100, 1),
        (128, 100, 5),
        (128, 100, 32),
        (128, 100, 100),
        (128, 100, 128),
    ]

    passed = 0
    failed = 0

    for vocab_size, batch_size, k in test_configs:
        result = bench_single_config(
            vocab_size,
            batch_size,
            k,
            num_warmup=2,
            num_iterations=20,
            timeout_seconds=5,
        )

        status = "PASS" if result.success else "FAIL"
        print(f"vocab={vocab_size:>3}, batch={batch_size:>3}, k={k:>3}: {status}")

        if result.success:
            passed += 1
        else:
            failed += 1
            print(f"  Error: {result.error}")

    print()
    print(f"Stress test complete: {passed}/{passed + failed} passed")
    return failed == 0


@torch.inference_mode()
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark bitonic top-k sampling")
    parser.add_argument("--stress", action="store_true", help="Run stress test only")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    args = parser.parse_args()

    if args.stress:
        success = run_stress_test()
        sys.exit(0 if success else 1)
    elif args.full:
        run_comparison_benchmark()
    else:
        # Default: run stress test first, then benchmark
        print("Running stress test first...")
        if run_stress_test():
            print("\nStress test passed. Running benchmark...\n")
            run_comparison_benchmark()
        else:
            print("\nStress test failed! Not running benchmark.")
            sys.exit(1)


if __name__ == "__main__":
    main()
