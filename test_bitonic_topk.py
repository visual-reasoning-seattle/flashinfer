"""
Simple test script for bitonic sort-based top-k sampling
"""

import torch
import flashinfer


def test_bitonic_topk_basic():
    """Test basic functionality with vocab size 64 and 128"""
    torch.manual_seed(42)

    # Test with vocab size 64
    print("Testing with vocab size 64...")
    batch_size = 10
    vocab_size = 64
    k = 10

    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # Get ground truth top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    # Run sampling multiple times to verify correctness
    num_trials = 100
    success_count = 0
    for _ in range(num_trials):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)

        # Check that samples are valid indices
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0), (
            f"Invalid sample indices: {samples}"
        )

        # Check that all samples are from top-k
        in_topk = mask[torch.arange(batch_size), samples] == 1
        if torch.all(in_topk):
            success_count += 1

    print(f"Vocab 64: {success_count}/{num_trials} trials passed")

    # Test with vocab size 128
    print("\nTesting with vocab size 128...")
    vocab_size = 128
    k = 20

    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    success_count = 0
    for _ in range(num_trials):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)

        assert torch.all(samples < vocab_size) and torch.all(samples >= 0), (
            f"Invalid sample indices: {samples}"
        )

        in_topk = mask[torch.arange(batch_size), samples] == 1
        if torch.all(in_topk):
            success_count += 1

    print(f"Vocab 128: {success_count}/{num_trials} trials passed")

    # Test with non-power-of-2 vocab size (should use original implementation)
    print("\nTesting with vocab size 111 (fallback to original)...")
    vocab_size = 111
    k = 10

    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    success_count = 0
    for _ in range(num_trials):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)

        assert torch.all(samples < vocab_size) and torch.all(samples >= 0), (
            f"Invalid sample indices: {samples}"
        )

        in_topk = mask[torch.arange(batch_size), samples] == 1
        if torch.all(in_topk):
            success_count += 1

    print(f"Vocab 111: {success_count}/{num_trials} trials passed")

    print("\nAll tests completed!")


if __name__ == "__main__":
    test_bitonic_topk_basic()
