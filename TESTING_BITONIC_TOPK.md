# Testing Bitonic Top-K Sampling Implementation

This document describes how to test the new bitonic sort-based top-k sampling implementation on a CUDA-enabled machine.

## Overview

The implementation adds a new register-resident bitonic sort-based kernel for top-k sampling that is optimized for small vocabulary sizes (64 and 128). It automatically dispatches to this fast path when appropriate, falling back to the original implementation for other vocabulary sizes.

## Prerequisites

- CUDA-capable GPU (Compute Capability 7.0 or higher recommended)
- CUDA Toolkit installed
- Python 3.8+
- PyTorch with CUDA support
- pytest

## Building from Source

```bash
# Navigate to the flashinfer directory
cd /path/to/flashinfer

# Install in development mode
pip install -e .

# Verify installation
python -c "import flashinfer; print('FlashInfer imported successfully')"
```

## Running Tests

### 1. Quick Smoke Test

Run the provided test script:

```bash
python test_bitonic_topk.py
```

This will test:
- Vocab size 64 (uses bitonic sort kernel)
- Vocab size 128 (uses bitonic sort kernel)
- Vocab size 111 (uses original kernel as fallback)

Expected output: All trials should pass, confirming that samples are always from the top-k elements.

### 2. Full Test Suite

Run the existing top-k sampling tests:

```bash
# Test basic top-k sampling
pytest tests/utils/test_sampling.py::test_top_k_sampling -v

# Test with variable k
pytest tests/utils/test_sampling.py::test_top_k_sampling_with_variable_k -v

# Test frequency distribution
pytest tests/utils/test_sampling.py::test_top_k_sampling_freq -v

# Run all sampling tests
pytest tests/utils/test_sampling.py -v
```

### 3. Specific Vocabulary Size Tests

To specifically test the vocabulary sizes that use the bitonic kernel:

```bash
# Test vocab size 64 and 128 specifically
pytest tests/utils/test_sampling.py::test_top_k_sampling -v \
  --batch-size 10 --vocab-size 64 --k 10

pytest tests/utils/test_sampling.py::test_top_k_sampling -v \
  --batch-size 10 --vocab-size 128 --k 20
```

## What to Verify

### 1. Correctness
- All sampled indices should be within valid range [0, vocab_size)
- All sampled indices should belong to the top-k elements
- Distribution of samples should match expected probabilities

### 2. Performance
The bitonic implementation should be faster than the original for vocab sizes 64 and 128. To benchmark:

```python
import torch
import flashinfer
import time

batch_size = 1000
vocab_size = 64  # or 128
k = 10

probs = torch.rand(batch_size, vocab_size, device="cuda:0")
probs = probs / probs.sum(dim=-1, keepdim=True)

# Warmup
for _ in range(10):
    flashinfer.sampling.top_k_sampling_from_probs(probs, k)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    samples = flashinfer.sampling.top_k_sampling_from_probs(probs, k)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Average time: {elapsed/100*1000:.3f} ms")
```

### 3. Compilation
Ensure the code compiles without errors:

```bash
# Check for compilation warnings
python -c "import flashinfer.sampling" 2>&1 | grep -i warning
```

## Potential Issues and Debugging

### Issue 1: Compilation Errors

If you see errors related to `__shfl_sync`:
- Ensure CUDA version is 9.0 or higher
- Check that the compute capability is set correctly

### Issue 2: Incorrect Samples

If samples are not from top-k:
- Check the bitonic sort logic (keys should be sorted in descending order)
- Verify the cumulative sum computation
- Add debug prints in the kernel to trace execution

### Issue 3: Performance Regression

If bitonic implementation is slower:
- Profile with `nsys` or `nvprof`
- Check register usage with `--ptxas-options=-v`
- Verify occupancy with `nvcc --resource-usage`

## Advanced Testing

### Memory Safety

```bash
# Run with CUDA memory checker
cuda-memcheck python test_bitonic_topk.py
```

### Stress Testing

```bash
# Test with many batches and iterations
python -c "
import torch
import flashinfer

for _ in range(1000):
    probs = torch.rand(100, 64, device='cuda:0')
    probs = probs / probs.sum(dim=-1, keepdim=True)
    samples = flashinfer.sampling.top_k_sampling_from_probs(probs, 10)
    assert torch.all(samples < 64)
print('Stress test passed!')
"
```

## Expected Behavior

- **Vocab size 64**: Uses `BitonicTopKSamplingFromProbKernel<64>`
- **Vocab size 128**: Uses `BitonicTopKSamplingFromProbKernel<128>`
- **Other sizes**: Falls back to `TopKSamplingFromProbKernel` (original implementation)

## Contact

If you encounter issues or have questions about this implementation, please:
1. Check the implementation in `include/flashinfer/sampling.cuh`
2. Review the bitonic sort macros in `include/flashinfer/bitonic_sort.cuh`
3. Open an issue on the repository with detailed error messages and environment info
