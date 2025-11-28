/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Portions of this file are adapted from:
 * Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0
 */

#ifndef FLASHINFER_BITONIC_SORT_CUH_
#define FLASHINFER_BITONIC_SORT_CUH_

#include <cuda_runtime.h>

namespace flashinfer {

// SHFL macro for warp shuffle operations
#define SHFL(var, srcLane) __shfl_sync(0xffffffff, var, srcLane)

// 64 element register resident bitonic merge sort
// To use, define:
// unsigned int otgx;
// type k0, k1;
// type v0, v1;
#define BITONICWARPEXCHANGE_64(mask)                       \
  key1 = k0;                                               \
  value1 = v0;                                             \
  otgx = tgx ^ mask;                                       \
  key2 = SHFL(k0, otgx);                                   \
  value2 = SHFL(v0, otgx);                                 \
  flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
  k0 = flag ? key1 : key2;                                 \
  v0 = flag ? value1 : value2;                             \
  key1 = k1;                                               \
  value1 = v1;                                             \
  key2 = SHFL(k1, otgx);                                   \
  value2 = SHFL(v1, otgx);                                 \
  flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
  k1 = flag ? key1 : key2;                                 \
  v1 = flag ? value1 : value2;

#define BITONICSORT32_64()   \
  BITONICWARPEXCHANGE_64(1)  \
  BITONICWARPEXCHANGE_64(3)  \
  BITONICWARPEXCHANGE_64(1)  \
  BITONICWARPEXCHANGE_64(7)  \
  BITONICWARPEXCHANGE_64(2)  \
  BITONICWARPEXCHANGE_64(1)  \
  BITONICWARPEXCHANGE_64(15) \
  BITONICWARPEXCHANGE_64(4)  \
  BITONICWARPEXCHANGE_64(2)  \
  BITONICWARPEXCHANGE_64(1)  \
  BITONICWARPEXCHANGE_64(31) \
  BITONICWARPEXCHANGE_64(8)  \
  BITONICWARPEXCHANGE_64(4)  \
  BITONICWARPEXCHANGE_64(2)  \
  BITONICWARPEXCHANGE_64(1)

#define BITONICMERGE64_64()        \
  otgx = 31 - tgx;                 \
  key1 = k0;                       \
  value1 = v0;                     \
  key2 = SHFL(k1, otgx);           \
  value2 = SHFL(v1, otgx);         \
  flag = (key1 > key2);            \
  k0 = flag ? key1 : key2;         \
  v0 = flag ? value1 : value2;     \
  key1 = flag ? key2 : key1;       \
  value1 = flag ? value2 : value1; \
  k1 = SHFL(key1, otgx);           \
  v1 = SHFL(value1, otgx);

#define BITONICSORT64_64()   \
  BITONICSORT32_64()         \
  BITONICMERGE64_64()        \
  BITONICWARPEXCHANGE_64(16) \
  BITONICWARPEXCHANGE_64(8)  \
  BITONICWARPEXCHANGE_64(4)  \
  BITONICWARPEXCHANGE_64(2)  \
  BITONICWARPEXCHANGE_64(1)

// 128 element register resident bitonic merge sort
// To use, define:
// unsigned int otgx;
// type k0, k1, k2, k3;
// type v0, v1, v2, v3;
#define BITONICWARPEXCHANGE_128(mask)                      \
  key1 = k0;                                               \
  value1 = v0;                                             \
  otgx = tgx ^ mask;                                       \
  key2 = SHFL(k0, otgx);                                   \
  value2 = SHFL(v0, otgx);                                 \
  flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
  k0 = flag ? key1 : key2;                                 \
  v0 = flag ? value1 : value2;                             \
  key1 = k1;                                               \
  value1 = v1;                                             \
  key2 = SHFL(k1, otgx);                                   \
  value2 = SHFL(v1, otgx);                                 \
  flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
  k1 = flag ? key1 : key2;                                 \
  v1 = flag ? value1 : value2;                             \
  key1 = k2;                                               \
  value1 = v2;                                             \
  key2 = SHFL(k2, otgx);                                   \
  value2 = SHFL(v2, otgx);                                 \
  flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
  k2 = flag ? key1 : key2;                                 \
  v2 = flag ? value1 : value2;                             \
  key1 = k3;                                               \
  value1 = v3;                                             \
  key2 = SHFL(k3, otgx);                                   \
  value2 = SHFL(v3, otgx);                                 \
  flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2); \
  k3 = flag ? key1 : key2;                                 \
  v3 = flag ? value1 : value2;

#define BITONICSORT32_128()   \
  BITONICWARPEXCHANGE_128(1)  \
  BITONICWARPEXCHANGE_128(3)  \
  BITONICWARPEXCHANGE_128(1)  \
  BITONICWARPEXCHANGE_128(7)  \
  BITONICWARPEXCHANGE_128(2)  \
  BITONICWARPEXCHANGE_128(1)  \
  BITONICWARPEXCHANGE_128(15) \
  BITONICWARPEXCHANGE_128(4)  \
  BITONICWARPEXCHANGE_128(2)  \
  BITONICWARPEXCHANGE_128(1)  \
  BITONICWARPEXCHANGE_128(31) \
  BITONICWARPEXCHANGE_128(8)  \
  BITONICWARPEXCHANGE_128(4)  \
  BITONICWARPEXCHANGE_128(2)  \
  BITONICWARPEXCHANGE_128(1)

#define BITONICMERGE64_128()       \
  otgx = 31 - tgx;                 \
  key1 = k0;                       \
  value1 = v0;                     \
  key2 = SHFL(k1, otgx);           \
  value2 = SHFL(v1, otgx);         \
  flag = (key1 > key2);            \
  k0 = flag ? key1 : key2;         \
  v0 = flag ? value1 : value2;     \
  key1 = flag ? key2 : key1;       \
  value1 = flag ? value2 : value1; \
  k1 = SHFL(key1, otgx);           \
  v1 = SHFL(value1, otgx);         \
  key1 = k2;                       \
  value1 = v2;                     \
  key2 = SHFL(k3, otgx);           \
  value2 = SHFL(v3, otgx);         \
  flag = (key1 > key2);            \
  k2 = flag ? key1 : key2;         \
  v2 = flag ? value1 : value2;     \
  key1 = flag ? key2 : key1;       \
  value1 = flag ? value2 : value1; \
  k3 = SHFL(key1, otgx);           \
  v3 = SHFL(value1, otgx);

#define BITONICSORT64_128()   \
  BITONICSORT32_128()         \
  BITONICMERGE64_128()        \
  BITONICWARPEXCHANGE_128(16) \
  BITONICWARPEXCHANGE_128(8)  \
  BITONICWARPEXCHANGE_128(4)  \
  BITONICWARPEXCHANGE_128(2)  \
  BITONICWARPEXCHANGE_128(1)

#define BITONICMERGE128_128()      \
  otgx = 31 - tgx;                 \
  key1 = k0;                       \
  value1 = v0;                     \
  key2 = SHFL(k3, otgx);           \
  value2 = SHFL(v3, otgx);         \
  flag = (key1 > key2);            \
  k0 = flag ? key1 : key2;         \
  v0 = flag ? value1 : value2;     \
  key1 = flag ? key2 : key1;       \
  value1 = flag ? value2 : value1; \
  k3 = SHFL(key1, otgx);           \
  v3 = SHFL(value1, otgx);         \
  key1 = k1;                       \
  value1 = v1;                     \
  key2 = SHFL(k2, otgx);           \
  value2 = SHFL(v2, otgx);         \
  flag = (key1 > key2);            \
  k1 = flag ? key1 : key2;         \
  v1 = flag ? value1 : value2;     \
  key1 = flag ? key2 : key1;       \
  value1 = flag ? value2 : value1; \
  k2 = SHFL(key1, otgx);           \
  v2 = SHFL(value1, otgx);

#define BITONICEXCHANGE32_128() \
  if (k0 < k1) {                \
    key1 = k0;                  \
    value1 = v0;                \
    k0 = k1;                    \
    v0 = v1;                    \
    k1 = key1;                  \
    v1 = value1;                \
  }                             \
  if (k2 < k3) {                \
    key1 = k2;                  \
    value1 = v2;                \
    k2 = k3;                    \
    v2 = v3;                    \
    k3 = key1;                  \
    v3 = value1;                \
  }

#define BITONICEXCHANGE64_128() \
  if (k0 < k2) {                \
    key1 = k0;                  \
    value1 = v0;                \
    k0 = k2;                    \
    v0 = v2;                    \
    k2 = key1;                  \
    v2 = value1;                \
  }                             \
  if (k1 < k3) {                \
    key1 = k1;                  \
    value1 = v1;                \
    k1 = k3;                    \
    v1 = v3;                    \
    k3 = key1;                  \
    v3 = value1;                \
  }

#define BITONICSORT128_128()  \
  BITONICSORT64_128()         \
  BITONICMERGE128_128()       \
  BITONICEXCHANGE32_128()     \
  BITONICWARPEXCHANGE_128(16) \
  BITONICWARPEXCHANGE_128(8)  \
  BITONICWARPEXCHANGE_128(4)  \
  BITONICWARPEXCHANGE_128(2)  \
  BITONICWARPEXCHANGE_128(1)

}  // namespace flashinfer

#endif  // FLASHINFER_BITONIC_SORT_CUH_
