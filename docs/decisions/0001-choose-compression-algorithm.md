# 0001 - Choose compression algorithm for model output data

**Date:** 2026-06-30  
**Status:** Development

## Context
RAWS outputs are very large. We have decided to test whether different types of
compression will reduce storage volume. To decide, we conduct a set of compression tests on 
available 1km discharge data from van Jaarsveld et al. 2025 
(https://public.yoda.uu.nl/geo/UU01/Q6EDB2.html).

## Tested Codecs

### Lossless
- **no-compression** — baseline
- **zlib-4** — universal, built into NetCDF4; moderate speed/ratio
- **zstd-3** — faster decompression, better ratio than zlib
- **lz4-blosc** — fast, lower ratio; shuffle preprocessing
- **blosclz-blosc** — BLOSCLZ algorithm with shuffle
- **lz4-bitshuffle** — LZ4 with bit-level shuffling
- **blosclz-bitshuffle** — BLOSCLZ with bit-level shuffling
- **zstd-bitshuffle** — Zstd with bit-level shuffling

### Lossy (error-bounded)
- **BitRound** — bit-precision reduction followed by compression (keepbits: 14, 10, 6)
  - With zstd compression
  - With PCO (P-Codec) serializer
- **Quantize** — decimal digit rounding with zstd (digits: 4, 3, 2)
- **ZFP** — floating-point error bounds (absolute tolerance: 1e-4, 1e-3, 1e-2 m³/s)
- **SZ3** — advanced lossy codec with absolute and relative error bounds
  - Absolute modes: 1e-4, 1e-3, 1e-2 m³/s
  - Relative modes: 1e-3, 1e-2 (fraction of magnitude)

## Decision
⌛No decision yet made, need to to the hydrobasins comparison on glooba water budgets

Need to consider what absolute error magnitude is applicable across all variables, especially since its 
applied globally. 

Need to refine the relative error threholds

## Consequences
⌛