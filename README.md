# MCZI

MCZI is a Rust toolkit for k-mer set construction and k-mer set subtraction.
It currently builds three user-facing binaries:

- `MC`: minimizer-based k-mer counting with optional file-of-filenames dataset-presence mode.
- `ZI`: zero-intersection subtraction of an SSHash-indexed FASTA/simplitig k-mer set from query input.
- `MCZI`: a single-process pipeline that runs MC on an index dataset, builds an SSHash index, then subtracts it from query input.

The implementation is reverse-complement aware everywhere k-mers are encoded or queried. Output FASTA is written as simplitigs, not one FASTA record per k-mer.

## Status

This repository is prepared for publication at:

```bash
https://github.com/Malfoy/MCZI
```

No push is required to use the code locally. Generated benchmark data, build output, perf files, and the local KMC checkout are ignored by git.

## Requirements

- Rust toolchain with edition 2024 support. Rust 1.85 or newer is recommended.
- A C/C++ build toolchain for native transitive dependencies.
- `pkg-config` is useful on Linux.
- Optional command-line tools for inspection and benchmarking: `gzip`, `xz`, `zstd`, `/usr/bin/time`, `pidstat`, and `perf`.

On Debian/Ubuntu-like systems:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config gzip xz-utils zstd
```

Install Rust with rustup if needed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

## Build

After cloning the uploaded repository:

```bash
git clone https://github.com/Malfoy/MCZI.git
cd MCZI
cargo build --release --bin MC --bin ZI --bin MCZI
```

The binaries are then available under `target/release/`.

For development builds:

```bash
cargo build --bin MC --bin ZI --bin MCZI
```

For the benchmark/helper binaries:

```bash
cargo build --release \
  --bin mc-minimizer-scan \
  --bin mc-minimizer-count \
  --bin mc-phase12 \
  --bin simulate-reads-from-genome
```

## Supported Input And Output Compression

Input parsing accepts:

- plain FASTA/FASTQ
- gzip: `.gz`, `.gzip`
- xz: `.xz`
- zstd: `.zst`, `.zstd`

Output compression is selected from the output filename extension:

- plain output for any other extension
- gzip for `.gz` or `.gzip`
- xz for `.xz`
- zstd for `.zst` or `.zstd`

MC can write FASTA or KFF. ZI and MCZI write FASTA simplitigs.

## FOFN Files

A FOFN is a text file containing one input path per line:

```text
# comments and blank lines are ignored
sample_a/read_set.fa.gz
sample_b/read_set.fa.zst
/absolute/path/sample_c.fa.xz
```

Relative paths are resolved relative to the FOFN file location.

## MC: K-mer Counting

Basic abundance mode counts canonical k-mers from one or more FASTA/FASTQ files and emits k-mers whose exact count is greater than `-x`.

```bash
target/release/MC \
  --input reads.fa.gz \
  --kmer-size 31 \
  --minimizer-size 21 \
  --threshold 5 \
  --format fasta \
  --output kmers.fa.zst \
  --threads 16
```

Equivalent short options:

```bash
target/release/MC -i reads.fa.gz -k 31 -m 21 -x 5 -o kmers.fa.zst --format fasta -t 16
```

Write KFF instead of FASTA:

```bash
target/release/MC \
  -i reads.fa.gz \
  -k 31 \
  -m 21 \
  -x 5 \
  --format kff \
  -o kmers.kff.zst \
  -t 16
```

Multiple inputs are accepted:

```bash
target/release/MC \
  -i lane1.fq.gz lane2.fq.gz lane3.fq.gz \
  -k 31 \
  -m 21 \
  -x 5 \
  -o kmers.fa.gz \
  --format fasta \
  -t 32
```

### MC FOFN Dataset-Presence Mode

In FOFN mode, `-x` changes meaning. It is no longer an abundance threshold.
It is a dataset-presence threshold, and MC emits k-mers present in more than `X` datasets from the FOFN.

```bash
target/release/MC \
  --fofn \
  --input datasets.fofn \
  --kmer-size 31 \
  --minimizer-size 21 \
  --threshold 5 \
  --format fasta \
  --output kmers_seen_in_more_than_5_datasets.fa.zst \
  --threads 16
```

Important FOFN semantics:

- A minimizer is incremented at most once per dataset during the minimizer phase.
- A k-mer is counted at most once per dataset during the k-mer phase.
- Datasets are processed one by one.
- Within a dataset, MC still uses multithreading.
- A minimizer that can no longer reach the dataset threshold is not kept.
- FOFN dataset-presence mode currently requires `k <= 32`.

FOFN mode can also write KFF:

```bash
target/release/MC \
  --fofn \
  -i datasets.fofn \
  -k 31 \
  -m 21 \
  -x 5 \
  --format kff \
  -o shared.kff.zst \
  -t 16
```

### MC Constraints

- `k` must be in `1..=64`.
- `m` must be in `1..=k` and `m <= 64`.
- Normal abundance mode uses compact `u8` minimizer counts, so `x < 255`.
- FOFN dataset-presence mode uses `u32` dataset counts and accepts high thresholds, but the current FOFN k-mer path requires `k <= 32`.

## ZI: Zero Intersection

ZI indexes a FASTA/simplitig file with SSHash and streams query FASTA/FASTQ input.
It writes a FASTA simplitig set containing query k-mers that are absent from the indexed k-mer set.

```bash
target/release/ZI \
  --index indexed_kmers.fa.zst \
  --input query.fa.gz \
  --kmer-size 31 \
  --minimizer-size 19 \
  --output query_minus_index.fa.zst \
  --threads 16 \
  --ram-limit-gib 16
```

The index file is usually MC FASTA output, but any FASTA/simplitig file with sequences of length at least `k` can be used.

For a query FOFN:

```bash
target/release/ZI \
  --index indexed_kmers.fa.zst \
  --fofn \
  --input query_files.fofn \
  -k 31 \
  -m 19 \
  -o query_minus_index.fa.gz \
  -t 16
```

ZI constraints come from `sshash-rs`:

- `k` must be odd.
- `k` must be in `[3, 63]`.
- `m` must be greater than 0 and less than `k`.
- If `-m` is omitted, ZI defaults to an odd minimizer length near 19 and below `k`.

## MCZI: MC Plus ZI In One Process

MCZI first counts the index input with MC, compacts the counted index k-mers into simplitigs, builds an SSHash index from those simplitigs, then subtracts that index from query input.

```bash
target/release/MCZI \
  --index-input index_reads.fa.gz \
  --query-input query_reads.fa.gz \
  --kmer-size 31 \
  --minimizer-size 21 \
  --threshold 5 \
  --output query_minus_mc_index.fa.zst \
  --threads 16 \
  --ram-limit-gib 16
```

Use FOFNs for either side:

```bash
target/release/MCZI \
  --index-fofn \
  --index-input index_datasets.fofn \
  --query-fofn \
  --query-input query_files.fofn \
  -k 31 \
  -m 21 \
  -x 5 \
  -o query_minus_mc_index.fa.zst \
  -t 16
```

For `--index-fofn`, the MC step uses the same dataset-presence semantics as `MC --fofn`: `-x X` keeps k-mers present in more than `X` index datasets, with one count per dataset.

MCZI constraints:

- `k` must be odd and in `[3, 63]` because the subtraction index uses `sshash-rs`.
- `m` must be greater than 0 and less than `k`.
- `m` is used both for MC minimizer processing and for SSHash construction.

## Phase Timing

The tools print phase timing to stderr as tab-separated records:

```text
MC_PHASE    phase_name    seconds
```

Examples:

- MC normal mode: `1_minimizer_counting`, `2_superkmer_partitioning`, `3_kmer_counting_and_output`, `total`
- MC FOFN mode: `1_dataset_minimizer_counting`, `2_dataset_kmer_counting`, `3_output_streaming`, `total`
- ZI: `1_sshash_indexing`, `2_query_subtraction`, `3_simplitig_output`, `total`
- MCZI: `1_mc_index_counting`, `2_mc_index_simplitigs`, `3_sshash_indexing`, `4_query_subtraction`, `5_simplitig_output`, `total`

MC also prints selected statistics such as partition bytes:

```text
MC_STAT     partition_bytes     bytes
```

## Useful Environment Variables

MC exposes a few tuning variables for experiments:

- `MC_MINIMIZER_ORDER=direct`: default. Uses `simd-minimizers` selected minimizer hashes directly.
- `MC_MINIMIZER_ORDER=antilex`: uses anti-lexicographic minimizer ordering through `simd-minimizers`.
- `MC_MINIMIZER_ORDER=simd-value`: extracts selected minimizer sequence values and hashes them in MC.
- `MC_KMC_STYLE=0`: disables KMC-style filtered super-kmer partitioning for the KFF abundance path.
- `MC_PHASE3_BUCKET_BITS=N`: bucket phase 3 partition processing. Current accepted range is `0..=10`.
- `MC_PHASE3_THREADS=N`: override the Rayon thread count used for phase 3 partition processing.

The production minimizer implementation is based on `simd-minimizers`; MC does not depend on ntHash.

## Under The Hood

### MC Normal Abundance Mode

MC is organized as a KMC-style multi-pass counter.

1. It parses FASTA/FASTQ with `helicase`, using DNA-only parsing and splitting on non-ACGT bases.
2. It computes canonical minimizers with `simd-minimizers`.
3. It stores minimizer counts in 1024 sharded hash tables. The shard is selected from the high bits of the 32-bit minimizer hash, and each shard is protected by its own mutex.
4. Normal minimizer counts are compact `u8` saturating counts. They only need to distinguish `<= x` from `> x`.
5. It runs a second streaming pass and writes only super-kmers whose minimizer passed the threshold.
6. Super-kmers are written in a compact binary partition format under a temporary `mc-partitions-*` directory.
7. It processes each partition independently, reconstructs canonical k-mers, counts them exactly, and emits only k-mers with count greater than `x`.
8. Output is FASTA simplitigs or KFF.

The point of the minimizer pass is to reduce disk traffic before the exact k-mer counting pass. Super-kmers whose minimizer did not pass the filter are not spilled.

### MC FOFN Dataset-Presence Mode

FOFN mode treats each file listed in the FOFN as one dataset.

The minimizer table stores a `u32` state per minimizer:

- the low 31 bits store dataset-presence count
- the high bit marks whether this minimizer has already been seen in the current dataset

That avoids building a separate per-dataset minimizer set just for deduplication. After a dataset finishes, MC clears the current-dataset marker, prunes minimizers that can no longer reach the threshold, and may block new minimizers that are already mathematically unable to pass.

The k-mer phase uses the same dataset-presence rule. For each dataset, candidate k-mers are spilled to binary partitions, sorted/deduplicated per dataset, then added once to global `u32` k-mer dataset counts.

### Simplitig Output

FASTA output is compacted into simplitigs.
MC starts from the emitted canonical k-mer set and greedily extends exact `k-1` overlaps forward and backward until no extension is available. The FASTA header reports how many k-mers are represented by the simplitig:

```text
>MC_simplitig_1 kmers=123
```

### KFF Output

MC can write a compact KFF-style file with canonical k-mers and counts.
Normal abundance mode writes compact one-byte counts for the optimized `k <= 32` path. FOFN mode writes four-byte dataset counts. When a compressed KFF output is requested, MC writes through the extension-selected compressor; paths that need seekable KFF output use a temporary uncompressed file before compression.

### ZI

ZI builds an `sshash-rs` canonical dictionary from the index FASTA/simplitigs.
It then streams query input and tests every canonical query k-mer against that dictionary. Query k-mers absent from the dictionary are deduplicated, sorted, and emitted as simplitigs.

### MCZI

MCZI keeps the MC-to-ZI pipeline in one process:

1. run MC on the index input
2. compact counted index k-mers to simplitigs in memory
3. build an SSHash dictionary from those simplitigs
4. stream query input and keep only absent k-mers
5. write absent query k-mers as FASTA simplitigs

This avoids writing the intermediate MC FASTA index to disk.

### Temporary Files

MC creates temporary partition directories named `mc-partitions-*` under the system temp directory.
ZI and MCZI create temporary SSHash build directories named `zi-sshash-*` and `mczi-sshash-*`.
These directories are removed on success and on handled errors.

## Testing

Run all tests:

```bash
cargo test
```

At this revision, `cargo test` runs 12 tests total:

- 10 library tests
- 1 `ZI` binary test
- 1 `MCZI` binary test

The tests cover canonical counting, gzip/xz input, compressed FASTA output, FOFN dataset-presence behavior, simplitig compaction, KFF writing, ZI subtraction, and MCZI subtraction.

Format check:

```bash
cargo fmt --check
```

## Benchmark Helpers

The repository has helper binaries and shell scripts under `bench/` for local performance work:

- `simulate-reads-from-genome`: simulate long reads from a genome.
- `mc-minimizer-scan`: benchmark parsing plus minimizer scanning.
- `mc-minimizer-count`: benchmark phase 1 minimizer counting.
- `mc-phase12`: benchmark minimizer counting plus filtered super-kmer partitioning.
- `bench/run_mc_time_pidstat.sh`: run MC with wall-clock, CPU, RSS, and pidstat collection.
- `bench/run_kmc_time_pidstat.sh`: run KMC with comparable timing collection.

Large generated benchmark directories should stay under `bench_runs/`, which is ignored by git.
