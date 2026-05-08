# MCZI

MCZI is a Rust toolkit for k-mer set construction and k-mer set subtraction.
It currently builds four user-facing binaries:

- `MC`: minimizer-count-based k-mer counting.
- `ZI`: zero-intersection subtraction of an SSHash-indexed FASTA/simplitig k-mer set from query input.
- `MCZI`: a single-process pipeline that runs MC on an index dataset, builds an SSHash index, then subtracts it from query input.
- `R`: reformer for compacting unitig FASTA records into longer simplitigs using exact `K-1` overlaps without splitting unitigs.

The implementation is reverse-complement aware everywhere k-mers are encoded or queried. Output FASTA is written as simplitigs, not one FASTA record per k-mer.

## Requirements

- Rust toolchain with edition 2024 support. Rust 1.85 or newer is recommended.
- A C/C++ build toolchain for native transitive dependencies.
- `pkg-config` is useful on Linux.

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
cargo build --release --bin MC --bin ZI --bin MCZI --bin R
```

The binaries are then available under `target/release/`.
MCZI links the vendored GGCAT Rust API directly; it does not call a `ggcat` executable.


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

MC can write FASTA or KFF. ZI, MCZI, and R write FASTA simplitigs.

## File of Files

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

MC options:

- `--input`, `-i`: one or more FASTA/FASTQ inputs, or FOFN paths with `--fofn`
- `--fofn`: treat `--input` paths as FOFNs
- `--kmer-size`, `-k`: k-mer size, `k <= 64`
- `--minimizer-size`, `-m`: minimizer size, `m <= k`
- `--threshold`, `-x`: strict threshold; normal mode keeps k-mers with count greater than `X`
- `--output`, `-o`: output path
- `--format fasta`: write FASTA simplitigs
- `--format kff`: write KFF with counts
- `--threads`, `-t`: Rayon worker count
- `--partition-count`: temporary partition file count for counting phases; default `1024`

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

### MC fof Dataset-Presence Mode

In fof mode, `-x` changes meaning. It is no longer an abundance threshold.
It is a dataset-presence threshold, and MC emits k-mers present in more than `X` datasets from the fof.

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

Important fof semantics:

- A minimizer is incremented at most once per dataset during the minimizer phase.
- A k-mer is counted at most once per dataset during the k-mer phase.
- Datasets are processed one by one.
- A minimizer that can no longer reach the dataset threshold is not kept.
- fof dataset-presence mode currently requires `k <= 32`.

fof mode can also write KFF:

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



## ZI: Zero Intersection

ZI indexes a FASTA (with no duplicated kmer) file with SSHash and streams query FASTA/FASTQ input.
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

ZI options:

- `--index`: FASTA/simplitig file to index with SSHash
- `--input`, `-i`: one or more query FASTA/FASTQ inputs, or FOFN paths with `--fofn`
- `--fofn`: treat `--input` paths as FOFNs
- `--kmer-size`, `-k`: odd k-mer size in `[3, 63]`
- `--minimizer-size`, `-m`: SSHash minimizer size; defaults to an odd value near 19 and below `k`
- `--output`, `-o`: output FASTA path
- `--threads`, `-t`: SSHash build threads; `0` means all cores
- `--ram-limit-gib`: SSHash build RAM limit

For a query FOF:

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
Index and final simplitig construction are delegated to the vendored GGCAT Rust API in `vendor/ggcat` for disk-backed, multi-threaded compaction inside the MCZI process.

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
  --output-suffix filtered \
  -t 16
```

For `--index-fofn`, the MC step uses the same dataset-presence semantics as `MC --fofn`: `-x X` keeps k-mers present in more than `X` index datasets, with one count per dataset.

When `--query-fofn` is set, MCZI writes one output file per query file instead of merging all query files into one output. By default each output is written next to its query input with `.filtered` inserted before FASTA/FASTQ and compression extensions:

```text
sample.fa.zst      -> sample.filtered.fa.zst
sample.fastq.gz    -> sample.filtered.fastq.gz
sample.unitigs.fna -> sample.unitigs.filtered.fna
```

Use `--output-suffix` to change `filtered`. If `--output` is also supplied in `--query-fofn` mode, it is treated as an output directory and the same suffixed filenames are written there.

MCZI options:

- `--index-input`: one or more index FASTA/FASTQ inputs, or index FOFNs with `--index-fofn`
- `--index-fofn`: treat `--index-input` paths as FOFNs and use MC dataset-presence semantics
- `--query-input`: one or more query FASTA/FASTQ inputs, or query FOFNs with `--query-fofn`
- `--query-fofn`: treat `--query-input` paths as FOFNs
- `--kmer-size`, `-k`: odd k-mer size in `[3, 63]`
- `--minimizer-size`, `-m`: MC minimizer size and SSHash minimizer size
- `--threshold`, `-x`: MC index threshold
- `--output`, `-o`: output FASTA path; required unless `--query-fofn` or `--output-mode no-output` is set. In `--query-fofn` mode this is optional and, if supplied, is an output directory
- `--output-suffix`: suffix for per-query outputs in `--query-fofn` mode; default `filtered`
- `--output-mode simplitig`: compact absent canonical query k-mers with in-process GGCAT
- `--output-mode regular`: stream unfiltered query-oriented sequence segments
- `--output-mode no-output`: write no FASTA output and report query filtering stats only
- `--reform-output`: apply in-process R-style `K-1` merging to the final output
- `--reform-abundance-mode mean|runs`: preserve `km:f` abundance during regular-output reforming
- `--threads`, `-t`: Rayon/GGCAT worker count
- `--partition-count`: temporary MC partition file count for index counting; default `1024`
- `--ram-limit-gib`: SSHash and GGCAT RAM limit

MCZI output modes:

- `--output-mode simplitig` is the default. MCZI streams absent canonical query k-mers to a temporary FASTA, then calls in-process GGCAT to write FASTA simplitigs.
- `--output-mode regular` streams query-oriented FASTA segments. It preserves original headers, query order, and orientation, keeps duplicates, and cuts a sequence whenever a `k`-mer window is found in the ZI index.
- `--output-mode no-output` writes no FASTA file. It scans the query and reports `query_kmers_filtered_by_zi`, `query_kmers_not_filtered_by_zi`, and `query_regular_output_nucleotides`, where the nucleotide count is the number of sequence bases that regular output would write, excluding headers and line breaks.
- `--reform-output` applies the in-process `R` merger to the MCZI output before writing the final file.
- `--reform-abundance-mode mean|runs` can be combined with `--output-mode regular --reform-output` to make the final reforming pass preserve `km:f` abundance headers. MCZI's simplitig/GGCAT intermediates do not carry valid per-unitig abundance, so abundance reforming is intentionally limited to regular output.

MCZI constraints:

- `k` must be odd and in `[3, 63]` because the subtraction index uses `sshash-rs`.
- `m` must be greater than 0 and less than `k`.
- `m` is used both for MC minimizer processing and for SSHash construction.

## R: Reformer

R takes a unitig FASTA/FASTQ file and a k-mer size, then merges whole unitig records through exact `K-1` overlaps. It never splits an input unitig sequence; a unitig is either used forward or reverse-complemented as one block.

```bash
target/release/R \
  --input unitigs.fa.zst \
  --kmer-size 31 \
  --output reformed_simplitigs.fa.zst \
  --abundance-mode mean \
  --threads 16
```

R stores normalized unitig sequence bytes in a temporary disk-backed store, keeps only record offsets/lengths and oriented `K-1` endpoints in RAM, sorts endpoints in parallel, and greedily selects orientation-compatible non-cycling joins.

R options:

- `--input`, `-i`: unitig FASTA/FASTQ input
- `--kmer-size`, `-k`: k-mer size used for exact `K-1` overlaps
- `--output`, `-o`: output FASTA path
- `--threads`, `-t`: Rayon worker count
- `--abundance-mode mean`: write weighted mean `km:f` abundance per output simplitig
- `--abundance-mode runs`: write run-length encoded per-k-mer `km:f` abundance

R expects each input unitig header to contain `km:f:<value>` when run from the CLI. The abundance can be emitted in two forms:

- `--abundance-mode mean` is the default. It writes one `km:f:<value>` per output simplitig, where the value is the mean abundance weighted by the number of k-mers contributed by each merged input unitig.
- `--abundance-mode runs` writes run-length encoded per-k-mer abundance as `km:f:<value>:<count>:...`, reverses run order when a unitig is reverse-complemented, and coalesces adjacent equal-value runs.

R constraints:

- `k` must be in `[2, 64]`.
- Input records must contain only A/C/G/T bases after case normalization.
- Input records shorter than `k` are rejected because they do not contain a full k-mer.
- Input `km:f` run lengths, when present as `value:count` pairs, must sum to the number of k-mers in the unitig.

## Under The Hood

### MC Normal Abundance Mode

MC is organized as a KMC-style multi-pass counter.

1. It parses FASTA/FASTQ with `helicase`, using DNA-only parsing and splitting on non-ACGT bases.
2. It computes canonical minimizers with `simd-minimizers`.
3. It stores minimizer counts in sharded hash tables. The shard count is controlled by `--partition-count` and defaults to `1024`.
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
2. write the selected index k-mers to a temporary FASTA
3. call the vendored GGCAT API with fast simplitigs and minimum multiplicity 1
4. build an SSHash dictionary from the GGCAT simplitig FASTA
5. stream query input and keep only absent k-mers
6. for `--output-mode simplitig`, write absent canonical k-mers to FASTA and call GGCAT for final simplitigs
7. for `--output-mode regular`, stream query-oriented unfiltered sequence segments directly
8. for `--output-mode no-output`, skip FASTA writing and report the regular-output nucleotide count
9. if `--reform-output` is set, pass the produced FASTA through the in-process `R` merger before writing the final output path

This keeps the large simplitig construction out of MCZI's in-memory hash-map builder, stays inside the MCZI executable, and avoids a user-visible intermediate index file.

### Phase, Resource, And Stat Logs

The tools write compact progress lines to stderr as soon as each phase finishes.

- `MC phase1 minimizer count <wall>s` is emitted by MC phases.
- `ZI_PHASE	<name>	<seconds>` is emitted by ZI phases.
- `R_PHASE	<name>	<seconds>` and `R_STAT	<name>	<value>` are emitted by R.
- `MCZI phase1 minimizer count <wall>s CPU <cpu>s <rss>MB RAM` is emitted by MCZI phases.
- `MCZI_STAT	<name>	<value>` is emitted for key MCZI counts.
- `MCZI_OUTPUT	<input>	<output>` is emitted for each per-file output in `--query-fofn` mode.

Large MCZI stat values are comma-grouped, for example `MCZI_STAT	query_kmers_scanned	9,832,424,437`.

MCZI currently reports:

- `index_minimizers_above_threshold`
- `index_kmers_above_threshold`
- `query_kmers_scanned`
- `query_kmers_filtered_by_zi`
- `query_kmers_not_filtered_by_zi`
- `query_regular_output_nucleotides` for regular and no-output paths
- `query_unique_kmers_not_filtered_by_zi` when a code path computes that value

R currently reports:

- `input_unitigs`
- `input_bases`
- `selected_overlaps`
- `output_simplitigs`
- `output_unitigs`
- `output_bases`

### Temporary Files

MC creates temporary partition directories named `mc-partitions-*` under the system temp directory.
ZI and MCZI create temporary SSHash build directories named `zi-sshash-*` and `mczi-sshash-*`.
MCZI also creates `mczi-ggcat-*` directories for the selected-kmer FASTA and GGCAT temporary files.
MCZI simplitig output creates `mczi-output-ggcat-*` directories for absent-kmer FASTA and final GGCAT temporary files.
These directories are removed on success and on handled errors.

## Testing

Run all tests:

```bash
cargo test
```

The tests cover canonical encoding, packed DNA helpers, minimizer run coverage, normal abundance counting, gzip/xz input, compressed FASTA output, FOFN expansion, FOFN dataset-presence behavior, validation errors, simplitig compaction, KFF writing, ZI subtraction, MCZI subtraction, MCZI output modes, MCZI resource/stat reporting contracts, R reforming, and R abundance preservation.

Format check:

```bash
cargo fmt --check
```
