use ahash::AHashMap;
use anyhow::{Context, Result, bail, ensure};
use flate2::Compression as GzipCompression;
use flate2::write::GzEncoder;
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use liblzma::read::XzDecoder;
use liblzma::write::XzEncoder;
use rayon::prelude::*;
use simd_minimizers::packed_seq::{AsciiSeq, PackedSeqVec, Seq, SeqVec};
use simd_minimizers::seq_hash::AntiLexHasher;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, mem};

type Count = u64;
type CompactCount = u8;
pub type EncodedKmer = u128;
type MinimizerHash = u32;

const BATCH_BASES: usize = 128 * 1024 * 1024;
const MINIMIZER_SHARDS: usize = 1024;
const MINIMIZER_SHARD_BITS: u32 = 10;
const MINIMIZER_BUFFER_FLUSH_LEN: usize = 16_384;
const KMER_COUNT_SHARDS: usize = 1024;
const KMER_COUNT_SHARD_BITS: u32 = 10;
const KMER_COUNT_BUFFER_FLUSH_LEN: usize = 16_384;
const MINIMIZER_TABLE_MIN_SHARD_CAPACITY: usize = 64;
const ESTIMATED_BYTES_PER_UNIQUE_MINIMIZER_HASH: u64 = 50;
const ESTIMATED_COMPRESSED_BYTES_PER_UNIQUE_MINIMIZER_HASH: u64 = 128;
const ESTIMATED_COMPRESSED_BYTES_PER_UNIQUE_KMER: u64 = 14;
const RAM_KMER_COUNT_MAX_INPUT_BYTES: u64 = 0;
const SUPERKMER_BUFFER_FLUSH_BYTES: usize = 4 * 1024 * 1024;
const PHASE3_IO_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_PHASE3_BUCKET_BITS: u32 = 8;
const KFF_ENCODING_ACGT: u8 = 0x1b;
const PACKED_READ_CACHE_MAGIC: &[u8; 8] = b"MCRD0001";
const DATASET_PRESENCE_SEEN_BIT: u32 = 1 << 31;
const DATASET_PRESENCE_COUNT_MASK: u32 = DATASET_PRESENCE_SEEN_BIT - 1;

#[derive(Clone, Copy, Debug)]
pub struct CounterConfig {
    pub k: usize,
    pub minimizer: usize,
    pub threshold: Count,
}

#[derive(Clone, Copy, Debug)]
enum MinimizerOrder {
    SimdValueHash,
    SimdDirectHash,
    AntiLex,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CountedKmer {
    pub encoded: EncodedKmer,
    pub count: Count,
}

#[derive(Clone, Copy, Debug)]
pub struct Phase12Stats {
    pub phase1: Duration,
    pub phase2: Duration,
    pub total: Duration,
    pub unique_minimizer_hashes: usize,
    pub partition_bytes: u64,
}

pub fn validate_config(config: CounterConfig) -> Result<()> {
    validate_shape_config(config)?;
    ensure!(
        config.threshold < CompactCount::MAX as Count,
        "threshold must be < 255 because compact minimizer counts saturate in u8"
    );
    Ok(())
}

fn validate_shape_config(config: CounterConfig) -> Result<()> {
    ensure!(config.k > 0, "k must be greater than 0");
    ensure!(
        config.k <= 64,
        "k must be <= 64 because MC stores k-mers in u128"
    );
    ensure!(
        config.minimizer > 0,
        "minimizer size must be greater than 0"
    );
    ensure!(config.minimizer <= config.k, "minimizer size must be <= k");
    ensure!(
        config.minimizer <= 64,
        "minimizer size must be <= 64 because MC stores minimizers in u128"
    );
    Ok(())
}

fn validate_dataset_presence_config(config: CounterConfig) -> Result<()> {
    validate_shape_config(config)?;
    ensure!(
        config.k <= 32,
        "FOFN dataset-presence mode currently requires k <= 32"
    );
    ensure!(
        config.threshold < DATASET_PRESENCE_COUNT_MASK as Count,
        "FOFN dataset-presence threshold must be < 2^31"
    );
    Ok(())
}

pub fn run_inputs_phase12(inputs: &[PathBuf], config: CounterConfig) -> Result<Phase12Stats> {
    run_inputs_phase12_with_order(inputs, config, MinimizerOrder::SimdValueHash)
}

pub fn run_inputs_phase12_direct(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<Phase12Stats> {
    run_inputs_phase12_with_order(inputs, config, MinimizerOrder::SimdDirectHash)
}

pub fn run_inputs_phase12_antilex(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<Phase12Stats> {
    run_inputs_phase12_with_order(inputs, config, MinimizerOrder::AntiLex)
}

fn run_inputs_phase12_with_order(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
) -> Result<Phase12Stats> {
    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let use_read_cache = should_cache_packed_reads(inputs);
    let partition_dir = create_partition_dir()?;
    let total_started = Instant::now();
    let result = (|| {
        let phase_started = Instant::now();
        let minimizer_counts =
            ShardedMinimizerCounts::new(config.threshold, estimate_unique_minimizer_hashes(inputs));
        count_input_minimizers_maybe_cached(
            inputs,
            config,
            order,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        let minimizer_counts = minimizer_counts.freeze();
        let phase1 = phase_started.elapsed();
        log_phase("1_minimizer_counting", phase1);

        let unique_minimizer_hashes = minimizer_counts.unique_hashes();
        let phase_started = Instant::now();
        write_filtered_superkmer_partitions_maybe_cached(
            inputs,
            config,
            order,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        if use_read_cache {
            remove_packed_read_caches(&partition_dir, inputs.len())?;
        }
        let phase2 = phase_started.elapsed();
        log_phase("2_superkmer_partitioning", phase2);

        let partition_bytes = directory_file_bytes(&partition_dir)?;
        let total = total_started.elapsed();
        log_phase("phase12_total", total);
        Ok(Phase12Stats {
            phase1,
            phase2,
            total,
            unique_minimizer_hashes,
            partition_bytes,
        })
    })();

    match fs::remove_dir_all(&partition_dir) {
        Ok(()) => result,
        Err(err) if result.is_err() => {
            let _ = err;
            result
        }
        Err(err) => {
            Err(err).with_context(|| format!("failed to remove {}", partition_dir.display()))
        }
    }
}

pub fn expand_fofns(fofns: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut inputs = Vec::new();

    for fofn in fofns {
        let text = fs::read_to_string(fofn)
            .with_context(|| format!("failed to read FOFN {}", fofn.display()))?;
        let base = fofn.parent().unwrap_or_else(|| Path::new("."));

        for (line_no, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let path = PathBuf::from(line);
            let path = if path.is_absolute() {
                path
            } else {
                base.join(path)
            };

            if path.as_os_str().is_empty() {
                bail!(
                    "empty path in FOFN {} at line {}",
                    fofn.display(),
                    line_no + 1
                );
            }
            inputs.push(path);
        }
    }

    ensure!(!inputs.is_empty(), "no inputs found in FOFN(s)");
    Ok(inputs)
}

pub fn count_inputs(inputs: &[PathBuf], config: CounterConfig) -> Result<Vec<CountedKmer>> {
    let mut counted = Vec::new();
    count_inputs_partitioned(inputs, config, |record| {
        counted.push(record.clone());
        Ok(())
    })?;

    counted.sort_unstable_by_key(|record| record.encoded);
    Ok(counted)
}

pub fn count_datasets(inputs: &[PathBuf], config: CounterConfig) -> Result<Vec<CountedKmer>> {
    let counts = count_dataset_presence_counts_u64(inputs, config)?;
    let mut counted = Vec::new();
    for shard in counts.shards.iter() {
        for (encoded, count) in shard.iter() {
            if count as Count > config.threshold {
                counted.push(CountedKmer {
                    encoded: encoded as EncodedKmer,
                    count: count as Count,
                });
            }
        }
    }
    counted.sort_unstable_by_key(|record| record.encoded);
    Ok(counted)
}

pub fn count_inputs_to_fasta_path(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    let total_started = Instant::now();
    let kmers = count_inputs(inputs, config)?;

    let phase_started = Instant::now();
    write_simplitig_fasta_path(output_path, config.k, &kmers)?;
    log_phase("4_simplitig_output", phase_started.elapsed());
    log_phase("fasta_total", total_started.elapsed());
    Ok(())
}

pub fn count_datasets_to_fasta_path(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    let total_started = Instant::now();
    let counts = count_dataset_presence_counts_u64(inputs, config)?;

    let phase_started = Instant::now();
    write_simplitig_fasta_u32_counts(output_path, config, &counts)?;
    log_phase("3_output_streaming", phase_started.elapsed());
    log_phase("total", total_started.elapsed());
    Ok(())
}

pub fn count_inputs_to_kff_path(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    if config.k <= 32 {
        return count_inputs_to_kff_path_u8(inputs, config, output_path);
    }

    if output_compression(output_path).is_some() {
        let temp_dir = create_partition_dir()?;
        let temp_path = temp_dir.join("output.kff");
        let result = (|| {
            count_inputs_to_kff_path(inputs, config, &temp_path)?;
            compress_existing_file(&temp_path, output_path)
        })();
        if result.is_ok() {
            fs::remove_dir_all(&temp_dir)
                .with_context(|| format!("failed to remove {}", temp_dir.display()))?;
        } else {
            let _ = fs::remove_dir_all(&temp_dir);
        }
        return result;
    }

    let output = File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut writer = BufWriter::new(output);
    write_kff_header(&mut writer, config, 0, false)?;
    let count_pos = writer.stream_position()? - 8;
    let mut emitted = 0u64;

    count_inputs_partitioned(inputs, config, |record| {
        write_packed_dna(&mut writer, record.encoded, config.k)?;
        write_u64(&mut writer, record.count)?;
        emitted += 1;
        Ok(())
    })?;

    writer.write_all(b"KFF")?;
    writer.flush()?;

    let mut output = writer
        .into_inner()
        .map_err(|err| anyhow::anyhow!("failed to flush {}: {}", output_path.display(), err))?;
    output.seek(SeekFrom::Start(count_pos))?;
    write_u64(&mut output, emitted)?;
    output.flush()?;

    Ok(())
}

pub fn count_datasets_to_kff_path(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    let total_started = Instant::now();
    let counts = count_dataset_presence_counts_u64(inputs, config)?;

    let phase_started = Instant::now();
    write_kff_u32_counts(output_path, config, &counts)?;
    log_phase("3_output_streaming", phase_started.elapsed());
    log_phase("total", total_started.elapsed());
    Ok(())
}

fn count_inputs_to_kff_path_u8(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    if use_kmc_style_partitioning() {
        return count_inputs_to_kff_path_u8_kmc_style(inputs, config, output_path);
    }

    if should_use_ram_kmer_count(inputs) {
        return count_inputs_to_kff_path_u8_ram(inputs, config, output_path);
    }

    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let use_read_cache = should_cache_packed_reads(inputs);
    let partition_dir = create_partition_dir()?;
    let total_started = Instant::now();
    let result = (|| {
        let phase_started = Instant::now();
        let minimizer_counts =
            ShardedMinimizerCounts::new(config.threshold, estimate_unique_minimizer_hashes(inputs));
        let order = kff_minimizer_order();
        count_input_minimizers_maybe_cached(
            inputs,
            config,
            order,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        let minimizer_counts = minimizer_counts.freeze();
        log_phase("1_minimizer_counting", phase_started.elapsed());

        let phase_started = Instant::now();
        write_filtered_superkmer_partitions_maybe_cached(
            inputs,
            config,
            order,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        if use_read_cache {
            remove_packed_read_caches(&partition_dir, inputs.len())?;
        }
        let partition_bytes = directory_file_bytes(&partition_dir)?;
        eprintln!("MC_STAT\tpartition_bytes\t{partition_bytes}");
        log_phase("2_superkmer_partitioning", phase_started.elapsed());

        let phase_started = Instant::now();
        let emitted = process_superkmer_partitions_to_kff_u8_fragments(&partition_dir, config)?;
        log_phase("3a_kmer_counting", phase_started.elapsed());

        let emit_started = Instant::now();
        emit_kff_u8_fragments(&partition_dir, config, emitted, output_path)?;
        log_phase("3b_output_streaming", emit_started.elapsed());
        log_phase("3_kmer_counting_and_output", phase_started.elapsed());
        Ok(())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&partition_dir)
            .with_context(|| format!("failed to remove {}", partition_dir.display()))?;
    } else {
        let _ = fs::remove_dir_all(&partition_dir);
    }

    result.inspect(|_| log_phase("total", total_started.elapsed()))
}

fn count_inputs_to_kff_path_u8_kmc_style(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let partition_dir = create_partition_dir()?;
    let total_started = Instant::now();
    let result = (|| {
        let phase_started = Instant::now();
        write_all_superkmer_partitions(inputs, config, kff_minimizer_order(), &partition_dir)?;
        let partition_bytes = directory_file_bytes(&partition_dir)?;
        eprintln!("MC_STAT\tpartition_bytes\t{partition_bytes}");
        log_phase("1_superkmer_partitioning", phase_started.elapsed());

        let phase_started = Instant::now();
        let emitted = process_superkmer_partitions_to_kff_u8_fragments(&partition_dir, config)?;
        log_phase("2a_kmer_counting", phase_started.elapsed());

        let emit_started = Instant::now();
        emit_kff_u8_fragments(&partition_dir, config, emitted, output_path)?;
        log_phase("2b_output_streaming", emit_started.elapsed());
        log_phase("2_kmer_counting_and_output", phase_started.elapsed());
        Ok(())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&partition_dir)
            .with_context(|| format!("failed to remove {}", partition_dir.display()))?;
    } else {
        let _ = fs::remove_dir_all(&partition_dir);
    }

    result.inspect(|_| log_phase("total", total_started.elapsed()))
}

fn count_dataset_presence_counts_u64(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<KmerCountsU64U32> {
    validate_dataset_presence_config(config)?;
    ensure!(!inputs.is_empty(), "at least one dataset file is required");

    let partition_dir = create_partition_dir()?;
    let result = (|| {
        let order = kff_minimizer_order();

        let phase_started = Instant::now();
        let minimizer_counts =
            ShardedDatasetMinimizerCounts::new(estimate_unique_minimizer_hashes(inputs));
        for (dataset_idx, path) in inputs.iter().enumerate() {
            let remaining_after = inputs.len() - dataset_idx - 1;
            let accept_new = 1 + remaining_after as Count > config.threshold;
            count_dataset_minimizer_presence(path, config, order, &minimizer_counts, accept_new)?;
            minimizer_counts.finish_dataset(remaining_after, config.threshold);
        }
        let minimizer_counts = minimizer_counts.freeze();
        let retained_minimizers = minimizer_counts.unique_hashes();
        log_phase("1_dataset_minimizer_counting", phase_started.elapsed());

        let phase_started = Instant::now();
        let kmer_counts = ShardedKmerCountsU64U32::new(estimate_unique_kmers_for_ram_count(inputs));
        if retained_minimizers != 0 {
            for (dataset_idx, path) in inputs.iter().enumerate() {
                let dataset_dir = dataset_presence_partition_dir(&partition_dir, dataset_idx);
                fs::create_dir(&dataset_dir)
                    .with_context(|| format!("failed to create {}", dataset_dir.display()))?;
                write_dataset_filtered_kmer_partitions(
                    path,
                    config,
                    order,
                    &minimizer_counts,
                    &dataset_dir,
                )?;
                process_dataset_kmer_presence_partitions(&dataset_dir, &kmer_counts)?;
                fs::remove_dir_all(&dataset_dir)
                    .with_context(|| format!("failed to remove {}", dataset_dir.display()))?;
            }
        }
        log_phase("2_dataset_kmer_counting", phase_started.elapsed());
        Ok(kmer_counts.freeze())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&partition_dir)
            .with_context(|| format!("failed to remove {}", partition_dir.display()))?;
    } else {
        let _ = fs::remove_dir_all(&partition_dir);
    }

    result
}

fn kff_minimizer_order() -> MinimizerOrder {
    match env::var("MC_MINIMIZER_ORDER").ok().as_deref() {
        Some("antilex") => MinimizerOrder::AntiLex,
        Some("simd-value" | "value") => MinimizerOrder::SimdValueHash,
        Some("direct" | "simd-direct") => MinimizerOrder::SimdDirectHash,
        _ => MinimizerOrder::SimdDirectHash,
    }
}

fn use_kmc_style_partitioning() -> bool {
    !matches!(
        env::var("MC_KMC_STYLE").ok().as_deref(),
        Some("0" | "false" | "no")
    )
}

fn count_inputs_to_kff_path_u8_ram(
    inputs: &[PathBuf],
    config: CounterConfig,
    output_path: &Path,
) -> Result<()> {
    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let use_read_cache = should_cache_packed_reads(inputs);
    let partition_dir = create_partition_dir()?;
    let total_started = Instant::now();
    let result = (|| {
        let phase_started = Instant::now();
        let minimizer_counts =
            ShardedMinimizerCounts::new(config.threshold, estimate_unique_minimizer_hashes(inputs));
        count_input_minimizers_maybe_cached(
            inputs,
            config,
            MinimizerOrder::SimdDirectHash,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        let minimizer_counts = minimizer_counts.freeze();
        log_phase("1_minimizer_counting", phase_started.elapsed());

        let phase_started = Instant::now();
        let kmer_counts = ShardedKmerCountsU64::new(
            config.threshold,
            estimate_unique_kmers_for_ram_count(inputs),
        );
        count_filtered_kmers_from_inputs_maybe_cached(
            inputs,
            config,
            MinimizerOrder::SimdDirectHash,
            &minimizer_counts,
            &kmer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        if use_read_cache {
            remove_packed_read_caches(&partition_dir, inputs.len())?;
        }
        let kmer_counts = kmer_counts.freeze();
        log_phase("2_filtered_kmer_counting", phase_started.elapsed());

        let phase_started = Instant::now();
        write_kff_u8_counts(output_path, config, &kmer_counts)?;
        log_phase("3_output_streaming", phase_started.elapsed());
        Ok(())
    })();

    if result.is_ok() {
        fs::remove_dir_all(&partition_dir)
            .with_context(|| format!("failed to remove {}", partition_dir.display()))?;
    } else {
        let _ = fs::remove_dir_all(&partition_dir);
    }

    result.inspect(|_| log_phase("total", total_started.elapsed()))
}

fn count_inputs_partitioned<F>(inputs: &[PathBuf], config: CounterConfig, mut emit: F) -> Result<()>
where
    F: FnMut(&CountedKmer) -> Result<()>,
{
    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let use_read_cache = should_cache_packed_reads(inputs);
    let partition_dir = create_partition_dir()?;
    let total_started = Instant::now();
    let result = (|| {
        let phase_started = Instant::now();
        let minimizer_counts =
            ShardedMinimizerCounts::new(config.threshold, estimate_unique_minimizer_hashes(inputs));
        count_input_minimizers_maybe_cached(
            inputs,
            config,
            MinimizerOrder::SimdDirectHash,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        let minimizer_counts = minimizer_counts.freeze();
        log_phase("1_minimizer_counting", phase_started.elapsed());

        let phase_started = Instant::now();
        write_filtered_superkmer_partitions_maybe_cached(
            inputs,
            config,
            MinimizerOrder::SimdDirectHash,
            &minimizer_counts,
            &partition_dir,
            use_read_cache,
        )?;
        if use_read_cache {
            remove_packed_read_caches(&partition_dir, inputs.len())?;
        }
        log_phase("2_superkmer_partitioning", phase_started.elapsed());

        let phase_started = Instant::now();
        process_superkmer_partitions(&partition_dir, config, &mut emit)
            .inspect(|_| log_phase("3_kmer_counting_and_output", phase_started.elapsed()))
    })();

    if result.is_ok() {
        fs::remove_dir_all(&partition_dir)
            .with_context(|| format!("failed to remove {}", partition_dir.display()))?;
    } else {
        let _ = fs::remove_dir_all(&partition_dir);
    }

    result.inspect(|_| log_phase("total", total_started.elapsed()))
}

fn log_phase(name: &str, elapsed: Duration) {
    eprintln!("MC_PHASE\t{name}\t{:.6}", elapsed.as_secs_f64());
}

pub fn write_fasta<W: Write>(mut writer: W, kmers: &[CountedKmer], k: usize) -> Result<()> {
    write_simplitigs(&mut writer, kmers, k)
}

pub fn write_fasta_path(output_path: &Path, kmers: &[CountedKmer], k: usize) -> Result<()> {
    let mut writer = create_output_writer(output_path)?;
    write_simplitigs(&mut writer, kmers, k)?;
    writer.finish()
}

pub fn simplitig_sequences(kmers: &[CountedKmer], k: usize) -> Result<Vec<Vec<u8>>> {
    let mut sequences = Vec::new();
    for_each_simplitig(kmers, k, |seq, _| {
        sequences.push(seq.to_vec());
        Ok(())
    })?;
    Ok(sequences)
}

pub fn write_kff<W: Write>(
    mut writer: W,
    kmers: &[CountedKmer],
    config: CounterConfig,
) -> Result<()> {
    validate_config(config)?;
    write_kff_header(&mut writer, config, kmers.len() as u64, true)?;

    for record in kmers {
        write_packed_dna(&mut writer, record.encoded, config.k)?;
        write_u64(&mut writer, record.count)?;
    }
    writer.write_all(b"KFF")?;

    Ok(())
}

enum OutputCompression {
    Gzip,
    Xz,
    Zstd,
}

pub struct OutputWriter {
    inner: OutputWriterInner,
}

enum OutputWriterInner {
    Plain(BufWriter<File>),
    Gzip(GzEncoder<BufWriter<File>>),
    Xz(XzEncoder<BufWriter<File>>),
    Zstd(zstd::stream::write::Encoder<'static, BufWriter<File>>),
}

impl OutputWriter {
    fn finish(self) -> Result<()> {
        match self.inner {
            OutputWriterInner::Plain(mut writer) => {
                writer.flush()?;
            }
            OutputWriterInner::Gzip(writer) => {
                let mut writer = writer.finish()?;
                writer.flush()?;
            }
            OutputWriterInner::Xz(writer) => {
                let mut writer = writer.finish()?;
                writer.flush()?;
            }
            OutputWriterInner::Zstd(writer) => {
                let mut writer = writer.finish()?;
                writer.flush()?;
            }
        }
        Ok(())
    }
}

impl Write for OutputWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.inner {
            OutputWriterInner::Plain(writer) => writer.write(buf),
            OutputWriterInner::Gzip(writer) => writer.write(buf),
            OutputWriterInner::Xz(writer) => writer.write(buf),
            OutputWriterInner::Zstd(writer) => writer.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.inner {
            OutputWriterInner::Plain(writer) => writer.flush(),
            OutputWriterInner::Gzip(writer) => writer.flush(),
            OutputWriterInner::Xz(writer) => writer.flush(),
            OutputWriterInner::Zstd(writer) => writer.flush(),
        }
    }
}

fn create_output_writer(output_path: &Path) -> Result<OutputWriter> {
    let file = File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    let inner = match output_compression(output_path) {
        Some(OutputCompression::Gzip) => {
            OutputWriterInner::Gzip(GzEncoder::new(writer, GzipCompression::fast()))
        }
        Some(OutputCompression::Xz) => OutputWriterInner::Xz(XzEncoder::new(writer, 6)),
        Some(OutputCompression::Zstd) => {
            OutputWriterInner::Zstd(zstd::stream::write::Encoder::new(writer, 0)?)
        }
        None => OutputWriterInner::Plain(writer),
    };
    Ok(OutputWriter { inner })
}

fn output_compression(path: &Path) -> Option<OutputCompression> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("gz" | "gzip") => Some(OutputCompression::Gzip),
        Some("xz") => Some(OutputCompression::Xz),
        Some("zst" | "zstd") => Some(OutputCompression::Zstd),
        _ => None,
    }
}

fn compress_existing_file(input_path: &Path, output_path: &Path) -> Result<()> {
    let input = File::open(input_path)
        .with_context(|| format!("failed to open {}", input_path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, input);
    let mut writer = create_output_writer(output_path)?;
    io::copy(&mut reader, &mut writer)
        .with_context(|| format!("failed to compress {}", output_path.display()))?;
    writer.finish()
}

pub fn with_xz_decompressed_path<T, F>(path: &Path, consume: F) -> Result<T>
where
    F: FnOnce(&Path) -> Result<T>,
{
    if !is_xz_path(path) {
        return consume(path);
    }

    let temp_dir = create_partition_dir()?;
    let temp_path = temp_dir.join("input.fa");
    let result = (|| {
        decompress_xz_file(path, &temp_path)?;
        consume(&temp_path)
    })();
    if result.is_ok() {
        fs::remove_dir_all(&temp_dir)
            .with_context(|| format!("failed to remove {}", temp_dir.display()))?;
    } else {
        let _ = fs::remove_dir_all(&temp_dir);
    }
    result
}

fn decompress_xz_file(input_path: &Path, output_path: &Path) -> Result<()> {
    let input = File::open(input_path)
        .with_context(|| format!("failed to open {}", input_path.display()))?;
    let mut reader = XzDecoder::new(input);
    let output = File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, output);
    io::copy(&mut reader, &mut writer)
        .with_context(|| format!("failed to decompress {}", input_path.display()))?;
    writer.flush()?;
    Ok(())
}

fn is_xz_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("xz"))
}

fn write_simplitig_fasta_path(output_path: &Path, k: usize, kmers: &[CountedKmer]) -> Result<()> {
    write_fasta_path(output_path, kmers, k)
}

fn write_simplitigs<W: Write>(writer: &mut W, kmers: &[CountedKmer], k: usize) -> Result<()> {
    for_each_simplitig(kmers, k, |seq, simplitig_idx| {
        write_simplitig_record(writer, simplitig_idx, seq, k)
    })
}

fn for_each_simplitig<F>(kmers: &[CountedKmer], k: usize, mut emit: F) -> Result<()>
where
    F: FnMut(&[u8], usize) -> Result<()>,
{
    validate_output_k(k)?;
    if kmers.is_empty() {
        return Ok(());
    }

    let mut remaining = AHashMap::with_capacity(kmers.len());
    for record in kmers {
        remaining.insert(record.encoded, ());
    }

    let mut seeds = remaining.keys().copied().collect::<Vec<_>>();
    seeds.sort_unstable();

    let mut simplitig_idx = 0usize;
    for seed in seeds {
        if remaining.remove(&seed).is_none() {
            continue;
        }

        let mut seq = decode_kmer(seed, k);
        extend_simplitig_forward(&mut seq, k, &mut remaining);
        reverse_complement_in_place(&mut seq);
        extend_simplitig_forward(&mut seq, k, &mut remaining);
        reverse_complement_in_place(&mut seq);

        simplitig_idx += 1;
        emit(&seq, simplitig_idx)?;
    }

    Ok(())
}

fn write_simplitig_record<W: Write>(
    writer: &mut W,
    idx: usize,
    seq: &[u8],
    k: usize,
) -> Result<()> {
    let kmer_count = seq.len().saturating_sub(k) + 1;
    writeln!(writer, ">MC_simplitig_{} kmers={}", idx, kmer_count)?;
    write_seq_lines(writer, seq)?;
    Ok(())
}

fn write_seq_lines<W: Write>(writer: &mut W, seq: &[u8]) -> Result<()> {
    for chunk in seq.chunks(80) {
        writer.write_all(chunk)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn extend_simplitig_forward(
    seq: &mut Vec<u8>,
    k: usize,
    remaining: &mut AHashMap<EncodedKmer, ()>,
) {
    const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

    loop {
        let mut extended = false;
        for base in BASES {
            let encoded = canonical_suffix_extension(seq, k, base);
            if remaining.remove(&encoded).is_some() {
                seq.push(base);
                extended = true;
                break;
            }
        }

        if !extended {
            return;
        }
    }
}

fn canonical_suffix_extension(seq: &[u8], k: usize, base: u8) -> EncodedKmer {
    let suffix_len = k - 1;
    let start = seq.len() - suffix_len;
    let high_shift = 2 * (k - 1);
    let mask = kmer_mask(k);
    let mut fwd = 0u128;
    let mut rev = 0u128;

    for &base in &seq[start..] {
        let bits = base_bits(base) as u128;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);
    }

    let bits = base_bits(base) as u128;
    fwd = ((fwd << 2) | bits) & mask;
    rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);
    fwd.min(rev)
}

fn reverse_complement_in_place(seq: &mut [u8]) {
    let len = seq.len();
    for idx in 0..(len / 2) {
        let left = complement_base(seq[idx]);
        let right = complement_base(seq[len - idx - 1]);
        seq[idx] = right;
        seq[len - idx - 1] = left;
    }
    if len % 2 == 1 {
        let mid = len / 2;
        seq[mid] = complement_base(seq[mid]);
    }
}

fn complement_base(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        _ => unreachable!("MC only stores A/C/G/T bases in encoded k-mers"),
    }
}

fn write_kff_header<W: Write>(
    writer: &mut W,
    config: CounterConfig,
    record_count: u64,
    ordered: bool,
) -> Result<()> {
    write_kff_header_with_count_size(writer, config, record_count, ordered, 8)
}

fn write_kff_header_with_count_size<W: Write>(
    mut writer: &mut W,
    config: CounterConfig,
    record_count: u64,
    ordered: bool,
    count_size: u64,
) -> Result<()> {
    validate_shape_config(config)?;
    let free_block = format!(
        "producer=MC\nk={}\nm={}\nthreshold={}\nfilter=count>threshold\ncanonical=1\n",
        config.k, config.minimizer, config.threshold
    );
    ensure!(
        free_block.len() <= u32::MAX as usize,
        "KFF free block is too large"
    );

    writer.write_all(b"KFF")?;
    writer.write_all(&[1, 0, KFF_ENCODING_ACGT, 1, 1])?;
    write_u32(&mut writer, free_block.len() as u32)?;
    writer.write_all(free_block.as_bytes())?;

    write_kff_values(
        &mut writer,
        &[
            ("k", config.k as u64),
            ("max", 1),
            ("data_size", count_size),
            ("ordered", u64::from(ordered)),
            ("m", config.minimizer as u64),
            ("threshold", config.threshold),
        ],
    )?;

    writer.write_all(b"r")?;
    write_u64(writer, record_count)?;
    Ok(())
}

fn validate_output_k(k: usize) -> Result<()> {
    ensure!(k > 0, "k must be greater than 0");
    ensure!(k <= 64, "k must be <= 64");
    Ok(())
}

fn count_file_minimizers_with_order(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &ShardedMinimizerCounts,
) -> Result<()> {
    for_fastx_batches(path, config.k, |batch| {
        batch
            .par_iter()
            .fold(MinimizerCountWorker::new, |mut worker, seq| {
                worker.add_seq(seq, config, order, minimizer_counts);
                worker
            })
            .for_each(|mut worker| {
                worker.flush(minimizer_counts);
            });
        Ok(())
    })?;
    Ok(())
}

fn count_file_minimizers_packed(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &ShardedMinimizerCounts,
) -> Result<()> {
    for_fastx_packed_batches(path, config.k, |batch| {
        batch
            .par_iter()
            .fold(MinimizerCountWorker::new, |mut worker, seq| {
                worker.add_packed_seq(seq, config, order, minimizer_counts);
                worker
            })
            .for_each(|mut worker| {
                worker.flush(minimizer_counts);
            });
        Ok(())
    })?;
    Ok(())
}

fn count_input_minimizers_maybe_cached(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &ShardedMinimizerCounts,
    partition_dir: &Path,
    use_read_cache: bool,
) -> Result<()> {
    let results: Result<Vec<_>> = inputs
        .par_iter()
        .enumerate()
        .map(|(input_idx, path)| {
            if use_read_cache {
                count_file_minimizers_packed_cached(
                    path,
                    &packed_read_cache_path(partition_dir, input_idx),
                    config,
                    order,
                    minimizer_counts,
                )
            } else {
                count_file_minimizers_packed(path, config, order, minimizer_counts)
            }
        })
        .collect();
    results?;
    Ok(())
}

fn count_file_minimizers_packed_cached(
    path: &Path,
    cache_path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &ShardedMinimizerCounts,
) -> Result<()> {
    let (sender, receiver) = mpsc::sync_channel::<Vec<PackedSeqVec>>(2);

    std::thread::scope(|scope| {
        let worker = scope.spawn(move || -> Result<()> {
            while let Ok(batch) = receiver.recv() {
                batch
                    .par_iter()
                    .fold(MinimizerCountWorker::new, |mut worker, seq| {
                        worker.add_packed_seq(seq, config, order, minimizer_counts);
                        worker
                    })
                    .for_each(|mut worker| {
                        worker.flush(minimizer_counts);
                    });
            }
            Ok(())
        });

        let parse_result = for_fastx_packed_batches_cached(path, config.k, cache_path, |batch| {
            sender
                .send(batch)
                .map_err(|_| anyhow::anyhow!("minimizer counting worker stopped"))?;
            Ok(())
        });
        drop(sender);

        let worker_result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("minimizer counting worker panicked"))?;
        parse_result.and(worker_result)
    })?;
    Ok(())
}

fn count_dataset_minimizer_presence(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &ShardedDatasetMinimizerCounts,
    accept_new: bool,
) -> Result<()> {
    for_fastx_packed_batches(path, config.k, |batch| {
        batch
            .into_par_iter()
            .fold(DatasetMinimizerPresenceWorker::new, |mut worker, seq| {
                worker.add_packed_seq(&seq, config, order, minimizer_counts, accept_new);
                worker
            })
            .for_each(|mut worker| {
                worker.flush(minimizer_counts, accept_new);
            });
        Ok(())
    })?;
    Ok(())
}

pub fn count_inputs_minimizer_phase(inputs: &[PathBuf], config: CounterConfig) -> Result<usize> {
    count_inputs_minimizer_phase_with_order(inputs, config, MinimizerOrder::SimdValueHash)
}

pub fn count_inputs_minimizer_phase_antilex(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<usize> {
    count_inputs_minimizer_phase_with_order(inputs, config, MinimizerOrder::AntiLex)
}

pub fn count_inputs_minimizer_phase_direct(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<usize> {
    count_inputs_minimizer_phase_with_order(inputs, config, MinimizerOrder::SimdDirectHash)
}

pub fn count_inputs_minimizer_phase_packed(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<usize> {
    count_inputs_minimizer_phase_packed_with_order(inputs, config, MinimizerOrder::SimdValueHash)
}

pub fn count_inputs_minimizer_phase_packed_direct(
    inputs: &[PathBuf],
    config: CounterConfig,
) -> Result<usize> {
    count_inputs_minimizer_phase_packed_with_order(inputs, config, MinimizerOrder::SimdDirectHash)
}

fn count_inputs_minimizer_phase_with_order(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
) -> Result<usize> {
    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let minimizer_counts =
        ShardedMinimizerCounts::new(config.threshold, estimate_unique_minimizer_hashes(inputs));
    let results: Result<Vec<_>> = inputs
        .par_iter()
        .map(|path| count_file_minimizers_with_order(path, config, order, &minimizer_counts))
        .collect();
    results?;
    let minimizer_counts = minimizer_counts.freeze();
    Ok(minimizer_counts.unique_hashes())
}

fn count_inputs_minimizer_phase_packed_with_order(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
) -> Result<usize> {
    validate_config(config)?;
    ensure!(!inputs.is_empty(), "at least one input file is required");

    let minimizer_counts =
        ShardedMinimizerCounts::new(config.threshold, estimate_unique_minimizer_hashes(inputs));
    let results: Result<Vec<_>> = inputs
        .par_iter()
        .map(|path| count_file_minimizers_packed(path, config, order, &minimizer_counts))
        .collect();
    results?;
    let minimizer_counts = minimizer_counts.freeze();
    Ok(minimizer_counts.unique_hashes())
}

fn write_filtered_superkmer_partitions(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    partition_dir: &Path,
) -> Result<()> {
    let writers = SuperkmerPartitionWriters::create(partition_dir)?;
    let result: Result<Vec<_>> = inputs
        .par_iter()
        .map(|path| {
            write_file_filtered_superkmers_packed(path, config, order, minimizer_counts, &writers)
        })
        .collect();
    result?;
    writers.finish()
}

fn write_all_superkmer_partitions(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
    partition_dir: &Path,
) -> Result<()> {
    let writers = SuperkmerPartitionWriters::create(partition_dir)?;
    let result: Result<Vec<_>> = inputs
        .par_iter()
        .map(|path| write_file_all_superkmers_packed(path, config, order, &writers))
        .collect();
    result?;
    writers.finish()
}

fn write_filtered_superkmer_partitions_maybe_cached(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    partition_dir: &Path,
    use_read_cache: bool,
) -> Result<()> {
    if use_read_cache {
        write_filtered_superkmer_partitions_from_caches(
            inputs.len(),
            config,
            order,
            minimizer_counts,
            partition_dir,
        )
    } else {
        write_filtered_superkmer_partitions(inputs, config, order, minimizer_counts, partition_dir)
    }
}

fn write_filtered_superkmer_partitions_from_caches(
    input_count: usize,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    partition_dir: &Path,
) -> Result<()> {
    let writers = SuperkmerPartitionWriters::create(partition_dir)?;
    let result: Result<Vec<_>> = (0..input_count)
        .into_par_iter()
        .map(|input_idx| {
            write_packed_read_cache_filtered_superkmers(
                &packed_read_cache_path(partition_dir, input_idx),
                config,
                order,
                minimizer_counts,
                &writers,
            )
        })
        .collect();
    result?;
    writers.finish()
}

fn write_file_all_superkmers_packed(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    writers: &SuperkmerPartitionWriters,
) -> Result<()> {
    let (sender, receiver) = mpsc::sync_channel::<Vec<PackedSeqVec>>(2);

    std::thread::scope(|scope| {
        let worker = scope.spawn(move || -> Result<()> {
            while let Ok(batch) = receiver.recv() {
                write_all_superkmer_batch(batch, config, order, writers)?;
            }
            Ok(())
        });

        let parse_result = for_fastx_packed_batches(path, config.k, |batch| {
            sender
                .send(batch)
                .map_err(|_| anyhow::anyhow!("super-kmer partition worker stopped"))?;
            Ok(())
        });
        drop(sender);

        let worker_result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("super-kmer partition worker panicked"))?;
        parse_result.and(worker_result)
    })?;
    Ok(())
}

fn write_all_superkmer_batch(
    batch: Vec<PackedSeqVec>,
    config: CounterConfig,
    order: MinimizerOrder,
    writers: &SuperkmerPartitionWriters,
) -> Result<()> {
    let buffers: Result<Vec<_>> = batch
        .into_par_iter()
        .fold(
            || Ok(SuperkmerWorkerBuffers::new()),
            |buffers, seq| {
                let mut buffers = buffers?;
                add_all_superkmers_packed(seq, config, order, writers, &mut buffers)?;
                Ok(buffers)
            },
        )
        .collect();

    for mut buffers in buffers? {
        buffers.superkmers.flush_all(writers)?;
    }
    Ok(())
}

fn write_file_filtered_superkmers_packed(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    writers: &SuperkmerPartitionWriters,
) -> Result<()> {
    for_fastx_packed_batches(path, config.k, |batch| {
        let buffers: Result<Vec<_>> = batch
            .into_par_iter()
            .fold(
                || Ok(SuperkmerWorkerBuffers::new()),
                |buffers, seq| {
                    let mut buffers = buffers?;
                    add_filtered_superkmers_packed(
                        seq,
                        config,
                        order,
                        minimizer_counts,
                        writers,
                        &mut buffers,
                    )?;
                    Ok(buffers)
                },
            )
            .collect();

        for mut buffers in buffers? {
            buffers.superkmers.flush_all(writers)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn write_packed_read_cache_filtered_superkmers(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    writers: &SuperkmerPartitionWriters,
) -> Result<()> {
    for_packed_read_cache_batches(path, config.k, |batch| {
        let buffers: Result<Vec<_>> = batch
            .into_par_iter()
            .fold(
                || Ok(SuperkmerWorkerBuffers::new()),
                |buffers, seq| {
                    let mut buffers = buffers?;
                    add_filtered_superkmers_packed(
                        seq,
                        config,
                        order,
                        minimizer_counts,
                        writers,
                        &mut buffers,
                    )?;
                    Ok(buffers)
                },
            )
            .collect();

        for mut buffers in buffers? {
            buffers.superkmers.flush_all(writers)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn count_filtered_kmers_from_inputs_maybe_cached(
    inputs: &[PathBuf],
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    kmer_counts: &ShardedKmerCountsU64,
    partition_dir: &Path,
    use_read_cache: bool,
) -> Result<()> {
    let results: Result<Vec<_>> = inputs
        .par_iter()
        .enumerate()
        .map(|(input_idx, path)| {
            if use_read_cache {
                count_filtered_kmers_from_packed_read_cache(
                    &packed_read_cache_path(partition_dir, input_idx),
                    config,
                    order,
                    minimizer_counts,
                    kmer_counts,
                )
            } else {
                count_filtered_kmers_from_file(path, config, order, minimizer_counts, kmer_counts)
            }
        })
        .collect();
    results?;
    Ok(())
}

fn count_filtered_kmers_from_file(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    kmer_counts: &ShardedKmerCountsU64,
) -> Result<()> {
    for_fastx_packed_batches(path, config.k, |batch| {
        batch
            .into_par_iter()
            .fold(KmerCountWorker::new, |mut worker, seq| {
                worker.add_filtered_seq(&seq, config, order, minimizer_counts, kmer_counts);
                worker
            })
            .for_each(|mut worker| {
                worker.flush(kmer_counts);
            });
        Ok(())
    })?;
    Ok(())
}

fn count_filtered_kmers_from_packed_read_cache(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    kmer_counts: &ShardedKmerCountsU64,
) -> Result<()> {
    for_packed_read_cache_batches(path, config.k, |batch| {
        batch
            .into_par_iter()
            .fold(KmerCountWorker::new, |mut worker, seq| {
                worker.add_filtered_seq(&seq, config, order, minimizer_counts, kmer_counts);
                worker
            })
            .for_each(|mut worker| {
                worker.flush(kmer_counts);
            });
        Ok(())
    })?;
    Ok(())
}

fn write_dataset_filtered_kmer_partitions(
    path: &Path,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &DatasetMinimizerCounts,
    dataset_dir: &Path,
) -> Result<()> {
    let writers = KmerPresencePartitionWriters::create(dataset_dir)?;
    for_fastx_packed_batches(path, config.k, |batch| {
        let buffers: Result<Vec<_>> = batch
            .into_par_iter()
            .fold(
                || Ok(DatasetKmerPresenceWorker::new()),
                |worker, seq| {
                    let mut worker = worker?;
                    worker.add_filtered_seq(seq, config, order, minimizer_counts, &writers)?;
                    Ok(worker)
                },
            )
            .collect();

        for mut worker in buffers? {
            worker.kmers.flush_all(&writers)?;
        }
        Ok(())
    })?;
    writers.finish()
}

fn process_dataset_kmer_presence_partitions(
    dataset_dir: &Path,
    kmer_counts: &ShardedKmerCountsU64U32,
) -> Result<()> {
    let results: Result<Vec<_>> = (0..KMER_COUNT_SHARDS)
        .into_par_iter()
        .map(|partition_idx| {
            process_dataset_kmer_presence_partition(dataset_dir, partition_idx, kmer_counts)
        })
        .collect();
    results?;
    Ok(())
}

fn process_dataset_kmer_presence_partition(
    dataset_dir: &Path,
    partition_idx: usize,
    kmer_counts: &ShardedKmerCountsU64U32,
) -> Result<()> {
    let path = dataset_kmer_presence_partition_path(dataset_dir, partition_idx);
    let mut kmers = Vec::with_capacity(estimated_partition_kmers(&path) / mem::size_of::<u64>());

    let file = File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(PHASE3_IO_BUFFER_BYTES, file);
    let mut buf = [0u8; 8];
    while let Some(encoded) = read_u64_opt_be(&mut reader, &mut buf)? {
        kmers.push(encoded);
    }

    kmers.sort_unstable();
    kmers.dedup();
    kmer_counts.add_buffer(partition_idx, &mut kmers);
    fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    Ok(())
}

fn process_superkmer_partitions<F>(
    partition_dir: &Path,
    config: CounterConfig,
    emit: &mut F,
) -> Result<()>
where
    F: FnMut(&CountedKmer) -> Result<()>,
{
    let count_started = Instant::now();
    let results: Result<Vec<_>> = (0..MINIMIZER_SHARDS)
        .into_par_iter()
        .map(|partition_idx| process_superkmer_partition(partition_dir, partition_idx, config))
        .collect();
    results?;
    log_phase("3a_kmer_counting", count_started.elapsed());

    let emit_started = Instant::now();
    for partition_idx in 0..MINIMIZER_SHARDS {
        let path = counted_partition_path(partition_dir, partition_idx);
        emit_counted_partition(&path, emit)?;
        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    }
    log_phase("3b_output_streaming", emit_started.elapsed());
    Ok(())
}
fn process_superkmer_partition(
    partition_dir: &Path,
    partition_idx: usize,
    config: CounterConfig,
) -> Result<()> {
    let superkmer_path = superkmer_partition_path(partition_dir, partition_idx);
    let counted_path = counted_partition_path(partition_dir, partition_idx);
    let mut counts = AHashMap::new();

    read_superkmer_partition(&superkmer_path, config, &mut counts)?;
    fs::remove_file(&superkmer_path)
        .with_context(|| format!("failed to remove {}", superkmer_path.display()))?;

    let file = File::create(&counted_path)
        .with_context(|| format!("failed to create {}", counted_path.display()))?;
    let mut writer = BufWriter::new(file);
    for (encoded, count) in counts {
        if count > config.threshold {
            writer.write_all(&encoded.to_be_bytes())?;
            write_u64(&mut writer, count)?;
        }
    }
    writer.flush()?;

    Ok(())
}

fn process_superkmer_partitions_to_kff_u8_fragments(
    partition_dir: &Path,
    config: CounterConfig,
) -> Result<u64> {
    let mut partition_indices = (0..MINIMIZER_SHARDS).collect::<Vec<_>>();
    partition_indices.sort_unstable_by_key(|&partition_idx| {
        std::cmp::Reverse(
            fs::metadata(superkmer_partition_path(partition_dir, partition_idx))
                .map(|metadata| metadata.len())
                .unwrap_or(0),
        )
    });

    let process_partitions = || -> Result<Vec<_>> {
        partition_indices
            .into_par_iter()
            .map(|partition_idx| {
                process_superkmer_partition_to_kff_u8(partition_dir, partition_idx, config)
            })
            .collect()
    };
    let results = if let Some(threads) = phase3_thread_count() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("failed to configure phase3 rayon thread pool")?
            .install(process_partitions)
    } else {
        process_partitions()
    };
    Ok(results?.into_iter().sum())
}

fn process_superkmer_partition_to_kff_u8(
    partition_dir: &Path,
    partition_idx: usize,
    config: CounterConfig,
) -> Result<u64> {
    let bucket_bits = phase3_bucket_bits();
    if bucket_bits > 0 {
        return process_superkmer_partition_to_kff_u8_bucketed(
            partition_dir,
            partition_idx,
            config,
            bucket_bits,
        );
    }

    let superkmer_path = superkmer_partition_path(partition_dir, partition_idx);
    let fragment_path = kff_fragment_partition_path(partition_dir, partition_idx);
    let mut kmers = Vec::with_capacity(estimated_partition_kmers(&superkmer_path));

    read_superkmer_partition_u64_kmers(&superkmer_path, config, &mut kmers)?;
    fs::remove_file(&superkmer_path)
        .with_context(|| format!("failed to remove {}", superkmer_path.display()))?;

    kmers.sort_unstable();

    let file = File::create(&fragment_path)
        .with_context(|| format!("failed to create {}", fragment_path.display()))?;
    let mut writer = BufWriter::with_capacity(PHASE3_IO_BUFFER_BYTES, file);
    let mut emitted = 0u64;
    let saturation = (config.threshold + 1).min(u8::MAX as Count) as u8;
    let mut idx = 0usize;
    while idx < kmers.len() {
        let encoded = kmers[idx];
        idx += 1;
        let mut count = 1u8;
        while idx < kmers.len() && kmers[idx] == encoded {
            if count < saturation {
                count += 1;
            }
            idx += 1;
        }

        if count as Count > config.threshold {
            write_kff_u8_record_u64(&mut writer, encoded, config.k, count)?;
            emitted += 1;
        }
    }
    writer.flush()?;

    Ok(emitted)
}

fn process_superkmer_partition_to_kff_u8_bucketed(
    partition_dir: &Path,
    partition_idx: usize,
    config: CounterConfig,
    bucket_bits: u32,
) -> Result<u64> {
    let superkmer_path = superkmer_partition_path(partition_dir, partition_idx);
    let fragment_path = kff_fragment_partition_path(partition_dir, partition_idx);
    let bucket_count = 1usize << bucket_bits;
    let per_bucket_capacity = (estimated_partition_kmers(&superkmer_path) / bucket_count).max(1024);
    let mut buckets = (0..bucket_count)
        .map(|_| Vec::with_capacity(per_bucket_capacity))
        .collect::<Vec<_>>();

    read_superkmer_partition_u64_buckets(&superkmer_path, config, bucket_bits, &mut buckets)?;
    fs::remove_file(&superkmer_path)
        .with_context(|| format!("failed to remove {}", superkmer_path.display()))?;

    let file = File::create(&fragment_path)
        .with_context(|| format!("failed to create {}", fragment_path.display()))?;
    let mut writer = BufWriter::with_capacity(PHASE3_IO_BUFFER_BYTES, file);
    let mut emitted = 0u64;
    let saturation = (config.threshold + 1).min(u8::MAX as Count) as u8;
    for bucket in &mut buckets {
        bucket.sort_unstable();
        emitted += write_sorted_kmers_u8(bucket, config, saturation, &mut writer)?;
        bucket.clear();
    }
    writer.flush()?;

    Ok(emitted)
}

fn write_sorted_kmers_u8<W: Write>(
    kmers: &[u64],
    config: CounterConfig,
    saturation: u8,
    writer: &mut W,
) -> Result<u64> {
    let mut emitted = 0u64;
    let mut idx = 0usize;
    while idx < kmers.len() {
        let encoded = kmers[idx];
        idx += 1;
        let mut count = 1u8;
        while idx < kmers.len() && kmers[idx] == encoded {
            if count < saturation {
                count += 1;
            }
            idx += 1;
        }

        if count as Count > config.threshold {
            write_kff_u8_record_u64(writer, encoded, config.k, count)?;
            emitted += 1;
        }
    }
    Ok(emitted)
}

fn emit_kff_u8_fragments(
    partition_dir: &Path,
    config: CounterConfig,
    record_count: u64,
    output_path: &Path,
) -> Result<()> {
    let mut writer = create_output_writer(output_path)?;
    write_kff_header_with_count_size(&mut writer, config, record_count, false, 1)?;

    for partition_idx in 0..MINIMIZER_SHARDS {
        let path = kff_fragment_partition_path(partition_dir, partition_idx);
        let file =
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut reader = BufReader::with_capacity(PHASE3_IO_BUFFER_BYTES, file);
        io::copy(&mut reader, &mut writer)
            .with_context(|| format!("failed to append {}", path.display()))?;
        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    }

    writer.write_all(b"KFF")?;
    writer.finish()
}

fn write_simplitig_fasta_u32_counts(
    output_path: &Path,
    config: CounterConfig,
    counts: &KmerCountsU64U32,
) -> Result<()> {
    let mut kmers = Vec::new();
    for shard in counts.shards.iter() {
        for (encoded, count) in shard.iter() {
            if count as Count > config.threshold {
                kmers.push(CountedKmer {
                    encoded: encoded as EncodedKmer,
                    count: count as Count,
                });
            }
        }
    }
    kmers.sort_unstable_by_key(|record| record.encoded);
    write_simplitig_fasta_path(output_path, config.k, &kmers)
}

fn write_kff_u32_counts(
    output_path: &Path,
    config: CounterConfig,
    counts: &KmerCountsU64U32,
) -> Result<()> {
    let record_count = counts.above_threshold_count(config.threshold);
    let mut writer = create_output_writer(output_path)?;
    write_kff_header_with_count_size(&mut writer, config, record_count, false, 4)?;

    for shard in counts.shards.iter() {
        for (encoded, count) in shard.iter() {
            if count as Count > config.threshold {
                write_kff_u32_record_u64(&mut writer, encoded, config.k, count)?;
            }
        }
    }

    writer.write_all(b"KFF")?;
    writer.finish()
}

fn write_kff_u8_counts(
    output_path: &Path,
    config: CounterConfig,
    counts: &KmerCountsU64,
) -> Result<()> {
    let record_count = counts.above_threshold_count(config.threshold);
    let mut writer = create_output_writer(output_path)?;
    write_kff_header_with_count_size(&mut writer, config, record_count, false, 1)?;

    for shard in counts.shards.iter() {
        for (encoded, count) in shard.iter() {
            if count as Count > config.threshold {
                write_kff_u8_record_u64(&mut writer, encoded, config.k, count)?;
            }
        }
    }

    writer.write_all(b"KFF")?;
    writer.finish()
}

fn emit_counted_partition<F>(path: &Path, emit: &mut F) -> Result<()>
where
    F: FnMut(&CountedKmer) -> Result<()>,
{
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut encoded_buf = [0u8; 16];
    let mut count_buf = [0u8; 8];

    while read_encoded(&mut reader, &mut encoded_buf)? {
        reader
            .read_exact(&mut count_buf)
            .with_context(|| format!("truncated counted k-mer record in {}", path.display()))?;
        let record = CountedKmer {
            encoded: u128::from_be_bytes(encoded_buf),
            count: u64::from_be_bytes(count_buf),
        };
        emit(&record)?;
    }

    Ok(())
}

fn read_encoded<R: Read>(reader: &mut R, encoded_buf: &mut [u8; 16]) -> Result<bool> {
    let mut first = [0u8; 1];
    let bytes = reader.read(&mut first)?;
    if bytes == 0 {
        return Ok(false);
    }

    encoded_buf[0] = first[0];
    reader.read_exact(&mut encoded_buf[1..])?;
    Ok(true)
}

fn read_superkmer_partition(
    path: &Path,
    config: CounterConfig,
    counts: &mut AHashMap<EncodedKmer, Count>,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut len_buf = [0u8; 4];
    let mut seq = Vec::new();

    while let Some(len) = read_record_len(&mut reader, &mut len_buf)? {
        seq.resize(packed_dna_bytes(len), 0);
        reader
            .read_exact(&mut seq)
            .with_context(|| format!("truncated super-kmer record in {}", path.display()))?;
        add_packed_superkmer_kmer_counts(&seq, len, config, counts);
    }

    Ok(())
}

fn read_superkmer_partition_u64_kmers(
    path: &Path,
    config: CounterConfig,
    kmers: &mut Vec<u64>,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(PHASE3_IO_BUFFER_BYTES, file);
    let mut len_buf = [0u8; 4];
    let mut seq = Vec::new();

    while let Some(len) = read_record_len(&mut reader, &mut len_buf)? {
        seq.resize(packed_dna_bytes(len), 0);
        reader
            .read_exact(&mut seq)
            .with_context(|| format!("truncated super-kmer record in {}", path.display()))?;
        append_packed_superkmer_kmers_u64(&seq, len, config, kmers);
    }

    Ok(())
}

fn read_superkmer_partition_u64_buckets(
    path: &Path,
    config: CounterConfig,
    bucket_bits: u32,
    buckets: &mut [Vec<u64>],
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(PHASE3_IO_BUFFER_BYTES, file);
    let mut len_buf = [0u8; 4];
    let mut seq = Vec::new();

    while let Some(len) = read_record_len(&mut reader, &mut len_buf)? {
        seq.resize(packed_dna_bytes(len), 0);
        reader
            .read_exact(&mut seq)
            .with_context(|| format!("truncated super-kmer record in {}", path.display()))?;
        append_packed_superkmer_kmers_u64_buckets(&seq, len, config, bucket_bits, buckets);
    }

    Ok(())
}

fn read_record_len<R: Read>(reader: &mut R, len_buf: &mut [u8; 4]) -> Result<Option<usize>> {
    let mut first = [0u8; 1];
    let bytes = reader.read(&mut first)?;
    if bytes == 0 {
        return Ok(None);
    }

    if first[0] != u8::MAX {
        return Ok(Some(first[0] as usize));
    }

    reader.read_exact(len_buf)?;
    Ok(Some(u32::from_be_bytes(*len_buf) as usize))
}

fn read_u32_opt<R: Read>(reader: &mut R, buf: &mut [u8; 4]) -> Result<Option<usize>> {
    let mut first = [0u8; 1];
    let bytes = reader.read(&mut first)?;
    if bytes == 0 {
        return Ok(None);
    }

    buf[0] = first[0];
    reader.read_exact(&mut buf[1..])?;
    Ok(Some(u32::from_be_bytes(*buf) as usize))
}

fn read_u64_opt_be<R: Read>(reader: &mut R, buf: &mut [u8; 8]) -> Result<Option<u64>> {
    let mut first = [0u8; 1];
    let bytes = reader.read(&mut first)?;
    if bytes == 0 {
        return Ok(None);
    }

    buf[0] = first[0];
    reader.read_exact(&mut buf[1..])?;
    Ok(Some(u64::from_be_bytes(*buf)))
}

fn remove_packed_read_caches(partition_dir: &Path, input_count: usize) -> Result<()> {
    for input_idx in 0..input_count {
        let path = packed_read_cache_path(partition_dir, input_idx);
        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(())
}

fn create_partition_dir() -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_nanos();

    for attempt in 0..1000 {
        let path =
            env::temp_dir().join(format!("mc-partitions-{}-{stamp}-{attempt}", process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("failed to create {}", path.display()));
            }
        }
    }

    bail!(
        "failed to create unique partition directory in {}",
        env::temp_dir().display()
    )
}

fn superkmer_partition_path(partition_dir: &Path, partition_idx: usize) -> PathBuf {
    partition_dir.join(format!("partition_{partition_idx:04}.skm"))
}

fn counted_partition_path(partition_dir: &Path, partition_idx: usize) -> PathBuf {
    partition_dir.join(format!("partition_{partition_idx:04}.cnt"))
}

fn kff_fragment_partition_path(partition_dir: &Path, partition_idx: usize) -> PathBuf {
    partition_dir.join(format!("partition_{partition_idx:04}.kfffrag"))
}

fn dataset_presence_partition_dir(partition_dir: &Path, dataset_idx: usize) -> PathBuf {
    partition_dir.join(format!("dataset_{dataset_idx:04}"))
}

fn dataset_kmer_presence_partition_path(dataset_dir: &Path, partition_idx: usize) -> PathBuf {
    dataset_dir.join(format!("kmer_presence_{partition_idx:04}.bin"))
}

fn saturating_compact_delta(delta: usize) -> CompactCount {
    delta.min(CompactCount::MAX as usize) as CompactCount
}

fn estimate_unique_minimizer_hashes(inputs: &[PathBuf]) -> usize {
    let bytes_per_hash = if should_cache_packed_reads(inputs) {
        ESTIMATED_COMPRESSED_BYTES_PER_UNIQUE_MINIMIZER_HASH
    } else {
        ESTIMATED_BYTES_PER_UNIQUE_MINIMIZER_HASH
    };
    let bytes = inputs
        .iter()
        .filter_map(|path| fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum::<u64>();
    (bytes / bytes_per_hash)
        .max(MINIMIZER_SHARDS as u64)
        .min(usize::MAX as u64) as usize
}

fn estimate_unique_kmers_for_ram_count(inputs: &[PathBuf]) -> usize {
    let bytes = input_file_bytes(inputs);
    (bytes / ESTIMATED_COMPRESSED_BYTES_PER_UNIQUE_KMER)
        .max(KMER_COUNT_SHARDS as u64)
        .min(usize::MAX as u64) as usize
}

fn input_file_bytes(inputs: &[PathBuf]) -> u64 {
    inputs
        .iter()
        .filter_map(|path| fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum::<u64>()
}

fn estimated_partition_kmers(path: &Path) -> usize {
    fs::metadata(path)
        .map(|metadata| metadata.len().min(usize::MAX as u64) as usize)
        .unwrap_or(1024)
        .max(1024)
}

fn phase3_bucket_bits() -> u32 {
    env::var("MC_PHASE3_BUCKET_BITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&bits| bits <= 10)
        .unwrap_or(DEFAULT_PHASE3_BUCKET_BITS)
}

fn phase3_thread_count() -> Option<usize> {
    env::var("MC_PHASE3_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
}

fn directory_file_bytes(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in
            fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let entry = entry.with_context(|| format!("failed to read {}", dir.display()))?;
            let metadata = entry
                .metadata()
                .with_context(|| format!("failed to stat {}", entry.path().display()))?;
            if metadata.is_dir() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }

    Ok(total)
}

fn minimizer_shard(hash: MinimizerHash) -> usize {
    (hash >> (MinimizerHash::BITS - MINIMIZER_SHARD_BITS)) as usize
}

fn kmer_count_shard(encoded: u64) -> usize {
    (spread_hash_u64(encoded) >> (u64::BITS - KMER_COUNT_SHARD_BITS)) as usize
}

fn minimizer_hash_u64(value: u64) -> MinimizerHash {
    let folded = value ^ value.rotate_right(17) ^ (value >> 32);
    (folded as MinimizerHash).wrapping_mul(0x9E37_79B1)
}

fn spread_hash_u32(mut value: u32) -> MinimizerHash {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn minimizer_hash_u128(value: u128) -> MinimizerHash {
    minimizer_hash_u64(value as u64 ^ ((value >> 64) as u64).rotate_left(23))
}

fn spread_hash_u64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^ (value >> 33)
}

fn add_count<K>(counts: &mut AHashMap<K, Count>, key: K, delta: Count)
where
    K: Eq + std::hash::Hash,
{
    let entry = counts.entry(key).or_insert(0);
    *entry = entry.saturating_add(delta);
}

#[cfg(test)]
const BASE_BITS: [u8; 256] = build_base_bits();

#[cfg(test)]
const fn build_base_bits() -> [u8; 256] {
    let mut lut = [0u8; 256];
    lut[b'A' as usize] = 0;
    lut[b'a' as usize] = 0;
    lut[b'C' as usize] = 1;
    lut[b'c' as usize] = 1;
    lut[b'G' as usize] = 3;
    lut[b'g' as usize] = 3;
    lut[b'T' as usize] = 2;
    lut[b't' as usize] = 2;
    lut
}

#[inline(always)]
fn base_bits(base: u8) -> u8 {
    match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => unreachable!("helicase split_non_actg should only yield A/C/G/T chunks"),
    }
}

fn kmer_mask(k: usize) -> u128 {
    if k == 64 {
        u128::MAX
    } else {
        (1u128 << (2 * k)) - 1
    }
}

fn kmer_mask_u64(k: usize) -> u64 {
    if k == 32 {
        u64::MAX
    } else {
        (1u64 << (2 * k)) - 1
    }
}

pub fn decode_kmer(encoded: EncodedKmer, k: usize) -> Vec<u8> {
    let mut seq = vec![0u8; k];
    for (idx, base) in seq.iter_mut().enumerate() {
        let shift = 2 * (k - idx - 1);
        *base = match (encoded >> shift) & 0b11 {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            3 => b'T',
            _ => unreachable!(),
        };
    }
    seq
}

fn write_kff_values<W: Write>(writer: &mut W, vars: &[(&str, u64)]) -> Result<()> {
    writer.write_all(b"v")?;
    write_u64(writer, vars.len() as u64)?;
    for (name, value) in vars {
        writer.write_all(name.as_bytes())?;
        writer.write_all(&[0])?;
        write_u64(writer, *value)?;
    }
    Ok(())
}

fn write_packed_dna<W: Write>(writer: &mut W, encoded: EncodedKmer, len: usize) -> Result<()> {
    let bytes = packed_dna_bytes(len);
    let packed = encoded.to_be_bytes();
    writer.write_all(&packed[packed.len() - bytes..])?;
    Ok(())
}

fn write_kff_u8_record_u64<W: Write>(
    writer: &mut W,
    encoded: u64,
    len: usize,
    count: u8,
) -> Result<()> {
    let bytes = packed_dna_bytes(len);
    let packed = encoded.to_be_bytes();
    let mut record = [0u8; 9];
    record[..bytes].copy_from_slice(&packed[packed.len() - bytes..]);
    record[bytes] = count;
    writer.write_all(&record[..bytes + 1])?;
    Ok(())
}

fn write_kff_u32_record_u64<W: Write>(
    writer: &mut W,
    encoded: u64,
    len: usize,
    count: u32,
) -> Result<()> {
    let bytes = packed_dna_bytes(len);
    let packed = encoded.to_be_bytes();
    let mut record = [0u8; 12];
    record[..bytes].copy_from_slice(&packed[packed.len() - bytes..]);
    record[bytes..bytes + 4].copy_from_slice(&count.to_be_bytes());
    writer.write_all(&record[..bytes + 4])?;
    Ok(())
}

fn packed_dna_bytes(len: usize) -> usize {
    len.div_ceil(4)
}

fn packed_read_cache_path(partition_dir: &Path, input_idx: usize) -> PathBuf {
    partition_dir.join(format!("input_{input_idx:04}.pread"))
}

fn should_cache_packed_reads(inputs: &[PathBuf]) -> bool {
    inputs.iter().any(|path| {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "gz" | "gzip" | "xz" | "zst" | "zstd"
                )
            })
            .unwrap_or(false)
    })
}

fn should_use_ram_kmer_count(inputs: &[PathBuf]) -> bool {
    should_cache_packed_reads(inputs) && input_file_bytes(inputs) <= RAM_KMER_COUNT_MAX_INPUT_BYTES
}

#[cfg(test)]
fn append_packed_dna_seq(out: &mut Vec<u8>, seq: &[u8]) {
    let start = out.len();
    out.resize(start + packed_dna_bytes(seq.len()), 0);

    let mut packed_idx = start;
    let mut chunks = seq.chunks_exact(4);
    for chunk in &mut chunks {
        out[packed_idx] = BASE_BITS[chunk[0] as usize]
            | (BASE_BITS[chunk[1] as usize] << 2)
            | (BASE_BITS[chunk[2] as usize] << 4)
            | (BASE_BITS[chunk[3] as usize] << 6);
        packed_idx += 1;
    }

    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut byte = 0u8;
        for (idx, &base) in rem.iter().enumerate() {
            byte |= BASE_BITS[base as usize] << (2 * idx);
        }
        out[packed_idx] = byte;
    }
}

fn append_packed_dna_range(out: &mut Vec<u8>, packed: &[u8], start: usize, len: usize) {
    if len == 0 {
        return;
    }

    let start_byte = start / 4;
    let bytes = packed_dna_bytes(len);
    if start % 4 == 0 {
        out.extend_from_slice(&packed[start_byte..start_byte + bytes]);
        return;
    }

    let shift = 2 * (start % 4);
    out.reserve(bytes);
    for byte_idx in 0..bytes {
        let src_idx = start_byte + byte_idx;
        let word =
            packed[src_idx] as u16 | ((packed.get(src_idx + 1).copied().unwrap_or(0) as u16) << 8);
        out.push(((word >> shift) & 0xff) as u8);
    }
}

fn packed_base_bits(packed: &[u8], idx: usize) -> u8 {
    let byte = packed[idx / 4];
    let shift = 2 * (idx % 4);
    let bits = (byte >> shift) & 0b11;
    bits ^ (bits >> 1)
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_be_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_be_bytes())?;
    Ok(())
}

fn append_record_len(out: &mut Vec<u8>, len: u32) {
    if len < u8::MAX as u32 {
        out.push(len as u8);
    } else {
        out.push(u8::MAX);
        out.extend_from_slice(&len.to_be_bytes());
    }
}

fn for_fastx_batches<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(&[Vec<u8>]) -> Result<()>,
{
    if is_xz_path(path) {
        return with_xz_decompressed_path(path, |plain_path| {
            for_fastx_batches(plain_path, min_len, consume)
        });
    }

    const CONFIG: Config = ParserOptions::default()
        .dna_string()
        .ignore_headers()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open FASTA/FASTQ input {}", path.display()))?;
    let mut batch = Vec::new();
    let mut bases = 0usize;

    while parser.next().is_some() {
        let seq = parser.get_dna_string();
        if seq.len() < min_len {
            continue;
        }
        bases += seq.len();
        batch.push(seq.to_vec());

        if bases >= BATCH_BASES {
            consume(&batch)?;
            batch.clear();
            bases = 0;
        }
    }

    if !batch.is_empty() {
        consume(&batch)?;
    }

    Ok(())
}

fn for_fastx_packed_batches<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(Vec<PackedSeqVec>) -> Result<()>,
{
    if is_xz_path(path) {
        return with_xz_decompressed_path(path, |plain_path| {
            for_fastx_packed_batches(plain_path, min_len, consume)
        });
    }

    const CONFIG: Config = ParserOptions::default()
        .dna_packed()
        .ignore_headers()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open FASTA/FASTQ input {}", path.display()))?;
    let mut batch = Vec::new();
    let mut bases = 0usize;

    while parser.next().is_some() {
        let seq = parser.get_packed_seq().to_vec();
        let len = seq.as_slice().len();
        if len < min_len {
            continue;
        }
        bases += len;
        batch.push(seq);

        if bases >= BATCH_BASES {
            consume(mem::take(&mut batch))?;
            bases = 0;
        }
    }

    if !batch.is_empty() {
        consume(batch)?;
    }

    Ok(())
}

fn for_fastx_packed_batches_cached<F>(
    path: &Path,
    min_len: usize,
    cache_path: &Path,
    mut consume: F,
) -> Result<()>
where
    F: FnMut(Vec<PackedSeqVec>) -> Result<()>,
{
    if is_xz_path(path) {
        return with_xz_decompressed_path(path, |plain_path| {
            for_fastx_packed_batches_cached(plain_path, min_len, cache_path, consume)
        });
    }

    const CONFIG: Config = ParserOptions::default()
        .dna_packed()
        .ignore_headers()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open FASTA/FASTQ input {}", path.display()))?;
    let cache = File::create(cache_path)
        .with_context(|| format!("failed to create {}", cache_path.display()))?;
    let mut cache = BufWriter::with_capacity(16 * 1024 * 1024, cache);
    cache.write_all(PACKED_READ_CACHE_MAGIC)?;

    let mut batch = Vec::new();
    let mut bases = 0usize;

    while parser.next().is_some() {
        let seq = parser.get_packed_seq().to_vec();
        let len = seq.as_slice().len();
        if len < min_len {
            continue;
        }
        bases += len;
        batch.push(seq);

        if bases >= BATCH_BASES {
            write_packed_read_cache_batch(&mut cache, &batch)?;
            consume(mem::take(&mut batch))?;
            bases = 0;
        }
    }

    if !batch.is_empty() {
        write_packed_read_cache_batch(&mut cache, &batch)?;
        consume(batch)?;
    }

    cache.flush()?;
    Ok(())
}

fn write_packed_read_cache_batch<W: Write>(writer: &mut W, batch: &[PackedSeqVec]) -> Result<()> {
    for seq in batch {
        let len = seq.len();
        ensure!(
            len <= u32::MAX as usize,
            "read is too large for packed-read cache format"
        );
        write_u32(writer, len as u32)?;
        writer.write_all(&seq.clone().into_raw())?;
    }
    Ok(())
}

fn for_packed_read_cache_batches<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(Vec<PackedSeqVec>) -> Result<()>,
{
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut magic = [0u8; PACKED_READ_CACHE_MAGIC.len()];
    reader
        .read_exact(&mut magic)
        .with_context(|| format!("truncated packed-read cache header in {}", path.display()))?;
    ensure!(
        magic == *PACKED_READ_CACHE_MAGIC,
        "invalid packed-read cache header in {}",
        path.display()
    );

    let mut batch = Vec::new();
    let mut bases = 0usize;
    let mut len_buf = [0u8; 4];

    while let Some(len) = read_u32_opt(&mut reader, &mut len_buf)? {
        let bytes = packed_dna_bytes(len);
        let mut raw = vec![0; bytes];
        reader
            .read_exact(&mut raw)
            .with_context(|| format!("truncated packed-read cache record in {}", path.display()))?;
        if len < min_len {
            continue;
        }

        bases += len;
        batch.push(PackedSeqVec::from_raw_parts(raw, len));
        if bases >= BATCH_BASES {
            consume(mem::take(&mut batch))?;
            bases = 0;
        }
    }

    if !batch.is_empty() {
        consume(batch)?;
    }

    Ok(())
}

fn add_minimizer_counts(
    seq: &[u8],
    config: CounterConfig,
    order: MinimizerOrder,
    counts: &ShardedMinimizerCounts,
    worker: &mut MinimizerCountWorker,
) {
    add_minimizer_counts_on_seq(AsciiSeq(seq), config, order, counts, worker);
}

fn add_minimizer_counts_on_seq<'s, S>(
    seq: S,
    config: CounterConfig,
    order: MinimizerOrder,
    counts: &ShardedMinimizerCounts,
    worker: &mut MinimizerCountWorker,
) where
    S: Seq<'s>,
{
    for_each_minimizer_run_on_seq(
        seq,
        config,
        order,
        &mut worker.minimizer_runs,
        |hash, start, end| {
            let delta = saturating_compact_delta(end - start);
            worker.updates.add(hash, delta, counts);
        },
    );
}

fn add_filtered_superkmers_packed(
    seq: PackedSeqVec,
    config: CounterConfig,
    order: MinimizerOrder,
    minimizer_counts: &MinimizerCounts,
    writers: &SuperkmerPartitionWriters,
    buffers: &mut SuperkmerWorkerBuffers,
) -> Result<()> {
    let seq_len = seq.len();
    let windows = seq_len.saturating_sub(config.k) + 1;
    if windows == 0 {
        return Ok(());
    }

    buffers.filtered_runs.clear();
    for_each_minimizer_run_on_seq(
        seq.as_slice(),
        config,
        order,
        &mut buffers.minimizer_runs,
        |hash, start, end| {
            if !minimizer_counts.is_above_threshold(hash, config.threshold) {
                return;
            }

            let superkmer_end = end + config.k - 1;
            if superkmer_end <= seq_len {
                buffers.filtered_runs.push((hash, start, superkmer_end));
            }
        },
    );

    let raw = seq.into_raw();
    for &(hash, start, end) in &buffers.filtered_runs {
        buffers
            .superkmers
            .add_packed_range(hash, &raw, start, end - start, writers)?;
    }

    Ok(())
}

fn add_all_superkmers_packed(
    seq: PackedSeqVec,
    config: CounterConfig,
    order: MinimizerOrder,
    writers: &SuperkmerPartitionWriters,
    buffers: &mut SuperkmerWorkerBuffers,
) -> Result<()> {
    let seq_len = seq.len();
    let windows = seq_len.saturating_sub(config.k) + 1;
    if windows == 0 {
        return Ok(());
    }

    buffers.filtered_runs.clear();
    for_each_minimizer_run_on_seq(
        seq.as_slice(),
        config,
        order,
        &mut buffers.minimizer_runs,
        |hash, start, end| {
            let superkmer_end = end + config.k - 1;
            if superkmer_end <= seq_len {
                buffers.filtered_runs.push((hash, start, superkmer_end));
            }
        },
    );

    let raw = seq.into_raw();
    for &(hash, start, end) in &buffers.filtered_runs {
        buffers
            .superkmers
            .add_packed_range(hash, &raw, start, end - start, writers)?;
    }

    Ok(())
}

#[cfg(test)]
fn add_superkmer_kmer_counts(
    seq: &[u8],
    config: CounterConfig,
    counts: &mut AHashMap<EncodedKmer, Count>,
) {
    if seq.len() < config.k {
        return;
    }

    let mut fwd = 0u128;
    let mut rev = 0u128;
    let mask = kmer_mask(config.k);
    let high_shift = 2 * (config.k - 1);

    for (idx, &base) in seq.iter().enumerate() {
        let bits = base_bits(base) as u128;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);

        if idx + 1 < config.k {
            continue;
        }
        add_count(counts, fwd.min(rev), 1);
    }
}

fn add_packed_superkmer_kmer_counts(
    packed: &[u8],
    len: usize,
    config: CounterConfig,
    counts: &mut AHashMap<EncodedKmer, Count>,
) {
    if len < config.k {
        return;
    }

    let mut fwd = 0u128;
    let mut rev = 0u128;
    let mask = kmer_mask(config.k);
    let high_shift = 2 * (config.k - 1);

    for idx in 0..len {
        let bits = packed_base_bits(packed, idx) as u128;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);

        if idx + 1 < config.k {
            continue;
        }
        add_count(counts, fwd.min(rev), 1);
    }
}

fn append_packed_superkmer_kmers_u64(
    packed: &[u8],
    len: usize,
    config: CounterConfig,
    kmers: &mut Vec<u64>,
) {
    if len < config.k {
        return;
    }

    let mut fwd = 0u64;
    let mut rev = 0u64;
    let mask = kmer_mask_u64(config.k);
    let high_shift = 2 * (config.k - 1);

    for idx in 0..len {
        let bits = packed_base_bits(packed, idx) as u64;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);

        if idx + 1 >= config.k {
            kmers.push(fwd.min(rev));
        }
    }
}

fn append_packed_superkmer_kmers_u64_buckets(
    packed: &[u8],
    len: usize,
    config: CounterConfig,
    bucket_bits: u32,
    buckets: &mut [Vec<u64>],
) {
    if len < config.k {
        return;
    }

    let mut fwd = 0u64;
    let mut rev = 0u64;
    let mask = kmer_mask_u64(config.k);
    let high_shift = 2 * (config.k - 1);
    let bucket_mask = (1usize << bucket_bits) - 1;

    for idx in 0..len {
        let bits = packed_base_bits(packed, idx) as u64;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);

        if idx + 1 >= config.k {
            let encoded = fwd.min(rev);
            buckets[encoded as usize & bucket_mask].push(encoded);
        }
    }
}

fn canonical_kmers_u64(packed: &[u8], len: usize, config: CounterConfig, out: &mut Vec<u64>) {
    out.clear();
    if len < config.k {
        return;
    }

    out.reserve(len - config.k + 1);
    let mut fwd = 0u64;
    let mut rev = 0u64;
    let mask = kmer_mask_u64(config.k);
    let high_shift = 2 * (config.k - 1);

    for idx in 0..len {
        let bits = packed_base_bits(packed, idx) as u64;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);

        if idx + 1 >= config.k {
            out.push(fwd.min(rev));
        }
    }
}

fn for_each_minimizer_run_on_seq<'s, S, F>(
    seq: S,
    config: CounterConfig,
    order: MinimizerOrder,
    buffers: &mut MinimizerRunBuffers,
    mut visit: F,
) where
    S: Seq<'s>,
    F: FnMut(MinimizerHash, usize, usize),
{
    if seq.len() < config.k {
        return;
    }
    let window = config.k - config.minimizer + 1;
    buffers.minimizer_positions.clear();
    buffers.super_kmer_starts.clear();
    let total_windows = seq.len() - config.k + 1;

    match order {
        MinimizerOrder::SimdValueHash => {
            let output = simd_minimizers::canonical_minimizers(config.minimizer, window)
                .super_kmers(&mut buffers.super_kmer_starts)
                .run(seq, &mut buffers.minimizer_positions);
            visit_value_output(
                output,
                config,
                &buffers.super_kmer_starts,
                total_windows,
                &mut visit,
            );
        }
        MinimizerOrder::SimdDirectHash => {
            buffers.minimizer_hashes.clear();
            simd_minimizers::canonical_minimizers(config.minimizer, window)
                .super_kmers(&mut buffers.super_kmer_starts)
                .run_hashes(seq, &mut buffers.minimizer_hashes);
            visit_direct_hash_runs(
                &buffers.minimizer_hashes,
                &buffers.super_kmer_starts,
                total_windows,
                &mut visit,
            );
        }
        MinimizerOrder::AntiLex => {
            let hasher = AntiLexHasher::<true>::new(config.minimizer);
            let output = simd_minimizers::canonical_minimizers(config.minimizer, window)
                .hasher(&hasher)
                .super_kmers(&mut buffers.super_kmer_starts)
                .run(seq, &mut buffers.minimizer_positions);
            visit_hash_output(
                output,
                &hasher,
                &buffers.super_kmer_starts,
                total_windows,
                &mut visit,
            );
        }
    }
}

fn visit_direct_hash_runs<F>(
    minimizer_hashes: &[MinimizerHash],
    super_kmer_starts: &[u32],
    total_windows: usize,
    visit: &mut F,
) where
    F: FnMut(MinimizerHash, usize, usize),
{
    visit_hash_runs(
        minimizer_hashes.iter().copied().map(spread_hash_u32),
        super_kmer_starts,
        total_windows,
        visit,
    );
}

fn visit_hash_runs<F>(
    minimizer_hashes: impl IntoIterator<Item = MinimizerHash>,
    super_kmer_starts: &[u32],
    total_windows: usize,
    visit: &mut F,
) where
    F: FnMut(MinimizerHash, usize, usize),
{
    for (idx, minimizer_hash) in minimizer_hashes.into_iter().enumerate() {
        let start = super_kmer_starts[idx] as usize;
        let end = super_kmer_starts
            .get(idx + 1)
            .map(|&value| value as usize)
            .unwrap_or(total_windows);

        if start < end {
            visit(minimizer_hash, start, end.min(total_windows));
        }
    }
}

fn visit_value_output<'s, F, SEQ>(
    output: simd_minimizers::Output<'_, true, SEQ>,
    config: CounterConfig,
    super_kmer_starts: &[u32],
    total_windows: usize,
    visit: &mut F,
) where
    F: FnMut(MinimizerHash, usize, usize),
    SEQ: simd_minimizers::packed_seq::Seq<'s>,
{
    if config.minimizer <= 32 {
        for (idx, minimizer) in output.values_u64().enumerate() {
            let start = super_kmer_starts[idx] as usize;
            let end = super_kmer_starts
                .get(idx + 1)
                .map(|&value| value as usize)
                .unwrap_or(total_windows);

            if start < end {
                visit(minimizer_hash_u64(minimizer), start, end.min(total_windows));
            }
        }
    } else {
        for (idx, minimizer) in output.values_u128().enumerate() {
            let start = super_kmer_starts[idx] as usize;
            let end = super_kmer_starts
                .get(idx + 1)
                .map(|&value| value as usize)
                .unwrap_or(total_windows);

            if start < end {
                visit(
                    minimizer_hash_u128(minimizer),
                    start,
                    end.min(total_windows),
                );
            }
        }
    }
}

fn visit_hash_output<'s, F, H, SEQ>(
    output: simd_minimizers::Output<'_, true, SEQ>,
    hasher: &H,
    super_kmer_starts: &[u32],
    total_windows: usize,
    visit: &mut F,
) where
    F: FnMut(MinimizerHash, usize, usize),
    H: simd_minimizers::seq_hash::KmerHasher,
    SEQ: simd_minimizers::packed_seq::Seq<'s>,
{
    for (idx, minimizer_hash) in output.hashes_u32(hasher).enumerate() {
        let start = super_kmer_starts[idx] as usize;
        let end = super_kmer_starts
            .get(idx + 1)
            .map(|&value| value as usize)
            .unwrap_or(total_windows);

        if start < end {
            visit(minimizer_hash, start, end.min(total_windows));
        }
    }
}

#[derive(Default)]
struct MinimizerRunBuffers {
    minimizer_positions: Vec<u32>,
    minimizer_hashes: Vec<u32>,
    super_kmer_starts: Vec<u32>,
}

struct MinimizerShardBuffers {
    buffers: Box<[Vec<(MinimizerHash, CompactCount)>]>,
}

impl MinimizerShardBuffers {
    fn new() -> Self {
        let buffers = (0..MINIMIZER_SHARDS).map(|_| Vec::new()).collect();
        Self { buffers }
    }

    fn add(&mut self, hash: MinimizerHash, delta: CompactCount, counts: &ShardedMinimizerCounts) {
        let shard_idx = minimizer_shard(hash);
        let buffer = &mut self.buffers[shard_idx];
        buffer.push((hash, delta));
        if buffer.len() >= MINIMIZER_BUFFER_FLUSH_LEN {
            counts.add_buffer(shard_idx, buffer);
        }
    }

    fn flush_all(&mut self, counts: &ShardedMinimizerCounts) {
        for (shard_idx, buffer) in self.buffers.iter_mut().enumerate() {
            counts.add_buffer(shard_idx, buffer);
        }
    }
}

struct DatasetMinimizerShardBuffers {
    buffers: Box<[Vec<MinimizerHash>]>,
}

impl DatasetMinimizerShardBuffers {
    fn new() -> Self {
        let buffers = (0..MINIMIZER_SHARDS).map(|_| Vec::new()).collect();
        Self { buffers }
    }

    fn add(
        &mut self,
        hash: MinimizerHash,
        counts: &ShardedDatasetMinimizerCounts,
        accept_new: bool,
    ) {
        let shard_idx = minimizer_shard(hash);
        let buffer = &mut self.buffers[shard_idx];
        buffer.push(hash);
        if buffer.len() >= MINIMIZER_BUFFER_FLUSH_LEN {
            counts.add_buffer(shard_idx, buffer, accept_new);
        }
    }

    fn flush_all(&mut self, counts: &ShardedDatasetMinimizerCounts, accept_new: bool) {
        for (shard_idx, buffer) in self.buffers.iter_mut().enumerate() {
            counts.add_buffer(shard_idx, buffer, accept_new);
        }
    }
}

struct MinimizerCountWorker {
    updates: MinimizerShardBuffers,
    minimizer_runs: MinimizerRunBuffers,
}

impl MinimizerCountWorker {
    fn new() -> Self {
        Self {
            updates: MinimizerShardBuffers::new(),
            minimizer_runs: MinimizerRunBuffers::default(),
        }
    }

    fn add_seq(
        &mut self,
        seq: &[u8],
        config: CounterConfig,
        order: MinimizerOrder,
        counts: &ShardedMinimizerCounts,
    ) {
        add_minimizer_counts(seq, config, order, counts, self);
    }

    fn add_packed_seq(
        &mut self,
        seq: &PackedSeqVec,
        config: CounterConfig,
        order: MinimizerOrder,
        counts: &ShardedMinimizerCounts,
    ) {
        add_minimizer_counts_on_seq(seq.as_slice(), config, order, counts, self);
    }

    fn flush(&mut self, counts: &ShardedMinimizerCounts) {
        self.updates.flush_all(counts);
    }
}

struct DatasetMinimizerPresenceWorker {
    updates: DatasetMinimizerShardBuffers,
    minimizer_runs: MinimizerRunBuffers,
}

impl DatasetMinimizerPresenceWorker {
    fn new() -> Self {
        Self {
            updates: DatasetMinimizerShardBuffers::new(),
            minimizer_runs: MinimizerRunBuffers::default(),
        }
    }

    fn add_packed_seq(
        &mut self,
        seq: &PackedSeqVec,
        config: CounterConfig,
        order: MinimizerOrder,
        counts: &ShardedDatasetMinimizerCounts,
        accept_new: bool,
    ) {
        for_each_minimizer_run_on_seq(
            seq.as_slice(),
            config,
            order,
            &mut self.minimizer_runs,
            |hash, _, _| {
                self.updates.add(hash, counts, accept_new);
            },
        );
    }

    fn flush(&mut self, counts: &ShardedDatasetMinimizerCounts, accept_new: bool) {
        self.updates.flush_all(counts, accept_new);
    }
}

struct KmerShardBuffers {
    buffers: Box<[Vec<u64>]>,
}

impl KmerShardBuffers {
    fn new() -> Self {
        let buffers = (0..KMER_COUNT_SHARDS).map(|_| Vec::new()).collect();
        Self { buffers }
    }

    fn add(&mut self, encoded: u64, counts: &ShardedKmerCountsU64) {
        let shard_idx = kmer_count_shard(encoded);
        let buffer = &mut self.buffers[shard_idx];
        buffer.push(encoded);
        if buffer.len() >= KMER_COUNT_BUFFER_FLUSH_LEN {
            counts.add_buffer(shard_idx, buffer);
        }
    }

    fn flush_all(&mut self, counts: &ShardedKmerCountsU64) {
        for (shard_idx, buffer) in self.buffers.iter_mut().enumerate() {
            counts.add_buffer(shard_idx, buffer);
        }
    }
}

struct KmerCountWorker {
    updates: KmerShardBuffers,
    minimizer_runs: MinimizerRunBuffers,
    canonical_kmers: Vec<u64>,
}

impl KmerCountWorker {
    fn new() -> Self {
        Self {
            updates: KmerShardBuffers::new(),
            minimizer_runs: MinimizerRunBuffers::default(),
            canonical_kmers: Vec::new(),
        }
    }

    fn add_filtered_seq(
        &mut self,
        seq: &PackedSeqVec,
        config: CounterConfig,
        order: MinimizerOrder,
        minimizer_counts: &MinimizerCounts,
        kmer_counts: &ShardedKmerCountsU64,
    ) {
        let windows = seq.len().saturating_sub(config.k) + 1;
        if windows == 0 {
            return;
        }

        let raw = seq.clone().into_raw();
        canonical_kmers_u64(&raw, seq.len(), config, &mut self.canonical_kmers);
        for_each_minimizer_run_on_seq(
            seq.as_slice(),
            config,
            order,
            &mut self.minimizer_runs,
            |hash, start, end| {
                if !minimizer_counts.is_above_threshold(hash, config.threshold) {
                    return;
                }

                for &encoded in &self.canonical_kmers[start..end.min(windows)] {
                    self.updates.add(encoded, kmer_counts);
                }
            },
        );
    }

    fn flush(&mut self, counts: &ShardedKmerCountsU64) {
        self.updates.flush_all(counts);
    }
}

struct DatasetKmerPresenceWorker {
    kmers: KmerPresenceShardBuffers,
    minimizer_runs: MinimizerRunBuffers,
    canonical_kmers: Vec<u64>,
    filtered_runs: Vec<(usize, usize)>,
}

impl DatasetKmerPresenceWorker {
    fn new() -> Self {
        Self {
            kmers: KmerPresenceShardBuffers::new(),
            minimizer_runs: MinimizerRunBuffers::default(),
            canonical_kmers: Vec::new(),
            filtered_runs: Vec::new(),
        }
    }

    fn add_filtered_seq(
        &mut self,
        seq: PackedSeqVec,
        config: CounterConfig,
        order: MinimizerOrder,
        minimizer_counts: &DatasetMinimizerCounts,
        writers: &KmerPresencePartitionWriters,
    ) -> Result<()> {
        let seq_len = seq.len();
        let windows = seq_len.saturating_sub(config.k) + 1;
        if windows == 0 {
            return Ok(());
        }

        self.filtered_runs.clear();
        for_each_minimizer_run_on_seq(
            seq.as_slice(),
            config,
            order,
            &mut self.minimizer_runs,
            |hash, start, end| {
                if minimizer_counts.is_above_threshold(hash, config.threshold) {
                    self.filtered_runs.push((start, end.min(windows)));
                }
            },
        );
        if self.filtered_runs.is_empty() {
            return Ok(());
        }

        let raw = seq.into_raw();
        canonical_kmers_u64(&raw, seq_len, config, &mut self.canonical_kmers);
        for &(start, end) in &self.filtered_runs {
            for &encoded in &self.canonical_kmers[start..end] {
                self.kmers.add(encoded, writers)?;
            }
        }
        Ok(())
    }
}

struct KmerPresenceShardBuffers {
    buffers: Box<[Vec<u8>]>,
}

impl KmerPresenceShardBuffers {
    fn new() -> Self {
        let buffers = (0..KMER_COUNT_SHARDS).map(|_| Vec::new()).collect();
        Self { buffers }
    }

    fn add(&mut self, encoded: u64, writers: &KmerPresencePartitionWriters) -> Result<()> {
        let partition_idx = kmer_count_shard(encoded);
        let buffer = &mut self.buffers[partition_idx];
        buffer.extend_from_slice(&encoded.to_be_bytes());
        if buffer.len() >= SUPERKMER_BUFFER_FLUSH_BYTES {
            writers.write_buffer(partition_idx, buffer)?;
        }
        Ok(())
    }

    fn flush_all(&mut self, writers: &KmerPresencePartitionWriters) -> Result<()> {
        for (partition_idx, buffer) in self.buffers.iter_mut().enumerate() {
            writers.write_buffer(partition_idx, buffer)?;
        }
        Ok(())
    }
}

fn minimizer_table_shard_capacity(estimated_unique_hashes: usize) -> usize {
    let per_shard = estimated_unique_hashes.div_ceil(MINIMIZER_SHARDS);
    minimizer_table_capacity_for_items(per_shard)
}

fn minimizer_table_capacity_for_items(items: usize) -> usize {
    (items * 10)
        .div_ceil(7)
        .max(MINIMIZER_TABLE_MIN_SHARD_CAPACITY)
        .next_power_of_two()
}

struct CompactMinimizerTable {
    keys: Vec<MinimizerHash>,
    counts: Vec<CompactCount>,
    items: usize,
    saturation: CompactCount,
}

impl CompactMinimizerTable {
    fn with_capacity(capacity: usize, saturation: CompactCount) -> Self {
        let capacity = capacity
            .max(MINIMIZER_TABLE_MIN_SHARD_CAPACITY)
            .next_power_of_two();
        Self {
            keys: vec![0; capacity],
            counts: vec![0; capacity],
            items: 0,
            saturation,
        }
    }

    fn len(&self) -> usize {
        self.items
    }

    fn get(&self, hash: MinimizerHash) -> Option<&CompactCount> {
        let mask = self.keys.len() - 1;
        let mut idx = hash as usize & mask;

        loop {
            let count = self.counts[idx];
            if count == 0 {
                return None;
            }
            if self.keys[idx] == hash {
                return Some(&self.counts[idx]);
            }
            idx = (idx + 1) & mask;
        }
    }

    fn add(&mut self, hash: MinimizerHash, delta: CompactCount) {
        if (self.items + 1) * 10 >= self.keys.len() * 7 {
            self.grow();
        }
        self.add_without_grow(hash, delta);
    }

    fn add_without_grow(&mut self, hash: MinimizerHash, delta: CompactCount) {
        let mask = self.keys.len() - 1;
        let mut idx = hash as usize & mask;

        loop {
            if self.counts[idx] == 0 {
                self.keys[idx] = hash;
                self.counts[idx] = delta.min(self.saturation);
                self.items += 1;
                return;
            }
            if self.keys[idx] == hash {
                if self.counts[idx] < self.saturation {
                    self.counts[idx] = self.counts[idx].saturating_add(delta).min(self.saturation);
                }
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let new_capacity = self.keys.len() * 2;
        let old_keys = mem::replace(&mut self.keys, vec![0; new_capacity]);
        let old_counts = mem::replace(&mut self.counts, vec![0; new_capacity]);
        self.items = 0;

        for (hash, count) in old_keys.into_iter().zip(old_counts) {
            if count != 0 {
                self.add_without_grow(hash, count);
            }
        }
    }
}

struct DatasetMinimizerTable {
    keys: Vec<MinimizerHash>,
    states: Vec<u32>,
    items: usize,
}

impl DatasetMinimizerTable {
    fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity
            .max(MINIMIZER_TABLE_MIN_SHARD_CAPACITY)
            .next_power_of_two();
        Self {
            keys: vec![0; capacity],
            states: vec![0; capacity],
            items: 0,
        }
    }

    fn len(&self) -> usize {
        self.items
    }

    fn get_count(&self, hash: MinimizerHash) -> Option<u32> {
        let mask = self.keys.len() - 1;
        let mut idx = hash as usize & mask;

        loop {
            let state = self.states[idx];
            if state == 0 {
                return None;
            }
            if self.keys[idx] == hash {
                return Some(state & DATASET_PRESENCE_COUNT_MASK);
            }
            idx = (idx + 1) & mask;
        }
    }

    fn add_seen_in_current_dataset(&mut self, hash: MinimizerHash, accept_new: bool) {
        if (self.items + 1) * 10 >= self.keys.len() * 7 {
            self.grow();
        }

        let mask = self.keys.len() - 1;
        let mut idx = hash as usize & mask;
        loop {
            let state = self.states[idx];
            if state == 0 {
                if accept_new {
                    self.keys[idx] = hash;
                    self.states[idx] = 1 | DATASET_PRESENCE_SEEN_BIT;
                    self.items += 1;
                }
                return;
            }
            if self.keys[idx] == hash {
                if state & DATASET_PRESENCE_SEEN_BIT == 0 {
                    let count = (state & DATASET_PRESENCE_COUNT_MASK)
                        .saturating_add(1)
                        .min(DATASET_PRESENCE_COUNT_MASK);
                    self.states[idx] = count | DATASET_PRESENCE_SEEN_BIT;
                }
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn finish_dataset(&mut self, remaining_datasets: usize, threshold: Count) {
        let mut retained = Vec::with_capacity(self.items);
        for (hash, state) in self.keys.iter().copied().zip(self.states.iter().copied()) {
            if state == 0 {
                continue;
            }
            let count = state & DATASET_PRESENCE_COUNT_MASK;
            if count as Count + remaining_datasets as Count > threshold {
                retained.push((hash, count));
            }
        }

        let new_capacity = minimizer_table_capacity_for_items(retained.len());
        self.keys = vec![0; new_capacity];
        self.states = vec![0; new_capacity];
        self.items = 0;

        for (hash, count) in retained {
            self.insert_state_without_grow(hash, count);
        }
    }

    fn grow(&mut self) {
        let new_capacity = self.keys.len() * 2;
        let old_keys = mem::replace(&mut self.keys, vec![0; new_capacity]);
        let old_states = mem::replace(&mut self.states, vec![0; new_capacity]);
        self.items = 0;

        for (hash, state) in old_keys.into_iter().zip(old_states) {
            if state != 0 {
                self.insert_state_without_grow(hash, state);
            }
        }
    }

    fn insert_state_without_grow(&mut self, hash: MinimizerHash, state: u32) {
        let mask = self.keys.len() - 1;
        let mut idx = hash as usize & mask;

        loop {
            if self.states[idx] == 0 {
                self.keys[idx] = hash;
                self.states[idx] = state;
                self.items += 1;
                return;
            }
            idx = (idx + 1) & mask;
        }
    }
}

struct CompactKmerTableU64 {
    keys: Vec<u64>,
    counts: Vec<u8>,
    items: usize,
    saturation: u8,
}

impl CompactKmerTableU64 {
    fn with_capacity(expected_items: usize, saturation: u8) -> Self {
        let capacity = (expected_items * 10)
            .div_ceil(7)
            .max(MINIMIZER_TABLE_MIN_SHARD_CAPACITY)
            .next_power_of_two();
        Self {
            keys: vec![0; capacity],
            counts: vec![0; capacity],
            items: 0,
            saturation,
        }
    }

    fn add(&mut self, encoded: u64) {
        if (self.items + 1) * 10 >= self.keys.len() * 7 {
            self.grow();
        }
        self.add_without_grow(encoded, 1);
    }

    fn add_without_grow(&mut self, encoded: u64, delta: u8) {
        let mask = self.keys.len() - 1;
        let mut idx = spread_hash_u64(encoded) as usize & mask;

        loop {
            let count = self.counts[idx];
            if count == 0 {
                self.keys[idx] = encoded;
                self.counts[idx] = delta.min(self.saturation);
                self.items += 1;
                return;
            }
            if self.keys[idx] == encoded {
                if count < self.saturation {
                    self.counts[idx] = count.saturating_add(delta).min(self.saturation);
                }
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn add_buffer(&mut self, updates: &mut Vec<u64>) {
        for encoded in updates.drain(..) {
            self.add(encoded);
        }
    }

    fn grow(&mut self) {
        let new_capacity = self.keys.len() * 2;
        let old_keys = mem::replace(&mut self.keys, vec![0; new_capacity]);
        let old_counts = mem::replace(&mut self.counts, vec![0; new_capacity]);
        self.items = 0;

        for (encoded, count) in old_keys.into_iter().zip(old_counts) {
            if count != 0 {
                self.add_without_grow(encoded, count);
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        self.keys
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .filter(|&(_, count)| count != 0)
    }
}

struct KmerTableU64U32 {
    keys: Vec<u64>,
    counts: Vec<u32>,
    items: usize,
}

impl KmerTableU64U32 {
    fn with_capacity(expected_items: usize) -> Self {
        let capacity = (expected_items * 10)
            .div_ceil(7)
            .max(MINIMIZER_TABLE_MIN_SHARD_CAPACITY)
            .next_power_of_two();
        Self {
            keys: vec![0; capacity],
            counts: vec![0; capacity],
            items: 0,
        }
    }

    fn add(&mut self, encoded: u64) {
        if (self.items + 1) * 10 >= self.keys.len() * 7 {
            self.grow();
        }
        self.add_without_grow(encoded, 1);
    }

    fn add_without_grow(&mut self, encoded: u64, delta: u32) {
        let mask = self.keys.len() - 1;
        let mut idx = spread_hash_u64(encoded) as usize & mask;

        loop {
            let count = self.counts[idx];
            if count == 0 {
                self.keys[idx] = encoded;
                self.counts[idx] = delta;
                self.items += 1;
                return;
            }
            if self.keys[idx] == encoded {
                self.counts[idx] = count.saturating_add(delta);
                return;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn add_buffer(&mut self, updates: &mut Vec<u64>) {
        for encoded in updates.drain(..) {
            self.add(encoded);
        }
    }

    fn grow(&mut self) {
        let new_capacity = self.keys.len() * 2;
        let old_keys = mem::replace(&mut self.keys, vec![0; new_capacity]);
        let old_counts = mem::replace(&mut self.counts, vec![0; new_capacity]);
        self.items = 0;

        for (encoded, count) in old_keys.into_iter().zip(old_counts) {
            if count != 0 {
                self.add_without_grow(encoded, count);
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.keys
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .filter(|&(_, count)| count != 0)
    }
}

struct ShardedMinimizerCounts {
    shards: Box<[Mutex<CompactMinimizerTable>]>,
}

impl ShardedMinimizerCounts {
    fn new(threshold: Count, estimated_unique_hashes: usize) -> Self {
        let saturation = (threshold + 1) as CompactCount;
        let capacity = minimizer_table_shard_capacity(estimated_unique_hashes);
        let shards = (0..MINIMIZER_SHARDS)
            .map(|_| Mutex::new(CompactMinimizerTable::with_capacity(capacity, saturation)))
            .collect();
        Self { shards }
    }

    fn add_buffer(&self, shard_idx: usize, updates: &mut Vec<(MinimizerHash, CompactCount)>) {
        if updates.is_empty() {
            return;
        }

        let mut shard = self.shards[shard_idx]
            .lock()
            .expect("minimizer shard mutex poisoned");
        for (hash, delta) in updates.drain(..) {
            shard.add(hash, delta);
        }
    }

    fn freeze(self) -> MinimizerCounts {
        let shards = self
            .shards
            .into_vec()
            .into_iter()
            .map(|shard| shard.into_inner().expect("minimizer shard mutex poisoned"))
            .collect();
        MinimizerCounts { shards }
    }
}

struct ShardedDatasetMinimizerCounts {
    shards: Box<[Mutex<DatasetMinimizerTable>]>,
}

impl ShardedDatasetMinimizerCounts {
    fn new(estimated_unique_hashes: usize) -> Self {
        let capacity = minimizer_table_shard_capacity(estimated_unique_hashes);
        let shards = (0..MINIMIZER_SHARDS)
            .map(|_| Mutex::new(DatasetMinimizerTable::with_capacity(capacity)))
            .collect();
        Self { shards }
    }

    fn add_buffer(&self, shard_idx: usize, updates: &mut Vec<MinimizerHash>, accept_new: bool) {
        if updates.is_empty() {
            return;
        }

        let mut shard = self.shards[shard_idx]
            .lock()
            .expect("dataset minimizer shard mutex poisoned");
        for hash in updates.drain(..) {
            shard.add_seen_in_current_dataset(hash, accept_new);
        }
    }

    fn finish_dataset(&self, remaining_datasets: usize, threshold: Count) {
        self.shards.par_iter().for_each(|shard| {
            shard
                .lock()
                .expect("dataset minimizer shard mutex poisoned")
                .finish_dataset(remaining_datasets, threshold);
        });
    }

    fn freeze(self) -> DatasetMinimizerCounts {
        let shards = self
            .shards
            .into_vec()
            .into_iter()
            .map(|shard| {
                shard
                    .into_inner()
                    .expect("dataset minimizer shard mutex poisoned")
            })
            .collect();
        DatasetMinimizerCounts { shards }
    }
}

struct MinimizerCounts {
    shards: Box<[CompactMinimizerTable]>,
}

impl MinimizerCounts {
    fn is_above_threshold(&self, hash: MinimizerHash, threshold: Count) -> bool {
        self.shards[minimizer_shard(hash)]
            .get(hash)
            .is_some_and(|&count| count as Count > threshold)
    }

    fn unique_hashes(&self) -> usize {
        self.shards.iter().map(CompactMinimizerTable::len).sum()
    }
}

struct DatasetMinimizerCounts {
    shards: Box<[DatasetMinimizerTable]>,
}

impl DatasetMinimizerCounts {
    fn is_above_threshold(&self, hash: MinimizerHash, threshold: Count) -> bool {
        self.shards[minimizer_shard(hash)]
            .get_count(hash)
            .is_some_and(|count| count as Count > threshold)
    }

    fn unique_hashes(&self) -> usize {
        self.shards.iter().map(DatasetMinimizerTable::len).sum()
    }
}

struct ShardedKmerCountsU64 {
    shards: Box<[Mutex<CompactKmerTableU64>]>,
}

impl ShardedKmerCountsU64 {
    fn new(threshold: Count, estimated_unique_kmers: usize) -> Self {
        let saturation = (threshold + 1) as u8;
        let capacity = estimated_unique_kmers.div_ceil(KMER_COUNT_SHARDS);
        let shards = (0..KMER_COUNT_SHARDS)
            .map(|_| Mutex::new(CompactKmerTableU64::with_capacity(capacity, saturation)))
            .collect();
        Self { shards }
    }

    fn add_buffer(&self, shard_idx: usize, updates: &mut Vec<u64>) {
        if updates.is_empty() {
            return;
        }

        let mut shard = self.shards[shard_idx]
            .lock()
            .expect("k-mer count shard mutex poisoned");
        shard.add_buffer(updates);
    }

    fn freeze(self) -> KmerCountsU64 {
        let shards = self
            .shards
            .into_vec()
            .into_iter()
            .map(|shard| {
                shard
                    .into_inner()
                    .expect("k-mer count shard mutex poisoned")
            })
            .collect();
        KmerCountsU64 { shards }
    }
}

struct KmerCountsU64 {
    shards: Box<[CompactKmerTableU64]>,
}

impl KmerCountsU64 {
    fn above_threshold_count(&self, threshold: Count) -> u64 {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .iter()
                    .filter(|&(_, count)| count as Count > threshold)
                    .count() as u64
            })
            .sum()
    }
}

struct ShardedKmerCountsU64U32 {
    shards: Box<[Mutex<KmerTableU64U32>]>,
}

impl ShardedKmerCountsU64U32 {
    fn new(estimated_unique_kmers: usize) -> Self {
        let capacity = estimated_unique_kmers.div_ceil(KMER_COUNT_SHARDS);
        let shards = (0..KMER_COUNT_SHARDS)
            .map(|_| Mutex::new(KmerTableU64U32::with_capacity(capacity)))
            .collect();
        Self { shards }
    }

    fn add_buffer(&self, shard_idx: usize, updates: &mut Vec<u64>) {
        if updates.is_empty() {
            return;
        }

        let mut shard = self.shards[shard_idx]
            .lock()
            .expect("k-mer dataset count shard mutex poisoned");
        shard.add_buffer(updates);
    }

    fn freeze(self) -> KmerCountsU64U32 {
        let shards = self
            .shards
            .into_vec()
            .into_iter()
            .map(|shard| {
                shard
                    .into_inner()
                    .expect("k-mer dataset count shard mutex poisoned")
            })
            .collect();
        KmerCountsU64U32 { shards }
    }
}

struct KmerCountsU64U32 {
    shards: Box<[KmerTableU64U32]>,
}

impl KmerCountsU64U32 {
    fn above_threshold_count(&self, threshold: Count) -> u64 {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .iter()
                    .filter(|&(_, count)| count as Count > threshold)
                    .count() as u64
            })
            .sum()
    }
}

struct KmerPresencePartitionWriters {
    writers: Box<[Mutex<BufWriter<File>>]>,
}

impl KmerPresencePartitionWriters {
    fn create(dataset_dir: &Path) -> Result<Self> {
        let writers: Result<Vec<_>> = (0..KMER_COUNT_SHARDS)
            .map(|partition_idx| {
                let path = dataset_kmer_presence_partition_path(dataset_dir, partition_idx);
                let file = File::create(&path)
                    .with_context(|| format!("failed to create {}", path.display()))?;
                Ok(Mutex::new(BufWriter::with_capacity(1024 * 1024, file)))
            })
            .collect();
        Ok(Self {
            writers: writers?.into_boxed_slice(),
        })
    }

    fn write_buffer(&self, partition_idx: usize, buffer: &mut Vec<u8>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }
        let mut writer = self.writers[partition_idx]
            .lock()
            .expect("k-mer presence partition mutex poisoned");
        writer.write_all(buffer)?;
        buffer.clear();
        Ok(())
    }

    fn finish(self) -> Result<()> {
        for writer in self.writers.into_vec() {
            writer
                .into_inner()
                .expect("k-mer presence partition mutex poisoned")
                .flush()?;
        }
        Ok(())
    }
}

struct SuperkmerPartitionWriters {
    writers: Box<[Mutex<BufWriter<File>>]>,
}

impl SuperkmerPartitionWriters {
    fn create(partition_dir: &Path) -> Result<Self> {
        let writers: Result<Vec<_>> = (0..MINIMIZER_SHARDS)
            .map(|partition_idx| {
                let path = superkmer_partition_path(partition_dir, partition_idx);
                let file = File::create(&path)
                    .with_context(|| format!("failed to create {}", path.display()))?;
                Ok(Mutex::new(BufWriter::with_capacity(1024 * 1024, file)))
            })
            .collect();
        Ok(Self {
            writers: writers?.into_boxed_slice(),
        })
    }

    fn write_buffer(&self, partition_idx: usize, buffer: &mut Vec<u8>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }
        let mut writer = self.writers[partition_idx]
            .lock()
            .expect("super-kmer partition mutex poisoned");
        writer.write_all(buffer)?;
        buffer.clear();
        Ok(())
    }

    fn finish(self) -> Result<()> {
        for writer in self.writers.into_vec() {
            writer
                .into_inner()
                .expect("super-kmer partition mutex poisoned")
                .flush()?;
        }
        Ok(())
    }
}

struct SuperkmerShardBuffers {
    buffers: Box<[Vec<u8>]>,
}

struct SuperkmerWorkerBuffers {
    superkmers: SuperkmerShardBuffers,
    minimizer_runs: MinimizerRunBuffers,
    filtered_runs: Vec<(MinimizerHash, usize, usize)>,
}

impl SuperkmerWorkerBuffers {
    fn new() -> Self {
        Self {
            superkmers: SuperkmerShardBuffers::new(),
            minimizer_runs: MinimizerRunBuffers::default(),
            filtered_runs: Vec::new(),
        }
    }
}

impl SuperkmerShardBuffers {
    fn new() -> Self {
        let buffers = (0..MINIMIZER_SHARDS).map(|_| Vec::new()).collect();
        Self { buffers }
    }

    fn add_packed_range(
        &mut self,
        hash: MinimizerHash,
        packed: &[u8],
        start: usize,
        len: usize,
        writers: &SuperkmerPartitionWriters,
    ) -> Result<()> {
        ensure!(
            len <= u32::MAX as usize,
            "super-kmer is too large for partition record format"
        );
        ensure!(
            start + len <= packed.len() * 4,
            "packed super-kmer range exceeds sequence length"
        );
        let partition_idx = minimizer_shard(hash);
        let buffer = &mut self.buffers[partition_idx];
        append_record_len(buffer, len as u32);
        append_packed_dna_range(buffer, packed, start, len);
        if buffer.len() >= SUPERKMER_BUFFER_FLUSH_BYTES {
            writers.write_buffer(partition_idx, buffer)?;
        }
        Ok(())
    }

    fn flush_all(&mut self, writers: &SuperkmerPartitionWriters) -> Result<()> {
        for (partition_idx, buffer) in self.buffers.iter_mut().enumerate() {
            writers.write_buffer(partition_idx, buffer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(seq: &[u8]) -> EncodedKmer {
        seq.iter().fold(0u128, |encoded, &base| {
            (encoded << 2) | base_bits(base) as u128
        })
    }

    #[test]
    fn decode_round_trips_acgt_encoding() {
        let seq = b"ACGTACGT";
        assert_eq!(decode_kmer(encode(seq), seq.len()), seq);
    }

    #[test]
    fn packed_superkmer_counts_match_ascii_counts() {
        let seq = b"ACGTACGTACG";
        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 1,
        };
        let mut packed = Vec::new();
        append_packed_dna_seq(&mut packed, seq);

        let mut ascii_counts = AHashMap::new();
        let mut packed_counts = AHashMap::new();
        add_superkmer_kmer_counts(seq, config, &mut ascii_counts);
        add_packed_superkmer_kmer_counts(&packed, seq.len(), config, &mut packed_counts);

        assert_eq!(packed.len(), packed_dna_bytes(seq.len()));
        assert_eq!(ascii_counts, packed_counts);
    }

    #[test]
    fn partitioned_counter_counts_canonical_kmers() {
        use std::io::Write as _;

        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 1,
        };
        let dir = create_partition_dir().unwrap();
        let path = dir.join("reads.fa");
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">read").unwrap();
        writeln!(file, "ACGTACGT").unwrap();
        drop(file);

        let kmers = count_inputs(&[path], config).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"ACG"),
            count: 4
        }));
        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"GTA"),
            count: 2
        }));
    }

    #[test]
    fn partitioned_counter_accepts_gzipped_fasta() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write as _;

        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 1,
        };
        let dir = create_partition_dir().unwrap();
        let path = dir.join("reads.fa.gz");
        let file = File::create(&path).unwrap();
        let mut writer = GzEncoder::new(file, Compression::fast());
        writeln!(writer, ">read").unwrap();
        writeln!(writer, "ACGTACGT").unwrap();
        writer.finish().unwrap();

        let kmers = count_inputs(&[path], config).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"ACG"),
            count: 4
        }));
        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"GTA"),
            count: 2
        }));
    }

    #[test]
    fn partitioned_counter_accepts_xz_fasta() {
        use liblzma::write::XzEncoder;
        use std::io::Write as _;

        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 1,
        };
        let dir = create_partition_dir().unwrap();
        let path = dir.join("reads.fa.xz");
        let file = File::create(&path).unwrap();
        let mut writer = XzEncoder::new(file, 1);
        writeln!(writer, ">read").unwrap();
        writeln!(writer, "ACGTACGT").unwrap();
        writer.finish().unwrap();

        let kmers = count_inputs(&[path], config).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"ACG"),
            count: 4
        }));
        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"GTA"),
            count: 2
        }));
    }

    #[test]
    fn dataset_presence_counter_counts_each_dataset_once() {
        use std::io::Write as _;

        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 2,
        };
        let dir = create_partition_dir().unwrap();
        let d1 = dir.join("dataset1.fa");
        let d2 = dir.join("dataset2.fa");
        let d3 = dir.join("dataset3.fa");

        let mut file = File::create(&d1).unwrap();
        writeln!(file, ">d1").unwrap();
        writeln!(file, "AAAACGACG").unwrap();
        drop(file);

        let mut file = File::create(&d2).unwrap();
        writeln!(file, ">d2").unwrap();
        writeln!(file, "AAAACGTTT").unwrap();
        drop(file);

        let mut file = File::create(&d3).unwrap();
        writeln!(file, ">d3").unwrap();
        writeln!(file, "GGGACGCCC").unwrap();
        drop(file);

        let kmers = count_datasets(&[d1, d2, d3], config).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(kmers.contains(&CountedKmer {
            encoded: encode(b"ACG"),
            count: 3
        }));
        assert!(!kmers.iter().any(|record| record.encoded == encode(b"AAA")));
    }

    #[test]
    fn dataset_presence_counter_accepts_high_threshold() {
        use std::io::Write as _;

        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 300,
        };
        let dir = create_partition_dir().unwrap();
        let d1 = dir.join("dataset1.fa");
        let d2 = dir.join("dataset2.fa");

        let mut file = File::create(&d1).unwrap();
        writeln!(file, ">d1").unwrap();
        writeln!(file, "ACGACG").unwrap();
        drop(file);

        let mut file = File::create(&d2).unwrap();
        writeln!(file, ">d2").unwrap();
        writeln!(file, "ACGACG").unwrap();
        drop(file);

        let kmers = count_datasets(&[d1, d2], config).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(kmers.is_empty());
    }

    #[test]
    fn fasta_output_compacts_kmers_into_simplitigs() {
        let kmers = vec![
            CountedKmer {
                encoded: encode(b"AAA"),
                count: 1,
            },
            CountedKmer {
                encoded: encode(b"AAC"),
                count: 1,
            },
            CountedKmer {
                encoded: encode(b"ACG"),
                count: 1,
            },
        ];
        let mut out = Vec::new();

        write_fasta(&mut out, &kmers, 3).unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(">MC_simplitig_1 kmers=3\n"));
        assert!(text.contains("AAACG\n"));
    }

    #[test]
    fn fasta_path_writes_compressed_outputs() {
        use flate2::read::GzDecoder;
        use liblzma::read::XzDecoder;
        use std::io::Read as _;

        let kmers = vec![
            CountedKmer {
                encoded: encode(b"AAA"),
                count: 1,
            },
            CountedKmer {
                encoded: encode(b"AAC"),
                count: 1,
            },
            CountedKmer {
                encoded: encode(b"ACG"),
                count: 1,
            },
        ];
        let dir = create_partition_dir().unwrap();

        let gz_path = dir.join("out.fa.gz");
        write_fasta_path(&gz_path, &kmers, 3).unwrap();
        let mut text = String::new();
        GzDecoder::new(File::open(&gz_path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        assert!(text.contains("AAACG\n"));

        let xz_path = dir.join("out.fa.xz");
        write_fasta_path(&xz_path, &kmers, 3).unwrap();
        let mut text = String::new();
        XzDecoder::new(File::open(&xz_path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        assert!(text.contains("AAACG\n"));

        let zst_path = dir.join("out.fa.zst");
        write_fasta_path(&zst_path, &kmers, 3).unwrap();
        let mut text = String::new();
        zstd::stream::read::Decoder::new(File::open(&zst_path).unwrap())
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        assert!(text.contains("AAACG\n"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn kff_has_required_signatures() {
        let config = CounterConfig {
            k: 3,
            minimizer: 2,
            threshold: 1,
        };
        let kmers = vec![CountedKmer {
            encoded: encode(b"ACG"),
            count: 4,
        }];
        let mut out = Vec::new();
        write_kff(&mut out, &kmers, config).unwrap();
        assert!(out.starts_with(b"KFF"));
        assert!(out.ends_with(b"KFF"));
    }
}
