use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use mc::{create_output_writer_with_zstd_workers, with_xz_decompressed_path};
use parallel_processor::fast_smart_bucket_sort::{SortKey, fast_smart_radix_sort};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process;
use std::ptr;
use std::slice;
use std::str;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SEQUENCE_STORE_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const OUTPUT_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const OUTPUT_RENDER_CHUNK_RECORDS: usize = 4096;
const OUTPUT_RENDER_BATCH_CHUNKS: usize = 128;
const SEQUENCE_RENDER_CHUNK_BASES: usize = 16 * 1024;
const OUTPUT_SORT_PREFIX_BASES: usize = 32;
const PACKED_AC_COUNTS: [u8; 256] = packed_ac_counts();
const PACKED_DECODED_BASES: [[u8; 4]; 256] = packed_decoded_bases();
const PACKED_REVERSE_COMPLEMENT_BASES: [[u8; 4]; 256] = packed_reverse_complement_bases();
const RAW_BASE_BITS: [u8; 256] = raw_base_bits_table();
const INVALID_BASE_BITS: u8 = 4;
const NO_LINK: u32 = u32::MAX;
const ENDPOINT_NODE_BITS: usize = 33;
const ORIENTATION_FORWARD: u8 = 0;
const ORIENTATION_REVERSE: u8 = 1;
const ORIENTATION_UNKNOWN: u8 = 2;

#[derive(Parser, Debug)]
#[command(name = "R")]
#[command(
    about = "Reform unitig FASTA records into simplitigs or unitigs using exact K-1 overlaps"
)]
struct Cli {
    #[arg(
        short,
        long,
        help = "Input unitig FASTA/FASTQ file; gzip, xz, and zstd accepted"
    )]
    input: PathBuf,

    #[arg(
        short = 'k',
        long,
        help = "K-mer size used to define K-1 unitig overlaps"
    )]
    kmer_size: usize,

    #[arg(
        short,
        long,
        help = "Output FASTA records; .gz, .xz, and .zst are compressed"
    )]
    output: PathBuf,

    #[arg(short, long, help = "Number of Rayon worker threads")]
    threads: Option<usize>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of zstd output worker threads; 0 derives the worker count from --threads"
    )]
    zstd_workers: u32,

    #[arg(
        long,
        value_enum,
        default_value_t = OutputMode::Simplitig,
        help = "Output mode: simplitig greedily covers overlaps; unitig only joins unambiguous overlaps"
    )]
    output_mode: OutputMode,

    #[arg(
        long,
        value_enum,
        default_value_t = SequenceStoreMode::Disk,
        help = "Where to keep normalized unitig sequences during reforming; memory uses 2-bit packing"
    )]
    sequence_store: SequenceStoreMode,

    #[arg(
        long,
        value_enum,
        default_value_t = OutputSort::None,
        help = "Optional output record sort for compression experiments"
    )]
    output_sort: OutputSort,

    #[arg(
        long,
        value_enum,
        default_value_t = OutputStrandTieBreak::None,
        help = "How to choose the strand when both strands have equal A+C"
    )]
    strand_tiebreak: OutputStrandTieBreak,

    #[arg(
        long,
        value_enum,
        default_value_t = AbundanceMode::Mean,
        help = "How to encode km:f abundance in output headers"
    )]
    abundance_mode: AbundanceMode,

    #[arg(
        long,
        help = "Do not require input km:f abundance headers and do not emit km:f output headers"
    )]
    no_abundance: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AbundanceMode {
    Mean,
    Runs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputMode {
    Simplitig,
    Unitig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SequenceStoreMode {
    Disk,
    Memory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputSort {
    None,
    Length,
    LengthAsc,
    Unitigs,
    UnitigsAsc,
    SequencePrefix,
    SequenceSuffix,
    PrefixLength,
    LengthBucketPrefix,
    #[value(name = "length-bucket-prefix-fast")]
    LengthBucketPrefixFast,
    #[value(name = "length-bucket16-prefix")]
    LengthBucket16Prefix,
    #[value(name = "length-bucket32-prefix")]
    LengthBucket32Prefix,
    #[value(name = "length-bucket128-prefix")]
    LengthBucket128Prefix,
    #[value(name = "length-bucket256-prefix")]
    LengthBucket256Prefix,
    #[value(name = "length-bucket-prefix-suffix")]
    LengthBucketPrefixSuffix,
    #[value(name = "length-bucket-prefix-ac")]
    LengthBucketPrefixAc,
    #[value(name = "length-bucket-unitigs-prefix")]
    LengthBucketUnitigsPrefix,
    #[value(name = "length-bucket16-unitigs-prefix")]
    LengthBucket16UnitigsPrefix,
    #[value(name = "length-bucket32-unitigs-prefix")]
    LengthBucket32UnitigsPrefix,
    #[value(name = "length-bucket128-unitigs-prefix")]
    LengthBucket128UnitigsPrefix,
    #[value(name = "length-bucket-unitigs-prefix-suffix")]
    LengthBucketUnitigsPrefixSuffix,
    #[value(name = "length-unitigs-prefix")]
    LengthUnitigsPrefix,
    #[value(name = "unitigs-length-prefix")]
    UnitigsLengthPrefix,
    AcRichness,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputStrandTieBreak {
    None,
    Prefix,
    #[value(name = "prefix-ac")]
    PrefixAc,
}

#[derive(Clone, Debug)]
pub struct ReformerConfig {
    pub input: PathBuf,
    pub kmer_size: usize,
    pub output: PathBuf,
    pub output_mode: OutputMode,
    pub sequence_store_mode: SequenceStoreMode,
    pub output_sort: OutputSort,
    pub strand_tiebreak: OutputStrandTieBreak,
    pub abundance_mode: Option<AbundanceMode>,
    pub zstd_workers: Option<u32>,
    pub emit_logs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OverlapKey {
    high: u64,
    low: u64,
}

impl OverlapKey {
    fn from_encoded(encoded: u128) -> Self {
        Self {
            high: (encoded >> 64) as u64,
            low: encoded as u64,
        }
    }
}

impl Ord for OverlapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.high.cmp(&other.high).then(self.low.cmp(&other.low))
    }
}

impl PartialOrd for OverlapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Endpoint<K> {
    key: K,
    node: u64,
}

impl<K> Endpoint<K> {
    fn new(key: K, unitig_id: usize, reverse: bool) -> Self {
        Self {
            key,
            node: ((unitig_id as u64) << 1) | reverse as u64,
        }
    }

    fn unitig_id(self) -> usize {
        (self.node >> 1) as usize
    }

    fn reverse(self) -> bool {
        self.node & 1 != 0
    }
}

#[derive(Clone, Copy, Debug)]
struct UnitigMeta {
    offset: u64,
    mean_abundance: f64,
    len: u32,
    kmers: u32,
    abundance_start: u32,
    abundance_len: u32,
}

#[derive(Debug)]
struct ReadIndex {
    unitigs: Vec<UnitigMeta>,
    abundance_runs: Vec<AbundanceRun>,
    endpoints: EndpointIndex,
    sequence_store: SequenceStoreBacking,
    total_bases: u64,
}

#[derive(Clone, Debug)]
struct Links {
    orientation: Vec<u8>,
    incoming: Vec<u32>,
    outgoing: Vec<u32>,
    selected: u64,
}

#[derive(Clone, Debug)]
pub struct OutputStats {
    pub simplitigs: u64,
    pub unitigs: u64,
    pub bases: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputPath {
    unitig_start: u32,
    unitig_len: u32,
}

#[derive(Debug)]
struct OutputPaths {
    paths: Vec<OutputPath>,
    unitigs: Vec<u32>,
}

impl OutputPaths {
    fn path(&self, path: OutputPath) -> &[u32] {
        let start = path.unitig_start as usize;
        let len = path.unitig_len as usize;
        &self.unitigs[start..start + len]
    }
}

#[derive(Debug)]
struct RenderedChunk {
    start_path: usize,
    bases: u64,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SortedOutputPath {
    key: OutputSortKey,
    original_idx: usize,
    path: OutputPath,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct OutputSortKey {
    primary: u64,
    secondary: u64,
    tertiary: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FastSortedOutputPath {
    key: u128,
    original_idx: u32,
    path: OutputPath,
    record_reverse: bool,
}

#[derive(Clone, Copy, Debug)]
struct AbundanceRun {
    value: f64,
    len: u64,
}

#[derive(Clone, Debug)]
enum EndpointIndex {
    U64 {
        starts: Vec<Endpoint<u64>>,
        ends: Vec<Endpoint<u64>>,
    },
    U128 {
        starts: Vec<Endpoint<OverlapKey>>,
        ends: Vec<Endpoint<OverlapKey>>,
    },
}

impl EndpointIndex {
    fn new(overlap: usize) -> Self {
        if overlap <= 32 {
            Self::U64 {
                starts: Vec::new(),
                ends: Vec::new(),
            }
        } else {
            Self::U128 {
                starts: Vec::new(),
                ends: Vec::new(),
            }
        }
    }

    fn push(
        &mut self,
        unitig_id: usize,
        prefix: u128,
        suffix: u128,
        reverse_start: u128,
        reverse_end: u128,
    ) {
        match self {
            Self::U64 { starts, ends } => {
                starts.push(Endpoint::new(prefix as u64, unitig_id, false));
                ends.push(Endpoint::new(suffix as u64, unitig_id, false));
                starts.push(Endpoint::new(reverse_start as u64, unitig_id, true));
                ends.push(Endpoint::new(reverse_end as u64, unitig_id, true));
            }
            Self::U128 { starts, ends } => {
                starts.push(Endpoint::new(
                    OverlapKey::from_encoded(prefix),
                    unitig_id,
                    false,
                ));
                ends.push(Endpoint::new(
                    OverlapKey::from_encoded(suffix),
                    unitig_id,
                    false,
                ));
                starts.push(Endpoint::new(
                    OverlapKey::from_encoded(reverse_start),
                    unitig_id,
                    true,
                ));
                ends.push(Endpoint::new(
                    OverlapKey::from_encoded(reverse_end),
                    unitig_id,
                    true,
                ));
            }
        }
    }

    fn into_links(self, unitig_count: usize, output_mode: OutputMode) -> Links {
        match self {
            Self::U64 { starts, ends } => link_unitigs_u64(starts, ends, unitig_count, output_mode),
            Self::U128 { starts, ends } => link_unitigs(starts, ends, unitig_count, output_mode),
        }
    }
}

#[derive(Default)]
struct UnambiguousScratch {
    candidates: Vec<(usize, usize)>,
    end_candidate_counts: Vec<usize>,
    start_candidate_counts: Vec<usize>,
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).map(|idx| idx as u32).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value] as usize;
        if parent == value {
            value
        } else {
            let root = self.find(parent);
            self.parent[value] = root as u32;
            root
        }
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root as u32;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_k(cli.kmer_size)?;

    if let Some(threads) = cli.threads {
        ensure!(threads > 0, "--threads must be greater than 0");
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to initialize Rayon thread pool")?;
    }

    let tmp_dir = create_temp_dir("r-reformer")?;
    let started = Instant::now();
    let result = reform_path(
        ReformerConfig {
            input: cli.input,
            kmer_size: cli.kmer_size,
            output: cli.output,
            output_mode: cli.output_mode,
            sequence_store_mode: cli.sequence_store,
            output_sort: cli.output_sort,
            strand_tiebreak: cli.strand_tiebreak,
            abundance_mode: if cli.no_abundance {
                None
            } else {
                Some(cli.abundance_mode)
            },
            zstd_workers: derive_zstd_workers(cli.zstd_workers, cli.threads),
            emit_logs: true,
        },
        &tmp_dir,
    );
    let _ = fs::remove_dir_all(&tmp_dir);
    log_phase("total", started.elapsed());
    result.map(|_| ())
}

pub fn reform_path(config: ReformerConfig, tmp_dir: &Path) -> Result<OutputStats> {
    validate_k(config.kmer_size)?;
    let overlap = config.kmer_size - 1;

    let phase_started = Instant::now();
    let read_index = read_unitigs(
        &config.input,
        config.kmer_size,
        overlap,
        tmp_dir,
        config.sequence_store_mode,
        config.abundance_mode,
    )?;
    if config.emit_logs {
        log_phase("1_read_and_endpoint_indexing", phase_started.elapsed());
        eprintln!("R_STAT\tinput_unitigs\t{}", read_index.unitigs.len());
        eprintln!("R_STAT\tinput_bases\t{}", read_index.total_bases);
    }
    ensure!(
        read_index.unitigs.len() < NO_LINK as usize,
        "R supports at most {} unitigs in one input",
        NO_LINK - 1
    );

    let phase_started = Instant::now();
    let links = read_index
        .endpoints
        .into_links(read_index.unitigs.len(), config.output_mode);
    if config.emit_logs {
        log_phase("2_overlap_linking", phase_started.elapsed());
        eprintln!("R_STAT\tselected_overlaps\t{}", links.selected);
    }

    let phase_started = Instant::now();
    let output_stats = write_reformed_simplitigs(
        &read_index.sequence_store,
        &read_index.unitigs,
        &read_index.abundance_runs,
        &links,
        overlap,
        &config.output,
        config.output_mode,
        config.output_sort,
        config.strand_tiebreak,
        config.abundance_mode,
        config.zstd_workers,
    )?;
    if config.emit_logs {
        log_phase("3_output_streaming", phase_started.elapsed());
        eprintln!(
            "R_STAT\t{}\t{}",
            output_record_stat_name(config.output_mode),
            output_stats.simplitigs
        );
        eprintln!("R_STAT\toutput_unitigs\t{}", output_stats.unitigs);
        eprintln!("R_STAT\toutput_bases\t{}", output_stats.bases);
    }

    Ok(output_stats)
}

fn validate_k(k: usize) -> Result<()> {
    ensure!(k >= 2, "k must be at least 2");
    ensure!(
        k <= 64,
        "k must be <= 64 because R stores K-1 overlaps in u128"
    );
    Ok(())
}

fn output_record_stat_name(output_mode: OutputMode) -> &'static str {
    match output_mode {
        OutputMode::Simplitig => "output_simplitigs",
        OutputMode::Unitig => "output_unitig_records",
    }
}

fn derive_zstd_workers(requested: u32, rayon_threads: Option<usize>) -> Option<u32> {
    if requested > 0 {
        return Some(requested);
    }
    rayon_threads
        .filter(|&threads| threads > 1)
        .map(|threads| threads.min(16) as u32)
        .filter(|&workers| workers > 0)
}

fn read_unitigs(
    input_path: &Path,
    k: usize,
    overlap: usize,
    tmp_dir: &Path,
    sequence_store_mode: SequenceStoreMode,
    abundance_mode: Option<AbundanceMode>,
) -> Result<ReadIndex> {
    let mut sequence_store = SequenceStoreBuilder::new(tmp_dir, sequence_store_mode)?;
    let mut unitigs = Vec::new();
    let mut abundance_runs = Vec::new();
    let mut endpoints = EndpointIndex::new(overlap);
    let mut offset = 0u64;
    let mut total_bases = 0u64;
    let mut normalized = Vec::new();

    for_unitig_records(input_path, |header, seq| {
        let unitig_idx = unitigs.len() + 1;
        ensure!(
            seq.len() >= k,
            "unitig record {unitig_idx} has length {}, shorter than k={k}",
            seq.len()
        );

        let id = unitigs.len();
        let kmer_count = (seq.len() - k + 1) as u64;
        let (mean_abundance, abundance_start, abundance_len) = match abundance_mode {
            Some(AbundanceMode::Mean) => {
                (parse_abundance_mean(header, kmer_count, unitig_idx)?, 0, 0)
            }
            Some(AbundanceMode::Runs) => {
                let parsed = parse_abundance_runs(header, kmer_count, unitig_idx)?;
                match parsed.runs {
                    ParsedAbundanceRuns::Simple => (parsed.mean, 0, 0),
                    ParsedAbundanceRuns::Runs(runs) => {
                        let abundance_start = checked_u32(
                            abundance_runs.len() as u64,
                            "number of abundance runs before this record",
                        )?;
                        abundance_runs.extend_from_slice(&runs);
                        let abundance_len = checked_u32(
                            (abundance_runs.len() as u64).saturating_sub(abundance_start as u64),
                            "number of abundance runs for one record",
                        )?;
                        (parsed.mean, abundance_start, abundance_len)
                    }
                }
            }
            None => (0.0, 0, 0),
        };
        let (prefix, suffix) = if sequence_store_mode == SequenceStoreMode::Memory {
            sequence_store.write_raw_dna_with_endpoints(seq, unitig_idx, overlap)?
        } else {
            normalized.clear();
            normalized.reserve(seq.len());
            for &base in seq {
                let normalized_base = normalized_base(base).with_context(|| {
                    format!("unitig record {unitig_idx} contains a non-ACGT base")
                })?;
                normalized.push(normalized_base);
            }
            let prefix = encode_bases(&normalized[..overlap]);
            let suffix = encode_bases(&normalized[normalized.len() - overlap..]);
            sequence_store.write_all(&normalized)?;
            (prefix, suffix)
        };
        let reverse_start = reverse_complement_encoded(suffix, overlap);
        let reverse_end = reverse_complement_encoded(prefix, overlap);
        let len = checked_u32(seq.len() as u64, "unitig length")?;
        let kmers = checked_u32(kmer_count, "unitig k-mer count")?;

        endpoints.push(id, prefix, suffix, reverse_start, reverse_end);

        unitigs.push(UnitigMeta {
            offset,
            mean_abundance,
            len,
            kmers,
            abundance_start,
            abundance_len,
        });
        offset = offset.saturating_add(seq.len() as u64);
        total_bases = total_bases.saturating_add(seq.len() as u64);
        Ok(())
    })?;

    let sequence_store = sequence_store.finish()?;
    ensure!(
        !unitigs.is_empty(),
        "input FASTA {} did not contain any unitig records",
        input_path.display()
    );

    Ok(ReadIndex {
        unitigs,
        abundance_runs,
        endpoints,
        sequence_store,
        total_bases,
    })
}

fn for_unitig_records<F>(path: &Path, mut consume: F) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> Result<()>,
{
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("xz"))
    {
        return with_xz_decompressed_path(path, |plain_path| {
            for_unitig_records(plain_path, consume)
        });
    }

    const CONFIG: Config = ParserOptions::default()
        .dna_string()
        .ignore_quality()
        .keep_non_actg()
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open FASTA/FASTQ input {}", path.display()))?;
    while parser.next().is_some() {
        consume(parser.get_header(), parser.get_dna_string())?;
    }
    Ok(())
}

fn link_unitigs<K>(
    mut starts: Vec<Endpoint<K>>,
    mut ends: Vec<Endpoint<K>>,
    unitig_count: usize,
    output_mode: OutputMode,
) -> Links
where
    K: Copy + Eq + Ord + Send + Sync,
{
    starts.par_sort_unstable_by(compare_endpoints);
    ends.par_sort_unstable_by(compare_endpoints);
    link_sorted_unitigs(starts, ends, unitig_count, output_mode)
}

fn link_unitigs_u64(
    mut starts: Vec<Endpoint<u64>>,
    mut ends: Vec<Endpoint<u64>>,
    unitig_count: usize,
    output_mode: OutputMode,
) -> Links {
    fast_smart_radix_sort::<_, EndpointU64SortKey, true>(&mut starts);
    fast_smart_radix_sort::<_, EndpointU64SortKey, true>(&mut ends);
    link_sorted_unitigs(starts, ends, unitig_count, output_mode)
}

struct EndpointU64SortKey;

impl SortKey<Endpoint<u64>> for EndpointU64SortKey {
    type KeyType = u128;
    const KEY_BITS: usize = u64::BITS as usize + ENDPOINT_NODE_BITS;

    #[inline(always)]
    fn compare(left: &Endpoint<u64>, right: &Endpoint<u64>) -> Ordering {
        compare_endpoints(left, right)
    }

    #[inline(always)]
    fn get_shifted(value: &Endpoint<u64>, rhs: u8) -> u8 {
        ((((value.key as u128) << ENDPOINT_NODE_BITS) | value.node as u128) >> rhs) as u8
    }
}

fn link_sorted_unitigs<K>(
    starts: Vec<Endpoint<K>>,
    ends: Vec<Endpoint<K>>,
    unitig_count: usize,
    output_mode: OutputMode,
) -> Links
where
    K: Copy + Eq + Ord,
{
    let mut orientation = vec![ORIENTATION_UNKNOWN; unitig_count];
    let mut incoming = vec![NO_LINK; unitig_count];
    let mut outgoing = vec![NO_LINK; unitig_count];
    let mut components = DisjointSet::new(unitig_count);
    let mut selected = 0u64;
    let mut unambiguous_scratch = UnambiguousScratch::default();

    let mut start_pos = 0usize;
    let mut end_pos = 0usize;
    while start_pos < starts.len() && end_pos < ends.len() {
        match starts[start_pos].key.cmp(&ends[end_pos].key) {
            Ordering::Less => start_pos = next_key_range(&starts, start_pos).1,
            Ordering::Greater => end_pos = next_key_range(&ends, end_pos).1,
            Ordering::Equal => {
                let (start_begin, start_end) = next_key_range(&starts, start_pos);
                let (end_begin, end_end) = next_key_range(&ends, end_pos);
                let group_selected = match output_mode {
                    OutputMode::Simplitig => select_group_links(
                        &starts[start_begin..start_end],
                        &ends[end_begin..end_end],
                        &mut orientation,
                        &mut incoming,
                        &mut outgoing,
                        &mut components,
                    ),
                    OutputMode::Unitig => select_unambiguous_group_links(
                        &starts[start_begin..start_end],
                        &ends[end_begin..end_end],
                        &mut orientation,
                        &mut incoming,
                        &mut outgoing,
                        &mut components,
                        &mut unambiguous_scratch,
                    ),
                };
                selected = selected.saturating_add(group_selected);
                start_pos = start_end;
                end_pos = end_end;
            }
        }
    }

    Links {
        orientation,
        incoming,
        outgoing,
        selected,
    }
}

fn compare_endpoints<K: Ord>(left: &Endpoint<K>, right: &Endpoint<K>) -> Ordering {
    left.key.cmp(&right.key).then(left.node.cmp(&right.node))
}

fn next_key_range<K: Copy + Eq>(endpoints: &[Endpoint<K>], start: usize) -> (usize, usize) {
    let key = endpoints[start].key;
    let mut end = start + 1;
    while end < endpoints.len() && endpoints[end].key == key {
        end += 1;
    }
    (start, end)
}

fn select_group_links<K>(
    starts: &[Endpoint<K>],
    ends: &[Endpoint<K>],
    orientation: &mut [u8],
    incoming: &mut [u32],
    outgoing: &mut [u32],
    components: &mut DisjointSet,
) -> u64
where
    K: Copy,
{
    let mut selected = 0u64;
    for &end in ends {
        for &start in starts {
            if select_link(end, start, orientation, incoming, outgoing, components) {
                selected += 1;
                break;
            }
        }
    }
    selected
}

fn select_unambiguous_group_links<K>(
    starts: &[Endpoint<K>],
    ends: &[Endpoint<K>],
    orientation: &mut [u8],
    incoming: &mut [u32],
    outgoing: &mut [u32],
    components: &mut DisjointSet,
    scratch: &mut UnambiguousScratch,
) -> u64
where
    K: Copy,
{
    if starts.len() == 1 && ends.len() == 1 {
        return u64::from(select_link(
            ends[0],
            starts[0],
            orientation,
            incoming,
            outgoing,
            components,
        ));
    }

    scratch.candidates.clear();
    scratch.end_candidate_counts.clear();
    scratch.start_candidate_counts.clear();
    scratch.end_candidate_counts.resize(ends.len(), 0);
    scratch.start_candidate_counts.resize(starts.len(), 0);

    for (end_idx, &end) in ends.iter().enumerate() {
        let from = end.unitig_id();
        if outgoing[from] != NO_LINK || !orientation_compatible(orientation[from], end.reverse()) {
            continue;
        }

        for (start_idx, &start) in starts.iter().enumerate() {
            let to = start.unitig_id();
            if from == to
                || incoming[to] != NO_LINK
                || !orientation_compatible(orientation[to], start.reverse())
                || components.find(from) == components.find(to)
            {
                continue;
            }

            scratch.candidates.push((end_idx, start_idx));
            scratch.end_candidate_counts[end_idx] += 1;
            scratch.start_candidate_counts[start_idx] += 1;
        }
    }

    let mut selected = 0u64;
    for &(end_idx, start_idx) in &scratch.candidates {
        if scratch.end_candidate_counts[end_idx] != 1
            || scratch.start_candidate_counts[start_idx] != 1
        {
            continue;
        }

        let end = ends[end_idx];
        let start = starts[start_idx];
        if select_link(end, start, orientation, incoming, outgoing, components) {
            selected += 1;
        }
    }

    selected
}

fn select_link<K: Copy>(
    end: Endpoint<K>,
    start: Endpoint<K>,
    orientation: &mut [u8],
    incoming: &mut [u32],
    outgoing: &mut [u32],
    components: &mut DisjointSet,
) -> bool {
    let from = end.unitig_id();
    let to = start.unitig_id();
    if from == to
        || outgoing[from] != NO_LINK
        || incoming[to] != NO_LINK
        || !orientation_compatible(orientation[from], end.reverse())
        || !orientation_compatible(orientation[to], start.reverse())
        || components.find(from) == components.find(to)
    {
        return false;
    }

    set_orientation(&mut orientation[from], end.reverse());
    set_orientation(&mut orientation[to], start.reverse());
    outgoing[from] = to as u32;
    incoming[to] = from as u32;
    components.union(from, to);
    true
}

fn orientation_compatible(current: u8, candidate: bool) -> bool {
    current == ORIENTATION_UNKNOWN || current == orientation_value(candidate)
}

fn set_orientation(current: &mut u8, candidate: bool) {
    if *current == ORIENTATION_UNKNOWN {
        *current = orientation_value(candidate);
    }
}

fn orientation_value(reverse: bool) -> u8 {
    if reverse {
        ORIENTATION_REVERSE
    } else {
        ORIENTATION_FORWARD
    }
}

fn is_reverse_orientation(orientation: u8) -> bool {
    orientation == ORIENTATION_REVERSE
}

fn link_target(link: u32) -> Option<usize> {
    if link == NO_LINK {
        None
    } else {
        Some(link as usize)
    }
}

fn checked_u32(value: u64, description: &str) -> Result<u32> {
    ensure!(
        value <= u32::MAX as u64,
        "{description} {value} exceeds {}",
        u32::MAX
    );
    Ok(value as u32)
}

#[derive(Debug)]
enum SequenceStoreBacking {
    Disk(PathBuf),
    Memory(PackedSequenceStore),
}

enum SequenceStoreBuilder {
    Disk {
        path: PathBuf,
        writer: BufWriter<File>,
    },
    Memory(PackedSequenceStoreBuilder),
}

impl SequenceStoreBuilder {
    fn new(tmp_dir: &Path, mode: SequenceStoreMode) -> Result<Self> {
        match mode {
            SequenceStoreMode::Disk => {
                let path = tmp_dir.join("unitigs.seq");
                let file = File::create(&path)
                    .with_context(|| format!("failed to create {}", path.display()))?;
                Ok(Self::Disk {
                    path,
                    writer: BufWriter::with_capacity(SEQUENCE_STORE_BUFFER_BYTES, file),
                })
            }
            SequenceStoreMode::Memory => Ok(Self::Memory(PackedSequenceStoreBuilder::new())),
        }
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            Self::Disk { writer, .. } => writer.write_all(bytes)?,
            Self::Memory(sequence_store) => sequence_store.write_all(bytes),
        }
        Ok(())
    }

    fn write_raw_dna_with_endpoints(
        &mut self,
        bytes: &[u8],
        unitig_idx: usize,
        overlap: usize,
    ) -> Result<(u128, u128)> {
        match self {
            Self::Disk { .. } => unreachable!("disk sequence store receives normalized bases"),
            Self::Memory(sequence_store) => {
                sequence_store.write_raw_dna_with_endpoints(bytes, unitig_idx, overlap)
            }
        }
    }

    fn finish(self) -> Result<SequenceStoreBacking> {
        match self {
            Self::Disk { path, mut writer } => {
                writer.flush()?;
                Ok(SequenceStoreBacking::Disk(path))
            }
            Self::Memory(sequence_store) => {
                Ok(SequenceStoreBacking::Memory(sequence_store.finish()))
            }
        }
    }
}

enum SequenceStore<'a> {
    Disk(MappedSequenceStore),
    Memory(&'a PackedSequenceStore),
}

unsafe impl<'a> Send for SequenceStore<'a> {}
unsafe impl<'a> Sync for SequenceStore<'a> {}

impl<'a> SequenceStore<'a> {
    fn open(backing: &'a SequenceStoreBacking) -> Result<Self> {
        match backing {
            SequenceStoreBacking::Disk(path) => MappedSequenceStore::open(path).map(Self::Disk),
            SequenceStoreBacking::Memory(sequence_store) => Ok(Self::Memory(sequence_store)),
        }
    }

    fn write_forward<W: Write>(
        &self,
        meta: UnitigMeta,
        start: usize,
        len: usize,
        writer: &mut FastaRecordWriter<'_, W>,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        match self {
            Self::Disk(sequence_store) => {
                let seq = sequence_store.sequence(meta);
                debug_assert!(start <= seq.len());
                debug_assert!(len <= seq.len() - start);
                writer.write(&seq[start..start + len])
            }
            Self::Memory(sequence_store) => {
                sequence_store.write_forward(meta.offset + start as u64, len, writer, scratch)
            }
        }
    }

    fn write_reverse_complemented<W: Write>(
        &self,
        meta: UnitigMeta,
        start: usize,
        len: usize,
        writer: &mut FastaRecordWriter<'_, W>,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        match self {
            Self::Disk(sequence_store) => {
                let seq = sequence_store.sequence(meta);
                debug_assert!(start <= seq.len());
                debug_assert!(len <= seq.len() - start);
                writer.write_reverse_complemented(&seq[start..start + len], scratch)
            }
            Self::Memory(sequence_store) => sequence_store.write_reverse_complemented(
                meta.offset + start as u64,
                len,
                writer,
                scratch,
            ),
        }
    }

    fn count_ac(
        &self,
        meta: UnitigMeta,
        start: usize,
        len: usize,
        reverse_complemented: bool,
    ) -> u64 {
        let ac = match self {
            Self::Disk(sequence_store) => {
                let seq = sequence_store.sequence(meta);
                debug_assert!(start <= seq.len());
                debug_assert!(len <= seq.len() - start);
                seq[start..start + len]
                    .iter()
                    .filter(|&&base| is_ac_base(base))
                    .count() as u64
            }
            Self::Memory(sequence_store) => {
                sequence_store.count_ac(meta.offset + start as u64, len)
            }
        };
        if reverse_complemented {
            len as u64 - ac
        } else {
            ac
        }
    }

    fn base_bits(&self, meta: UnitigMeta, offset: usize) -> u8 {
        match self {
            Self::Disk(sequence_store) => {
                let seq = sequence_store.sequence(meta);
                debug_assert!(offset < seq.len());
                base_bits(seq[offset])
            }
            Self::Memory(sequence_store) => sequence_store.base_bits(meta.offset + offset as u64),
        }
    }
}

#[derive(Debug, Default)]
struct PackedSequenceStoreBuilder {
    data: Vec<u8>,
    bases: u64,
}

impl PackedSequenceStoreBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn write_all(&mut self, bases: &[u8]) {
        let mut pos = 0usize;

        while pos < bases.len() && (self.bases & 3) != 0 {
            self.push(bases[pos]);
            pos += 1;
        }

        let packed_end = pos + ((bases.len() - pos) / 4) * 4;
        self.data.reserve((packed_end - pos) / 4);
        while pos < packed_end {
            self.data.push(pack_four_bases(&bases[pos..pos + 4]));
            self.bases += 4;
            pos += 4;
        }

        while pos < bases.len() {
            self.push(bases[pos]);
            pos += 1;
        }
    }

    fn write_raw_dna_with_endpoints(
        &mut self,
        bases: &[u8],
        unitig_idx: usize,
        overlap: usize,
    ) -> Result<(u128, u128)> {
        let mut pos = 0usize;
        let suffix_start = bases.len() - overlap;
        let mut prefix = 0u128;
        let mut suffix = 0u128;

        while pos < bases.len() && (self.bases & 3) != 0 {
            let bits = raw_base_bits(bases[pos], unitig_idx)?;
            update_endpoint_encoding(bits, pos, overlap, suffix_start, &mut prefix, &mut suffix);
            self.push_bits(bits);
            pos += 1;
        }

        let packed_end = pos + ((bases.len() - pos) / 4) * 4;
        self.data.reserve((packed_end - pos) / 4);
        while pos < packed_end {
            let b0 = RAW_BASE_BITS[bases[pos] as usize];
            let b1 = RAW_BASE_BITS[bases[pos + 1] as usize];
            let b2 = RAW_BASE_BITS[bases[pos + 2] as usize];
            let b3 = RAW_BASE_BITS[bases[pos + 3] as usize];
            if (b0 | b1 | b2 | b3) >= INVALID_BASE_BITS {
                invalid_raw_base(unitig_idx)?;
            }
            update_endpoint_encoding(b0, pos, overlap, suffix_start, &mut prefix, &mut suffix);
            update_endpoint_encoding(b1, pos + 1, overlap, suffix_start, &mut prefix, &mut suffix);
            update_endpoint_encoding(b2, pos + 2, overlap, suffix_start, &mut prefix, &mut suffix);
            update_endpoint_encoding(b3, pos + 3, overlap, suffix_start, &mut prefix, &mut suffix);
            self.data.push((b0 << 6) | (b1 << 4) | (b2 << 2) | b3);
            self.bases += 4;
            pos += 4;
        }

        while pos < bases.len() {
            let bits = raw_base_bits(bases[pos], unitig_idx)?;
            update_endpoint_encoding(bits, pos, overlap, suffix_start, &mut prefix, &mut suffix);
            self.push_bits(bits);
            pos += 1;
        }

        Ok((prefix, suffix))
    }

    fn push(&mut self, base: u8) {
        self.push_bits(base_bits(base));
    }

    fn push_bits(&mut self, bits: u8) {
        let byte_idx = (self.bases / 4) as usize;
        let shift = 6 - (((self.bases as usize) & 3) * 2);
        if byte_idx == self.data.len() {
            self.data.push(0);
        }
        self.data[byte_idx] |= bits << shift;
        self.bases += 1;
    }

    fn finish(self) -> PackedSequenceStore {
        PackedSequenceStore {
            data: self.data,
            bases: self.bases,
        }
    }
}

#[derive(Debug)]
struct PackedSequenceStore {
    data: Vec<u8>,
    bases: u64,
}

impl PackedSequenceStore {
    fn write_forward<W: Write>(
        &self,
        base_offset: u64,
        len: usize,
        writer: &mut FastaRecordWriter<'_, W>,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        debug_assert!(base_offset <= self.bases);
        debug_assert!(len as u64 <= self.bases - base_offset);
        let mut written = 0usize;
        while written < len {
            let take = SEQUENCE_RENDER_CHUNK_BASES.min(len - written);
            scratch.clear();
            scratch.reserve(take);
            self.append_forward_bases(base_offset + written as u64, take, scratch);
            writer.write(scratch)?;
            written += take;
        }
        Ok(())
    }

    fn write_reverse_complemented<W: Write>(
        &self,
        base_offset: u64,
        len: usize,
        writer: &mut FastaRecordWriter<'_, W>,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        debug_assert!(base_offset <= self.bases);
        debug_assert!(len as u64 <= self.bases - base_offset);
        let mut remaining = len;
        while remaining != 0 {
            let take = SEQUENCE_RENDER_CHUNK_BASES.min(remaining);
            let chunk_start = base_offset + (remaining - take) as u64;
            scratch.clear();
            scratch.reserve(take);
            self.append_reverse_complemented_bases(chunk_start, take, scratch);
            writer.write(scratch)?;
            remaining -= take;
        }
        Ok(())
    }

    fn append_forward_bases(&self, mut base_offset: u64, mut len: usize, output: &mut Vec<u8>) {
        while len != 0 && (base_offset & 3) != 0 {
            output.push(base_from_bits(self.base_bits(base_offset)));
            base_offset += 1;
            len -= 1;
        }

        let byte_count = len / 4;
        let byte_start = (base_offset / 4) as usize;
        if byte_count != 0 {
            let output_start = output.len();
            output.resize(output_start + byte_count * 4, 0);
            for (slot, &byte) in output[output_start..]
                .chunks_exact_mut(4)
                .zip(&self.data[byte_start..byte_start + byte_count])
            {
                slot.copy_from_slice(&PACKED_DECODED_BASES[byte as usize]);
            }
        }
        base_offset += (byte_count * 4) as u64;
        len -= byte_count * 4;

        while len != 0 {
            output.push(base_from_bits(self.base_bits(base_offset)));
            base_offset += 1;
            len -= 1;
        }
    }

    fn append_reverse_complemented_bases(
        &self,
        base_offset: u64,
        mut len: usize,
        output: &mut Vec<u8>,
    ) {
        let mut end_offset = base_offset + len as u64;

        while len != 0 && (end_offset & 3) != 0 {
            end_offset -= 1;
            output.push(base_from_bits(self.base_bits(end_offset) ^ 0b11));
            len -= 1;
        }

        let byte_count = len / 4;
        if byte_count != 0 {
            let byte_end = (end_offset / 4) as usize;
            let byte_start = byte_end - byte_count;
            let output_start = output.len();
            output.resize(output_start + byte_count * 4, 0);
            for (slot, &byte) in output[output_start..]
                .chunks_exact_mut(4)
                .zip(self.data[byte_start..byte_end].iter().rev())
            {
                slot.copy_from_slice(&PACKED_REVERSE_COMPLEMENT_BASES[byte as usize]);
            }
            end_offset -= (byte_count * 4) as u64;
            len -= byte_count * 4;
        }

        while len != 0 {
            end_offset -= 1;
            output.push(base_from_bits(self.base_bits(end_offset) ^ 0b11));
            len -= 1;
        }
    }

    fn count_ac(&self, base_offset: u64, len: usize) -> u64 {
        debug_assert!(base_offset <= self.bases);
        debug_assert!(len as u64 <= self.bases - base_offset);
        let mut offset = base_offset;
        let mut remaining = len;
        let mut count = 0u64;

        while remaining != 0 && (offset & 3) != 0 {
            count += u64::from(packed_base_is_ac(self.base_bits(offset)));
            offset += 1;
            remaining -= 1;
        }

        while remaining >= 4 {
            count += PACKED_AC_COUNTS[self.data[(offset / 4) as usize] as usize] as u64;
            offset += 4;
            remaining -= 4;
        }

        while remaining != 0 {
            count += u64::from(packed_base_is_ac(self.base_bits(offset)));
            offset += 1;
            remaining -= 1;
        }

        count
    }

    fn base_bits(&self, base_offset: u64) -> u8 {
        debug_assert!(base_offset < self.bases);
        let byte = self.data[(base_offset / 4) as usize];
        let shift = 6 - (((base_offset as usize) & 3) * 2);
        (byte >> shift) & 0b11
    }
}

struct MappedSequenceStore {
    ptr: *const u8,
    len: usize,
    _file: File,
}

// The mapped sequence store is read-only and stays valid for the lifetime of the struct.
unsafe impl Send for MappedSequenceStore {}
unsafe impl Sync for MappedSequenceStore {}

impl MappedSequenceStore {
    fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let len = usize::try_from(
            file.metadata()
                .with_context(|| format!("failed to stat {}", path.display()))?
                .len(),
        )
        .context("sequence store exceeds addressable memory")?;
        ensure!(len > 0, "sequence store {} is empty", path.display());

        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("failed to mmap {}", path.display()));
        }

        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len,
            _file: file,
        })
    }

    fn sequence(&self, meta: UnitigMeta) -> &[u8] {
        let offset = meta.offset as usize;
        let len = meta.len as usize;
        debug_assert!(offset <= self.len);
        debug_assert!(len <= self.len - offset);
        unsafe { slice::from_raw_parts(self.ptr.add(offset), len) }
    }
}

impl Drop for MappedSequenceStore {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.cast_mut().cast(), self.len);
        }
    }
}

struct ChunkedWriter<W: Write> {
    inner: W,
    buffer: Vec<u8>,
    capacity: usize,
}

impl<W: Write> ChunkedWriter<W> {
    fn with_capacity(capacity: usize, inner: W) -> Self {
        Self {
            inner,
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn flush_buffer(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.inner.write_all(&self.buffer)?;
            self.buffer.clear();
        }
        Ok(())
    }

    fn finish(mut self) -> io::Result<W> {
        self.flush_buffer()?;
        Ok(self.inner)
    }
}

impl<W: Write> Write for ChunkedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_all(buf)?;
        Ok(buf.len())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        if buf.len() >= self.capacity {
            self.flush_buffer()?;
            return self.inner.write_all(buf);
        }
        if self.buffer.len() + buf.len() > self.capacity {
            self.flush_buffer()?;
        }
        self.buffer.extend_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.inner.flush()
    }
}

fn write_reformed_simplitigs(
    sequence_store_backing: &SequenceStoreBacking,
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    overlap: usize,
    output_path: &Path,
    output_mode: OutputMode,
    output_sort: OutputSort,
    strand_tiebreak: OutputStrandTieBreak,
    abundance_mode: Option<AbundanceMode>,
    zstd_workers: Option<u32>,
) -> Result<OutputStats> {
    let sequence_store = SequenceStore::open(sequence_store_backing)?;
    let mut output_paths = collect_output_paths(links, unitigs.len())?;
    let record_reverses = sort_output_paths(
        &mut output_paths,
        unitigs,
        links,
        overlap,
        &sequence_store,
        output_sort,
        strand_tiebreak,
    );
    let writer = create_output_writer_with_zstd_workers(output_path, zstd_workers)?;
    let mut writer = ChunkedWriter::with_capacity(OUTPUT_BUFFER_BYTES, writer);
    let mut stats = OutputStats {
        simplitigs: output_paths.paths.len() as u64,
        unitigs: output_paths.unitigs.len() as u64,
        bases: 0,
    };

    write_output_paths(
        &output_paths,
        unitigs,
        abundance_runs,
        links,
        overlap,
        &sequence_store,
        record_reverses.as_deref(),
        &mut writer,
        output_mode,
        strand_tiebreak,
        abundance_mode,
        &mut stats,
    )?;

    writer.finish()?.finish()?;
    Ok(stats)
}

fn collect_output_paths(links: &Links, unitig_count: usize) -> Result<OutputPaths> {
    let mut visited = vec![false; unitig_count];
    let mut paths = Vec::new();
    let mut path_unitigs = Vec::with_capacity(unitig_count);

    for unitig_id in 0..unitig_count {
        if links.incoming[unitig_id] == NO_LINK && !visited[unitig_id] {
            collect_output_path(
                unitig_id,
                links,
                &mut visited,
                &mut paths,
                &mut path_unitigs,
            )?;
        }
    }

    for unitig_id in 0..unitig_count {
        if !visited[unitig_id] {
            collect_output_path(
                unitig_id,
                links,
                &mut visited,
                &mut paths,
                &mut path_unitigs,
            )?;
        }
    }

    Ok(OutputPaths {
        paths,
        unitigs: path_unitigs,
    })
}

fn collect_output_path(
    start: usize,
    links: &Links,
    visited: &mut [bool],
    paths: &mut Vec<OutputPath>,
    path_unitigs: &mut Vec<u32>,
) -> Result<()> {
    let path_start = checked_u32(path_unitigs.len() as u64, "number of output path unitigs")?;
    let mut path_len = 0u64;
    let mut current = Some(start);
    while let Some(unitig_id) = current {
        if visited[unitig_id] {
            break;
        }
        path_unitigs.push(checked_u32(unitig_id as u64, "unitig id")?);
        visited[unitig_id] = true;
        path_len += 1;
        current = link_target(links.outgoing[unitig_id]);
    }

    if path_len > 0 {
        paths.push(OutputPath {
            unitig_start: path_start,
            unitig_len: checked_u32(path_len, "number of unitigs in one output path")?,
        });
    }

    Ok(())
}

fn sort_output_paths(
    output_paths: &mut OutputPaths,
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    output_sort: OutputSort,
    strand_tiebreak: OutputStrandTieBreak,
) -> Option<Vec<bool>> {
    if output_sort == OutputSort::None || output_paths.paths.len() <= 1 {
        return None;
    }
    if matches!(
        output_sort,
        OutputSort::LengthBucketPrefix
            | OutputSort::LengthBucketPrefixFast
            | OutputSort::LengthBucket16Prefix
            | OutputSort::LengthBucket32Prefix
            | OutputSort::LengthBucket128Prefix
            | OutputSort::LengthBucket256Prefix
    ) {
        return Some(sort_output_paths_length_bucket_prefix_fast(
            output_paths,
            unitigs,
            links,
            overlap,
            sequence_store,
            strand_tiebreak,
            length_bucket_size(output_sort).unwrap_or(64),
        ));
    }

    let mut keyed_paths: Vec<_> = output_paths
        .paths
        .par_iter()
        .copied()
        .enumerate()
        .map(|(original_idx, path)| SortedOutputPath {
            key: output_sort_key(
                output_paths.path(path),
                unitigs,
                links,
                overlap,
                sequence_store,
                output_sort,
                strand_tiebreak,
            ),
            original_idx,
            path,
        })
        .collect();

    keyed_paths.par_sort_unstable_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then(left.original_idx.cmp(&right.original_idx))
    });

    for (slot, keyed_path) in output_paths.paths.iter_mut().zip(keyed_paths) {
        *slot = keyed_path.path;
    }
    None
}

fn sort_output_paths_length_bucket_prefix_fast(
    output_paths: &mut OutputPaths,
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    strand_tiebreak: OutputStrandTieBreak,
    bucket_size: u64,
) -> Vec<bool> {
    let mut keyed_paths: Vec<_> = output_paths
        .paths
        .par_iter()
        .copied()
        .enumerate()
        .map(|(original_idx, path)| {
            let path_unitigs = output_paths.path(path);
            let record_reverse = choose_record_reverse(
                path_unitigs,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path_unitigs, unitigs, overlap);
            let bucket = u64::MAX - bases / bucket_size;
            let prefix = output_path_prefix_key(
                path_unitigs,
                unitigs,
                links,
                overlap,
                sequence_store,
                record_reverse,
            );
            FastSortedOutputPath {
                key: ((bucket as u128) << 64) | prefix as u128,
                original_idx: original_idx as u32,
                path,
                record_reverse,
            }
        })
        .collect();

    keyed_paths.par_sort_unstable_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then(left.original_idx.cmp(&right.original_idx))
    });

    let mut record_reverses = Vec::with_capacity(output_paths.paths.len());
    for (slot, keyed_path) in output_paths.paths.iter_mut().zip(keyed_paths) {
        *slot = keyed_path.path;
        record_reverses.push(keyed_path.record_reverse);
    }
    record_reverses
}

fn length_bucket_size(output_sort: OutputSort) -> Option<u64> {
    match output_sort {
        OutputSort::LengthBucket16Prefix => Some(16),
        OutputSort::LengthBucket32Prefix => Some(32),
        OutputSort::LengthBucketPrefix | OutputSort::LengthBucketPrefixFast => Some(64),
        OutputSort::LengthBucket128Prefix => Some(128),
        OutputSort::LengthBucket256Prefix => Some(256),
        _ => None,
    }
}

fn length_bucket_unitigs_size(output_sort: OutputSort) -> Option<u64> {
    match output_sort {
        OutputSort::LengthBucket16UnitigsPrefix => Some(16),
        OutputSort::LengthBucket32UnitigsPrefix => Some(32),
        OutputSort::LengthBucketUnitigsPrefix | OutputSort::LengthBucketUnitigsPrefixSuffix => {
            Some(64)
        }
        OutputSort::LengthBucket128UnitigsPrefix => Some(128),
        _ => None,
    }
}

fn output_sort_key(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    output_sort: OutputSort,
    strand_tiebreak: OutputStrandTieBreak,
) -> OutputSortKey {
    match output_sort {
        OutputSort::None => OutputSortKey {
            primary: 0,
            secondary: 0,
            tertiary: 0,
        },
        OutputSort::Length => {
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - bases,
                secondary: path.len() as u64,
                tertiary: 0,
            }
        }
        OutputSort::LengthAsc => {
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: bases,
                secondary: path.len() as u64,
                tertiary: 0,
            }
        }
        OutputSort::Unitigs => OutputSortKey {
            primary: u64::MAX - path.len() as u64,
            secondary: u64::MAX - output_path_bases(path, unitigs, overlap),
            tertiary: 0,
        },
        OutputSort::UnitigsAsc => OutputSortKey {
            primary: path.len() as u64,
            secondary: output_path_bases(path, unitigs, overlap),
            tertiary: 0,
        },
        OutputSort::SequencePrefix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            OutputSortKey {
                primary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
                secondary: output_path_bases(path, unitigs, overlap),
                tertiary: 0,
            }
        }
        OutputSort::SequenceSuffix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            OutputSortKey {
                primary: output_path_suffix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
                secondary: output_path_bases(path, unitigs, overlap),
                tertiary: 0,
            }
        }
        OutputSort::PrefixLength => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            OutputSortKey {
                primary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
                secondary: u64::MAX - output_path_bases(path, unitigs, overlap),
                tertiary: path.len() as u64,
            }
        }
        OutputSort::LengthBucketPrefix
        | OutputSort::LengthBucketPrefixFast
        | OutputSort::LengthBucket16Prefix
        | OutputSort::LengthBucket32Prefix
        | OutputSort::LengthBucket128Prefix
        | OutputSort::LengthBucket256Prefix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - bases / length_bucket_size(output_sort).unwrap_or(64),
                secondary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
                tertiary: u64::MAX - bases,
            }
        }
        OutputSort::LengthBucketPrefixSuffix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - bases / 64,
                secondary: output_path_prefix_suffix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
                tertiary: u64::MAX - bases,
            }
        }
        OutputSort::LengthBucketPrefixAc => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            let (ac, _) = output_path_ac_bases(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                record_reverse,
            );
            let density = if bases == 0 {
                0
            } else {
                ac.saturating_mul(1_000_000) / bases
            };
            OutputSortKey {
                primary: u64::MAX - bases / 64,
                secondary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
                tertiary: u64::MAX - density,
            }
        }
        OutputSort::LengthBucketUnitigsPrefix
        | OutputSort::LengthBucket16UnitigsPrefix
        | OutputSort::LengthBucket32UnitigsPrefix
        | OutputSort::LengthBucket128UnitigsPrefix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - bases / length_bucket_unitigs_size(output_sort).unwrap_or(64),
                secondary: u64::MAX - path.len() as u64,
                tertiary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
            }
        }
        OutputSort::LengthBucketUnitigsPrefixSuffix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - bases / 64,
                secondary: u64::MAX - path.len() as u64,
                tertiary: output_path_prefix_suffix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
            }
        }
        OutputSort::LengthUnitigsPrefix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - bases,
                secondary: u64::MAX - path.len() as u64,
                tertiary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
            }
        }
        OutputSort::UnitigsLengthPrefix => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let bases = output_path_bases(path, unitigs, overlap);
            OutputSortKey {
                primary: u64::MAX - path.len() as u64,
                secondary: u64::MAX - bases,
                tertiary: output_path_prefix_key(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverse,
                ),
            }
        }
        OutputSort::AcRichness => {
            let record_reverse = choose_record_reverse(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                strand_tiebreak,
            );
            let (ac, bases) = output_path_ac_bases(
                path,
                unitigs,
                links,
                overlap,
                sequence_store,
                record_reverse,
            );
            let density = if bases == 0 {
                0
            } else {
                ac.saturating_mul(1_000_000) / bases
            };
            OutputSortKey {
                primary: u64::MAX - density,
                secondary: u64::MAX - bases,
                tertiary: 0,
            }
        }
    }
}

fn output_path_bases(path: &[u32], unitigs: &[UnitigMeta], overlap: usize) -> u64 {
    path.iter()
        .enumerate()
        .map(|(path_idx, &unitig_id)| {
            let len = unitigs[unitig_id as usize].len as usize;
            if path_idx == 0 {
                len as u64
            } else {
                (len - overlap) as u64
            }
        })
        .sum()
}

fn output_path_ac_bases(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverse: bool,
) -> (u64, u64) {
    let mut ac = 0u64;
    let mut bases = 0u64;
    for (path_idx, &unitig_id) in path.iter().enumerate() {
        let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
        ac = ac.saturating_add(sequence_store.count_ac(
            segment.unitig,
            segment.start,
            segment.len,
            segment.reverse_complemented ^ record_reverse,
        ));
        bases = bases.saturating_add(segment.len as u64);
    }
    (ac, bases)
}

fn output_path_prefix_key(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverse: bool,
) -> u64 {
    let mut key = 0u64;
    let mut used = 0usize;
    if record_reverse {
        for (path_idx, &unitig_id) in path.iter().enumerate().rev() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            append_segment_prefix_key(
                sequence_store,
                segment,
                !segment.reverse_complemented,
                &mut key,
                &mut used,
            );
            if used == OUTPUT_SORT_PREFIX_BASES {
                break;
            }
        }
    } else {
        for (path_idx, &unitig_id) in path.iter().enumerate() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            append_segment_prefix_key(
                sequence_store,
                segment,
                segment.reverse_complemented,
                &mut key,
                &mut used,
            );
            if used == OUTPUT_SORT_PREFIX_BASES {
                break;
            }
        }
    }
    key << ((OUTPUT_SORT_PREFIX_BASES - used) * 2)
}

fn output_path_prefix_ac_count(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverse: bool,
) -> u64 {
    let mut count = 0u64;
    let mut used = 0usize;
    if record_reverse {
        for (path_idx, &unitig_id) in path.iter().enumerate().rev() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            append_segment_prefix_ac_count(
                sequence_store,
                segment,
                !segment.reverse_complemented,
                &mut count,
                &mut used,
            );
            if used == OUTPUT_SORT_PREFIX_BASES {
                break;
            }
        }
    } else {
        for (path_idx, &unitig_id) in path.iter().enumerate() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            append_segment_prefix_ac_count(
                sequence_store,
                segment,
                segment.reverse_complemented,
                &mut count,
                &mut used,
            );
            if used == OUTPUT_SORT_PREFIX_BASES {
                break;
            }
        }
    }
    count
}

fn output_path_prefix_suffix_key(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverse: bool,
) -> u64 {
    let prefix = output_path_prefix_key(
        path,
        unitigs,
        links,
        overlap,
        sequence_store,
        record_reverse,
    ) >> 32;
    let suffix = output_path_suffix_key(
        path,
        unitigs,
        links,
        overlap,
        sequence_store,
        record_reverse,
    ) >> 32;
    (prefix << 32) | suffix
}

fn append_segment_prefix_key(
    sequence_store: &SequenceStore,
    segment: OutputSegment,
    reverse_complemented: bool,
    key: &mut u64,
    used: &mut usize,
) {
    let take = (OUTPUT_SORT_PREFIX_BASES - *used).min(segment.len);
    for idx in 0..take {
        let base_idx = if reverse_complemented {
            segment.start + segment.len - 1 - idx
        } else {
            segment.start + idx
        };
        let mut bits = sequence_store.base_bits(segment.unitig, base_idx);
        if reverse_complemented {
            bits ^= 0b11;
        }
        *key = (*key << 2) | bits as u64;
        *used += 1;
    }
}

fn append_segment_prefix_ac_count(
    sequence_store: &SequenceStore,
    segment: OutputSegment,
    reverse_complemented: bool,
    count: &mut u64,
    used: &mut usize,
) {
    let take = (OUTPUT_SORT_PREFIX_BASES - *used).min(segment.len);
    for idx in 0..take {
        let base_idx = if reverse_complemented {
            segment.start + segment.len - 1 - idx
        } else {
            segment.start + idx
        };
        let mut bits = sequence_store.base_bits(segment.unitig, base_idx);
        if reverse_complemented {
            bits ^= 0b11;
        }
        *count += u64::from(packed_base_is_ac(bits));
        *used += 1;
    }
}

fn output_path_suffix_key(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverse: bool,
) -> u64 {
    let mut suffix = [0u8; OUTPUT_SORT_PREFIX_BASES];
    let mut used = 0usize;
    if record_reverse {
        for (path_idx, &unitig_id) in path.iter().enumerate() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            append_segment_suffix_bits(
                sequence_store,
                segment,
                !segment.reverse_complemented,
                &mut suffix,
                &mut used,
            );
            if used == OUTPUT_SORT_PREFIX_BASES {
                break;
            }
        }
    } else {
        for (path_idx, &unitig_id) in path.iter().enumerate().rev() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            append_segment_suffix_bits(
                sequence_store,
                segment,
                segment.reverse_complemented,
                &mut suffix,
                &mut used,
            );
            if used == OUTPUT_SORT_PREFIX_BASES {
                break;
            }
        }
    }

    let mut key = 0u64;
    for &bits in suffix[..used].iter().rev() {
        key = (key << 2) | bits as u64;
    }
    key << ((OUTPUT_SORT_PREFIX_BASES - used) * 2)
}

fn append_segment_suffix_bits(
    sequence_store: &SequenceStore,
    segment: OutputSegment,
    reverse_complemented: bool,
    suffix: &mut [u8; OUTPUT_SORT_PREFIX_BASES],
    used: &mut usize,
) {
    let take = (OUTPUT_SORT_PREFIX_BASES - *used).min(segment.len);
    for idx in 0..take {
        let base_idx = if reverse_complemented {
            segment.start + idx
        } else {
            segment.start + segment.len - 1 - idx
        };
        let mut bits = sequence_store.base_bits(segment.unitig, base_idx);
        if reverse_complemented {
            bits ^= 0b11;
        }
        suffix[*used] = bits;
        *used += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn write_output_paths<W: Write>(
    output_paths: &OutputPaths,
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverses: Option<&[bool]>,
    writer: &mut W,
    output_mode: OutputMode,
    strand_tiebreak: OutputStrandTieBreak,
    abundance_mode: Option<AbundanceMode>,
    stats: &mut OutputStats,
) -> Result<()> {
    let chunk_records = OUTPUT_RENDER_CHUNK_RECORDS.max(1);
    let batch_records = chunk_records.saturating_mul(OUTPUT_RENDER_BATCH_CHUNKS.max(1));
    for batch_start in (0..output_paths.paths.len()).step_by(batch_records) {
        let batch_end = (batch_start + batch_records).min(output_paths.paths.len());
        let chunk_starts: Vec<_> = (batch_start..batch_end).step_by(chunk_records).collect();
        let mut chunks = chunk_starts
            .into_par_iter()
            .map(|chunk_start| {
                let chunk_end = (chunk_start + chunk_records).min(batch_end);
                render_output_chunk(
                    chunk_start,
                    chunk_end,
                    output_paths,
                    unitigs,
                    abundance_runs,
                    links,
                    overlap,
                    sequence_store,
                    record_reverses,
                    output_mode,
                    strand_tiebreak,
                    abundance_mode,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        chunks.par_sort_unstable_by_key(|chunk| chunk.start_path);
        for chunk in chunks {
            writer.write_all(&chunk.bytes)?;
            stats.bases = stats.bases.saturating_add(chunk.bases);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_output_chunk(
    start_path: usize,
    end_path: usize,
    output_paths: &OutputPaths,
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverses: Option<&[bool]>,
    output_mode: OutputMode,
    strand_tiebreak: OutputStrandTieBreak,
    abundance_mode: Option<AbundanceMode>,
) -> Result<RenderedChunk> {
    let records = end_path - start_path;
    let mut bytes = Vec::with_capacity(records.saturating_mul(96));
    let mut buffer = Vec::with_capacity(SEQUENCE_RENDER_CHUNK_BASES);
    let mut bases = 0u64;

    for path_idx in start_path..end_path {
        let path = output_paths.path(output_paths.paths[path_idx]);
        bases = bases.saturating_add(render_path_record(
            (path_idx + 1) as u64,
            path,
            unitigs,
            abundance_runs,
            links,
            overlap,
            sequence_store,
            record_reverses.map(|values| values[path_idx]),
            &mut bytes,
            &mut buffer,
            output_mode,
            strand_tiebreak,
            abundance_mode,
        )?);
    }

    Ok(RenderedChunk {
        start_path,
        bases,
        bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn render_path_record<W: Write>(
    record_idx: u64,
    path: &[u32],
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    record_reverse: Option<bool>,
    writer: &mut W,
    buffer: &mut Vec<u8>,
    output_mode: OutputMode,
    strand_tiebreak: OutputStrandTieBreak,
    abundance_mode: Option<AbundanceMode>,
) -> Result<u64> {
    let record_reverse = record_reverse.unwrap_or_else(|| {
        choose_record_reverse(
            path,
            unitigs,
            links,
            overlap,
            sequence_store,
            strand_tiebreak,
        )
    });
    write_header(
        writer,
        record_idx,
        path,
        unitigs,
        abundance_runs,
        links,
        output_mode,
        record_reverse,
        abundance_mode,
    )?;

    let mut simplitig_bases = 0u64;
    let mut record_writer = FastaRecordWriter::new(writer);
    if record_reverse {
        for (path_idx, &unitig_id) in path.iter().enumerate().rev() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            write_output_segment(
                sequence_store,
                segment,
                !segment.reverse_complemented,
                &mut record_writer,
                buffer,
            )?;
            simplitig_bases = simplitig_bases.saturating_add(segment.len as u64);
        }
    } else {
        for (path_idx, &unitig_id) in path.iter().enumerate() {
            let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
            write_output_segment(
                sequence_store,
                segment,
                segment.reverse_complemented,
                &mut record_writer,
                buffer,
            )?;
            simplitig_bases = simplitig_bases.saturating_add(segment.len as u64);
        }
    }
    record_writer.finish()?;

    Ok(simplitig_bases)
}

#[derive(Clone, Copy)]
struct OutputSegment {
    unitig: UnitigMeta,
    start: usize,
    len: usize,
    reverse_complemented: bool,
}

fn choose_record_reverse(
    path: &[u32],
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
    sequence_store: &SequenceStore,
    strand_tiebreak: OutputStrandTieBreak,
) -> bool {
    let mut ac = 0u64;
    let mut bases = 0u64;
    for (path_idx, &unitig_id) in path.iter().enumerate() {
        let segment = output_segment(path_idx, unitig_id, unitigs, links, overlap);
        ac = ac.saturating_add(sequence_store.count_ac(
            segment.unitig,
            segment.start,
            segment.len,
            segment.reverse_complemented,
        ));
        bases = bases.saturating_add(segment.len as u64);
    }
    match ac.saturating_mul(2).cmp(&bases) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => match strand_tiebreak {
            OutputStrandTieBreak::None => false,
            OutputStrandTieBreak::Prefix => {
                output_path_prefix_key(path, unitigs, links, overlap, sequence_store, true)
                    < output_path_prefix_key(path, unitigs, links, overlap, sequence_store, false)
            }
            OutputStrandTieBreak::PrefixAc => {
                let reverse_ac = output_path_prefix_ac_count(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    true,
                );
                let forward_ac = output_path_prefix_ac_count(
                    path,
                    unitigs,
                    links,
                    overlap,
                    sequence_store,
                    false,
                );
                reverse_ac > forward_ac
                    || (reverse_ac == forward_ac
                        && output_path_prefix_key(
                            path,
                            unitigs,
                            links,
                            overlap,
                            sequence_store,
                            true,
                        ) < output_path_prefix_key(
                            path,
                            unitigs,
                            links,
                            overlap,
                            sequence_store,
                            false,
                        ))
            }
        },
    }
}

fn output_segment(
    path_idx: usize,
    unitig_id: u32,
    unitigs: &[UnitigMeta],
    links: &Links,
    overlap: usize,
) -> OutputSegment {
    let unitig_id = unitig_id as usize;
    let unitig = unitigs[unitig_id];
    let unitig_len = unitig.len as usize;
    if is_reverse_orientation(links.orientation[unitig_id]) {
        OutputSegment {
            unitig,
            start: 0,
            len: if path_idx == 0 {
                unitig_len
            } else {
                unitig_len - overlap
            },
            reverse_complemented: true,
        }
    } else {
        let start = if path_idx == 0 { 0 } else { overlap };
        OutputSegment {
            unitig,
            start,
            len: unitig_len - start,
            reverse_complemented: false,
        }
    }
}

fn write_output_segment<W: Write>(
    sequence_store: &SequenceStore,
    segment: OutputSegment,
    reverse_complemented: bool,
    writer: &mut FastaRecordWriter<'_, W>,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    if reverse_complemented {
        sequence_store.write_reverse_complemented(
            segment.unitig,
            segment.start,
            segment.len,
            writer,
            scratch,
        )
    } else {
        sequence_store.write_forward(segment.unitig, segment.start, segment.len, writer, scratch)
    }
}

fn write_header<W: Write>(
    writer: &mut W,
    _record_idx: u64,
    path: &[u32],
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    _output_mode: OutputMode,
    record_reverse: bool,
    abundance_mode: Option<AbundanceMode>,
) -> Result<()> {
    writer.write_all(b">A")?;
    match abundance_mode {
        Some(AbundanceMode::Mean) => {
            let mean = path_weighted_mean(path, unitigs);
            writer.write_all(b" km:f:")?;
            write_rounded_abundance(writer, mean)?;
        }
        Some(AbundanceMode::Runs) => {
            write!(writer, " km:f:")?;
            write_abundance_runs(writer, path, unitigs, abundance_runs, links, record_reverse)?;
        }
        None => {}
    }
    writer.write_all(b"\n")?;
    Ok(())
}

fn path_weighted_mean(path: &[u32], unitigs: &[UnitigMeta]) -> f64 {
    let mut weighted_sum = 0.0f64;
    let mut total = 0u64;
    for &unitig_id in path {
        let unitig = unitigs[unitig_id as usize];
        weighted_sum += unitig.mean_abundance * unitig.kmers as f64;
        total = total.saturating_add(unitig.kmers as u64);
    }
    if total == 0 {
        0.0
    } else {
        weighted_sum / total as f64
    }
}

fn write_abundance_runs<W: Write>(
    writer: &mut W,
    path: &[u32],
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    record_reverse: bool,
) -> Result<()> {
    let mut first = true;
    let mut pending: Option<AbundanceRun> = None;
    if record_reverse {
        for &unitig_id in path.iter().rev() {
            write_unitig_abundance_runs(
                writer,
                unitig_id,
                unitigs,
                abundance_runs,
                links,
                true,
                &mut first,
                &mut pending,
            )?;
        }
    } else {
        for &unitig_id in path {
            write_unitig_abundance_runs(
                writer,
                unitig_id,
                unitigs,
                abundance_runs,
                links,
                false,
                &mut first,
                &mut pending,
            )?;
        }
    }
    flush_abundance_run(writer, &mut first, pending)
}

#[allow(clippy::too_many_arguments)]
fn write_unitig_abundance_runs<W: Write>(
    writer: &mut W,
    unitig_id: u32,
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    record_reverse: bool,
    first: &mut bool,
    pending: &mut Option<AbundanceRun>,
) -> Result<()> {
    let unitig_id = unitig_id as usize;
    let unitig = unitigs[unitig_id];
    let abundance_start = unitig.abundance_start as usize;
    let abundance_len = unitig.abundance_len as usize;
    if abundance_len == 0 {
        append_abundance_run(
            writer,
            first,
            pending,
            AbundanceRun {
                value: unitig.mean_abundance,
                len: unitig.kmers as u64,
            },
        )?;
    } else {
        let runs = &abundance_runs[abundance_start..abundance_start + abundance_len];
        if is_reverse_orientation(links.orientation[unitig_id]) ^ record_reverse {
            for &run in runs.iter().rev() {
                append_abundance_run(writer, first, pending, run)?;
            }
        } else {
            for &run in runs {
                append_abundance_run(writer, first, pending, run)?;
            }
        }
    }
    Ok(())
}

fn append_abundance_run<W: Write>(
    writer: &mut W,
    first: &mut bool,
    pending: &mut Option<AbundanceRun>,
    run: AbundanceRun,
) -> Result<()> {
    let run = AbundanceRun {
        value: rounded_abundance(run.value),
        len: run.len,
    };
    if let Some(current) = pending {
        if current.value == run.value {
            current.len = current.len.saturating_add(run.len);
            return Ok(());
        }
        let finished = *current;
        flush_abundance_run(writer, first, Some(finished))?;
    }
    *pending = Some(run);
    Ok(())
}

fn flush_abundance_run<W: Write>(
    writer: &mut W,
    first: &mut bool,
    run: Option<AbundanceRun>,
) -> Result<()> {
    if let Some(run) = run {
        if !*first {
            writer.write_all(b":")?;
        }
        *first = false;
        write_rounded_abundance(writer, run.value)?;
        write!(writer, ":{}", run.len)?;
    }
    Ok(())
}

fn rounded_abundance(value: f64) -> f64 {
    value.round()
}

fn write_rounded_abundance<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    let mut formatted = FixedFormatBuffer::<64>::new();
    std::fmt::write(
        &mut formatted,
        format_args!("{:.0}", rounded_abundance(value)),
    )
    .map_err(|_| anyhow::anyhow!("formatted abundance exceeded fixed buffer"))?;
    writer.write_all(formatted.as_bytes())?;
    Ok(())
}

struct FixedFormatBuffer<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> FixedFormatBuffer<N> {
    fn new() -> Self {
        Self {
            bytes: [0; N],
            len: 0,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl<const N: usize> std::fmt::Write for FixedFormatBuffer<N> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        let bytes = value.as_bytes();
        if self.len + bytes.len() > N {
            return Err(std::fmt::Error);
        }
        self.bytes[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ParsedAbundance {
    mean: f64,
    runs: ParsedAbundanceRuns,
}

#[derive(Clone, Debug)]
enum ParsedAbundanceRuns {
    Simple,
    Runs(Vec<AbundanceRun>),
}

fn parse_abundance_runs(
    header: &[u8],
    kmer_count: u64,
    unitig_idx: usize,
) -> Result<ParsedAbundance> {
    let rest = abundance_payload(header, unitig_idx)?;
    parse_abundance_payload(rest, kmer_count, unitig_idx, true)
}

fn parse_abundance_mean(header: &[u8], kmer_count: u64, unitig_idx: usize) -> Result<f64> {
    let rest = abundance_payload(header, unitig_idx)?;
    parse_abundance_payload(rest, kmer_count, unitig_idx, false).map(|parsed| parsed.mean)
}

fn abundance_payload(header: &[u8], unitig_idx: usize) -> Result<&[u8]> {
    let mut pos = 0usize;
    let mut rest = None;
    while pos < header.len() {
        while pos < header.len() && is_ascii_whitespace_byte(header[pos]) {
            pos += 1;
        }
        let start = pos;
        while pos < header.len() && !is_ascii_whitespace_byte(header[pos]) {
            pos += 1;
        }
        if let Some(payload) = strip_abundance_prefix(&header[start..pos]) {
            rest = Some(payload);
            break;
        }
    }

    let Some(rest) = rest else {
        anyhow::bail!("unitig record {unitig_idx} header is missing km:f/ka:f abundance");
    };
    Ok(rest)
}

fn is_ascii_whitespace_byte(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn strip_abundance_prefix(token: &[u8]) -> Option<&[u8]> {
    token
        .strip_prefix(b"km:f:")
        .or_else(|| token.strip_prefix(b"ka:f:"))
}

fn parse_abundance_payload(
    rest: &[u8],
    kmer_count: u64,
    unitig_idx: usize,
    keep_runs: bool,
) -> Result<ParsedAbundance> {
    let mut fields = rest.split(|&byte| byte == b':');
    let Some(first) = fields.next() else {
        anyhow::bail!("unitig record {unitig_idx} has an empty km:f/ka:f abundance");
    };
    ensure!(
        !first.is_empty(),
        "unitig record {unitig_idx} has an empty km:f/ka:f abundance"
    );

    let Some(first_len) = fields.next() else {
        let value = parse_abundance_value(first, unitig_idx)?;
        return Ok(ParsedAbundance {
            mean: value,
            runs: if keep_runs {
                ParsedAbundanceRuns::Simple
            } else {
                ParsedAbundanceRuns::Runs(Vec::new())
            },
        });
    };

    let mut runs = Vec::new();
    let mut total = 0u64;
    let mut weighted_sum = 0.0f64;

    let mut value_field = first;
    let mut len_field = first_len;
    loop {
        let value = parse_abundance_value(value_field, unitig_idx)?;
        let len = parse_abundance_run_len(len_field, unitig_idx)?;
        total = total.saturating_add(len);
        weighted_sum += value * len as f64;
        if keep_runs {
            runs.push(AbundanceRun { value, len });
        }

        let Some(next_value) = fields.next() else {
            break;
        };
        let Some(next_len) = fields.next() else {
            anyhow::bail!(
                "unitig record {unitig_idx} has invalid km:f/ka:f run encoding; expected value:count pairs"
            );
        };
        value_field = next_value;
        len_field = next_len;
    }

    ensure!(
        total == kmer_count,
        "unitig record {unitig_idx} km:f/ka:f run lengths sum to {total}, expected {kmer_count}"
    );

    Ok(ParsedAbundance {
        mean: weighted_sum / total as f64,
        runs: if keep_runs {
            ParsedAbundanceRuns::Runs(runs)
        } else {
            ParsedAbundanceRuns::Runs(Vec::new())
        },
    })
}

fn parse_abundance_run_len(value: &[u8], unitig_idx: usize) -> Result<u64> {
    let Some(len) = parse_u64_bytes(value) else {
        anyhow::bail!("unitig record {unitig_idx} has invalid km:f/ka:f run length");
    };
    ensure!(
        len > 0,
        "unitig record {unitig_idx} has a zero-length km:f run"
    );
    Ok(len)
}

fn parse_abundance_value(value: &[u8], unitig_idx: usize) -> Result<f64> {
    let value = match parse_simple_decimal_bytes(value) {
        Some(value) => value,
        None => {
            let value = str::from_utf8(value).with_context(|| {
                format!("unitig record {unitig_idx} has invalid km:f/ka:f abundance")
            })?;
            value.parse::<f64>().with_context(|| {
                format!("unitig record {unitig_idx} has invalid km:f/ka:f abundance")
            })?
        }
    };
    ensure!(
        value.is_finite(),
        "unitig record {unitig_idx} has non-finite km:f/ka:f abundance"
    );
    Ok(value)
}

fn parse_u64_bytes(value: &[u8]) -> Option<u64> {
    if value.is_empty() {
        return None;
    }
    let mut parsed = 0u64;
    for &byte in value {
        if !byte.is_ascii_digit() {
            return None;
        }
        parsed = parsed
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
    }
    Some(parsed)
}

fn parse_simple_decimal_bytes(value: &[u8]) -> Option<f64> {
    if value.is_empty() {
        return None;
    }

    let mut pos = 0usize;
    let sign = match value[0] {
        b'+' => {
            pos = 1;
            1.0
        }
        b'-' => {
            pos = 1;
            -1.0
        }
        _ => 1.0,
    };

    let mut parsed = 0.0f64;
    let mut digits = 0usize;
    while pos < value.len() && value[pos].is_ascii_digit() {
        parsed = parsed * 10.0 + f64::from(value[pos] - b'0');
        pos += 1;
        digits += 1;
    }

    if pos < value.len() && value[pos] == b'.' {
        pos += 1;
        let mut scale = 0.1f64;
        while pos < value.len() && value[pos].is_ascii_digit() {
            parsed += f64::from(value[pos] - b'0') * scale;
            scale *= 0.1;
            pos += 1;
            digits += 1;
        }
    }

    if digits == 0 || pos != value.len() {
        return None;
    }
    Some(sign * parsed)
}

struct FastaRecordWriter<'a, W: Write> {
    writer: &'a mut W,
    has_sequence: bool,
}

impl<'a, W: Write> FastaRecordWriter<'a, W> {
    fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            has_sequence: false,
        }
    }

    fn write(&mut self, seq: &[u8]) -> Result<()> {
        if !seq.is_empty() {
            self.writer.write_all(seq)?;
            self.has_sequence = true;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.has_sequence {
            self.writer.write_all(b"\n")?;
            self.has_sequence = false;
        }
        Ok(())
    }

    fn write_reverse_complemented(&mut self, seq: &[u8], scratch: &mut Vec<u8>) -> Result<()> {
        const RC_CHUNK_BYTES: usize = 16 * 1024;
        let mut end = seq.len();
        while end != 0 {
            let start = end.saturating_sub(RC_CHUNK_BYTES);
            let chunk = &seq[start..end];
            scratch.clear();
            scratch.reserve(chunk.len());
            for &base in chunk.iter().rev() {
                scratch.push(complement_base(base));
            }
            self.write(&scratch[..])?;
            end = start;
        }
        Ok(())
    }
}

fn normalized_base(base: u8) -> Result<u8> {
    match base {
        b'A' | b'a' => Ok(b'A'),
        b'C' | b'c' => Ok(b'C'),
        b'G' | b'g' => Ok(b'G'),
        b'T' | b't' => Ok(b'T'),
        _ => anyhow::bail!("invalid base {}", base as char),
    }
}

fn raw_base_bits(base: u8, unitig_idx: usize) -> Result<u8> {
    let bits = RAW_BASE_BITS[base as usize];
    if bits < INVALID_BASE_BITS {
        Ok(bits)
    } else {
        invalid_raw_base(unitig_idx)
    }
}

fn invalid_raw_base(unitig_idx: usize) -> Result<u8> {
    anyhow::bail!("unitig record {unitig_idx} contains a non-ACGT base")
}

fn base_bits(base: u8) -> u8 {
    debug_assert!(matches!(base, b'A' | b'C' | b'G' | b'T'));
    ((base >> 1) ^ (base >> 2)) & 0b11
}

fn is_ac_base(base: u8) -> bool {
    matches!(base, b'A' | b'C')
}

fn packed_base_is_ac(bits: u8) -> bool {
    bits < 2
}

const fn raw_base_bits_table() -> [u8; 256] {
    let mut bits = [INVALID_BASE_BITS; 256];
    bits[b'A' as usize] = 0;
    bits[b'a' as usize] = 0;
    bits[b'C' as usize] = 1;
    bits[b'c' as usize] = 1;
    bits[b'G' as usize] = 2;
    bits[b'g' as usize] = 2;
    bits[b'T' as usize] = 3;
    bits[b't' as usize] = 3;
    bits
}

const fn packed_ac_counts() -> [u8; 256] {
    let mut counts = [0u8; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut count = 0u8;
        let mut shift = 0usize;
        while shift < 8 {
            if ((byte >> shift) & 0b11) < 2 {
                count += 1;
            }
            shift += 2;
        }
        counts[byte] = count;
        byte += 1;
    }
    counts
}

const fn packed_decoded_bases() -> [[u8; 4]; 256] {
    let mut decoded = [[0u8; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        decoded[byte][0] = base_from_bits(((byte >> 6) & 0b11) as u8);
        decoded[byte][1] = base_from_bits(((byte >> 4) & 0b11) as u8);
        decoded[byte][2] = base_from_bits(((byte >> 2) & 0b11) as u8);
        decoded[byte][3] = base_from_bits((byte & 0b11) as u8);
        byte += 1;
    }
    decoded
}

const fn packed_reverse_complement_bases() -> [[u8; 4]; 256] {
    let mut decoded = [[0u8; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        decoded[byte][0] = base_from_bits(((byte & 0b11) as u8) ^ 0b11);
        decoded[byte][1] = base_from_bits((((byte >> 2) & 0b11) as u8) ^ 0b11);
        decoded[byte][2] = base_from_bits((((byte >> 4) & 0b11) as u8) ^ 0b11);
        decoded[byte][3] = base_from_bits((((byte >> 6) & 0b11) as u8) ^ 0b11);
        byte += 1;
    }
    decoded
}

const fn base_from_bits(bits: u8) -> u8 {
    match bits {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => panic!("packed bases use two bits"),
    }
}

fn pack_four_bases(bases: &[u8]) -> u8 {
    (base_bits(bases[0]) << 6)
        | (base_bits(bases[1]) << 4)
        | (base_bits(bases[2]) << 2)
        | base_bits(bases[3])
}

#[inline(always)]
fn update_endpoint_encoding(
    bits: u8,
    pos: usize,
    overlap: usize,
    suffix_start: usize,
    prefix: &mut u128,
    suffix: &mut u128,
) {
    if pos < overlap {
        *prefix = (*prefix << 2) | bits as u128;
    }
    if pos >= suffix_start {
        *suffix = (*suffix << 2) | bits as u128;
    }
}

fn encode_bases(seq: &[u8]) -> u128 {
    seq.iter().fold(0u128, |encoded, &base| {
        (encoded << 2) | base_bits(base) as u128
    })
}

fn reverse_complement_encoded(mut encoded: u128, len: usize) -> u128 {
    let mut rc = 0u128;
    for _ in 0..len {
        rc = (rc << 2) | ((!encoded) & 0b11);
        encoded >>= 2;
    }
    rc
}

fn complement_base(base: u8) -> u8 {
    match base {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        _ => unreachable!("unitig bases are normalized before storage"),
    }
}

fn create_temp_dir(prefix: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{stamp}", process::id()));
    fs::create_dir(&path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(path)
}

fn log_phase(name: &str, elapsed: Duration) {
    eprintln!("R_PHASE\t{name}\t{:.6}", elapsed.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reform_text(
        input_text: &[u8],
        k: usize,
        abundance_mode: Option<AbundanceMode>,
    ) -> Result<(String, OutputStats)> {
        reform_text_with_mode(input_text, k, OutputMode::Simplitig, abundance_mode)
    }

    fn reform_text_with_mode(
        input_text: &[u8],
        k: usize,
        output_mode: OutputMode,
        abundance_mode: Option<AbundanceMode>,
    ) -> Result<(String, OutputStats)> {
        reform_text_with_store(
            input_text,
            k,
            output_mode,
            abundance_mode,
            SequenceStoreMode::Disk,
        )
    }

    fn reform_text_with_store(
        input_text: &[u8],
        k: usize,
        output_mode: OutputMode,
        abundance_mode: Option<AbundanceMode>,
        sequence_store_mode: SequenceStoreMode,
    ) -> Result<(String, OutputStats)> {
        reform_text_with_store_and_sort(
            input_text,
            k,
            output_mode,
            abundance_mode,
            sequence_store_mode,
            OutputSort::None,
        )
    }

    fn reform_text_with_store_and_sort(
        input_text: &[u8],
        k: usize,
        output_mode: OutputMode,
        abundance_mode: Option<AbundanceMode>,
        sequence_store_mode: SequenceStoreMode,
        output_sort: OutputSort,
    ) -> Result<(String, OutputStats)> {
        let dir = create_temp_dir("r-test").unwrap();
        let input = dir.join("unitigs.fa");
        let output = dir.join("out.fa");
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        fs::write(&input, input_text).unwrap();

        let result = reform_path(
            ReformerConfig {
                input,
                kmer_size: k,
                output: output.clone(),
                output_mode,
                sequence_store_mode,
                output_sort,
                strand_tiebreak: OutputStrandTieBreak::None,
                abundance_mode,
                zstd_workers: None,
                emit_logs: false,
            },
            &tmp,
        );
        let text = fs::read_to_string(&output).unwrap_or_default();
        let _ = fs::remove_dir_all(&dir);
        result.map(|stats| (text, stats))
    }

    fn reform_error(input_text: &[u8], k: usize, abundance_mode: Option<AbundanceMode>) -> String {
        reform_text(input_text, k, abundance_mode)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn reforms_unitigs_with_reverse_complement() {
        let (text, stats) = reform_text(
            b">u1 km:f:3.0\nAAACG\n>u2 km:f:4.0\nTTTCG\n",
            3,
            Some(AbundanceMode::Runs),
        )
        .unwrap();

        assert!(text.contains("CGAAAACG"));
        assert!(text.contains("km:f:4:3:3:3"));
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 2);
        assert_eq!(stats.bases, 8);
    }

    #[test]
    fn packed_memory_sequence_store_matches_disk_output() {
        let input =
            b">u1 km:f:1.0:1:2.0:2\nAAACG\n>u2 km:f:3.0:1:4.0:2\nTTTCG\n>u3 km:f:5.0:3\nGGGTA\n";
        let (disk_text, disk_stats) = reform_text_with_store(
            input,
            3,
            OutputMode::Simplitig,
            Some(AbundanceMode::Runs),
            SequenceStoreMode::Disk,
        )
        .unwrap();
        let (memory_text, memory_stats) = reform_text_with_store(
            input,
            3,
            OutputMode::Simplitig,
            Some(AbundanceMode::Runs),
            SequenceStoreMode::Memory,
        )
        .unwrap();

        assert_eq!(memory_text, disk_text);
        assert_eq!(memory_stats.simplitigs, disk_stats.simplitigs);
        assert_eq!(memory_stats.unitigs, disk_stats.unitigs);
        assert_eq!(memory_stats.bases, disk_stats.bases);
    }

    #[test]
    fn output_strand_reverses_sequence_and_run_abundance_to_maximize_ac() {
        let (text, stats) = reform_text_with_store(
            b">u1 km:f:1.0:1:2.0:1:3.0:1\nTTTTC\n",
            3,
            OutputMode::Simplitig,
            Some(AbundanceMode::Runs),
            SequenceStoreMode::Disk,
        )
        .unwrap();

        assert!(text.contains(">A km:f:3:1:2:1:1:1"));
        assert!(text.contains("GAAAA"));
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 1);
        assert_eq!(stats.bases, 5);
    }

    #[test]
    fn output_sequence_is_not_wrapped() {
        let seq = "A".repeat(120);
        let input = format!(">u1 km:f:1.0\n{seq}\n");
        let (text, stats) = reform_text(input.as_bytes(), 3, Some(AbundanceMode::Mean)).unwrap();
        let lines: Vec<_> = text.lines().collect();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1], seq);
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 1);
        assert_eq!(stats.bases, 120);
    }

    #[test]
    fn output_sort_sequence_prefix_orders_records() {
        let (text, stats) = reform_text_with_store_and_sort(
            b">u1 km:f:1.0\nTTTTC\n>u2 km:f:2.0\nCCCCG\n",
            5,
            OutputMode::Simplitig,
            Some(AbundanceMode::Mean),
            SequenceStoreMode::Disk,
            OutputSort::SequencePrefix,
        )
        .unwrap();

        assert_eq!(text.lines().nth(1), Some("CCCCG"));
        assert_eq!(stats.simplitigs, 2);
        assert_eq!(stats.unitigs, 2);
        assert_eq!(stats.bases, 10);
    }

    #[test]
    fn preserves_unitig_sequence_without_breaking() {
        let (text, stats) = reform_text(
            b">u1 km:f:3.0\nAAACG\n>u2 km:f:4.0\nCGTTT\n",
            3,
            Some(AbundanceMode::Mean),
        )
        .unwrap();

        assert!(text.contains("AAACGTTT"));
        assert!(text.contains(">A km:f:4"));
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 2);
        assert_eq!(stats.bases, 8);
    }

    #[test]
    fn unitig_mode_writes_unitig_headers_and_merges_unambiguous_paths() {
        let (text, stats) = reform_text_with_mode(
            b">u1 km:f:3.0\nAAACG\n>u2 km:f:4.0\nCGTTT\n",
            3,
            OutputMode::Unitig,
            Some(AbundanceMode::Mean),
        )
        .unwrap();

        assert!(text.contains(">A km:f:4"));
        assert!(text.contains("AAACGTTT"));
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 2);
        assert_eq!(stats.bases, 8);
    }

    #[test]
    fn unitig_mode_does_not_join_branching_overlaps() {
        let (text, stats) = reform_text_with_mode(
            b">u1 km:f:1.0\nCCAA\n>u2 km:f:2.0\nAAGG\n>u3 km:f:3.0\nAATT\n",
            3,
            OutputMode::Unitig,
            Some(AbundanceMode::Mean),
        )
        .unwrap();

        assert_eq!(
            text.lines().filter(|line| line.starts_with(">A ")).count(),
            3
        );
        assert!(!text.contains("CCAAGG"));
        assert!(!text.contains("CCAATT"));
        assert_eq!(stats.simplitigs, 3);
        assert_eq!(stats.unitigs, 3);
        assert_eq!(stats.bases, 12);
    }

    #[test]
    fn weighted_mean_uses_kmer_counts_not_unitig_counts() {
        let (text, _) = reform_text(
            b">u1 km:f:2.0\nAAAACG\n>u2 km:f:5.0\nCGTTT\n",
            3,
            Some(AbundanceMode::Mean),
        )
        .unwrap();

        assert!(text.contains("AAAACGTTT"));
        assert!(text.contains("km:f:3"));
    }

    #[test]
    fn run_abundance_coalesces_adjacent_equal_values() {
        let (text, _) = reform_text(
            b">u1 km:f:2.0:1:3.0:2\nAAACG\n>u2 km:f:3.0:1:4.0:2\nCGTTT\n",
            3,
            Some(AbundanceMode::Runs),
        )
        .unwrap();

        assert!(text.contains("AAACGTTT"));
        assert!(text.contains("km:f:2:1:3:3:4:2"));
    }

    #[test]
    fn run_abundance_reverses_when_unitig_is_reverse_complemented() {
        let (text, _) = reform_text(
            b">u1 km:f:1.0:1:2.0:2\nAAACG\n>u2 km:f:4.0:1:5.0:2\nTTTCG\n",
            3,
            Some(AbundanceMode::Runs),
        )
        .unwrap();

        assert!(text.contains("CGAAAACG"));
        assert!(text.contains("km:f:5:2:4:1:1:1:2:2"));
    }

    #[test]
    fn accepts_ka_abundance_tag() {
        let (text, stats) = reform_text(
            b">u1 ka:f:3.0\nAAACG\n>u2 ka:f:4.0\nCGTTT\n",
            3,
            Some(AbundanceMode::Mean),
        )
        .unwrap();

        assert!(text.contains("AAACGTTT"));
        assert!(text.contains("km:f:4"));
        assert_eq!(stats.simplitigs, 1);
    }

    #[test]
    fn headers_only_emit_fixed_name_and_rounded_abundance() {
        let (text, _) =
            reform_text(b">u1 km:f:12.5\nAAACG\n", 3, Some(AbundanceMode::Mean)).unwrap();
        let header = text.lines().next().unwrap();

        assert_eq!(header, ">A km:f:13");
        assert!(!header.contains("unitigs="));
        assert!(!header.contains("R_simplitig"));
        assert!(!header.contains("R_unitig"));

        let (text, _) = reform_text(
            b">u1 km:f:2.49:1:2.5:2\nAAACG\n",
            3,
            Some(AbundanceMode::Runs),
        )
        .unwrap();
        assert_eq!(text.lines().next().unwrap(), ">A km:f:2:1:3:2");
    }

    #[test]
    fn missing_abundance_is_rejected_when_abundance_mode_is_enabled() {
        let err = reform_error(b">u1\nAAACG\n", 3, Some(AbundanceMode::Mean));

        assert!(err.contains("missing km:f/ka:f abundance"));
    }

    #[test]
    fn missing_abundance_is_allowed_for_internal_topology_only_mode() {
        let (text, _) = reform_text(b">u1\nAAACG\n>u2\nCGTTT\n", 3, None).unwrap();

        assert!(text.contains(">A\n"));
        assert!(!text.contains("km:f:"));
        assert!(text.contains("AAACGTTT"));
    }

    #[test]
    fn invalid_abundance_run_sum_is_rejected() {
        let err = reform_error(
            b">u1 km:f:2.0:1:3.0:1\nAAACG\n",
            3,
            Some(AbundanceMode::Runs),
        );

        assert!(err.contains("run lengths sum to 2, expected 3"));
    }

    #[test]
    fn non_finite_abundance_is_rejected() {
        let err = reform_error(b">u1 km:f:NaN\nAAACG\n", 3, Some(AbundanceMode::Mean));

        assert!(err.contains("non-finite km:f/ka:f abundance"));
    }

    #[test]
    fn short_unitig_is_rejected() {
        let err = reform_error(b">u1 km:f:2.0\nAA\n", 3, Some(AbundanceMode::Mean));

        assert!(err.contains("shorter than k=3"));
    }

    #[test]
    fn non_acgt_unitig_is_rejected() {
        let err = reform_error(b">u1 km:f:2.0\nAANC\n", 3, Some(AbundanceMode::Mean));

        assert!(err.contains("contains a non-ACGT base"));
    }

    #[test]
    fn compressed_output_extension_is_honored() {
        use std::io::Read as _;

        let dir = create_temp_dir("r-test-compressed").unwrap();
        let input = dir.join("unitigs.fa");
        let output = dir.join("out.fa.zst");
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        fs::write(&input, b">u1 km:f:3.0\nAAACG\n").unwrap();

        reform_path(
            ReformerConfig {
                input,
                kmer_size: 3,
                output: output.clone(),
                output_mode: OutputMode::Simplitig,
                sequence_store_mode: SequenceStoreMode::Disk,
                output_sort: OutputSort::None,
                strand_tiebreak: OutputStrandTieBreak::None,
                abundance_mode: Some(AbundanceMode::Mean),
                zstd_workers: None,
                emit_logs: false,
            },
            &tmp,
        )
        .unwrap();

        let mut text = String::new();
        zstd::stream::read::Decoder::new(File::open(&output).unwrap())
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(text.contains(">A km:f:3"));
        assert!(text.contains("AAACG"));
    }
}
