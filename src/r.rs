use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use mc::{create_output_writer, with_xz_decompressed_path};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::str;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const OUTPUT_LINE_WIDTH: usize = 80;
const SEQUENCE_STORE_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(name = "R")]
#[command(about = "Reform unitig FASTA records into longer simplitigs using exact K-1 overlaps")]
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
        help = "Output FASTA simplitigs; .gz, .xz, and .zst are compressed"
    )]
    output: PathBuf,

    #[arg(short, long, help = "Number of Rayon worker threads")]
    threads: Option<usize>,

    #[arg(
        long,
        value_enum,
        default_value_t = AbundanceMode::Mean,
        help = "How to encode km:f abundance in output headers"
    )]
    abundance_mode: AbundanceMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AbundanceMode {
    Mean,
    Runs,
}

#[derive(Clone, Debug)]
pub struct ReformerConfig {
    pub input: PathBuf,
    pub kmer_size: usize,
    pub output: PathBuf,
    pub abundance_mode: Option<AbundanceMode>,
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
struct Endpoint {
    key: OverlapKey,
    node: u64,
}

impl Endpoint {
    fn new(key: OverlapKey, unitig_id: usize, reverse: bool) -> Self {
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
    len: u64,
    kmers: u64,
    mean_abundance: f64,
    abundance_start: usize,
    abundance_len: usize,
}

#[derive(Clone, Debug)]
struct ReadIndex {
    unitigs: Vec<UnitigMeta>,
    abundance_runs: Vec<AbundanceRun>,
    starts: Vec<Endpoint>,
    ends: Vec<Endpoint>,
    sequence_store_path: PathBuf,
    total_bases: u64,
}

#[derive(Clone, Debug)]
struct Links {
    orientation: Vec<Option<bool>>,
    incoming: Vec<Option<usize>>,
    outgoing: Vec<Option<usize>>,
    selected: u64,
}

#[derive(Clone, Debug)]
pub struct OutputStats {
    pub simplitigs: u64,
    pub unitigs: u64,
    pub bases: u64,
}

#[derive(Clone, Copy, Debug)]
struct AbundanceRun {
    value: f64,
    len: u64,
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent == value {
            value
        } else {
            let root = self.find(parent);
            self.parent[value] = root;
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
        self.parent[right_root] = left_root;
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
            abundance_mode: Some(cli.abundance_mode),
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
        config.abundance_mode,
    )?;
    if config.emit_logs {
        log_phase("1_read_and_endpoint_indexing", phase_started.elapsed());
        eprintln!("R_STAT\tinput_unitigs\t{}", read_index.unitigs.len());
        eprintln!("R_STAT\tinput_bases\t{}", read_index.total_bases);
    }

    let phase_started = Instant::now();
    let links = link_unitigs(read_index.starts, read_index.ends, read_index.unitigs.len());
    if config.emit_logs {
        log_phase("2_overlap_linking", phase_started.elapsed());
        eprintln!("R_STAT\tselected_overlaps\t{}", links.selected);
    }

    let phase_started = Instant::now();
    let output_stats = write_reformed_simplitigs(
        &read_index.sequence_store_path,
        &read_index.unitigs,
        &read_index.abundance_runs,
        &links,
        overlap,
        &config.output,
        config.abundance_mode,
    )?;
    if config.emit_logs {
        log_phase("3_output_streaming", phase_started.elapsed());
        eprintln!("R_STAT\toutput_simplitigs\t{}", output_stats.simplitigs);
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

fn read_unitigs(
    input_path: &Path,
    k: usize,
    overlap: usize,
    tmp_dir: &Path,
    abundance_mode: Option<AbundanceMode>,
) -> Result<ReadIndex> {
    let sequence_store_path = tmp_dir.join("unitigs.seq");
    let sequence_store = File::create(&sequence_store_path)
        .with_context(|| format!("failed to create {}", sequence_store_path.display()))?;
    let mut writer = BufWriter::with_capacity(SEQUENCE_STORE_BUFFER_BYTES, sequence_store);
    let mut unitigs = Vec::new();
    let mut abundance_runs = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut offset = 0u64;
    let mut total_bases = 0u64;

    for_unitig_records(input_path, |header, seq| {
        let unitig_idx = unitigs.len() + 1;
        ensure!(
            seq.len() >= k,
            "unitig record {unitig_idx} has length {}, shorter than k={k}",
            seq.len()
        );

        let mut normalized = Vec::with_capacity(seq.len());
        for &base in seq {
            normalized.push(
                normalized_base(base).with_context(|| {
                    format!("unitig record {unitig_idx} contains a non-ACGT base")
                })?,
            );
        }

        let prefix = encode_bases(&normalized[..overlap]);
        let suffix = encode_bases(&normalized[normalized.len() - overlap..]);
        let reverse_start = reverse_complement_encoded(suffix, overlap);
        let reverse_end = reverse_complement_encoded(prefix, overlap);

        let id = unitigs.len();
        let kmer_count = (normalized.len() - k + 1) as u64;
        let (mean_abundance, abundance_start, abundance_len) = if abundance_mode.is_some() {
            let parsed = parse_abundance_runs(header, kmer_count, unitig_idx)?;
            let abundance_start = abundance_runs.len();
            abundance_runs.extend_from_slice(&parsed.runs);
            (
                parsed.mean,
                abundance_start,
                abundance_runs.len() - abundance_start,
            )
        } else {
            (0.0, 0, 0)
        };

        starts.push(Endpoint::new(OverlapKey::from_encoded(prefix), id, false));
        ends.push(Endpoint::new(OverlapKey::from_encoded(suffix), id, false));
        starts.push(Endpoint::new(
            OverlapKey::from_encoded(reverse_start),
            id,
            true,
        ));
        ends.push(Endpoint::new(
            OverlapKey::from_encoded(reverse_end),
            id,
            true,
        ));

        writer.write_all(&normalized)?;
        unitigs.push(UnitigMeta {
            offset,
            len: normalized.len() as u64,
            kmers: kmer_count,
            mean_abundance,
            abundance_start,
            abundance_len,
        });
        offset = offset.saturating_add(normalized.len() as u64);
        total_bases = total_bases.saturating_add(normalized.len() as u64);
        Ok(())
    })?;

    writer.flush()?;
    ensure!(
        !unitigs.is_empty(),
        "input FASTA {} did not contain any unitig records",
        input_path.display()
    );

    Ok(ReadIndex {
        unitigs,
        abundance_runs,
        starts,
        ends,
        sequence_store_path,
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

fn link_unitigs(mut starts: Vec<Endpoint>, mut ends: Vec<Endpoint>, unitig_count: usize) -> Links {
    starts.par_sort_unstable_by(compare_endpoints);
    ends.par_sort_unstable_by(compare_endpoints);

    let mut orientation = vec![None; unitig_count];
    let mut incoming = vec![None; unitig_count];
    let mut outgoing = vec![None; unitig_count];
    let mut components = DisjointSet::new(unitig_count);
    let mut selected = 0u64;

    let mut start_pos = 0usize;
    let mut end_pos = 0usize;
    while start_pos < starts.len() && end_pos < ends.len() {
        match starts[start_pos].key.cmp(&ends[end_pos].key) {
            Ordering::Less => start_pos = next_key_range(&starts, start_pos).1,
            Ordering::Greater => end_pos = next_key_range(&ends, end_pos).1,
            Ordering::Equal => {
                let (start_begin, start_end) = next_key_range(&starts, start_pos);
                let (end_begin, end_end) = next_key_range(&ends, end_pos);
                selected = selected.saturating_add(select_group_links(
                    &starts[start_begin..start_end],
                    &ends[end_begin..end_end],
                    &mut orientation,
                    &mut incoming,
                    &mut outgoing,
                    &mut components,
                ));
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

fn compare_endpoints(left: &Endpoint, right: &Endpoint) -> Ordering {
    left.key.cmp(&right.key).then(left.node.cmp(&right.node))
}

fn next_key_range(endpoints: &[Endpoint], start: usize) -> (usize, usize) {
    let key = endpoints[start].key;
    let mut end = start + 1;
    while end < endpoints.len() && endpoints[end].key == key {
        end += 1;
    }
    (start, end)
}

fn select_group_links(
    starts: &[Endpoint],
    ends: &[Endpoint],
    orientation: &mut [Option<bool>],
    incoming: &mut [Option<usize>],
    outgoing: &mut [Option<usize>],
    components: &mut DisjointSet,
) -> u64 {
    let mut selected = 0u64;
    for &end in ends {
        let from = end.unitig_id();
        if outgoing[from].is_some() || !orientation_compatible(orientation[from], end.reverse()) {
            continue;
        }

        for &start in starts {
            let to = start.unitig_id();
            if from == to
                || incoming[to].is_some()
                || !orientation_compatible(orientation[to], start.reverse())
                || components.find(from) == components.find(to)
            {
                continue;
            }

            orientation[from].get_or_insert(end.reverse());
            orientation[to].get_or_insert(start.reverse());
            outgoing[from] = Some(to);
            incoming[to] = Some(from);
            components.union(from, to);
            selected += 1;
            break;
        }
    }
    selected
}

fn orientation_compatible(current: Option<bool>, candidate: bool) -> bool {
    current.is_none_or(|value| value == candidate)
}

fn write_reformed_simplitigs(
    sequence_store_path: &Path,
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    overlap: usize,
    output_path: &Path,
    abundance_mode: Option<AbundanceMode>,
) -> Result<OutputStats> {
    let mut sequence_store = File::open(sequence_store_path)
        .with_context(|| format!("failed to open {}", sequence_store_path.display()))?;
    let mut writer = create_output_writer(output_path)?;
    let mut visited = vec![false; unitigs.len()];
    let mut path = Vec::new();
    let mut buffer = Vec::new();
    let mut stats = OutputStats {
        simplitigs: 0,
        unitigs: 0,
        bases: 0,
    };

    for unitig_id in 0..unitigs.len() {
        if links.incoming[unitig_id].is_none() && !visited[unitig_id] {
            write_path(
                unitig_id,
                unitigs,
                abundance_runs,
                links,
                overlap,
                &mut sequence_store,
                &mut writer,
                &mut visited,
                &mut path,
                &mut buffer,
                &mut stats,
                abundance_mode,
            )?;
        }
    }

    for unitig_id in 0..unitigs.len() {
        if !visited[unitig_id] {
            write_path(
                unitig_id,
                unitigs,
                abundance_runs,
                links,
                overlap,
                &mut sequence_store,
                &mut writer,
                &mut visited,
                &mut path,
                &mut buffer,
                &mut stats,
                abundance_mode,
            )?;
        }
    }

    writer.finish()?;
    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn write_path<W: Write>(
    start: usize,
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    overlap: usize,
    sequence_store: &mut File,
    writer: &mut W,
    visited: &mut [bool],
    path: &mut Vec<usize>,
    buffer: &mut Vec<u8>,
    stats: &mut OutputStats,
    abundance_mode: Option<AbundanceMode>,
) -> Result<()> {
    path.clear();
    let mut current = Some(start);
    while let Some(unitig_id) = current {
        if visited[unitig_id] {
            break;
        }
        path.push(unitig_id);
        visited[unitig_id] = true;
        current = links.outgoing[unitig_id];
    }

    if path.is_empty() {
        return Ok(());
    }

    stats.simplitigs += 1;
    let simplitig_idx = stats.simplitigs;
    write_header(
        writer,
        simplitig_idx,
        path,
        unitigs,
        abundance_runs,
        links,
        abundance_mode,
    )?;

    let mut simplitig_bases = 0u64;
    let mut record_writer = FastaRecordWriter::new(writer);
    for (path_idx, &unitig_id) in path.iter().enumerate() {
        read_unitig_sequence(sequence_store, unitigs[unitig_id], buffer)?;
        if links.orientation[unitig_id].unwrap_or(false) {
            reverse_complement_in_place(buffer);
        }

        let seq = if path_idx == 0 {
            &buffer[..]
        } else {
            &buffer[overlap..]
        };
        record_writer.write(seq)?;
        simplitig_bases = simplitig_bases.saturating_add(seq.len() as u64);
    }
    record_writer.finish()?;

    stats.unitigs = stats.unitigs.saturating_add(path.len() as u64);
    stats.bases = stats.bases.saturating_add(simplitig_bases);
    Ok(())
}

fn read_unitig_sequence(file: &mut File, meta: UnitigMeta, buffer: &mut Vec<u8>) -> Result<()> {
    buffer.resize(meta.len as usize, 0);
    file.seek(SeekFrom::Start(meta.offset))?;
    file.read_exact(buffer)?;
    Ok(())
}

fn write_header<W: Write>(
    writer: &mut W,
    simplitig_idx: u64,
    path: &[usize],
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
    abundance_mode: Option<AbundanceMode>,
) -> Result<()> {
    write!(
        writer,
        ">R_simplitig_{simplitig_idx} unitigs={}",
        path.len()
    )?;
    match abundance_mode {
        Some(AbundanceMode::Mean) => {
            let mean = path_weighted_mean(path, unitigs);
            write!(writer, " km:f:{}", format_abundance(mean))?;
        }
        Some(AbundanceMode::Runs) => {
            write!(writer, " km:f:")?;
            write_abundance_runs(writer, path, unitigs, abundance_runs, links)?;
        }
        None => {}
    }
    writer.write_all(b"\n")?;
    Ok(())
}

fn path_weighted_mean(path: &[usize], unitigs: &[UnitigMeta]) -> f64 {
    let mut weighted_sum = 0.0f64;
    let mut total = 0u64;
    for &unitig_id in path {
        let unitig = unitigs[unitig_id];
        weighted_sum += unitig.mean_abundance * unitig.kmers as f64;
        total = total.saturating_add(unitig.kmers);
    }
    if total == 0 {
        0.0
    } else {
        weighted_sum / total as f64
    }
}

fn write_abundance_runs<W: Write>(
    writer: &mut W,
    path: &[usize],
    unitigs: &[UnitigMeta],
    abundance_runs: &[AbundanceRun],
    links: &Links,
) -> Result<()> {
    let mut first = true;
    let mut pending: Option<AbundanceRun> = None;
    for &unitig_id in path {
        let unitig = unitigs[unitig_id];
        let runs =
            &abundance_runs[unitig.abundance_start..unitig.abundance_start + unitig.abundance_len];
        if links.orientation[unitig_id].unwrap_or(false) {
            for &run in runs.iter().rev() {
                append_abundance_run(writer, &mut first, &mut pending, run)?;
            }
        } else {
            for &run in runs {
                append_abundance_run(writer, &mut first, &mut pending, run)?;
            }
        }
    }
    flush_abundance_run(writer, &mut first, pending)
}

fn append_abundance_run<W: Write>(
    writer: &mut W,
    first: &mut bool,
    pending: &mut Option<AbundanceRun>,
    run: AbundanceRun,
) -> Result<()> {
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
        write!(writer, "{}:{}", format_abundance(run.value), run.len)?;
    }
    Ok(())
}

fn format_abundance(value: f64) -> String {
    let mut formatted = format!("{value:.6}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.push('0');
    }
    formatted
}

#[derive(Clone, Debug)]
struct ParsedAbundance {
    mean: f64,
    runs: Vec<AbundanceRun>,
}

fn parse_abundance_runs(
    header: &[u8],
    kmer_count: u64,
    unitig_idx: usize,
) -> Result<ParsedAbundance> {
    let header = str::from_utf8(header)
        .with_context(|| format!("unitig record {unitig_idx} has a non-UTF8 header"))?;
    let Some(rest) = header
        .split_ascii_whitespace()
        .find_map(|token| token.strip_prefix("km:f:"))
    else {
        anyhow::bail!("unitig record {unitig_idx} header is missing km:f abundance");
    };

    let fields = rest.split(':').collect::<Vec<_>>();
    ensure!(
        !fields.is_empty(),
        "unitig record {unitig_idx} has an empty km:f abundance"
    );

    if fields.len() == 1 {
        let value = parse_abundance_value(fields[0], unitig_idx)?;
        return Ok(ParsedAbundance {
            mean: value,
            runs: vec![AbundanceRun {
                value,
                len: kmer_count,
            }],
        });
    }

    ensure!(
        fields.len() % 2 == 0,
        "unitig record {unitig_idx} has invalid km:f run encoding; expected value:count pairs"
    );

    let mut runs = Vec::with_capacity(fields.len() / 2);
    let mut total = 0u64;
    let mut weighted_sum = 0.0f64;
    for pair in fields.chunks_exact(2) {
        let value = parse_abundance_value(pair[0], unitig_idx)?;
        let len = pair[1]
            .parse::<u64>()
            .with_context(|| format!("unitig record {unitig_idx} has invalid km:f run length"))?;
        ensure!(
            len > 0,
            "unitig record {unitig_idx} has a zero-length km:f run"
        );
        total = total.saturating_add(len);
        weighted_sum += value * len as f64;
        runs.push(AbundanceRun { value, len });
    }
    ensure!(
        total == kmer_count,
        "unitig record {unitig_idx} km:f run lengths sum to {total}, expected {kmer_count}"
    );

    Ok(ParsedAbundance {
        mean: weighted_sum / total as f64,
        runs,
    })
}

fn parse_abundance_value(value: &str, unitig_idx: usize) -> Result<f64> {
    let value = value
        .parse::<f64>()
        .with_context(|| format!("unitig record {unitig_idx} has invalid km:f abundance"))?;
    ensure!(
        value.is_finite(),
        "unitig record {unitig_idx} has non-finite km:f abundance"
    );
    Ok(value)
}

struct FastaRecordWriter<'a, W: Write> {
    writer: &'a mut W,
    line_len: usize,
}

impl<'a, W: Write> FastaRecordWriter<'a, W> {
    fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            line_len: 0,
        }
    }

    fn write(&mut self, mut seq: &[u8]) -> Result<()> {
        while !seq.is_empty() {
            let remaining = OUTPUT_LINE_WIDTH - self.line_len;
            let take = remaining.min(seq.len());
            self.writer.write_all(&seq[..take])?;
            self.line_len += take;
            seq = &seq[take..];
            if self.line_len == OUTPUT_LINE_WIDTH {
                self.writer.write_all(b"\n")?;
                self.line_len = 0;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.line_len != 0 {
            self.writer.write_all(b"\n")?;
            self.line_len = 0;
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

fn base_bits(base: u8) -> u128 {
    match base {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => unreachable!("unitig bases are normalized before encoding"),
    }
}

fn encode_bases(seq: &[u8]) -> u128 {
    seq.iter()
        .fold(0u128, |encoded, &base| (encoded << 2) | base_bits(base))
}

fn reverse_complement_encoded(mut encoded: u128, len: usize) -> u128 {
    let mut rc = 0u128;
    for _ in 0..len {
        rc = (rc << 2) | ((!encoded) & 0b11);
        encoded >>= 2;
    }
    rc
}

fn reverse_complement_in_place(seq: &mut [u8]) {
    seq.reverse();
    for base in seq {
        *base = match *base {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            _ => unreachable!("unitig bases are normalized before storage"),
        };
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
                abundance_mode,
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
        assert!(text.contains("km:f:4.0:3:3.0:3"));
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 2);
        assert_eq!(stats.bases, 8);
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
        assert!(text.contains("km:f:3.5"));
        assert_eq!(stats.simplitigs, 1);
        assert_eq!(stats.unitigs, 2);
        assert_eq!(stats.bases, 8);
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
        assert!(text.contains("km:f:3.285714"));
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
        assert!(text.contains("km:f:2.0:1:3.0:3:4.0:2"));
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
        assert!(text.contains("km:f:5.0:2:4.0:1:1.0:1:2.0:2"));
    }

    #[test]
    fn missing_abundance_is_rejected_when_abundance_mode_is_enabled() {
        let err = reform_error(b">u1\nAAACG\n", 3, Some(AbundanceMode::Mean));

        assert!(err.contains("missing km:f abundance"));
    }

    #[test]
    fn missing_abundance_is_allowed_for_internal_topology_only_mode() {
        let (text, _) = reform_text(b">u1\nAAACG\n>u2\nCGTTT\n", 3, None).unwrap();

        assert!(text.contains(">R_simplitig_1 unitigs=2\n"));
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

        assert!(err.contains("non-finite km:f abundance"));
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
                abundance_mode: Some(AbundanceMode::Mean),
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

        assert!(text.contains(">R_simplitig_1 unitigs=1 km:f:3.0"));
        assert!(text.contains("AAACG"));
    }
}
