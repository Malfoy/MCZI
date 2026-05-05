use anyhow::{Context, Result, ensure};
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use rayon::prelude::*;
use simd_minimizers::packed_seq::{AsciiSeq, Seq};
use simd_minimizers::seq_hash::AntiLexHasher;
use std::env;
use std::path::Path;
use std::time::Instant;

const BATCH_BASES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
struct ConfigArgs {
    k: usize,
    m: usize,
    mode: Mode,
}

#[derive(Clone, Copy, Default)]
struct ScanStats {
    reads: u64,
    bases: u64,
    runs: u64,
    checksum: u128,
}

impl ScanStats {
    fn add(self, other: Self) -> Self {
        Self {
            reads: self.reads + other.reads,
            bases: self.bases + other.bases,
            runs: self.runs + other.runs,
            checksum: self.checksum ^ other.checksum.rotate_left((self.runs & 127) as u32),
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Parse,
    PackedParse,
    Copy,
    PositionsNoSuper,
    Positions,
    PositionsReuse,
    Values64,
    Values64Reuse,
    Values64AntiLex,
    Values128,
    ScalarPositions,
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "parse" => Ok(Self::Parse),
            "packed-parse" => Ok(Self::PackedParse),
            "copy" => Ok(Self::Copy),
            "positions-nosuper" => Ok(Self::PositionsNoSuper),
            "positions" => Ok(Self::Positions),
            "positions-reuse" => Ok(Self::PositionsReuse),
            "values64" => Ok(Self::Values64),
            "values64-reuse" => Ok(Self::Values64Reuse),
            "values64-antilex" => Ok(Self::Values64AntiLex),
            "values128" => Ok(Self::Values128),
            "scalar-positions" => Ok(Self::ScalarPositions),
            _ => bail_mode(value),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::PackedParse => "packed-parse",
            Self::Copy => "copy",
            Self::PositionsNoSuper => "positions-nosuper",
            Self::Positions => "positions",
            Self::PositionsReuse => "positions-reuse",
            Self::Values64 => "values64",
            Self::Values64Reuse => "values64-reuse",
            Self::Values64AntiLex => "values64-antilex",
            Self::Values128 => "values128",
            Self::ScalarPositions => "scalar-positions",
        }
    }
}

fn bail_mode(value: &str) -> Result<Mode> {
    anyhow::bail!(
        "unknown mode {value}; expected parse, packed-parse, copy, positions-nosuper, positions, positions-reuse, values64, values64-reuse, values64-antilex, values128, or scalar-positions"
    )
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mode = Mode::parse(
        &args
            .next()
            .context("usage: mc-minimizer-scan <mode> <reads.fa> [k=31] [m=21] [threads=8]")?,
    )?;
    let input = args
        .next()
        .context("usage: mc-minimizer-scan <mode> <reads.fa> [k=31] [m=21] [threads=8]")?;
    let k = parse_arg(args.next(), 31, "k")?;
    let m = parse_arg(args.next(), 21, "m")?;
    let threads = parse_arg(args.next(), 8, "threads")?;
    ensure!(args.next().is_none(), "too many arguments");
    ensure!(k > 0, "k must be greater than 0");
    ensure!(m > 0 && m <= k, "m must be in 1..=k");

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("failed to configure rayon thread pool")?;

    let config = ConfigArgs { k, m, mode };
    let started = Instant::now();
    let stats = match mode {
        Mode::Parse => parse_file(Path::new(&input), config)?,
        Mode::PackedParse => parse_packed_file(Path::new(&input), config)?,
        _ => scan_file(Path::new(&input), config)?,
    };
    let elapsed = started.elapsed().as_secs_f64();

    println!("mode\t{}", mode.as_str());
    println!("reads\t{}", stats.reads);
    println!("bases\t{}", stats.bases);
    println!("minimizer_runs\t{}", stats.runs);
    println!("checksum\t{:032x}", stats.checksum);
    println!("elapsed_seconds\t{elapsed:.6}");
    eprintln!("MC_MINIMIZER_SCAN\t{}\t{elapsed:.6}", mode.as_str());

    Ok(())
}

fn parse_arg(value: Option<String>, default: usize, name: &str) -> Result<usize> {
    match value {
        Some(value) => value
            .parse()
            .with_context(|| format!("failed to parse {name} as usize: {value}")),
        None => Ok(default),
    }
}

fn parse_file(path: &Path, config: ConfigArgs) -> Result<ScanStats> {
    let mut total = ScanStats::default();

    const CONFIG: Config = ParserOptions::default()
        .dna_string()
        .ignore_headers()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open FASTA/FASTQ input {}", path.display()))?;

    while parser.next().is_some() {
        let seq = parser.get_dna_string();
        if seq.len() < config.k {
            continue;
        }
        total.reads += 1;
        total.bases += seq.len() as u64;
        total.checksum ^= (seq.len() as u128).rotate_left((total.reads & 127) as u32);
    }

    Ok(total)
}

fn parse_packed_file(path: &Path, config: ConfigArgs) -> Result<ScanStats> {
    let mut total = ScanStats::default();

    const CONFIG: Config = ParserOptions::default()
        .dna_packed()
        .ignore_headers()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open FASTA/FASTQ input {}", path.display()))?;

    while parser.next().is_some() {
        let seq = parser.get_packed_seq();
        if seq.len() < config.k {
            continue;
        }
        total.reads += 1;
        total.bases += seq.len() as u64;
        total.checksum ^= (seq.len() as u128).rotate_left((total.reads & 127) as u32);
    }

    Ok(total)
}

fn scan_file(path: &Path, config: ConfigArgs) -> Result<ScanStats> {
    let mut total = ScanStats::default();
    for_fastx_batches(path, config.k, |batch| {
        let batch_stats = match config.mode {
            Mode::Copy => batch
                .iter()
                .map(|seq| copy_seq(seq, config))
                .fold(ScanStats::default(), ScanStats::add),
            Mode::PositionsReuse | Mode::Values64Reuse => batch
                .par_iter()
                .fold(ReusableScan::default, |mut scanner, seq| {
                    scanner.stats = scanner.stats.add(scanner.scan_seq(seq, config));
                    scanner
                })
                .map(|scanner| scanner.stats)
                .reduce(ScanStats::default, ScanStats::add),
            _ => batch
                .par_iter()
                .map(|seq| scan_seq(seq, config))
                .reduce(ScanStats::default, ScanStats::add),
        };
        total = total.add(batch_stats);
        Ok(())
    })?;
    Ok(total)
}

fn copy_seq(seq: &[u8], _config: ConfigArgs) -> ScanStats {
    ScanStats {
        reads: 1,
        bases: seq.len() as u64,
        runs: 0,
        checksum: (seq.len() as u128).rotate_left((seq[0] & 127) as u32),
    }
}

#[derive(Default)]
struct ReusableScan {
    minimizer_positions: Vec<u32>,
    super_kmer_starts: Vec<u32>,
    stats: ScanStats,
}

impl ReusableScan {
    fn scan_seq(&mut self, seq: &[u8], config: ConfigArgs) -> ScanStats {
        if seq.len() < config.k {
            return ScanStats::default();
        }

        self.minimizer_positions.clear();
        self.super_kmer_starts.clear();

        let window = config.k - config.m + 1;
        let ascii_seq = AsciiSeq(seq);
        let output = simd_minimizers::canonical_minimizers(config.m, window)
            .super_kmers(&mut self.super_kmer_starts)
            .run(ascii_seq, &mut self.minimizer_positions);

        let mut checksum = 0u128;
        match config.mode {
            Mode::PositionsReuse => {
                for (idx, &pos) in self.minimizer_positions.iter().enumerate() {
                    checksum ^= (pos as u128).rotate_left((idx & 127) as u32);
                }
            }
            Mode::Values64Reuse => {
                for (idx, minimizer) in output.values_u64().enumerate() {
                    checksum ^= (minimizer as u128).rotate_left((idx & 127) as u32);
                }
            }
            _ => unreachable!("reusable scan only handles reuse modes"),
        }

        ScanStats {
            reads: 1,
            bases: seq.len() as u64,
            runs: self.minimizer_positions.len() as u64,
            checksum,
        }
    }
}

fn scan_seq(seq: &[u8], config: ConfigArgs) -> ScanStats {
    if seq.len() < config.k {
        return ScanStats::default();
    }

    let window = config.k - config.m + 1;
    let ascii_seq = AsciiSeq(seq);
    let mut minimizer_positions = Vec::new();
    let mut super_kmer_starts = Vec::new();

    if let Mode::PositionsNoSuper = config.mode {
        simd_minimizers::canonical_minimizers(config.m, window)
            .run(ascii_seq, &mut minimizer_positions);

        let mut checksum = 0u128;
        for (idx, &pos) in minimizer_positions.iter().enumerate() {
            checksum ^= (pos as u128).rotate_left((idx & 127) as u32);
        }

        return ScanStats {
            reads: 1,
            bases: seq.len() as u64,
            runs: minimizer_positions.len() as u64,
            checksum,
        };
    }

    if let Mode::ScalarPositions = config.mode {
        simd_minimizers::canonical_minimizers(config.m, window)
            .super_kmers(&mut super_kmer_starts)
            .run_scalar(ascii_seq, &mut minimizer_positions);

        let mut checksum = 0u128;
        for (idx, &pos) in minimizer_positions.iter().enumerate() {
            checksum ^= (pos as u128).rotate_left((idx & 127) as u32);
        }

        return ScanStats {
            reads: 1,
            bases: seq.len() as u64,
            runs: minimizer_positions.len() as u64,
            checksum,
        };
    }

    if let Mode::Values64AntiLex = config.mode {
        let hasher = AntiLexHasher::<true>::new(config.m);
        let output = simd_minimizers::canonical_minimizers(config.m, window)
            .hasher(&hasher)
            .super_kmers(&mut super_kmer_starts)
            .run(ascii_seq, &mut minimizer_positions);

        let mut checksum = 0u128;
        for (idx, minimizer) in output.values_u64().enumerate() {
            checksum ^= (minimizer as u128).rotate_left((idx & 127) as u32);
        }

        return ScanStats {
            reads: 1,
            bases: seq.len() as u64,
            runs: minimizer_positions.len() as u64,
            checksum,
        };
    }

    let output = simd_minimizers::canonical_minimizers(config.m, window)
        .super_kmers(&mut super_kmer_starts)
        .run(ascii_seq, &mut minimizer_positions);

    let mut checksum = 0u128;
    match config.mode {
        Mode::PositionsNoSuper | Mode::Positions => {
            for (idx, &pos) in minimizer_positions.iter().enumerate() {
                checksum ^= (pos as u128).rotate_left((idx & 127) as u32);
            }
        }
        Mode::Values64 => {
            for (idx, minimizer) in output.values_u64().enumerate() {
                checksum ^= (minimizer as u128).rotate_left((idx & 127) as u32);
            }
        }
        Mode::Values64AntiLex => unreachable!("anti-lex mode returns before default hasher run"),
        Mode::Values128 => {
            for (idx, minimizer) in output.values_u128().enumerate() {
                checksum ^= minimizer.rotate_left((idx & 127) as u32);
            }
        }
        Mode::ScalarPositions => unreachable!("scalar-positions returns before SIMD run"),
        Mode::PositionsReuse | Mode::Values64Reuse => {
            unreachable!("reuse modes do not call scan_seq")
        }
        Mode::Parse | Mode::PackedParse | Mode::Copy => {
            unreachable!("parse/copy modes do not call scan_seq")
        }
    }

    ScanStats {
        reads: 1,
        bases: seq.len() as u64,
        runs: minimizer_positions.len() as u64,
        checksum,
    }
}

fn for_fastx_batches<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(&[Vec<u8>]) -> Result<()>,
{
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
