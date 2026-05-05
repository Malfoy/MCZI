use anyhow::{Context, Result, bail, ensure};
use flate2::Compression;
use flate2::write::GzEncoder;
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

struct RunConfig {
    out_dir: PathBuf,
    genome_path: PathBuf,
    depth: usize,
    read_len: usize,
    errors_per_read: usize,
    read_seed: u64,
}

impl RunConfig {
    fn parse() -> Result<Self> {
        let args: Vec<String> = env::args().collect();
        if args.len() != 7 {
            eprintln!(
                "usage: {} OUT_DIR GENOME_FASTA DEPTH READ_LEN ERROR_RATE READ_SEED",
                args.first()
                    .map(String::as_str)
                    .unwrap_or("simulate-reads-from-genome")
            );
            std::process::exit(2);
        }

        let out_dir = PathBuf::from(&args[1]);
        let genome_path = PathBuf::from(&args[2]);
        let depth = parse_usize("DEPTH", &args[3])?;
        let read_len = parse_usize("READ_LEN", &args[4])?;
        let error_rate = parse_f64("ERROR_RATE", &args[5])?;
        let read_seed = parse_u64("READ_SEED", &args[6])?;

        ensure!(depth > 0, "DEPTH must be greater than zero");
        ensure!(read_len > 0, "READ_LEN must be greater than zero");
        ensure!(
            error_rate.is_finite() && error_rate >= 0.0,
            "ERROR_RATE must be a finite non-negative value"
        );

        let errors_per_read = (read_len as f64 * error_rate).round() as usize;
        ensure!(
            errors_per_read <= read_len,
            "ERROR_RATE produces more errors than read bases"
        );

        Ok(Self {
            out_dir,
            genome_path,
            depth,
            read_len,
            errors_per_read,
            read_seed,
        })
    }
}

fn parse_usize(name: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("invalid {name} value {value:?}"))
}

fn parse_u64(name: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .with_context(|| format!("invalid {name} value {value:?}"))
}

fn parse_f64(name: &str, value: &str) -> Result<f64> {
    value
        .parse::<f64>()
        .with_context(|| format!("invalid {name} value {value:?}"))
}

fn main() -> Result<()> {
    let config = RunConfig::parse()?;
    let started = Instant::now();
    fs::create_dir_all(&config.out_dir)
        .with_context(|| format!("failed to create {}", config.out_dir.display()))?;

    eprintln!("loading genome: {}", config.genome_path.display());
    let genome = load_genome(&config.genome_path)?;
    ensure!(
        genome.len() >= config.read_len,
        "genome has {} bases but READ_LEN is {}",
        genome.len(),
        config.read_len
    );
    eprintln!("loaded genome: {} bp", genome.len());

    let read_count = checked_read_count(genome.len(), config.depth, config.read_len)?;
    let requested_bases = read_count
        .checked_mul(config.read_len)
        .context("read_count * read_len overflows usize")?;
    if requested_bases != genome.len() * config.depth {
        eprintln!(
            "warning: depth is truncated to {requested_bases} read bases because read count must be integral"
        );
    }

    let reads_path = config.out_dir.join("reads.fa.gz");
    let tmp_reads_path = config.out_dir.join("reads.fa.gz.tmp");
    let manifest_path = config.out_dir.join("simulation_manifest.txt");

    write_reads_fasta_gz(&tmp_reads_path, &genome, &config, read_count)?;
    fs::rename(&tmp_reads_path, &reads_path).with_context(|| {
        format!(
            "failed to rename {} to {}",
            tmp_reads_path.display(),
            reads_path.display()
        )
    })?;
    write_manifest(
        &manifest_path,
        &config,
        genome.len(),
        &reads_path,
        read_count,
        requested_bases,
        started.elapsed().as_secs_f64(),
    )?;

    eprintln!(
        "done: reads={}, manifest={}, elapsed={:.3}s",
        reads_path.display(),
        manifest_path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn load_genome(path: &Path) -> Result<Vec<u8>> {
    const CONFIG: Config = ParserOptions::default()
        .dna_string()
        .ignore_headers()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let mut parser = FastxParser::<CONFIG>::from_file(path)
        .with_context(|| format!("failed to open genome {}", path.display()))?;
    let mut genome = Vec::with_capacity(estimate_genome_bases(path));

    while parser.next().is_some() {
        genome.extend_from_slice(parser.get_dna_string());
    }

    if genome.is_empty() {
        bail!(
            "genome {} did not contain any A/C/G/T bases",
            path.display()
        );
    }
    Ok(genome)
}

fn estimate_genome_bases(path: &Path) -> usize {
    fs::metadata(path)
        .map(|metadata| metadata.len().min(usize::MAX as u64) as usize)
        .unwrap_or(0)
}

fn checked_read_count(genome_len: usize, depth: usize, read_len: usize) -> Result<usize> {
    let requested_bases = genome_len
        .checked_mul(depth)
        .context("genome_len * depth overflows usize")?;
    Ok(requested_bases / read_len)
}

fn write_reads_fasta_gz(
    path: &Path,
    genome: &[u8],
    config: &RunConfig,
    read_count: usize,
) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let writer = BufWriter::with_capacity(64 * 1024 * 1024, file);
    let mut writer = GzEncoder::new(writer, Compression::fast());
    let mut rng = SplitMix64::new(config.read_seed);
    let mut read = vec![0u8; config.read_len];
    let mut positions = Vec::with_capacity(config.errors_per_read);
    let start_upper = genome.len() - config.read_len + 1;

    for read_id in 0..read_count {
        let start = rng.gen_range(start_upper);
        read.copy_from_slice(&genome[start..start + config.read_len]);
        add_substitution_errors(&mut read, config.errors_per_read, &mut rng, &mut positions);

        writeln!(writer, ">read_{read_id} start={start}")?;
        writer.write_all(&read)?;
        writer.write_all(b"\n")?;

        if read_id > 0 && read_id % 100_000 == 0 {
            eprintln!("wrote {read_id}/{read_count} reads");
        }
    }

    writer.try_finish()?;
    Ok(())
}

fn add_substitution_errors(
    read: &mut [u8],
    errors_per_read: usize,
    rng: &mut SplitMix64,
    positions: &mut Vec<usize>,
) {
    positions.clear();
    while positions.len() < errors_per_read {
        let pos = rng.gen_range(read.len());
        if positions.contains(&pos) {
            continue;
        }
        positions.push(pos);
    }

    for &pos in positions.iter() {
        read[pos] = substitute_base(read[pos], rng.gen_range(3));
    }
}

fn substitute_base(base: u8, offset: usize) -> u8 {
    match base {
        b'A' => [b'C', b'G', b'T'][offset],
        b'C' => [b'A', b'G', b'T'][offset],
        b'G' => [b'A', b'C', b'T'][offset],
        b'T' => [b'A', b'C', b'G'][offset],
        _ => BASES[offset],
    }
}

fn write_manifest(
    path: &Path,
    config: &RunConfig,
    genome_bases: usize,
    reads_path: &Path,
    read_count: usize,
    read_bases: usize,
    elapsed_seconds: f64,
) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "source_genome={}", config.genome_path.display())?;
    writeln!(writer, "genome_bases={genome_bases}")?;
    writeln!(writer, "depth={}", config.depth)?;
    writeln!(writer, "read_len={}", config.read_len)?;
    writeln!(writer, "read_count={read_count}")?;
    writeln!(writer, "read_bases={read_bases}")?;
    writeln!(writer, "read_output={}", reads_path.display())?;
    writeln!(writer, "read_format=fasta_gzip")?;
    writeln!(writer, "gzip_level=fast")?;
    writeln!(writer, "error_model=substitution")?;
    writeln!(writer, "errors_per_read={}", config.errors_per_read)?;
    writeln!(
        writer,
        "realized_error_rate={:.8}",
        config.errors_per_read as f64 / config.read_len as f64
    )?;
    writeln!(writer, "read_seed={}", config.read_seed)?;
    writeln!(writer, "elapsed_seconds={elapsed_seconds:.3}")?;
    Ok(())
}
