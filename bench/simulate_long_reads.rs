use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
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

struct Config {
    out_dir: PathBuf,
    genome_len: usize,
    depth: usize,
    read_len: usize,
    errors_per_read: usize,
    genome_seed: u64,
    read_seed: u64,
}

impl Config {
    fn parse() -> io::Result<Self> {
        let args: Vec<String> = env::args().collect();
        if args.len() != 8 {
            eprintln!(
                "usage: {} OUT_DIR GENOME_LEN DEPTH READ_LEN ERROR_RATE GENOME_SEED READ_SEED",
                args.first().map(String::as_str).unwrap_or("simulate_long_reads")
            );
            std::process::exit(2);
        }

        let out_dir = PathBuf::from(&args[1]);
        let genome_len = parse_usize("GENOME_LEN", &args[2])?;
        let depth = parse_usize("DEPTH", &args[3])?;
        let read_len = parse_usize("READ_LEN", &args[4])?;
        let error_rate = parse_f64("ERROR_RATE", &args[5])?;
        let genome_seed = parse_u64("GENOME_SEED", &args[6])?;
        let read_seed = parse_u64("READ_SEED", &args[7])?;

        if genome_len < read_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GENOME_LEN must be at least READ_LEN",
            ));
        }
        if genome_len.checked_mul(depth).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GENOME_LEN * DEPTH overflows usize",
            ));
        }

        let errors_per_read = (read_len as f64 * error_rate).round() as usize;
        if errors_per_read > read_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ERROR_RATE produces more errors than read bases",
            ));
        }

        Ok(Self {
            out_dir,
            genome_len,
            depth,
            read_len,
            errors_per_read,
            genome_seed,
            read_seed,
        })
    }
}

fn parse_usize(name: &str, value: &str) -> io::Result<usize> {
    value.parse::<usize>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {name} value {value:?}: {err}"),
        )
    })
}

fn parse_u64(name: &str, value: &str) -> io::Result<u64> {
    value.parse::<u64>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {name} value {value:?}: {err}"),
        )
    })
}

fn parse_f64(name: &str, value: &str) -> io::Result<f64> {
    value.parse::<f64>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {name} value {value:?}: {err}"),
        )
    })
}

fn main() -> io::Result<()> {
    let config = Config::parse()?;
    std::fs::create_dir_all(&config.out_dir)?;

    let genome_path = config.out_dir.join("genome_3g.fa");
    let reads_path = config.out_dir.join("reads_3g_10x_10kb_e0.001.fa");
    let manifest_path = config.out_dir.join("simulation_manifest.txt");

    let started = Instant::now();
    eprintln!(
        "generating genome: {} bp, read length {}, depth {}, substitutions/read {}",
        config.genome_len, config.read_len, config.depth, config.errors_per_read
    );

    let mut genome = vec![0u8; config.genome_len];
    fill_random_genome(&mut genome, config.genome_seed);
    write_genome_fasta(&genome_path, &genome)?;

    let read_count = config.genome_len * config.depth / config.read_len;
    let requested_bases = read_count * config.read_len;
    if requested_bases != config.genome_len * config.depth {
        eprintln!(
            "warning: depth is truncated to {} read bases because read count must be integral",
            requested_bases
        );
    }

    write_reads_fasta(&reads_path, &genome, &config, read_count)?;
    write_manifest(
        &manifest_path,
        &config,
        read_count,
        requested_bases,
        started.elapsed().as_secs_f64(),
    )?;

    eprintln!(
        "done: genome={}, reads={}, elapsed={:.3}s",
        genome_path.display(),
        reads_path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn fill_random_genome(genome: &mut [u8], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let mut bits = 0u64;
    let mut available = 0usize;

    for base in genome {
        if available < 2 {
            bits = rng.next_u64();
            available = 64;
        }
        *base = BASES[(bits & 3) as usize];
        bits >>= 2;
        available -= 2;
    }
}

fn write_genome_fasta(path: &PathBuf, genome: &[u8]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);
    writer.write_all(b">random_genome_3g\n")?;
    for chunk in genome.chunks(80) {
        writer.write_all(chunk)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()
}

fn write_reads_fasta(
    path: &PathBuf,
    genome: &[u8],
    config: &Config,
    read_count: usize,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, file);
    let mut rng = SplitMix64::new(config.read_seed);
    let mut read = vec![0u8; config.read_len];
    let start_upper = genome.len() - config.read_len + 1;

    for read_id in 0..read_count {
        let start = rng.gen_range(start_upper);
        read.copy_from_slice(&genome[start..start + config.read_len]);
        add_substitution_errors(&mut read, config.errors_per_read, &mut rng);

        writeln!(writer, ">read_{read_id} start={start}")?;
        writer.write_all(&read)?;
        writer.write_all(b"\n")?;

        if read_id > 0 && read_id % 100_000 == 0 {
            eprintln!("wrote {read_id}/{read_count} reads");
        }
    }
    writer.flush()
}

fn add_substitution_errors(read: &mut [u8], errors_per_read: usize, rng: &mut SplitMix64) {
    let mut positions = [usize::MAX; 64];
    assert!(
        errors_per_read <= positions.len(),
        "increase positions scratch buffer"
    );

    let mut filled = 0usize;
    while filled < errors_per_read {
        let pos = rng.gen_range(read.len());
        if positions[..filled].contains(&pos) {
            continue;
        }
        positions[filled] = pos;
        filled += 1;
    }

    for &pos in &positions[..errors_per_read] {
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
    path: &PathBuf,
    config: &Config,
    read_count: usize,
    read_bases: usize,
    elapsed_seconds: f64,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "genome_len={}", config.genome_len)?;
    writeln!(writer, "depth={}", config.depth)?;
    writeln!(writer, "read_len={}", config.read_len)?;
    writeln!(writer, "read_count={read_count}")?;
    writeln!(writer, "read_bases={read_bases}")?;
    writeln!(writer, "error_model=substitution")?;
    writeln!(writer, "errors_per_read={}", config.errors_per_read)?;
    writeln!(
        writer,
        "realized_error_rate={:.8}",
        config.errors_per_read as f64 / config.read_len as f64
    )?;
    writeln!(writer, "genome_seed={}", config.genome_seed)?;
    writeln!(writer, "read_seed={}", config.read_seed)?;
    writeln!(writer, "elapsed_seconds={elapsed_seconds:.3}")?;
    Ok(())
}
