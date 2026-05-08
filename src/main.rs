use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use mc::{
    CounterConfig, DEFAULT_PARTITION_COUNT, count_datasets_to_fasta_path,
    count_datasets_to_kff_path, count_inputs_to_fasta_path, count_inputs_to_kff_path, expand_fofns,
};
use rayon::ThreadPoolBuilder;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "MC")]
#[command(about = "Parallel two-pass pseudo k-mer counter using minimizer pre-counting")]
struct Cli {
    #[arg(
        short,
        long,
        required = true,
        num_args = 1..,
        help = "Input FASTA/FASTQ file(s), optionally gzip, xz, or zstd compressed"
    )]
    input: Vec<PathBuf>,

    #[arg(
        long,
        help = "Treat --input path(s) as file-of-filenames; -x is a dataset-presence threshold"
    )]
    fofn: bool,

    #[arg(short = 'k', long, help = "K-mer size; must be <= 64")]
    kmer_size: usize,

    #[arg(short = 'm', long, help = "Minimizer size; must be <= k")]
    minimizer_size: usize,

    #[arg(
        short = 'x',
        long,
        default_value_t = 1,
        help = "Emit k-mers seen more than this many times"
    )]
    threshold: u64,

    #[arg(
        short,
        long,
        help = "Output FASTA or KFF file; .gz, .xz, and .zst are compressed"
    )]
    output: PathBuf,

    #[arg(long, value_enum, default_value_t = OutputFormat::Fasta, help = "Output format")]
    format: OutputFormat,

    #[arg(short, long, help = "Number of Rayon worker threads")]
    threads: Option<usize>,

    #[arg(
        long,
        default_value_t = DEFAULT_PARTITION_COUNT,
        help = "Number of temporary partition files used by counting phases"
    )]
    partition_count: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Fasta,
    Kff,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        ensure!(threads > 0, "--threads must be greater than 0");
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to initialize Rayon thread pool")?;
    }

    let config = CounterConfig {
        k: cli.kmer_size,
        minimizer: cli.minimizer_size,
        threshold: cli.threshold,
        partition_count: cli.partition_count,
    };

    if cli.fofn {
        let datasets = expand_fofns(&cli.input)?;
        match cli.format {
            OutputFormat::Fasta => count_datasets_to_fasta_path(&datasets, config, &cli.output)?,
            OutputFormat::Kff => count_datasets_to_kff_path(&datasets, config, &cli.output)?,
        }
    } else {
        let inputs = cli.input.clone();
        match cli.format {
            OutputFormat::Fasta => count_inputs_to_fasta_path(&inputs, config, &cli.output)?,
            OutputFormat::Kff => count_inputs_to_kff_path(&inputs, config, &cli.output)?,
        }
    }

    Ok(())
}
