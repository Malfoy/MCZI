use ahash::AHashSet;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use mc::{
    CountedKmer, CounterConfig, EncodedKmer, count_datasets, count_inputs, expand_fofns,
    simplitig_sequences, with_xz_decompressed_path, write_fasta_path,
};
use rayon::ThreadPoolBuilder;
use sshash_lib::{BuildConfiguration, Dictionary, DictionaryBuilder, dispatch_on_k};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(name = "MCZI")]
#[command(about = "Run MC indexing and ZI subtraction in one process")]
struct Cli {
    #[arg(
        long,
        required = true,
        num_args = 1..,
        help = "MC index-set input FASTA/FASTQ file(s), or FOFN path(s) with --index-fofn; gzip, xz, and zstd accepted"
    )]
    index_input: Vec<PathBuf>,

    #[arg(long, help = "Treat --index-input path(s) as MC file-of-filenames")]
    index_fofn: bool,

    #[arg(
        long,
        required = true,
        num_args = 1..,
        help = "Query FASTA/FASTQ file(s), or FOFN path(s) with --query-fofn; gzip, xz, and zstd accepted"
    )]
    query_input: Vec<PathBuf>,

    #[arg(long, help = "Treat --query-input path(s) as file-of-filenames")]
    query_fofn: bool,

    #[arg(short = 'k', long, help = "K-mer size; must be odd and in [3, 63]")]
    kmer_size: usize,

    #[arg(short = 'm', long, help = "MC minimizer size; also used for SSHash")]
    minimizer_size: usize,

    #[arg(
        short = 'x',
        long,
        default_value_t = 1,
        help = "MC threshold for the index set"
    )]
    threshold: u64,

    #[arg(
        short,
        long,
        help = "Output FASTA containing query k-mers absent from the MC index set; .gz, .xz, and .zst are compressed"
    )]
    output: PathBuf,

    #[arg(short, long, help = "Number of worker threads")]
    threads: Option<usize>,

    #[arg(long, default_value_t = 8, help = "SSHash build RAM limit in GiB")]
    ram_limit_gib: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_k(cli.kmer_size)?;
    ensure!(
        cli.minimizer_size > 0 && cli.minimizer_size < cli.kmer_size,
        "minimizer size must be > 0 and < k"
    );

    if let Some(threads) = cli.threads {
        ensure!(threads > 0, "--threads must be greater than 0");
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to initialize Rayon thread pool")?;
    }

    let started = Instant::now();
    let index_inputs = if cli.index_fofn {
        expand_fofns(&cli.index_input)?
    } else {
        cli.index_input.clone()
    };
    let query_inputs = if cli.query_fofn {
        expand_fofns(&cli.query_input)?
    } else {
        cli.query_input.clone()
    };
    ensure!(
        !index_inputs.is_empty(),
        "at least one index input is required"
    );
    ensure!(
        !query_inputs.is_empty(),
        "at least one query input is required"
    );

    let config = CounterConfig {
        k: cli.kmer_size,
        minimizer: cli.minimizer_size,
        threshold: cli.threshold,
    };

    let phase_started = Instant::now();
    let index_kmers = if cli.index_fofn {
        count_datasets(&index_inputs, config)?
    } else {
        count_inputs(&index_inputs, config)?
    };
    log_phase("1_mc_index_counting", phase_started.elapsed());

    let phase_started = Instant::now();
    let index_simplitigs = simplitig_sequences(&index_kmers, cli.kmer_size)?;
    log_phase("2_mc_index_simplitigs", phase_started.elapsed());

    let phase_started = Instant::now();
    let novel = if index_simplitigs.is_empty() {
        collect_all_query_kmers(&query_inputs, cli.kmer_size)?
    } else {
        let tmp_dir = create_temp_dir("mczi-sshash")?;
        let dictionary_result = build_dictionary_from_simplitigs(
            index_simplitigs,
            cli.kmer_size,
            cli.minimizer_size,
            cli.threads.unwrap_or(0),
            cli.ram_limit_gib,
            &tmp_dir,
        );
        let _ = fs::remove_dir_all(&tmp_dir);
        let dictionary = dictionary_result?;
        log_phase("3_sshash_indexing", phase_started.elapsed());

        let query_started = Instant::now();
        let novel = collect_query_kmers_absent_from_index(&dictionary, &query_inputs)?;
        log_phase("4_query_subtraction", query_started.elapsed());
        novel
    };

    let phase_started = Instant::now();
    write_novel_simplitigs(&cli.output, cli.kmer_size, novel)?;
    log_phase("5_simplitig_output", phase_started.elapsed());
    log_phase("total", started.elapsed());

    Ok(())
}

fn validate_k(k: usize) -> Result<()> {
    ensure!(k >= 3 && k <= 63, "k must be in [3, 63] for sshash-rs");
    ensure!(k % 2 == 1, "k must be odd for sshash-rs");
    Ok(())
}

fn build_dictionary_from_simplitigs(
    simplitigs: Vec<Vec<u8>>,
    k: usize,
    m: usize,
    threads: usize,
    ram_limit_gib: usize,
    tmp_dir: &Path,
) -> Result<Dictionary> {
    let sequences = simplitigs
        .into_iter()
        .filter(|seq| seq.len() >= k)
        .map(String::from_utf8)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("MC simplitigs contained non-UTF8 DNA")?;
    ensure!(
        !sequences.is_empty(),
        "MC index set did not contain any simplitig of length >= k"
    );

    let mut config = BuildConfiguration::new(k, m)
        .map_err(|err| anyhow::anyhow!("invalid SSHash config: {err}"))?;
    config.canonical = true;
    config.num_threads = threads;
    config.ram_limit_gib = ram_limit_gib;
    config.verbose = false;
    config.tmp_dirname = tmp_dir.to_path_buf();

    DictionaryBuilder::new(config)
        .map_err(|err| anyhow::anyhow!("failed to create SSHash builder: {err}"))?
        .build_from_sequences(sequences)
        .map_err(|err| anyhow::anyhow!("failed to build SSHash dictionary: {err}"))
}

fn collect_query_kmers_absent_from_index(
    dictionary: &Dictionary,
    query_inputs: &[PathBuf],
) -> Result<AHashSet<EncodedKmer>> {
    let mut novel = AHashSet::new();
    let k = dictionary.k();
    dispatch_on_k!(k, K => {
        let mut engine = dictionary.create_streaming_query::<K>();
        for path in query_inputs {
            for_fastx_sequences(path, k, |seq| {
                engine.reset();
                for window in seq.windows(k) {
                    let lookup = engine.lookup(window);
                    if !lookup.is_found() {
                        novel.insert(canonical_encoded_kmer(window));
                    }
                }
                Ok(())
            })?;
        }
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(novel)
}

fn collect_all_query_kmers(query_inputs: &[PathBuf], k: usize) -> Result<AHashSet<EncodedKmer>> {
    let mut kmers = AHashSet::new();
    for path in query_inputs {
        for_fastx_sequences(path, k, |seq| {
            for window in seq.windows(k) {
                kmers.insert(canonical_encoded_kmer(window));
            }
            Ok(())
        })?;
    }
    Ok(kmers)
}

fn write_novel_simplitigs(
    output_path: &Path,
    k: usize,
    novel: AHashSet<EncodedKmer>,
) -> Result<()> {
    let mut kmers = novel
        .into_iter()
        .map(|encoded| CountedKmer { encoded, count: 1 })
        .collect::<Vec<_>>();
    kmers.sort_unstable_by_key(|record| record.encoded);

    write_fasta_path(output_path, &kmers, k)
}

fn for_fastx_sequences<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("xz"))
    {
        return with_xz_decompressed_path(path, |plain_path| {
            for_fastx_sequences(plain_path, min_len, consume)
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
    while parser.next().is_some() {
        let seq = parser.get_dna_string();
        if seq.len() >= min_len {
            consume(seq)?;
        }
    }
    Ok(())
}

fn canonical_encoded_kmer(seq: &[u8]) -> EncodedKmer {
    let k = seq.len();
    let high_shift = 2 * (k - 1);
    let mask = kmer_mask(k);
    let mut fwd = 0u128;
    let mut rev = 0u128;

    for &base in seq {
        let bits = mc_base_bits(base) as u128;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);
    }

    fwd.min(rev)
}

fn mc_base_bits(base: u8) -> u8 {
    match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => unreachable!("helicase split_non_actg should only yield A/C/G/T chunks"),
    }
}

fn kmer_mask(k: usize) -> u128 {
    (1u128 << (2 * k)) - 1
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
    eprintln!("MCZI_PHASE\t{name}\t{:.6}", elapsed.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mczi_subtracts_mc_index_kmers() {
        let dir = create_temp_dir("mczi-test").unwrap();
        let index = dir.join("index.fa");
        let query = dir.join("query.fa");
        let output = dir.join("out.fa");

        fs::write(&index, b">idx\nAAACG\n").unwrap();
        fs::write(&query, b">query\nAAATG\n").unwrap();

        let config = CounterConfig {
            k: 3,
            minimizer: 1,
            threshold: 0,
        };
        let index_kmers = count_inputs(&[index], config).unwrap();
        let simplitigs = simplitig_sequences(&index_kmers, 3).unwrap();
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        let dictionary = build_dictionary_from_simplitigs(simplitigs, 3, 1, 1, 1, &tmp).unwrap();
        let novel = collect_query_kmers_absent_from_index(&dictionary, &[query]).unwrap();
        write_novel_simplitigs(&output, 3, novel).unwrap();

        let text = fs::read_to_string(&output).unwrap();
        assert!(text.contains("AATG") || text.contains("CATT"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
