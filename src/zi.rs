use ahash::AHashSet;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use mc::{CountedKmer, EncodedKmer, expand_fofns, with_xz_decompressed_path, write_fasta_path};
use sshash_lib::{BuildConfiguration, Dictionary, DictionaryBuilder, dispatch_on_k};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(name = "ZI")]
#[command(about = "Subtract an SSHash-indexed k-mer set from FASTA/FOFN FASTA input")]
struct Cli {
    #[arg(
        long,
        help = "FASTA containing k-mers/simplitigs to index and subtract; gzip, xz, and zstd accepted"
    )]
    index: PathBuf,

    #[arg(
        short,
        long,
        required = true,
        num_args = 1..,
        help = "Query FASTA file(s), or FOFN path(s) when --fofn is set; gzip, xz, and zstd accepted"
    )]
    input: Vec<PathBuf>,

    #[arg(long, help = "Treat --input path(s) as file-of-filenames")]
    fofn: bool,

    #[arg(short = 'k', long, help = "K-mer size; must be odd and in [3, 63]")]
    kmer_size: usize,

    #[arg(
        short = 'm',
        long,
        help = "SSHash minimizer size; defaults to 19, adjusted below k"
    )]
    minimizer_size: Option<usize>,

    #[arg(
        short,
        long,
        help = "Output FASTA containing query k-mers absent from the index; .gz, .xz, and .zst are compressed"
    )]
    output: PathBuf,

    #[arg(
        short,
        long,
        default_value_t = 0,
        help = "SSHash build threads; 0 means all cores"
    )]
    threads: usize,

    #[arg(long, default_value_t = 8, help = "SSHash build RAM limit in GiB")]
    ram_limit_gib: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let k = cli.kmer_size;
    validate_k(k)?;
    let m = cli
        .minimizer_size
        .unwrap_or_else(|| default_minimizer_size(k));
    ensure!(m > 0 && m < k, "minimizer size must be > 0 and < k");

    let query_inputs = if cli.fofn {
        expand_fofns(&cli.input)?
    } else {
        cli.input.clone()
    };
    ensure!(
        !query_inputs.is_empty(),
        "at least one query FASTA is required"
    );

    let started = Instant::now();
    let tmp_dir = create_temp_dir("zi-sshash")?;
    let dictionary_result =
        build_dictionary(&cli.index, k, m, cli.threads, cli.ram_limit_gib, &tmp_dir);
    let _ = fs::remove_dir_all(&tmp_dir);
    let dictionary = dictionary_result?;
    log_phase("1_sshash_indexing", started.elapsed());

    let phase_started = Instant::now();
    let novel = collect_query_kmers_absent_from_index(&dictionary, &query_inputs)?;
    log_phase("2_query_subtraction", phase_started.elapsed());

    let phase_started = Instant::now();
    write_novel_simplitigs(&cli.output, k, novel)?;
    log_phase("3_simplitig_output", phase_started.elapsed());
    log_phase("total", started.elapsed());

    Ok(())
}

fn validate_k(k: usize) -> Result<()> {
    ensure!(k >= 3 && k <= 63, "k must be in [3, 63] for sshash-rs");
    ensure!(k % 2 == 1, "k must be odd for sshash-rs");
    Ok(())
}

fn default_minimizer_size(k: usize) -> usize {
    let m = 19.min(k - 2);
    if m % 2 == 1 { m } else { m - 1 }
}

fn build_dictionary(
    index_path: &Path,
    k: usize,
    m: usize,
    threads: usize,
    ram_limit_gib: usize,
    tmp_dir: &Path,
) -> Result<Dictionary> {
    let sequences = read_fasta_sequences(index_path)?;
    ensure!(
        !sequences.is_empty(),
        "index FASTA {} did not contain any sequence of length >= k",
        index_path.display()
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

fn read_fasta_sequences(path: &Path) -> Result<Vec<String>> {
    let mut sequences = Vec::new();
    for_fasta_sequences(path, 1, |seq| {
        sequences.push(String::from_utf8_lossy(seq).into_owned());
        Ok(())
    })?;
    Ok(sequences)
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
            for_fasta_sequences(path, k, |seq| {
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

fn for_fasta_sequences<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("xz"))
    {
        return with_xz_decompressed_path(path, |plain_path| {
            for_fasta_sequences(plain_path, min_len, consume)
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
        .with_context(|| format!("failed to open FASTA input {}", path.display()))?;
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
    if k == 64 {
        u128::MAX
    } else {
        (1u128 << (2 * k)) - 1
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

fn log_phase(name: &str, elapsed: std::time::Duration) {
    eprintln!("ZI_PHASE\t{name}\t{:.6}", elapsed.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtracts_indexed_canonical_kmers() {
        let dir = create_temp_dir("zi-test").unwrap();
        let index = dir.join("index.fa");
        let query = dir.join("query.fa");
        let output = dir.join("out.fa");

        fs::write(&index, b">idx\nAAACG\n").unwrap();
        fs::write(&query, b">query\nAAATG\n").unwrap();

        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        let dictionary = build_dictionary(&index, 3, 1, 1, 1, &tmp).unwrap();
        let novel = collect_query_kmers_absent_from_index(&dictionary, &[query]).unwrap();
        write_novel_simplitigs(&output, 3, novel).unwrap();

        let text = fs::read_to_string(&output).unwrap();
        assert!(text.contains("AATG") || text.contains("CATT"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
