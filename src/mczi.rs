use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use ggcat_api::{
    ExtraElaboration, GGCATConfig, GGCATInstance, GeneralSequenceBlockData, MessageLevel,
};
use helicase::input::FromFile;
use helicase::{Config, FastxParser, HelicaseParser, ParserOptions};
use mc::{
    CounterConfig, DEFAULT_PARTITION_COUNT, EncodedKmer, PhaseMetrics,
    count_datasets_to_kmer_fasta_path_silent_stats, count_inputs_to_kmer_fasta_path_silent_stats,
    create_output_writer, decode_kmer, expand_fofns, log_resource_phase, measure_resource_phase,
    with_xz_decompressed_path,
};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use sshash_lib::{BuildConfiguration, Dictionary, DictionaryBuilder, dispatch_on_k};
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

#[allow(dead_code)]
#[path = "r.rs"]
mod reformer;

const FASTA_OUTPUT_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum OutputMode {
    Simplitig,
    Regular,
    NoOutput,
}

#[derive(Clone, Debug, Default)]
struct QuerySubtractionStats {
    scanned_kmers: u64,
    filtered_kmers: u64,
    not_filtered_kmers: u64,
    regular_output_nucleotides: Option<u64>,
    unique_not_filtered_kmers: Option<u64>,
}

impl QuerySubtractionStats {
    fn with_counts(scanned_kmers: u64, not_filtered_kmers: u64) -> Self {
        Self {
            scanned_kmers,
            filtered_kmers: scanned_kmers.saturating_sub(not_filtered_kmers),
            not_filtered_kmers,
            regular_output_nucleotides: None,
            unique_not_filtered_kmers: None,
        }
    }

    fn with_regular_output(
        scanned_kmers: u64,
        not_filtered_kmers: u64,
        regular_output_nucleotides: u64,
    ) -> Self {
        Self {
            scanned_kmers,
            filtered_kmers: scanned_kmers.saturating_sub(not_filtered_kmers),
            not_filtered_kmers,
            regular_output_nucleotides: Some(regular_output_nucleotides),
            unique_not_filtered_kmers: None,
        }
    }

    fn regular_output_zero() -> Self {
        Self {
            regular_output_nucleotides: Some(0),
            ..Self::default()
        }
    }

    fn add_regular_output_nucleotides(&mut self, nucleotides: usize) {
        let current = self.regular_output_nucleotides.get_or_insert(0);
        *current = current.saturating_add(nucleotides as u64);
    }

    fn merged(left: Self, right: Self) -> Self {
        let regular_output_nucleotides = match (
            left.regular_output_nucleotides,
            right.regular_output_nucleotides,
        ) {
            (Some(left), Some(right)) => Some(left.saturating_add(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        Self {
            scanned_kmers: left.scanned_kmers.saturating_add(right.scanned_kmers),
            filtered_kmers: left.filtered_kmers.saturating_add(right.filtered_kmers),
            not_filtered_kmers: left
                .not_filtered_kmers
                .saturating_add(right.not_filtered_kmers),
            regular_output_nucleotides,
            unique_not_filtered_kmers: None,
        }
    }
}

#[derive(Clone, Debug)]
struct QueryOutputJob {
    input_path: PathBuf,
    output_path: PathBuf,
}

#[derive(Clone, Debug)]
enum QueryOutputPlan {
    Single {
        query_inputs: Vec<PathBuf>,
        output_path: PathBuf,
    },
    PerInput {
        jobs: Vec<QueryOutputJob>,
    },
}

impl QueryOutputPlan {
    fn all_query_inputs(&self) -> Vec<PathBuf> {
        match self {
            QueryOutputPlan::Single { query_inputs, .. } => query_inputs.clone(),
            QueryOutputPlan::PerInput { jobs } => {
                jobs.iter().map(|job| job.input_path.clone()).collect()
            }
        }
    }

    fn is_null_output(&self) -> bool {
        matches!(
            self,
            QueryOutputPlan::Single { output_path, .. } if is_null_output_path(output_path)
        )
    }
}

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
        help = "Output FASTA path. Required unless --query-fofn or --output-mode no-output is set. In --query-fofn mode this is optional; if set, it is an output directory"
    )]
    output: Option<PathBuf>,

    #[arg(
        long,
        default_value = "filtered",
        help = "Suffix inserted into each output filename in --query-fofn mode"
    )]
    output_suffix: String,

    #[arg(
        long,
        value_enum,
        default_value_t = OutputMode::Simplitig,
        help = "Output mode: simplitig compacts absent canonical k-mers with in-process GGCAT; regular streams query-oriented unfiltered sequence segments; no-output reports stats only"
    )]
    output_mode: OutputMode,

    #[arg(
        long,
        help = "Apply in-process R-style K-1 unitig reforming to the MCZI output before writing the final file"
    )]
    reform_output: bool,

    #[arg(
        long,
        value_enum,
        requires = "reform_output",
        help = "When reforming regular MCZI output, preserve km:f abundance with R's mean or runs mode"
    )]
    reform_abundance_mode: Option<reformer::AbundanceMode>,

    #[arg(short, long, help = "Number of worker threads")]
    threads: Option<usize>,

    #[arg(
        long,
        default_value_t = DEFAULT_PARTITION_COUNT,
        help = "Number of temporary MC partition files used by index counting"
    )]
    partition_count: usize,

    #[arg(long, default_value_t = 8, help = "SSHash build RAM limit in GiB")]
    ram_limit_gib: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (result, total_phase) = measure_resource_phase("total", || run(cli));
    log_mczi_phase(&total_phase);
    result
}

fn run(cli: Cli) -> Result<()> {
    validate_k(cli.kmer_size)?;
    ensure!(
        cli.minimizer_size > 0 && cli.minimizer_size < cli.kmer_size,
        "minimizer size must be > 0 and < k"
    );
    ensure!(
        cli.reform_abundance_mode.is_none() || cli.output_mode == OutputMode::Regular,
        "--reform-abundance-mode requires --output-mode regular because MCZI simplitig intermediates do not carry valid km:f abundance"
    );
    ensure!(
        cli.output_mode != OutputMode::NoOutput || !cli.reform_output,
        "--reform-output cannot be combined with --output-mode no-output"
    );
    ensure!(
        cli.output_mode != OutputMode::NoOutput || cli.output.is_none(),
        "--output is not used with --output-mode no-output"
    );
    ensure!(
        cli.output_mode == OutputMode::NoOutput || cli.query_fofn || cli.output.is_some(),
        "--output is required unless --query-fofn or --output-mode no-output is set"
    );

    if let Some(threads) = cli.threads {
        ensure!(threads > 0, "--threads must be greater than 0");
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to initialize Rayon thread pool")?;
    }

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
    let output_plan = if cli.output_mode == OutputMode::NoOutput {
        None
    } else {
        Some(build_query_output_plan(
            cli.query_fofn,
            &query_inputs,
            cli.output.as_deref(),
            &cli.output_suffix,
        )?)
    };

    let config = CounterConfig {
        k: cli.kmer_size,
        minimizer: cli.minimizer_size,
        threshold: cli.threshold,
        partition_count: cli.partition_count,
    };
    ensure!(
        config.partition_count > 0,
        "--partition-count must be greater than 0"
    );

    if cli.index_fofn && cli.threshold >= index_inputs.len() as u64 {
        eprintln!(
            "MCZI_WARN\tindex_fofn_threshold_excludes_all_kmers\tindex_inputs\t{}\tthreshold\t{}\tmessage\tno k-mer can pass a dataset-presence threshold greater than or equal to the number of index datasets",
            index_inputs.len(),
            cli.threshold
        );
        log_zero_phase("1_index_minimizer_presence_counting");
        log_zero_phase("2_index_kmer_partition_counting");
        log_mczi_stat("index_minimizers_above_threshold", 0);
        log_mczi_stat("index_kmers_above_threshold", 0);
        log_zero_phase("3_ggcat_simplitigs");
        if cli.output_mode == OutputMode::NoOutput {
            return finish_empty_index_no_output(&query_inputs, cli.kmer_size);
        }
        return finish_empty_index_query(
            output_plan
                .as_ref()
                .expect("output plan is set outside no-output mode"),
            cli.output_mode,
            cli.kmer_size,
            cli.minimizer_size,
            cli.ram_limit_gib,
            cli.threads,
            cli.reform_output,
            cli.reform_abundance_mode,
        );
    }

    let ggcat_tmp_dir = create_temp_dir("mczi-ggcat")?;
    let selected_kmers_path = ggcat_tmp_dir.join("mc-index-selected-kmers.fa");

    let selected_stats_result = if cli.index_fofn {
        count_datasets_to_kmer_fasta_path_silent_stats(&index_inputs, config, &selected_kmers_path)
    } else {
        count_inputs_to_kmer_fasta_path_silent_stats(&index_inputs, config, &selected_kmers_path)
    };
    let selected_stats = match selected_stats_result {
        Ok(stats) => stats,
        Err(err) => {
            let _ = fs::remove_dir_all(&ggcat_tmp_dir);
            return Err(err);
        }
    };
    log_mczi_stat(
        "index_minimizers_above_threshold",
        selected_stats.minimizers_above_threshold,
    );
    log_mczi_stat("index_kmers_above_threshold", selected_stats.selected_kmers);
    let selected_count = selected_stats.selected_kmers;

    if selected_count == 0 {
        let _ = fs::remove_dir_all(&ggcat_tmp_dir);
        log_zero_phase("3_ggcat_simplitigs");
        if cli.output_mode == OutputMode::NoOutput {
            return finish_empty_index_no_output(&query_inputs, cli.kmer_size);
        }
        return finish_empty_index_query(
            output_plan
                .as_ref()
                .expect("output plan is set outside no-output mode"),
            cli.output_mode,
            cli.kmer_size,
            cli.minimizer_size,
            cli.ram_limit_gib,
            cli.threads,
            cli.reform_output,
            cli.reform_abundance_mode,
        );
    }

    let (index_simplitigs_path_result, ggcat_phase) =
        measure_resource_phase("3_ggcat_simplitigs", || {
            build_index_simplitigs_with_ggcat(
                &selected_kmers_path,
                cli.kmer_size,
                cli.minimizer_size,
                cli.ram_limit_gib,
                cli.threads,
                &ggcat_tmp_dir,
            )
        });
    log_mczi_phase(&ggcat_phase);
    let index_simplitigs_path = match index_simplitigs_path_result {
        Ok(path) => path,
        Err(err) => {
            let _ = fs::remove_dir_all(&ggcat_tmp_dir);
            return Err(err);
        }
    };

    let (dictionary_result, sshash_phase) = measure_resource_phase("4_sshash_indexing", || {
        let tmp_dir = create_temp_dir("mczi-sshash")?;
        let result = build_dictionary_from_fasta_path(
            &index_simplitigs_path,
            cli.kmer_size,
            cli.minimizer_size,
            cli.threads.unwrap_or(0),
            cli.ram_limit_gib,
            &tmp_dir,
        );
        let _ = fs::remove_dir_all(&tmp_dir);
        let _ = fs::remove_dir_all(&ggcat_tmp_dir);
        result
    });
    log_mczi_phase(&sshash_phase);
    let dictionary = dictionary_result?;

    if cli.output_mode == OutputMode::NoOutput {
        let (query_result, query_phase) =
            measure_resource_phase("5_query_subtraction_no_output", || {
                scan_regular_query_output_absent_from_index(&dictionary, &query_inputs)
            });
        log_mczi_phase(&query_phase);
        let query_stats = query_result?;
        log_query_stats(&query_stats);
        log_zero_phase(output_zero_phase_name(cli.output_mode));
        return Ok(());
    }

    let output_plan = output_plan.expect("output plan is set outside no-output mode");
    if output_plan.is_null_output() {
        let all_query_inputs = output_plan.all_query_inputs();
        let (query_result, query_phase) = measure_resource_phase("5_query_subtraction", || {
            scan_query_kmers_absent_from_index(&dictionary, &all_query_inputs)
        });
        log_mczi_phase(&query_phase);
        let query_stats = query_result?;
        log_query_stats(&query_stats);
        log_zero_phase(output_zero_phase_name(cli.output_mode));
        return Ok(());
    }

    write_indexed_query_outputs(
        &dictionary,
        &output_plan,
        cli.output_mode,
        cli.kmer_size,
        cli.minimizer_size,
        cli.ram_limit_gib,
        cli.threads,
        cli.reform_output,
        cli.reform_abundance_mode,
    )?;

    Ok(())
}

fn finish_empty_index_query(
    output_plan: &QueryOutputPlan,
    output_mode: OutputMode,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    reform_output: bool,
    reform_abundance_mode: Option<reformer::AbundanceMode>,
) -> Result<()> {
    log_zero_phase("4_sshash_indexing");
    if output_plan.is_null_output() {
        let query_inputs = output_plan.all_query_inputs();
        let (query_result, query_phase) = measure_resource_phase("5_query_subtraction", || {
            let query_kmers = scan_all_query_kmers(&query_inputs, k)?;
            Ok(QuerySubtractionStats::with_counts(query_kmers, query_kmers))
        });
        log_mczi_phase(&query_phase);
        let query_stats = query_result?;
        log_query_stats(&query_stats);
        log_zero_phase(output_zero_phase_name(output_mode));
        return Ok(());
    }

    match output_plan {
        QueryOutputPlan::Single {
            query_inputs,
            output_path,
        } => write_empty_index_query_output(
            output_path,
            output_mode,
            k,
            minimizer,
            ram_limit_gib,
            threads,
            reform_output,
            reform_abundance_mode,
            query_inputs,
        ),
        QueryOutputPlan::PerInput { jobs } => {
            for job in jobs {
                log_mczi_output(&job.input_path, &job.output_path);
                write_empty_index_query_output(
                    &job.output_path,
                    output_mode,
                    k,
                    minimizer,
                    ram_limit_gib,
                    threads,
                    reform_output,
                    reform_abundance_mode,
                    std::slice::from_ref(&job.input_path),
                )?;
            }
            Ok(())
        }
    }
}

fn finish_empty_index_no_output(query_inputs: &[PathBuf], k: usize) -> Result<()> {
    log_zero_phase("4_sshash_indexing");
    let (query_result, query_phase) =
        measure_resource_phase("5_query_subtraction_no_output", || {
            scan_regular_query_output_without_index(query_inputs, k)
        });
    log_mczi_phase(&query_phase);
    let query_stats = query_result?;
    log_query_stats(&query_stats);
    log_zero_phase(output_zero_phase_name(OutputMode::NoOutput));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_empty_index_query_output(
    output_path: &Path,
    output_mode: OutputMode,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    reform_output: bool,
    reform_abundance_mode: Option<reformer::AbundanceMode>,
    query_inputs: &[PathBuf],
) -> Result<()> {
    match output_mode {
        OutputMode::Simplitig => {
            let output_tmp_dir = create_temp_dir("mczi-output-ggcat")?;
            let all_kmers_path = output_tmp_dir.join("query-all-kmers.fa");
            let result = (|| {
                let (query_result, query_phase) =
                    measure_resource_phase("5_query_subtraction", || {
                        write_all_query_kmers_to_fasta_path(query_inputs, k, &all_kmers_path)
                    });
                log_mczi_phase(&query_phase);
                let query_stats = query_result?;
                log_query_stats(&query_stats);

                let (output_result, output_phase) =
                    measure_resource_phase("6_ggcat_simplitig_output", || {
                        write_ggcat_simplitigs_from_kmer_fasta(
                            &all_kmers_path,
                            output_path,
                            k,
                            minimizer,
                            ram_limit_gib,
                            threads,
                            &output_tmp_dir,
                            query_stats.not_filtered_kmers,
                            reform_output,
                            reform_abundance_mode,
                        )
                    });
                log_mczi_phase(&output_phase);
                output_result
            })();
            let _ = fs::remove_dir_all(&output_tmp_dir);
            result
        }
        OutputMode::Regular => {
            if reform_output {
                let output_tmp_dir = create_temp_dir("mczi-regular-reform")?;
                let regular_path = output_tmp_dir.join("regular-output.fa");
                let result = (|| {
                    let (output_result, output_phase) =
                        measure_resource_phase("5_query_subtraction_regular_output", || {
                            write_regular_query_output_without_index(query_inputs, k, &regular_path)
                        });
                    log_mczi_phase(&output_phase);
                    let query_stats = output_result?;
                    log_query_stats(&query_stats);

                    let (reform_result, reform_phase) =
                        measure_resource_phase("6_reform_output", || {
                            reform_output_path(
                                &regular_path,
                                output_path,
                                k,
                                &output_tmp_dir,
                                reform_abundance_mode,
                            )
                        });
                    log_mczi_phase(&reform_phase);
                    reform_result
                })();
                let _ = fs::remove_dir_all(&output_tmp_dir);
                return result;
            }
            let (output_result, output_phase) =
                measure_resource_phase("5_query_subtraction_regular_output", || {
                    write_regular_query_output_without_index(query_inputs, k, output_path)
                });
            log_mczi_phase(&output_phase);
            let query_stats = output_result?;
            log_query_stats(&query_stats);
            Ok(())
        }
        OutputMode::NoOutput => {
            anyhow::bail!("--output-mode no-output does not write query output")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_indexed_query_outputs(
    dictionary: &Dictionary,
    output_plan: &QueryOutputPlan,
    output_mode: OutputMode,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    reform_output: bool,
    reform_abundance_mode: Option<reformer::AbundanceMode>,
) -> Result<()> {
    match output_plan {
        QueryOutputPlan::Single {
            query_inputs,
            output_path,
        } => write_indexed_query_output(
            dictionary,
            query_inputs,
            output_path,
            output_mode,
            k,
            minimizer,
            ram_limit_gib,
            threads,
            reform_output,
            reform_abundance_mode,
        ),
        QueryOutputPlan::PerInput { jobs } => {
            for job in jobs {
                log_mczi_output(&job.input_path, &job.output_path);
                write_indexed_query_output(
                    dictionary,
                    std::slice::from_ref(&job.input_path),
                    &job.output_path,
                    output_mode,
                    k,
                    minimizer,
                    ram_limit_gib,
                    threads,
                    reform_output,
                    reform_abundance_mode,
                )?;
            }
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_indexed_query_output(
    dictionary: &Dictionary,
    query_inputs: &[PathBuf],
    output_path: &Path,
    output_mode: OutputMode,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    reform_output: bool,
    reform_abundance_mode: Option<reformer::AbundanceMode>,
) -> Result<()> {
    match output_mode {
        OutputMode::Simplitig => {
            let output_tmp_dir = create_temp_dir("mczi-output-ggcat")?;
            let absent_kmers_path = output_tmp_dir.join("query-absent-kmers.fa");
            let result = (|| {
                let (query_result, query_phase) =
                    measure_resource_phase("5_query_subtraction", || {
                        write_absent_query_kmers_to_fasta_path(
                            dictionary,
                            query_inputs,
                            &absent_kmers_path,
                        )
                    });
                log_mczi_phase(&query_phase);
                let query_stats = query_result?;
                log_query_stats(&query_stats);

                let (output_result, output_phase) =
                    measure_resource_phase("6_ggcat_simplitig_output", || {
                        write_ggcat_simplitigs_from_kmer_fasta(
                            &absent_kmers_path,
                            output_path,
                            k,
                            minimizer,
                            ram_limit_gib,
                            threads,
                            &output_tmp_dir,
                            query_stats.not_filtered_kmers,
                            reform_output,
                            reform_abundance_mode,
                        )
                    });
                log_mczi_phase(&output_phase);
                output_result
            })();
            let _ = fs::remove_dir_all(&output_tmp_dir);
            result
        }
        OutputMode::Regular => {
            if reform_output {
                let output_tmp_dir = create_temp_dir("mczi-regular-reform")?;
                let regular_path = output_tmp_dir.join("regular-output.fa");
                let result = (|| {
                    let (output_result, output_phase) =
                        measure_resource_phase("5_query_subtraction_regular_output", || {
                            write_regular_query_output_absent_from_index(
                                dictionary,
                                query_inputs,
                                &regular_path,
                            )
                        });
                    log_mczi_phase(&output_phase);
                    let query_stats = output_result?;
                    log_query_stats(&query_stats);

                    let (reform_result, reform_phase) =
                        measure_resource_phase("6_reform_output", || {
                            reform_output_path(
                                &regular_path,
                                output_path,
                                k,
                                &output_tmp_dir,
                                reform_abundance_mode,
                            )
                        });
                    log_mczi_phase(&reform_phase);
                    reform_result
                })();
                let _ = fs::remove_dir_all(&output_tmp_dir);
                return result;
            }
            let (output_result, output_phase) =
                measure_resource_phase("5_query_subtraction_regular_output", || {
                    write_regular_query_output_absent_from_index(
                        dictionary,
                        query_inputs,
                        output_path,
                    )
                });
            log_mczi_phase(&output_phase);
            let query_stats = output_result?;
            log_query_stats(&query_stats);
            Ok(())
        }
        OutputMode::NoOutput => {
            anyhow::bail!("--output-mode no-output does not write query output")
        }
    }
}

fn build_query_output_plan(
    query_fofn: bool,
    query_inputs: &[PathBuf],
    output: Option<&Path>,
    output_suffix: &str,
) -> Result<QueryOutputPlan> {
    if !query_fofn {
        let output_path = output
            .context("--output is required unless --query-fofn is set")?
            .to_path_buf();
        return Ok(QueryOutputPlan::Single {
            query_inputs: query_inputs.to_vec(),
            output_path,
        });
    }

    if let Some(output_path) = output {
        if is_null_output_path(output_path) {
            return Ok(QueryOutputPlan::Single {
                query_inputs: query_inputs.to_vec(),
                output_path: output_path.to_path_buf(),
            });
        }
    }

    let normalized_suffix = normalized_output_suffix(output_suffix)?;
    if let Some(output_dir) = output {
        if output_dir.exists() {
            ensure!(
                output_dir.is_dir(),
                "--output must be a directory in --query-fofn mode"
            );
        } else {
            fs::create_dir_all(output_dir)
                .with_context(|| format!("failed to create {}", output_dir.display()))?;
        }
    }

    let mut jobs = Vec::with_capacity(query_inputs.len());
    let mut seen_outputs = HashSet::with_capacity(query_inputs.len());
    for input_path in query_inputs {
        let output_path = query_fofn_output_path(input_path, output, normalized_suffix)?;
        ensure!(
            output_path != *input_path,
            "output path {} would overwrite query input {}; choose a non-empty --output-suffix",
            output_path.display(),
            input_path.display()
        );
        ensure!(
            seen_outputs.insert(output_path.clone()),
            "multiple query inputs map to the same output path {}; use --output or --output-suffix to disambiguate",
            output_path.display()
        );
        jobs.push(QueryOutputJob {
            input_path: input_path.clone(),
            output_path,
        });
    }

    Ok(QueryOutputPlan::PerInput { jobs })
}

fn normalized_output_suffix(output_suffix: &str) -> Result<&str> {
    let suffix = output_suffix.trim();
    let suffix = suffix.strip_prefix('.').unwrap_or(suffix);
    ensure!(!suffix.is_empty(), "--output-suffix must not be empty");
    ensure!(
        !suffix.contains(std::path::MAIN_SEPARATOR),
        "--output-suffix must be a filename suffix, not a path"
    );
    Ok(suffix)
}

fn query_fofn_output_path(
    input_path: &Path,
    output_dir: Option<&Path>,
    output_suffix: &str,
) -> Result<PathBuf> {
    let file_name = input_path
        .file_name()
        .with_context(|| format!("query input {} has no filename", input_path.display()))?;
    let file_name = file_name
        .to_str()
        .with_context(|| format!("query input filename {:?} is not UTF-8", file_name))?;
    let suffixed_file_name = suffixed_query_file_name(file_name, output_suffix);
    let directory = output_dir
        .or_else(|| input_path.parent())
        .unwrap_or_else(|| Path::new("."));
    Ok(directory.join(suffixed_file_name))
}

fn suffixed_query_file_name(file_name: &str, output_suffix: &str) -> String {
    let (without_compression, compression_ext) =
        split_known_suffix(file_name, &[".zstd", ".gzip", ".zst", ".gz", ".xz"])
            .unwrap_or((file_name, ""));
    let (stem, sequence_ext) = split_known_suffix(
        without_compression,
        &[".fasta", ".fastq", ".fna", ".fa", ".fq"],
    )
    .unwrap_or((without_compression, ""));
    format!("{stem}.{output_suffix}{sequence_ext}{compression_ext}")
}

fn split_known_suffix<'a>(value: &'a str, suffixes: &[&str]) -> Option<(&'a str, &'a str)> {
    suffixes.iter().find_map(|suffix| {
        if value.len() > suffix.len()
            && value[value.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
        {
            Some(value.split_at(value.len() - suffix.len()))
        } else {
            None
        }
    })
}

fn validate_k(k: usize) -> Result<()> {
    ensure!(k >= 3 && k <= 63, "k must be in [3, 63] for sshash-rs");
    ensure!(k % 2 == 1, "k must be odd for sshash-rs");
    Ok(())
}

fn output_zero_phase_name(output_mode: OutputMode) -> &'static str {
    match output_mode {
        OutputMode::Simplitig => "6_ggcat_simplitig_output",
        OutputMode::Regular => "6_regular_output",
        OutputMode::NoOutput => "6_no_output",
    }
}

fn build_index_simplitigs_with_ggcat(
    selected_kmers_path: &Path,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    tmp_dir: &Path,
) -> Result<PathBuf> {
    let output_path = tmp_dir.join("mc-index-ggcat-simplitigs.fa");
    build_simplitigs_with_ggcat(
        selected_kmers_path,
        &output_path,
        k,
        minimizer,
        ram_limit_gib,
        threads,
        tmp_dir,
        "index",
    )?;
    Ok(output_path)
}

fn build_simplitigs_with_ggcat(
    input_path: &Path,
    output_path: &Path,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    tmp_dir: &Path,
    label: &str,
) -> Result<()> {
    let ggcat_temp_dir = tmp_dir.join("ggcat-temp");
    fs::create_dir(&ggcat_temp_dir)
        .with_context(|| format!("failed to create {}", ggcat_temp_dir.display()))?;

    let threads_count = threads.unwrap_or_else(available_threads);
    let instance = GGCATInstance::create(GGCATConfig {
        temp_dir: Some(ggcat_temp_dir),
        memory: ram_limit_gib as f64,
        prefer_memory: false,
        total_threads_count: threads_count,
        intermediate_compression_level: None,
        stats_file: None,
        messages_callback: Some(ggcat_message_callback),
        disk_optimization_level: 5,
    })
    .context("failed to initialize in-process GGCAT")?;

    instance
        .build_graph(
            vec![GeneralSequenceBlockData::FASTA((
                input_path.to_path_buf(),
                None,
            ))],
            output_path.to_path_buf(),
            None,
            k,
            threads_count,
            false,
            Some(minimizer),
            false,
            1,
            ExtraElaboration::FastSimplitigs,
            None,
            5,
        )
        .with_context(|| format!("in-process GGCAT {label} simplitig construction failed"))?;
    ensure!(
        output_path.exists(),
        "ggcat did not create expected output {}",
        output_path.display()
    );
    Ok(())
}

fn ggcat_message_callback(level: MessageLevel, message: &str) {
    match level {
        MessageLevel::Info => {}
        MessageLevel::Warning => eprintln!("GGCAT_WARNING\t{message}"),
        MessageLevel::Error => eprintln!("GGCAT_ERROR\t{message}"),
        MessageLevel::UnrecoverableError => eprintln!("GGCAT_UNRECOVERABLE_ERROR\t{message}"),
    }
}

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(16)
}

fn build_dictionary_from_fasta_path(
    fasta_path: &Path,
    k: usize,
    m: usize,
    threads: usize,
    ram_limit_gib: usize,
    tmp_dir: &Path,
) -> Result<Dictionary> {
    let mut sequences = Vec::new();
    for_fastx_sequences(fasta_path, k, |seq| {
        sequences.push(
            std::str::from_utf8(seq)
                .context("GGCAT simplitig contained non-UTF8 DNA")?
                .to_owned(),
        );
        Ok(())
    })?;
    build_dictionary_from_sequences(sequences, k, m, threads, ram_limit_gib, tmp_dir)
}

#[cfg(test)]
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
    build_dictionary_from_sequences(sequences, k, m, threads, ram_limit_gib, tmp_dir)
}

fn build_dictionary_from_sequences(
    sequences: Vec<String>,
    k: usize,
    m: usize,
    threads: usize,
    ram_limit_gib: usize,
    tmp_dir: &Path,
) -> Result<Dictionary> {
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

fn scan_query_kmers_absent_from_index(
    dictionary: &Dictionary,
    query_inputs: &[PathBuf],
) -> Result<QuerySubtractionStats> {
    let k = dictionary.k();
    let stats = dispatch_on_k!(k, K => {
        query_inputs
            .par_iter()
            .map(|path| {
                let mut scanned_kmers = 0u64;
                let mut not_filtered_kmers = 0u64;
                let mut engine = dictionary.create_streaming_query::<K>();
            for_fastx_sequences(path, k, |seq| {
                engine.reset();
                for window in seq.windows(k) {
                    scanned_kmers = scanned_kmers.saturating_add(1);
                    if !engine.lookup(window).is_found() {
                        not_filtered_kmers = not_filtered_kmers.saturating_add(1);
                    }
                }
                Ok(())
            })?;
                Ok::<_, anyhow::Error>(QuerySubtractionStats::with_counts(
                    scanned_kmers,
                    not_filtered_kmers,
                ))
            })
            .try_reduce(QuerySubtractionStats::default, |left, right| {
                let scanned_kmers = left.scanned_kmers.saturating_add(right.scanned_kmers);
                let not_filtered_kmers = left
                    .not_filtered_kmers
                    .saturating_add(right.not_filtered_kmers);
                Ok(QuerySubtractionStats::with_counts(
                    scanned_kmers,
                    not_filtered_kmers,
                ))
            })
    })?;
    Ok(stats)
}

fn scan_regular_query_output_absent_from_index(
    dictionary: &Dictionary,
    query_inputs: &[PathBuf],
) -> Result<QuerySubtractionStats> {
    let k = dictionary.k();
    let stats = dispatch_on_k!(k, K => {
        query_inputs
            .par_iter()
            .map(|path| {
                let mut stats = QuerySubtractionStats::regular_output_zero();
                let mut engine = dictionary.create_streaming_query::<K>();
                for_fastx_sequences_with_headers(path, k, |_header, seq| {
                    engine.reset();
                    process_regular_output_sequence(
                        seq,
                        k,
                        &mut stats,
                        |window| engine.lookup(window).is_found(),
                        |_| Ok(()),
                    )
                })?;
                stats.filtered_kmers = stats
                    .scanned_kmers
                    .saturating_sub(stats.not_filtered_kmers);
                Ok::<_, anyhow::Error>(stats)
            })
            .try_reduce(QuerySubtractionStats::regular_output_zero, |left, right| {
                Ok(QuerySubtractionStats::merged(left, right))
            })
    })?;
    Ok(stats)
}

fn scan_all_query_kmers(query_inputs: &[PathBuf], k: usize) -> Result<u64> {
    query_inputs
        .par_iter()
        .map(|path| {
            let mut kmers = 0u64;
            for_fastx_sequences(path, k, |seq| {
                if seq.len() >= k {
                    kmers = kmers.saturating_add((seq.len() - k + 1) as u64);
                }
                Ok(())
            })?;
            Ok::<_, anyhow::Error>(kmers)
        })
        .try_reduce(|| 0, |left, right| Ok(left.saturating_add(right)))
}

fn scan_regular_query_output_without_index(
    query_inputs: &[PathBuf],
    k: usize,
) -> Result<QuerySubtractionStats> {
    query_inputs
        .par_iter()
        .map(|path| {
            let mut scanned_kmers = 0u64;
            let mut regular_output_nucleotides = 0u64;
            for_fastx_sequences_with_headers(path, k, |_header, seq| {
                scanned_kmers = scanned_kmers.saturating_add((seq.len() - k + 1) as u64);
                regular_output_nucleotides =
                    regular_output_nucleotides.saturating_add(seq.len() as u64);
                Ok(())
            })?;
            Ok::<_, anyhow::Error>(QuerySubtractionStats::with_regular_output(
                scanned_kmers,
                scanned_kmers,
                regular_output_nucleotides,
            ))
        })
        .try_reduce(QuerySubtractionStats::regular_output_zero, |left, right| {
            Ok(QuerySubtractionStats::merged(left, right))
        })
}

fn is_null_output_path(path: &Path) -> bool {
    path == Path::new("/dev/null")
}

fn write_absent_query_kmers_to_fasta_path(
    dictionary: &Dictionary,
    query_inputs: &[PathBuf],
    output_path: &Path,
) -> Result<QuerySubtractionStats> {
    let output = File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(FASTA_OUTPUT_BUFFER_BYTES, output);
    let mut scanned_kmers = 0u64;
    let mut not_filtered_kmers = 0u64;
    let k = dictionary.k();

    dispatch_on_k!(k, K => {
        let mut engine = dictionary.create_streaming_query::<K>();
        for path in query_inputs {
            for_fastx_sequences(path, k, |seq| {
                engine.reset();
                try_for_each_canonical_encoded_kmer(seq, k, |start, encoded| {
                    scanned_kmers = scanned_kmers.saturating_add(1);
                    if !engine.lookup(&seq[start..start + k]).is_found() {
                        not_filtered_kmers = not_filtered_kmers.saturating_add(1);
                        write_kmer_fasta_record(
                            &mut writer,
                            not_filtered_kmers,
                            encoded,
                            k,
                            "MCZI_absent_kmer",
                        )?;
                    }
                    Ok(())
                })
            })?;
        }
        Ok::<_, anyhow::Error>(())
    })?;
    writer.flush()?;
    Ok(QuerySubtractionStats::with_counts(
        scanned_kmers,
        not_filtered_kmers,
    ))
}

fn write_all_query_kmers_to_fasta_path(
    query_inputs: &[PathBuf],
    k: usize,
    output_path: &Path,
) -> Result<QuerySubtractionStats> {
    let output = File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(FASTA_OUTPUT_BUFFER_BYTES, output);
    let mut scanned_kmers = 0u64;

    for path in query_inputs {
        for_fastx_sequences(path, k, |seq| {
            try_for_each_canonical_encoded_kmer(seq, k, |_, encoded| {
                scanned_kmers = scanned_kmers.saturating_add(1);
                write_kmer_fasta_record(&mut writer, scanned_kmers, encoded, k, "MCZI_query_kmer")
            })
        })?;
    }

    writer.flush()?;
    Ok(QuerySubtractionStats::with_counts(
        scanned_kmers,
        scanned_kmers,
    ))
}

fn write_ggcat_simplitigs_from_kmer_fasta(
    input_path: &Path,
    output_path: &Path,
    k: usize,
    minimizer: usize,
    ram_limit_gib: usize,
    threads: Option<usize>,
    tmp_dir: &Path,
    input_kmers: u64,
    reform_output: bool,
    reform_abundance_mode: Option<reformer::AbundanceMode>,
) -> Result<()> {
    if input_kmers == 0 {
        let writer = create_output_writer(output_path)?;
        return writer.finish();
    }

    let ggcat_output_path = tmp_dir.join("query-absent-ggcat-simplitigs.fa");
    build_simplitigs_with_ggcat(
        input_path,
        &ggcat_output_path,
        k,
        minimizer,
        ram_limit_gib,
        threads,
        tmp_dir,
        "query-output",
    )?;
    if reform_output {
        reform_output_path(
            &ggcat_output_path,
            output_path,
            k,
            tmp_dir,
            reform_abundance_mode,
        )
    } else {
        copy_fasta_path_to_output(&ggcat_output_path, output_path)
    }
}

fn reform_output_path(
    input_path: &Path,
    output_path: &Path,
    k: usize,
    tmp_dir: &Path,
    abundance_mode: Option<reformer::AbundanceMode>,
) -> Result<()> {
    if fs::metadata(input_path)
        .with_context(|| format!("failed to stat {}", input_path.display()))?
        .len()
        == 0
    {
        return create_output_writer(output_path)?.finish();
    }

    reformer::reform_path(
        reformer::ReformerConfig {
            input: input_path.to_path_buf(),
            kmer_size: k,
            output: output_path.to_path_buf(),
            output_mode: reformer::OutputMode::Simplitig,
            sequence_store_mode: reformer::SequenceStoreMode::Disk,
            output_sort: reformer::OutputSort::None,
            strand_tiebreak: reformer::OutputStrandTieBreak::None,
            abundance_mode,
            zstd_workers: None,
            emit_logs: false,
        },
        tmp_dir,
    )
    .map(|_| ())
}

fn copy_fasta_path_to_output(input_path: &Path, output_path: &Path) -> Result<()> {
    let input = File::open(input_path)
        .with_context(|| format!("failed to open {}", input_path.display()))?;
    let mut reader = BufReader::with_capacity(FASTA_OUTPUT_BUFFER_BYTES, input);
    let mut writer = create_output_writer(output_path)?;
    io::copy(&mut reader, &mut writer)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    writer.finish()
}

fn write_regular_query_output_absent_from_index(
    dictionary: &Dictionary,
    query_inputs: &[PathBuf],
    output_path: &Path,
) -> Result<QuerySubtractionStats> {
    let mut writer = create_output_writer(output_path)?;
    let mut stats = QuerySubtractionStats::regular_output_zero();
    let k = dictionary.k();

    dispatch_on_k!(k, K => {
        let mut engine = dictionary.create_streaming_query::<K>();
        for path in query_inputs {
            for_fastx_sequences_with_headers(path, k, |header, seq| {
                engine.reset();
                process_regular_output_sequence(
                    seq,
                    k,
                    &mut stats,
                    |window| engine.lookup(window).is_found(),
                    |segment| write_regular_segment(&mut writer, header, segment),
                )
            })?;
        }
        Ok::<_, anyhow::Error>(())
    })?;
    writer.finish()?;
    stats.filtered_kmers = stats.scanned_kmers.saturating_sub(stats.not_filtered_kmers);
    Ok(stats)
}

fn write_regular_query_output_without_index(
    query_inputs: &[PathBuf],
    k: usize,
    output_path: &Path,
) -> Result<QuerySubtractionStats> {
    let mut writer = create_output_writer(output_path)?;
    let mut stats = QuerySubtractionStats::regular_output_zero();
    for path in query_inputs {
        for_fastx_sequences_with_headers(path, k, |header, seq| {
            stats.scanned_kmers = stats
                .scanned_kmers
                .saturating_add((seq.len() - k + 1) as u64);
            stats.not_filtered_kmers = stats
                .not_filtered_kmers
                .saturating_add((seq.len() - k + 1) as u64);
            stats.add_regular_output_nucleotides(seq.len());
            write_regular_segment(&mut writer, header, seq)
        })?;
    }
    writer.finish()?;
    stats.filtered_kmers = stats.scanned_kmers.saturating_sub(stats.not_filtered_kmers);
    Ok(stats)
}

fn process_regular_output_sequence<IsFiltered, Emit>(
    seq: &[u8],
    k: usize,
    stats: &mut QuerySubtractionStats,
    mut is_filtered: IsFiltered,
    mut emit: Emit,
) -> Result<()>
where
    IsFiltered: FnMut(&[u8]) -> bool,
    Emit: FnMut(&[u8]) -> Result<()>,
{
    let mut run_start = None;
    for (start, window) in seq.windows(k).enumerate() {
        stats.scanned_kmers = stats.scanned_kmers.saturating_add(1);
        if is_filtered(window) {
            if let Some(segment_start) = run_start.take() {
                let segment = &seq[segment_start..start + k - 1];
                stats.add_regular_output_nucleotides(segment.len());
                emit(segment)?;
            }
        } else {
            stats.not_filtered_kmers = stats.not_filtered_kmers.saturating_add(1);
            run_start.get_or_insert(start);
        }
    }
    if let Some(segment_start) = run_start {
        let segment = &seq[segment_start..];
        stats.add_regular_output_nucleotides(segment.len());
        emit(segment)?;
    }
    Ok(())
}

fn write_kmer_fasta_record<W: Write>(
    writer: &mut W,
    idx: u64,
    encoded: EncodedKmer,
    k: usize,
    prefix: &str,
) -> Result<()> {
    writeln!(writer, ">{prefix}_{idx}")?;
    write_sequence_lines(writer, &decode_kmer(encoded, k))
}

fn write_regular_segment<W: Write>(writer: &mut W, header: &[u8], seq: &[u8]) -> Result<()> {
    writer.write_all(b">")?;
    writer.write_all(header)?;
    writer.write_all(b"\n")?;
    write_sequence_lines(writer, seq)
}

fn write_sequence_lines<W: Write>(writer: &mut W, seq: &[u8]) -> Result<()> {
    for chunk in seq.chunks(80) {
        writer.write_all(chunk)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
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

    let Some(mut parser) = open_fastx_parser::<CONFIG>(path, "FASTA/FASTQ")? else {
        return Ok(());
    };
    while parser.next().is_some() {
        let seq = parser.get_dna_string();
        if seq.len() >= min_len {
            consume(seq)?;
        }
    }
    Ok(())
}

fn for_fastx_sequences_with_headers<F>(path: &Path, min_len: usize, mut consume: F) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> Result<()>,
{
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("xz"))
    {
        return with_xz_decompressed_path(path, |plain_path| {
            for_fastx_sequences_with_headers(plain_path, min_len, consume)
        });
    }

    const CONFIG: Config = ParserOptions::default()
        .dna_string()
        .ignore_quality()
        .split_non_actg()
        .return_record(false)
        .config();

    let Some(mut parser) = open_fastx_parser::<CONFIG>(path, "FASTA/FASTQ")? else {
        return Ok(());
    };
    while parser.next().is_some() {
        let seq = parser.get_dna_string();
        if seq.len() >= min_len {
            consume(parser.get_header(), seq)?;
        }
    }
    Ok(())
}

fn open_fastx_parser<const CONFIG: Config>(
    path: &Path,
    input_kind: &str,
) -> Result<Option<FastxParser<'static, CONFIG>>> {
    match FastxParser::<CONFIG>::from_file(path) {
        Ok(parser) => Ok(Some(parser)),
        Err(err) if is_invalid_fastx_record_start(&err) => {
            eprintln!(
                "MCZI_WARN\tskipped_invalid_fastx_input\tpath\t{}\terror\t{}",
                path.display(),
                err
            );
            Ok(None)
        }
        Err(err) => Err(err)
            .with_context(|| format!("failed to open {input_kind} input {}", path.display())),
    }
}

fn is_invalid_fastx_record_start(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::Other && err.to_string().starts_with("Invalid record start")
}

#[cfg(test)]
fn for_each_canonical_encoded_kmer<F>(seq: &[u8], k: usize, mut visit: F)
where
    F: FnMut(usize, EncodedKmer),
{
    let result = try_for_each_canonical_encoded_kmer(seq, k, |start, encoded| {
        visit(start, encoded);
        Ok(())
    });
    debug_assert!(result.is_ok());
}

fn try_for_each_canonical_encoded_kmer<F>(seq: &[u8], k: usize, mut visit: F) -> Result<()>
where
    F: FnMut(usize, EncodedKmer) -> Result<()>,
{
    if seq.len() < k {
        return Ok(());
    }

    let high_shift = 2 * (k - 1);
    let mask = kmer_mask(k);
    let mut fwd = 0u128;
    let mut rev = 0u128;

    for (idx, &base) in seq.iter().enumerate() {
        let bits = mc_base_bits(base) as u128;
        fwd = ((fwd << 2) | bits) & mask;
        rev = (rev >> 2) | ((bits ^ 0b11) << high_shift);

        if idx + 1 >= k {
            visit(idx + 1 - k, fwd.min(rev))?;
        }
    }

    Ok(())
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

fn log_mczi_phase(metrics: &PhaseMetrics) {
    log_resource_phase("MCZI_PHASE", metrics);
}

fn log_zero_phase(name: &'static str) {
    log_mczi_phase(&PhaseMetrics::zero(name));
}

fn log_mczi_stat(name: &str, value: u64) {
    eprintln!("MCZI_STAT\t{name}\t{}", format_stat_u64(value));
}

fn log_mczi_output(input_path: &Path, output_path: &Path) {
    eprintln!(
        "MCZI_OUTPUT\t{}\t{}",
        input_path.display(),
        output_path.display()
    );
}

fn format_stat_u64(value: u64) -> String {
    let digits = value.to_string();
    let first_group_len = digits.len() % 3;
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    let mut idx = 0usize;

    if first_group_len != 0 {
        formatted.push_str(&digits[..first_group_len]);
        idx = first_group_len;
    }

    while idx < digits.len() {
        if !formatted.is_empty() {
            formatted.push(',');
        }
        formatted.push_str(&digits[idx..idx + 3]);
        idx += 3;
    }

    formatted
}

fn log_query_stats(stats: &QuerySubtractionStats) {
    log_mczi_stat("query_kmers_scanned", stats.scanned_kmers);
    log_mczi_stat("query_kmers_filtered_by_zi", stats.filtered_kmers);
    log_mczi_stat("query_kmers_not_filtered_by_zi", stats.not_filtered_kmers);
    if let Some(regular_output_nucleotides) = stats.regular_output_nucleotides {
        log_mczi_stat(
            "query_regular_output_nucleotides",
            regular_output_nucleotides,
        );
    }
    if let Some(unique_not_filtered_kmers) = stats.unique_not_filtered_kmers {
        log_mczi_stat(
            "query_unique_kmers_not_filtered_by_zi",
            unique_not_filtered_kmers,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashSet;
    use mc::{count_inputs, simplitig_sequences};
    use std::sync::Mutex;

    static GGCAT_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn mczi_subtracts_mc_index_kmers() {
        let _guard = GGCAT_TEST_LOCK.lock().unwrap();
        let dir = create_temp_dir("mczi-test").unwrap();
        let index = dir.join("index.fa");
        let query = dir.join("query.fa");
        let output = dir.join("out.fa");
        let regular_output = dir.join("regular.fa");
        let absent_kmers = dir.join("absent.fa");
        let ggcat_tmp = dir.join("ggcat-output");

        fs::write(&index, b">idx\nAAACG\n").unwrap();
        fs::write(&query, b">query\nAAATG\n").unwrap();
        fs::create_dir(&ggcat_tmp).unwrap();

        let config = CounterConfig {
            k: 3,
            minimizer: 1,
            threshold: 0,
            partition_count: DEFAULT_PARTITION_COUNT,
        };
        let index_kmers = count_inputs(&[index], config).unwrap();
        let simplitigs = simplitig_sequences(&index_kmers, 3).unwrap();
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        let dictionary = build_dictionary_from_simplitigs(simplitigs, 3, 1, 1, 1, &tmp).unwrap();
        let stats =
            write_absent_query_kmers_to_fasta_path(&dictionary, &[query.clone()], &absent_kmers)
                .unwrap();
        assert_eq!(stats.scanned_kmers, 3);
        assert_eq!(stats.not_filtered_kmers, 2);
        assert_eq!(stats.unique_not_filtered_kmers, None);
        write_ggcat_simplitigs_from_kmer_fasta(
            &absent_kmers,
            &output,
            3,
            1,
            1,
            Some(1),
            &ggcat_tmp,
            stats.not_filtered_kmers,
            false,
            None,
        )
        .unwrap();

        let text = fs::read_to_string(&output).unwrap();
        assert!(text.contains("AATG") || text.contains("CATT"));

        let regular_stats =
            write_regular_query_output_absent_from_index(&dictionary, &[query], &regular_output)
                .unwrap();
        assert_eq!(regular_stats.scanned_kmers, 3);
        assert_eq!(regular_stats.filtered_kmers, 1);
        assert_eq!(regular_stats.not_filtered_kmers, 2);
        assert_eq!(regular_stats.regular_output_nucleotides, Some(4));
        let regular_text = fs::read_to_string(&regular_output).unwrap();
        assert_eq!(regular_text, ">query\nAATG\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ggcat_empty_index_output_preserves_query_kmers() {
        let _guard = GGCAT_TEST_LOCK.lock().unwrap();
        let dir = create_temp_dir("mczi-empty-index-test").unwrap();
        let query = dir.join("query.fa");
        let output = dir.join("out.fa");
        let kmers = dir.join("query-kmers.fa");
        let ggcat_tmp = dir.join("ggcat-output");
        fs::create_dir(&ggcat_tmp).unwrap();

        let mut query_text = String::new();
        let bases = [b'A', b'C', b'G', b'T'];
        let mut record_idx = 0usize;
        for &a in &bases {
            for &b in &bases {
                for &c in &bases {
                    record_idx += 1;
                    query_text.push_str(&format!(
                        ">q{record_idx}\n{}{}{}\n",
                        a as char, b as char, c as char
                    ));
                }
            }
        }
        fs::write(&query, query_text.as_bytes()).unwrap();

        let stats = write_all_query_kmers_to_fasta_path(&[query], 3, &kmers).unwrap();
        write_ggcat_simplitigs_from_kmer_fasta(
            &kmers,
            &output,
            3,
            1,
            1,
            Some(1),
            &ggcat_tmp,
            stats.not_filtered_kmers,
            false,
            None,
        )
        .unwrap();

        let output_text = fs::read_to_string(&output).unwrap();
        let expected = encoded_kmers_from_fasta_text(&query_text, 3);
        let actual = encoded_kmers_from_fasta_text(&output_text, 3);

        assert_eq!(actual, expected);
        assert_eq!(stats.scanned_kmers, 64);
        assert_eq!(stats.not_filtered_kmers, 64);
        assert_eq!(stats.unique_not_filtered_kmers, None);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn regular_output_cuts_on_filtered_kmers_and_preserves_headers() {
        let dir = create_temp_dir("mczi-regular-cut-test").unwrap();
        let query = dir.join("query.fa");
        let output = dir.join("regular.fa");
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        fs::write(&query, b">q1 km:f:7.0\nCCCAAAGGG\n").unwrap();

        let dictionary =
            build_dictionary_from_simplitigs(vec![b"AAA".to_vec()], 3, 1, 1, 1, &tmp).unwrap();
        let stats =
            write_regular_query_output_absent_from_index(&dictionary, &[query], &output).unwrap();
        let text = fs::read_to_string(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 7);
        assert_eq!(stats.filtered_kmers, 1);
        assert_eq!(stats.not_filtered_kmers, 6);
        assert_eq!(stats.regular_output_nucleotides, Some(10));
        assert_eq!(text, ">q1 km:f:7.0\nCCCAA\n>q1 km:f:7.0\nAAGGG\n");
    }

    #[test]
    fn regular_output_without_index_preserves_records_and_stats() {
        let dir = create_temp_dir("mczi-regular-no-index-test").unwrap();
        let query = dir.join("query.fa");
        let output = dir.join("regular.fa");
        fs::write(&query, b">q1 km:f:2.0\nAAACG\n>q2 km:f:3.0\nCCCG\n").unwrap();

        let stats = write_regular_query_output_without_index(&[query], 3, &output).unwrap();
        let text = fs::read_to_string(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 5);
        assert_eq!(stats.filtered_kmers, 0);
        assert_eq!(stats.not_filtered_kmers, 5);
        assert_eq!(stats.regular_output_nucleotides, Some(9));
        assert_eq!(text, ">q1 km:f:2.0\nAAACG\n>q2 km:f:3.0\nCCCG\n");
    }

    #[test]
    fn regular_output_without_index_skips_invalid_fastx_query_inputs() {
        let dir = create_temp_dir("mczi-skip-invalid-query-output-test").unwrap();
        let invalid = dir.join("queries.txt");
        let valid = dir.join("query.fa");
        let output = dir.join("regular.fa");
        fs::write(&invalid, b"not a FASTA or FASTQ file\n").unwrap();
        fs::write(&valid, b">q\nAAAC\n").unwrap();

        let stats =
            write_regular_query_output_without_index(&[invalid, valid], 3, &output).unwrap();
        let text = fs::read_to_string(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 2);
        assert_eq!(stats.filtered_kmers, 0);
        assert_eq!(stats.not_filtered_kmers, 2);
        assert_eq!(stats.regular_output_nucleotides, Some(4));
        assert_eq!(text, ">q\nAAAC\n");
    }

    #[test]
    fn no_output_scan_reports_regular_nucleotide_count() {
        let dir = create_temp_dir("mczi-no-output-scan-test").unwrap();
        let query = dir.join("query.fa");
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        fs::write(&query, b">q1 km:f:7.0\nCCCAAAGGG\n").unwrap();

        let dictionary =
            build_dictionary_from_simplitigs(vec![b"AAA".to_vec()], 3, 1, 1, 1, &tmp).unwrap();
        let stats = scan_regular_query_output_absent_from_index(&dictionary, &[query]).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 7);
        assert_eq!(stats.filtered_kmers, 1);
        assert_eq!(stats.not_filtered_kmers, 6);
        assert_eq!(stats.regular_output_nucleotides, Some(10));
    }

    #[test]
    fn no_output_scan_without_index_reports_regular_nucleotide_count() {
        let dir = create_temp_dir("mczi-no-output-empty-index-scan-test").unwrap();
        let query = dir.join("query.fa");
        fs::write(&query, b">q1\nAAACG\n>short\nAA\n").unwrap();

        let stats = scan_regular_query_output_without_index(&[query], 3).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 3);
        assert_eq!(stats.filtered_kmers, 0);
        assert_eq!(stats.not_filtered_kmers, 3);
        assert_eq!(stats.regular_output_nucleotides, Some(5));
    }

    #[test]
    fn mczi_stat_values_are_comma_grouped() {
        assert_eq!(format_stat_u64(0), "0");
        assert_eq!(format_stat_u64(572400), "572,400");
        assert_eq!(format_stat_u64(9832424437), "9,832,424,437");
        assert_eq!(format_stat_u64(30162227617), "30,162,227,617");
    }

    #[test]
    fn no_output_scan_without_index_skips_invalid_fastx_query_inputs() {
        let dir = create_temp_dir("mczi-skip-invalid-query-test").unwrap();
        let invalid = dir.join("queries.txt");
        let valid = dir.join("query.fa");
        fs::write(&invalid, b"not a FASTA or FASTQ file\n").unwrap();
        fs::write(&valid, b">q\nAAAC\n").unwrap();

        let stats = scan_regular_query_output_without_index(&[invalid, valid], 3).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 2);
        assert_eq!(stats.filtered_kmers, 0);
        assert_eq!(stats.not_filtered_kmers, 2);
        assert_eq!(stats.regular_output_nucleotides, Some(4));
    }

    #[test]
    fn scan_query_stats_count_filtered_and_absent_kmers() {
        let dir = create_temp_dir("mczi-scan-stats-test").unwrap();
        let query1 = dir.join("query1.fa");
        let query2 = dir.join("query2.fa");
        let tmp = dir.join("tmp");
        fs::create_dir(&tmp).unwrap();
        fs::write(&query1, b">q1\nAAAT\n").unwrap();
        fs::write(&query2, b">q2\nCCCC\n").unwrap();

        let dictionary =
            build_dictionary_from_simplitigs(vec![b"AAA".to_vec()], 3, 1, 1, 1, &tmp).unwrap();
        let stats = scan_query_kmers_absent_from_index(&dictionary, &[query1, query2]).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 4);
        assert_eq!(stats.filtered_kmers, 1);
        assert_eq!(stats.not_filtered_kmers, 3);
        assert_eq!(stats.unique_not_filtered_kmers, None);
    }

    #[test]
    fn all_query_kmer_writer_keeps_duplicate_windows_for_ggcat_input() {
        let dir = create_temp_dir("mczi-all-kmers-test").unwrap();
        let query = dir.join("query.fa");
        let output = dir.join("kmers.fa");
        fs::write(&query, b">q\nAAAA\n").unwrap();

        let stats = write_all_query_kmers_to_fasta_path(&[query], 3, &output).unwrap();
        let text = fs::read_to_string(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(stats.scanned_kmers, 2);
        assert_eq!(stats.not_filtered_kmers, 2);
        assert_eq!(text.matches(">MCZI_query_kmer_").count(), 2);
        assert_eq!(text.matches("\nAAA\n").count(), 2);
    }

    #[test]
    fn reform_output_path_preserves_regular_abundance_mean() {
        let dir = create_temp_dir("mczi-reform-mean-test").unwrap();
        let input = dir.join("regular.fa");
        let output = dir.join("out.fa");
        fs::write(&input, b">q1 km:f:2.0\nAAAACG\n>q2 km:f:5.0\nCGTTT\n").unwrap();

        reform_output_path(
            &input,
            &output,
            3,
            &dir,
            Some(reformer::AbundanceMode::Mean),
        )
        .unwrap();
        let text = fs::read_to_string(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(text.contains("AAAACGTTT"));
        assert!(text.contains(">A km:f:3"));
    }

    #[test]
    fn reform_output_path_preserves_regular_abundance_runs() {
        let dir = create_temp_dir("mczi-reform-runs-test").unwrap();
        let input = dir.join("regular.fa");
        let output = dir.join("out.fa");
        fs::write(
            &input,
            b">q1 km:f:2.0:1:3.0:3\nAAAACG\n>q2 km:f:3.0:1:4.0:2\nCGTTT\n",
        )
        .unwrap();

        reform_output_path(
            &input,
            &output,
            3,
            &dir,
            Some(reformer::AbundanceMode::Runs),
        )
        .unwrap();
        let text = fs::read_to_string(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(text.contains("AAAACGTTT"));
        assert!(text.contains(">A km:f:2:1:3:4:4:2"));
    }

    #[test]
    fn reform_output_path_writes_empty_output_for_empty_input() {
        let dir = create_temp_dir("mczi-reform-empty-test").unwrap();
        let input = dir.join("regular.fa");
        let output = dir.join("out.fa");
        fs::write(&input, b"").unwrap();

        reform_output_path(&input, &output, 3, &dir, None).unwrap();
        let bytes = fs::read(&output).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(bytes.is_empty());
    }

    #[test]
    fn zero_kmer_ggcat_output_path_writes_empty_compressed_output() {
        use std::io::Read as _;

        let dir = create_temp_dir("mczi-empty-ggcat-output-test").unwrap();
        let input = dir.join("empty.fa");
        let output = dir.join("out.fa.zst");
        fs::write(&input, b"").unwrap();

        write_ggcat_simplitigs_from_kmer_fasta(
            &input,
            &output,
            3,
            1,
            1,
            Some(1),
            &dir,
            0,
            false,
            None,
        )
        .unwrap();
        let mut text = String::new();
        zstd::stream::read::Decoder::new(File::open(&output).unwrap())
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(text.is_empty());
    }

    #[test]
    fn simplitig_output_rejects_abundance_reforming() {
        let err = run(Cli {
            index_input: Vec::new(),
            index_fofn: false,
            query_input: Vec::new(),
            query_fofn: false,
            kmer_size: 3,
            minimizer_size: 1,
            threshold: 0,
            output: Some(PathBuf::from("out.fa")),
            output_suffix: "filtered".to_owned(),
            output_mode: OutputMode::Simplitig,
            reform_output: true,
            reform_abundance_mode: Some(reformer::AbundanceMode::Mean),
            threads: None,
            partition_count: DEFAULT_PARTITION_COUNT,
            ram_limit_gib: 1,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("--reform-abundance-mode requires --output-mode regular"));
    }

    #[test]
    fn no_output_mode_does_not_require_output() {
        let dir = create_temp_dir("mczi-no-output-no-path-test").unwrap();
        let index = dir.join("index.fa");
        let index_fofn = dir.join("index.fofn");
        let query = dir.join("query.fa");
        fs::write(&index, b">idx\nAAA\n").unwrap();
        fs::write(&index_fofn, format!("{}\n", index.display())).unwrap();
        fs::write(&query, b">q\nCCCG\n").unwrap();

        run(Cli {
            index_input: vec![index_fofn],
            index_fofn: true,
            query_input: vec![query],
            query_fofn: false,
            kmer_size: 3,
            minimizer_size: 1,
            threshold: 1,
            output: None,
            output_suffix: "filtered".to_owned(),
            output_mode: OutputMode::NoOutput,
            reform_output: false,
            reform_abundance_mode: None,
            threads: None,
            partition_count: DEFAULT_PARTITION_COUNT,
            ram_limit_gib: 1,
        })
        .unwrap();

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_output_mode_rejects_reform_output() {
        let err = run(Cli {
            index_input: Vec::new(),
            index_fofn: false,
            query_input: Vec::new(),
            query_fofn: false,
            kmer_size: 3,
            minimizer_size: 1,
            threshold: 0,
            output: None,
            output_suffix: "filtered".to_owned(),
            output_mode: OutputMode::NoOutput,
            reform_output: true,
            reform_abundance_mode: None,
            threads: None,
            partition_count: DEFAULT_PARTITION_COUNT,
            ram_limit_gib: 1,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("--reform-output cannot be combined with --output-mode no-output"));
    }

    #[test]
    fn query_fofn_output_names_insert_suffix_before_sequence_and_compression_extensions() {
        assert_eq!(
            suffixed_query_file_name("sample.fa.zst", "filtered"),
            "sample.filtered.fa.zst"
        );
        assert_eq!(
            suffixed_query_file_name("sample.fastq.gz", "clean"),
            "sample.clean.fastq.gz"
        );
        assert_eq!(
            suffixed_query_file_name("sample.unitigs.fna.xz", "mczi"),
            "sample.unitigs.mczi.fna.xz"
        );
        assert_eq!(
            suffixed_query_file_name("sample", "filtered"),
            "sample.filtered"
        );
    }

    #[test]
    fn query_fofn_output_plan_uses_optional_output_directory() {
        let dir = create_temp_dir("mczi-output-plan-test").unwrap();
        let output_dir = dir.join("outputs");
        let query = dir.join("query.fa.gz");
        fs::write(&query, b">q\nAAA\n").unwrap();

        let plan = build_query_output_plan(true, &[query], Some(&output_dir), ".kept").unwrap();
        let QueryOutputPlan::PerInput { jobs } = plan else {
            panic!("expected per-input query output plan");
        };
        let output_path = jobs[0].output_path.clone();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(output_path, output_dir.join("query.kept.fa.gz"));
    }

    #[test]
    fn query_fofn_regular_empty_index_writes_one_suffixed_output_per_query_file() {
        let dir = create_temp_dir("mczi-query-fofn-outputs-test").unwrap();
        let index = dir.join("index.fa");
        let index_fofn = dir.join("index.fofn");
        let query1 = dir.join("query1.fa");
        let query2 = dir.join("query2.fa");
        let query_fofn = dir.join("queries.fofn");
        fs::write(&index, b">idx\nAAA\n").unwrap();
        fs::write(&index_fofn, format!("{}\n", index.display())).unwrap();
        fs::write(&query1, b">q1\nCCCG\n").unwrap();
        fs::write(&query2, b">q2\nTTTA\n").unwrap();
        fs::write(
            &query_fofn,
            format!("{}\n{}\n", query1.display(), query2.display()),
        )
        .unwrap();

        run(Cli {
            index_input: vec![index_fofn],
            index_fofn: true,
            query_input: vec![query_fofn],
            query_fofn: true,
            kmer_size: 3,
            minimizer_size: 1,
            threshold: 1,
            output: None,
            output_suffix: "filtered".to_owned(),
            output_mode: OutputMode::Regular,
            reform_output: false,
            reform_abundance_mode: None,
            threads: None,
            partition_count: DEFAULT_PARTITION_COUNT,
            ram_limit_gib: 1,
        })
        .unwrap();

        let out1 = fs::read_to_string(dir.join("query1.filtered.fa")).unwrap();
        let out2 = fs::read_to_string(dir.join("query2.filtered.fa")).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(out1, ">q1\nCCCG\n");
        assert_eq!(out2, ">q2\nTTTA\n");
    }

    fn encoded_kmers_from_fasta_text(text: &str, k: usize) -> AHashSet<EncodedKmer> {
        let mut kmers = AHashSet::new();
        let mut seq = Vec::new();
        for line in text.lines() {
            if line.starts_with('>') {
                insert_sequence_kmers(&seq, k, &mut kmers);
                seq.clear();
            } else {
                seq.extend_from_slice(line.as_bytes());
            }
        }
        insert_sequence_kmers(&seq, k, &mut kmers);
        kmers
    }

    fn insert_sequence_kmers(seq: &[u8], k: usize, kmers: &mut AHashSet<EncodedKmer>) {
        for_each_canonical_encoded_kmer(seq, k, |_, encoded| {
            kmers.insert(encoded);
        });
    }
}
