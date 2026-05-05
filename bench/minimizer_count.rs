use anyhow::{Context, Result, ensure};
use mc::{
    CounterConfig, count_inputs_minimizer_phase, count_inputs_minimizer_phase_antilex,
    count_inputs_minimizer_phase_direct, count_inputs_minimizer_phase_packed,
    count_inputs_minimizer_phase_packed_direct,
};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .context("usage: mc-minimizer-count <reads.fa> [k=31] [m=21] [threshold=5] [threads=8]")?;
    let k = parse_arg(args.next(), 31, "k")?;
    let m = parse_arg(args.next(), 21, "m")?;
    let threshold = parse_arg(args.next(), 5, "threshold")?;
    let threads = parse_arg(args.next(), 8, "threads")?;
    let order = args.next().unwrap_or_else(|| "simd-value".to_string());
    ensure!(args.next().is_none(), "too many arguments");
    ensure!(
        order == "simd-value"
            || order == "value"
            || order == "direct"
            || order == "antilex"
            || order == "packed"
            || order == "packed-direct",
        "order must be simd-value, direct, antilex, packed, or packed-direct"
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("failed to configure rayon thread pool")?;

    let config = CounterConfig {
        k,
        minimizer: m,
        threshold: threshold as u64,
    };
    let started = Instant::now();
    let unique_hashes = if order == "antilex" {
        count_inputs_minimizer_phase_antilex(&[PathBuf::from(input)], config)?
    } else if order == "direct" {
        count_inputs_minimizer_phase_direct(&[PathBuf::from(input)], config)?
    } else if order == "packed" {
        count_inputs_minimizer_phase_packed(&[PathBuf::from(input)], config)?
    } else if order == "packed-direct" {
        count_inputs_minimizer_phase_packed_direct(&[PathBuf::from(input)], config)?
    } else if order == "simd-value" || order == "value" {
        count_inputs_minimizer_phase(&[PathBuf::from(input)], config)?
    } else {
        unreachable!()
    };
    let elapsed = started.elapsed().as_secs_f64();

    println!("order\t{order}");
    println!("k\t{k}");
    println!("m\t{m}");
    println!("threshold\t{threshold}");
    println!("threads\t{threads}");
    println!("unique_minimizer_hashes\t{unique_hashes}");
    println!("elapsed_seconds\t{elapsed:.6}");
    eprintln!("MC_MINIMIZER_COUNT\t{elapsed:.6}");

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
