use anyhow::{Context, Result, ensure};
use mc::{
    CounterConfig, run_inputs_phase12, run_inputs_phase12_antilex, run_inputs_phase12_direct,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .context("usage: mc-phase12 <reads.fa> [k=31] [m=21] [threshold=5] [threads=8]")?;
    let k = parse_arg(args.next(), 31, "k")?;
    let m = parse_arg(args.next(), 21, "m")?;
    let threshold = parse_arg(args.next(), 5, "threshold")?;
    let threads = parse_arg(args.next(), 8, "threads")?;
    let order = args.next().unwrap_or_else(|| "simd-value".to_string());
    ensure!(args.next().is_none(), "too many arguments");
    ensure!(threads > 0, "threads must be greater than 0");
    ensure!(
        order == "simd-value" || order == "value" || order == "direct" || order == "antilex",
        "order must be simd-value, direct, or antilex"
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
    let stats = match order.as_str() {
        "direct" => run_inputs_phase12_direct(&[PathBuf::from(input)], config)?,
        "antilex" => run_inputs_phase12_antilex(&[PathBuf::from(input)], config)?,
        "simd-value" | "value" => run_inputs_phase12(&[PathBuf::from(input)], config)?,
        _ => unreachable!(),
    };

    println!("order\t{order}");
    println!("k\t{k}");
    println!("m\t{m}");
    println!("threshold\t{threshold}");
    println!("threads\t{threads}");
    println!("unique_minimizer_hashes\t{}", stats.unique_minimizer_hashes);
    println!("phase1_seconds\t{:.6}", stats.phase1.as_secs_f64());
    println!("phase2_seconds\t{:.6}", stats.phase2.as_secs_f64());
    println!("total_phase12_seconds\t{:.6}", stats.total.as_secs_f64());
    println!("partition_bytes\t{}", stats.partition_bytes);

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
