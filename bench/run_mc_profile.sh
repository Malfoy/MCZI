#!/usr/bin/env bash
set -u

run_dir="${1:-bench_runs/3g_10x_10kb_e001}"
threads="${THREADS:-8}"
bin="${MC_BIN:-target/release/MC}"
reads="$run_dir/reads_3g_10x_10kb_e0.001.fa"
output="$run_dir/mc_k31_m21_x5.kff"
perf_log="$run_dir/mc.perf.stat"
time_log="$run_dir/mc.time.stderr.log"
stdout_log="$run_dir/mc.stdout.log"
pidstat_log="$run_dir/mc.pidstat.log"

mkdir -p "$run_dir"
rm -f "$output" "$perf_log" "$time_log" "$stdout_log" "$pidstat_log"

pidstat -h -r -u -d -C '^MC$' 5 >"$pidstat_log" &
pidstat_pid=$!

perf stat \
  -e duration_time,user_time,system_time \
  -o "$perf_log" \
  -- \
  /usr/bin/time -v \
  "$bin" \
  --input "$reads" \
  --kmer-size 31 \
  --minimizer-size 21 \
  --threshold 5 \
  --output "$output" \
  --format kff \
  --threads "$threads" \
  >"$stdout_log" 2>"$time_log"
status=$?

kill "$pidstat_pid" 2>/dev/null
wait "$pidstat_pid" 2>/dev/null

exit "$status"
