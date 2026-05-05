#!/usr/bin/env bash
set -u

run_dir="${1:-bench_runs/3g_10x_10kb_e001}"
threads="${THREADS:-8}"
bin="${MC_BIN:-target/release/MC}"
reads="$run_dir/reads_3g_10x_10kb_e0.001.fa"
output="$run_dir/mc_k31_m21_x5.kff"
time_log="$run_dir/mc.no_perf.time.stderr.log"
stdout_log="$run_dir/mc.no_perf.stdout.log"
pidstat_log="$run_dir/mc.no_perf.pidstat.log"
temp_log="$run_dir/mc.no_perf.temp_usage.tsv"
tmp_dir="$run_dir/tmp"

mkdir -p "$run_dir"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"
rm -f "$output" "$time_log" "$stdout_log" "$pidstat_log" "$temp_log"

pidstat -h -r -u -d -C '^MC$' 5 >"$pidstat_log" &
pidstat_pid=$!

(
  while true; do
    bytes=$(du -sb "$tmp_dir" 2>/dev/null | awk '{print $1}')
    bytes=${bytes:-0}
    printf '%(%s)T\t%s\n' -1 "$bytes"
    sleep 1
  done
) >"$temp_log" &
temp_pid=$!

/usr/bin/time -v \
  env TMPDIR="$tmp_dir" \
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

kill "$pidstat_pid" "$temp_pid" 2>/dev/null
wait "$pidstat_pid" "$temp_pid" 2>/dev/null

exit "$status"
