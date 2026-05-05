#!/usr/bin/env bash
set -u

run_dir="${1:-bench_runs/300m_10x_10kb_e001}"
threads="${THREADS:-8}"
kmc_bin="${KMC_BIN:-tools/KMC/bin/kmc}"
reads="$run_dir/reads_3g_10x_10kb_e0.001.fa"
out_prefix="$run_dir/kmc_k31_ci5"
tmp_dir="$run_dir/kmc_tmp"
time_log="$run_dir/kmc.time.stderr.log"
stdout_log="$run_dir/kmc.stdout.log"
stderr_log="$run_dir/kmc.stderr.log"
pidstat_log="$run_dir/kmc.pidstat.log"
temp_log="$run_dir/kmc.temp_usage.tsv"
summary_json="$run_dir/kmc.summary.json"

mkdir -p "$run_dir"
rm -rf "$tmp_dir" "$out_prefix.kmc_pre" "$out_prefix.kmc_suf"
mkdir -p "$tmp_dir"
rm -f "$time_log" "$stdout_log" "$stderr_log" "$pidstat_log" "$temp_log" "$summary_json"

pidstat -h -r -u -d -C '^kmc$' 1 >"$pidstat_log" &
pidstat_pid=$!

(
  while true; do
    bytes=$(du -sb "$tmp_dir" 2>/dev/null | awk '{print $1}')
    bytes=${bytes:-0}
    printf '%(%s)T\t%s\n' -1 "$bytes"
    sleep 0.2
  done
) >"$temp_log" &
temp_pid=$!

/usr/bin/time -v -o "$time_log" \
  "$kmc_bin" \
  -fa \
  -k31 \
  -ci5 \
  -t"$threads" \
  -m24 \
  -j"$summary_json" \
  "$reads" \
  "$out_prefix" \
  "$tmp_dir" \
  >"$stdout_log" 2>"$stderr_log"
status=$?

kill "$pidstat_pid" "$temp_pid" 2>/dev/null
wait "$pidstat_pid" "$temp_pid" 2>/dev/null

exit "$status"
