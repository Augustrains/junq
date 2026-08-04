"""Launch isolated Warlock/Python bottom-training workers in parallel."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "envs" / "afsim_units.json"
DEFAULT_TRAIN = ROOT / "train" / "train_bottom_mappo.py"

def parse_args():
    p = argparse.ArgumentParser(description="Launch one isolated Warlock + Python training process per UDP port.")
    p.add_argument("--workers", type=int, required=True, help="Number of parallel Warlock/Python workers.")
    p.add_argument("--base-port", type=int, default=50050)
    p.add_argument("--simulation-clock-rate", type=float, default=40.0)
    p.add_argument("--bottom-decisions-per-hour", type=float, default=50.0)
    p.add_argument("--config-path", default=str(DEFAULT_CONFIG))
    p.add_argument("--scenario-file", default="", help="Source scenario; default uses scenario_file in config.")
    p.add_argument("--udp-target-address", default="127.0.0.1")
    p.add_argument("--device", default="cpu", help="Passed to every training worker; use explicit cuda:N only when intended.")
    p.add_argument("--output-dir", default=str(ROOT / "parallel_runs"))
    p.add_argument("--train-script", default=str(DEFAULT_TRAIN))
    p.add_argument("--train-arg", action="append", default=[], help="Extra argument forwarded to every training process; repeat as needed.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def rewrite_scenario(source: Path, worker: int, port: int, address: str, clock_rate: float, destination: Path):
    text = source.read_text(encoding="utf-8")
    block = "udpnet\n   port {0}\n   address {1}\nend_udpnet".format(port, address)
    text, count = re.subn(r"(?ms)^udpnet\s*\n.*?^end_udpnet", block, text, count=1)
    if count != 1:
        raise ValueError("source scenario must contain exactly one active udpnet block: {0}".format(source))
    line = "clock_rate {0:.12g}".format(clock_rate)
    text, count = re.subn(r"(?m)^\s*clock_rate\s+\S+.*$", line, text, count=1)
    if count == 0:
        text = text.rstrip() + "\n" + line + "\n"
    destination.write_text(text, encoding="utf-8")

def main():
    args = parse_args()
    if args.workers <= 0 or not 0 < args.base_port + args.workers - 1 < 65536:
        raise ValueError("invalid --workers/--base-port range")
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_path).resolve(); config = json.loads(config_path.read_text(encoding="utf-8"))
    scenario_cfg = config.setdefault("scenario", {})
    scenario_dir = Path(scenario_cfg["scenario_dir"]).resolve()
    source = Path(args.scenario_file).resolve() if args.scenario_file else (scenario_dir / scenario_cfg["scenario_file"]).resolve()
    if not source.is_file(): raise FileNotFoundError(source)
    processes = []
    for worker in range(args.workers):
        port = args.base_port + worker; workdir = root / "worker_{0}".format(worker); workdir.mkdir(exist_ok=True)
        # Keep the scenario beside its source so all relative include_once paths remain valid.
        scenario = source.parent / "{0}.parallel_worker_{1}.port_{2}{3}".format(source.stem, worker, port, source.suffix)
        rewrite_scenario(source, worker, port, args.udp_target_address, args.simulation_clock_rate, scenario)
        worker_config = json.loads(json.dumps(config)); sc = worker_config["scenario"]
        sc["scenario_dir"] = str(scenario.parent); sc["scenario_file"] = scenario.name
        sc["warlock_log_path"] = str(workdir / "warlock.log")
        cfg = workdir / "env.json"; cfg.write_text(json.dumps(worker_config, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = [sys.executable, str(Path(args.train_script).resolve()), "--bind", "--auto-start-warlock", "--config-path", str(cfg), "--local-address", "0.0.0.0:{0}".format(port), "--simulation-clock-rate", str(args.simulation_clock_rate), "--bottom-decisions-per-hour", str(args.bottom_decisions_per_hour), "--device", args.device, "--checkpoint-dir", str(workdir / "checkpoints"), "--metrics-file", str(workdir / "metrics.jsonl")] + list(args.train_arg)
        print("WORKER_{0}_COMMAND=".format(worker) + subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            log = open(workdir / "train.log", "w", encoding="utf-8", buffering=1)
            processes.append((worker, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True), log))
    if args.dry_run: return 0
    results = {}
    try:
        for worker, process, log in processes: results[worker] = process.wait()
    except KeyboardInterrupt:
        for _, process, _ in processes:
            if process.poll() is None: process.terminate()
        raise
    finally:
        for _, _, log in processes: log.close()
    print("PARALLEL_TRAINING_EXIT_CODES=" + json.dumps(results, sort_keys=True))
    return 0 if all(code == 0 for code in results.values()) else 1
if __name__ == "__main__": raise SystemExit(main())