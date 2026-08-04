"""Validate native AFSIM decision pauses using the real Mission scenario.
This collects no training trajectory and performs no network update.
"""
import argparse, json, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from envs.afsim_env import AFSIMIslandEnv

def p95(values):
    values=sorted(values)
    return values[max(0, min(len(values)-1, int((len(values)-1)*.95)))] if values else 0.0

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--workers',type=int,default=4); a.add_argument('--base-port',type=int,default=50050)
    a.add_argument('--steps',type=int,default=50); a.add_argument('--pause-timeout',type=float,default=45)
    a.add_argument('--launch-mission',action='store_true'); a.add_argument('--report-file',required=True)
    a.add_argument('--scenario-dir',default=r'D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air\scenarios')
    a.add_argument('--mission',default=r'D:\junq\afsim_work\afsim-2.9.0-win64_bin\bin_release\mission.exe')
    args=a.parse_args(); scenario_dir=Path(args.scenario_dir); processes=[]; envs=[]
    try:
        if args.launch_mission:
            for i in range(args.workers):
                name=f'island_assault_linux_train.worker_{i}.txt'
                processes.append(subprocess.Popen([args.mission, f'scenarios\\{name}'], cwd=str(scenario_dir.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for i in range(args.workers):
            env=AFSIMIslandEnv(bind=True,auto_start_warlock=False,local_address=('0.0.0.0',args.base_port+i))
            env.native_decision_pause_control=True; env.native_decision_pause_timeout=args.pause_timeout
            if not env.wait_for_platforms(['red_recon_1','red_attack_1','red_transport_1'],timeout=45):
                raise RuntimeError(f'worker {i} platform registration timeout')
            envs.append(env)
        hist=[[env._current_sim_time()] for env in envs]; walls=[[] for _ in envs]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for step in range(args.steps):
                started=time.monotonic(); futures=[ex.submit(env.step_flat) for env in envs]
                for i,f in enumerate(futures):
                    f.result(); hist[i].append(envs[i]._current_sim_time()); walls[i].append(time.monotonic()-started)
                print(f'native_pause_progress step={step+1}/{args.steps} sim_times={[round(x[-1],3) for x in hist]}',flush=True)
        workers={}
        for i in range(args.workers):
            d=[b-a for a,b in zip(hist[i],hist[i][1:])]
            effective=args.steps*3600/sum(d) if sum(d)>0 else 0
            workers[str(args.base_port+i)]={'steps':args.steps,'steps_with_sim_progress':sum(x>0 for x in d),'sim_elapsed_seconds':sum(d),'mean_step_sim_seconds':statistics.mean(d),'p95_step_sim_seconds':p95(d),'mean_step_wall_seconds':statistics.mean(walls[i]),'effective_decisions_per_sim_hour':effective,'passed':47.5<=effective<=52.5 and p95(d)<=80 and sum(x>0 for x in d)==args.steps}
        report={'mode':'real_mission_native_pause_no_actions_no_update','workers':workers,'passed':all(x['passed'] for x in workers.values())}
        Path(args.report_file).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print('NATIVE_PAUSE_BENCHMARK='+json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        for env in envs:
            try: env.close()
            except Exception: pass
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill()
if __name__=='__main__': main()