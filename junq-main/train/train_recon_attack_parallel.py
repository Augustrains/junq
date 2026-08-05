"""Synchronous multi-Warlock recon/attack HAPPO with one shared learner."""
from __future__ import annotations
import argparse, base64, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.checkpointing import append_metrics_jsonl, load_training_checkpoint, save_training_checkpoint
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from train.happo_trainer import HAPPOConfig,HAPPOTrainer
from train.rule_driven_rollout_collector import RuleDrivenRolloutCollector
from train.recon_attack_stage import high_quality_landing_status,recon_attack_terminal_bonus
from train.stage_episode_buffer import TaskTrajectoryBuffer

def args_parser():
 p=argparse.ArgumentParser(description="Aggregate trajectories from multiple Warlocks into one HAPPO learner")
 p.add_argument('--workers',type=int,required=True); p.add_argument('--base-port',type=int,default=50050)
 p.add_argument('--share-policy-by-type',action='store_true')
 p.add_argument('--config-path',default=''); p.add_argument('--device',default='auto')
 p.add_argument('--updates',type=int,default=1); p.add_argument('--episodes-per-worker',type=int,default=100)
 p.add_argument('--max-episode-steps',type=int,default=5000); p.add_argument('--recon-min-samples',type=int,default=2048); p.add_argument('--attack-min-samples',type=int,default=2048)
 p.add_argument('--hidden-size',type=int,default=128); p.add_argument('--update-epochs',type=int,default=4); p.add_argument('--minibatch-size',type=int,default=64); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--gamma',type=float,default=.99); p.add_argument('--gae-lambda',type=float,default=.95)
 p.add_argument('--bottom-decisions-per-hour',type=float,default=50); p.add_argument('--simulation-clock-rate',type=float,default=15); p.add_argument('--decision-seconds',type=float,default=0); p.add_argument('--adaptive-decision-timing',action='store_true')
 p.add_argument('--platform-timeout',type=float,default=120); p.add_argument('--platform-state-stall-seconds',type=float,default=30)
 p.add_argument('--bottom-global-reward-weight',type=float,default=.1); p.add_argument('--bottom-global-reward-clip',type=float,default=10); p.add_argument('--deadline-miss-penalty',type=float,default=-50)
 p.add_argument('--min-recon-alive',type=int,default=3); p.add_argument('--min-attack-alive',type=int,default=3); p.add_argument('--min-loaded-transports',type=int,default=1)
 p.add_argument('--checkpoint-dir',default=str(ROOT/'checkpoints'/'happo_recon_attack_parallel')); p.add_argument('--metrics-file',default=''); p.add_argument('--resume',default='')
 p.add_argument('--warlock-ssh-target',required=True); p.add_argument('--warlock-ssh-port',type=int,default=22); p.add_argument('--warlock-ssh-key',default=''); p.add_argument('--warlock-task-prefix',default='AFSIM-Warlock-'); p.add_argument('--warlock-start-delay',type=float,default=2)
 p.add_argument('--enable-negative-rewards',action='store_true'); return p.parse_args()

def ssh(args,command):
 cmd=['ssh','-p',str(args.warlock_ssh_port)];
 if args.warlock_ssh_key: cmd += ['-i',args.warlock_ssh_key]
 cmd += ['-o','BatchMode=yes','-o','ConnectTimeout=15',args.warlock_ssh_target,command]
 subprocess.run(cmd,check=True)

def windows_powershell(args, script):
 script = "$ProgressPreference='SilentlyContinue'; " + script
 encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
 ssh(args, "powershell -NoProfile -EncodedCommand {0}".format(encoded))

def task_command(args,worker,action):
 task=(args.warlock_task_prefix+str(worker)).replace("'","''")
 if action=='start':
  body="Start-ScheduledTask -TaskName '{0}'".format(task)
 elif action=='restart':
  # Windows OpenSSH parses nested quotes unreliably.  Send this through
  # PowerShell -EncodedCommand, then wait for the previous Mission to exit.
  body=("Stop-ScheduledTask -TaskName '{0}' -ErrorAction SilentlyContinue; "
        "$deadline=(Get-Date).AddSeconds(20); "
        "do {{ Start-Sleep -Milliseconds 250; $state=(Get-ScheduledTask -TaskName '{0}').State }} "
        "while ($state -eq 'Running' -and (Get-Date) -lt $deadline); "
        "if ($state -eq 'Running') {{ throw 'task did not stop: {0}' }}; "
        "Start-ScheduledTask -TaskName '{0}'").format(task)
 else:
  body=("Stop-ScheduledTask -TaskName '{0}' -ErrorAction SilentlyContinue; "
        "$deadline=(Get-Date).AddSeconds(20); "
        "do {{ Start-Sleep -Milliseconds 250; $state=(Get-ScheduledTask -TaskName '{0}').State }} "
        "while ($state -eq 'Running' -and (Get-Date) -lt $deadline); "
        "if ($state -eq 'Running') {{ throw 'task did not stop: {0}' }}").format(task)
 windows_powershell(args, body)

def start_worker(args,w,startup_attempt=1):
 env=w['env']
 task_command(args,w['id'],'stop')
 discarded=env.discard_pending_udp_messages()
 env.prepare_for_scenario_restart()
 task_command(args,w['id'],'start')
 time.sleep(max(0,args.warlock_start_delay))
 print('worker_restart_clean',w['id'],'startup_attempt',startup_attempt,'discarded_udp',discarded,flush=True)
 names=list(env.platforms)
 initial_timeout=min(float(args.platform_timeout),20.0) if env.native_decision_pause_control else args.platform_timeout
 ready=env.wait_for_platforms(names,timeout=initial_timeout)
 retries=0
 while not ready and env.native_decision_pause_control and retries<3:
  retries+=1; env.native_decision_ready=False; env._send({'MsgType':'SimRestart'})
  env._drain_messages(timeout=env.native_decision_pause_timeout,until_decision_ready=True)
  ready=env.wait_for_platforms(names,timeout=2.0)
  print('worker_registration_retry',w['id'],'attempt',retries,'decision_ready',env.native_decision_ready,'registered',sum(p.platform_id is not None for p in env.platforms.values()),flush=True)
 if ready and env.native_decision_pause_control:
  # The initial native boundary is T=0, so a timestamp of zero is valid.
  if not getattr(env,'native_decision_ready_seen',False):
   env.native_decision_ready=False
   env._drain_messages(timeout=env.native_decision_pause_timeout,until_decision_ready=True)
  ready=bool(getattr(env,'native_decision_ready_seen',False))
  print('worker_native_pause_ready',w['id'],'port',w['port'],'ready',ready,'sim',env.native_decision_ready_time,flush=True)
  if ready:
   try:
    env.verify_native_decision_pause()
    print('worker_native_pause_verified',w['id'],'port',w['port'],'sim',env.native_decision_ready_time,flush=True)
   except RuntimeError as error:
    ready=False
    print('worker_native_pause_verification_failed',w['id'],'port',w['port'],'error',error,flush=True)
 w['episode_start_decision_time']=float(env.native_decision_ready_time) if env.native_decision_pause_control else float(env._current_sim_time())
 print('worker_ready',w['id'],'port',w['port'],'ready',ready,'registered',sum(p.platform_id is not None for p in env.platforms.values()),flush=True)
 if not ready:
  if startup_attempt < 2:
   print('worker_startup_relaunch',w['id'],'after_failed_attempt',startup_attempt,flush=True)
   return start_worker(args,w,startup_attempt=startup_attempt+1)
  raise RuntimeError(
   'worker {0} platforms did not register or native pause did not arm after {1} clean launches '
   '(registered={2}, decision_ready_seen={3})'.format(
    w['id'], startup_attempt,
    sum(p.platform_id is not None for p in env.platforms.values()),
    bool(getattr(env,'native_decision_ready_seen',False)),
   )
  )
 collector_kwargs=dict(bottom_global_reward_weight=args.bottom_global_reward_weight,bottom_global_reward_clip=args.bottom_global_reward_clip,bottom_agent_types=('recon','attack'))
 if args.adaptive_decision_timing: collector_kwargs['adaptive_decision_timing']=True
 w['collector']=RuleDrivenRolloutCollector(w['api'],**collector_kwargs)
def start_workers_parallel(executor, args, workers):
 futures={w['id']:executor.submit(start_worker,args,w) for w in workers}
 for worker_id,future in futures.items(): future.result()

def collect_parallel(executor, active_workers, done_worker_ids, trainer, return_errors=False):
 futures={w['id']:executor.submit(w['collector'].collect,trainer,n_steps=1,reset=False) for w in active_workers if w['id'] not in done_worker_ids}
 results={}
 for worker_id,future in futures.items():
  try:
   results[worker_id]=future.result()
  except Exception as error:
   if not return_errors:
    raise
   results[worker_id]={"_collection_error": str(error), "_collection_error_type": type(error).__name__}
 return results
def main():
 args=args_parser();
 if args.workers<1: raise ValueError('--workers must be positive')
 device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
 timing=resolve_bottom_decision_timing(args.bottom_decisions_per_hour,args.simulation_clock_rate,args.decision_seconds)
 workers=[]
 for i in range(args.workers):
  env=AFSIMIslandEnv(config_path=args.config_path or None,bind=True,auto_start_warlock=False,local_address=('0.0.0.0',args.base_port+i)); env.set_negative_rewards_enabled(args.enable_negative_rewards); apply_bottom_decision_timing(env,timing)
  api=AFSIMRLInterface(env,reward_profile='recon_attack_stage'); workers.append({'id':i,'port':args.base_port+i,'env':env,'api':api,'buffer':TaskTrajectoryBuffer(('recon','attack')),'episode':0})
 specs=workers[0]['api'].get_bottom_agent_specs(); bottom={k:specs[k] for k in ('recon','attack')}
 trainer=HAPPOTrainer(bottom,global_state_dim=int(specs['global']['obs_dim']),agent_types=('recon','attack'),hidden_sizes=(args.hidden_size,args.hidden_size),config=HAPPOConfig(gamma=args.gamma,gae_lambda=args.gae_lambda,learning_rate=args.lr,update_epochs=args.update_epochs,minibatch_size=args.minibatch_size,share_policy_by_type=args.share_policy_by_type),device=device,trainable_agent_types=('recon','attack'))
 global_buffer=TaskTrajectoryBuffer(('recon','attack')); update_id=0
 if args.resume:
  state=load_training_checkpoint(args.resume); trainer.load_state_dict(state.get('trainer',state)); global_buffer.load_state_dict(state.get('stage_buffer',{})); update_id=int(state.get('update',0))
 ckpt=Path(args.checkpoint_dir); ckpt.mkdir(parents=True,exist_ok=True); metrics=args.metrics_file or str(ckpt/'metrics.jsonl')
 minimum={'recon':args.recon_min_samples,'attack':args.attack_min_samples}
 executor=ThreadPoolExecutor(max_workers=args.workers,thread_name_prefix='afsim_actor')
 print('shared_policy_object',id(trainer),'workers',args.workers,'share_policy_by_type',args.share_policy_by_type,'simulation_clock_rate',args.simulation_clock_rate,'adaptive_decision_timing',args.adaptive_decision_timing,flush=True)
 try:
  start_workers_parallel(executor,args,workers)
  while update_id < args.updates and any(w['episode']<args.episodes_per_worker for w in workers):
   active=[w for w in workers if w['episode']<args.episodes_per_worker]
   done=set(); reasons={}
   for step in range(1,args.max_episode_steps+1):
    sampling = collect_parallel(executor,active,done,trainer)
    for w in active:
     if w['id'] in done: continue
     rollout=sampling[w['id']]; w['buffer'].append_rollout(rollout)
     if w['env'].platform_state_age_seconds()>args.platform_state_stall_seconds: raise RuntimeError('worker {0} state stream stalled'.format(w['id']))
     summary=rollout.summary(); status=high_quality_landing_status(w['env'],args.min_recon_alive,args.min_attack_alive,args.min_loaded_transports)
     reason=''
     if status['landing_combat_conditions_met']: reason='combat_landing_ready'
     elif status['landing_time_override_met']: reason='landing_deadline_missed'
     elif summary.get('terminal'): reason=summary.get('done_reason','environment_terminal')
     elif step==args.max_episode_steps: reason='stage_time_limit'
     if reason:
      bonus=recon_attack_terminal_bonus(status,reason,w['env']._current_sim_time(),deadline_miss_penalty=args.deadline_miss_penalty,negative_rewards_enabled=w['env'].negative_rewards_enabled)
      w['buffer'].apply_episode_global_reward(bonus); w['buffer'].finish_episode(end_reason=reason); moved=global_buffer.merge_completed_from(w['buffer'],w['id']); w['episode']+=1; done.add(w['id']); reasons[w['id']]=reason
      print('worker_trajectory_merged',w['id'],'episode',w['episode'],'reason',reason,'steps',step,'moved',moved,'global',global_buffer.assigned_counts(),flush=True)
    if len(done)==len(active): break
   if global_buffer.ready(minimum):
    counts=global_buffer.assigned_counts(); result=trainer.update_all(global_buffer.to_batches()); update_id+=1; global_buffer.clear_after_update()
    for worker_state in workers: worker_state['buffer'].policy_version=update_id
    payload={'update':update_id,'episode':sum(w['episode'] for w in workers),'trainer':trainer.state_dict(),'stage_buffer':global_buffer.state_dict(include_active=False),'samples':counts,'workers':args.workers,'share_policy_by_type':args.share_policy_by_type,'simulation_clock_rate':args.simulation_clock_rate}
    _,latest=save_training_checkpoint(payload,ckpt,'bottom_happo_recon_attack_parallel',update_id); append_metrics_jsonl(metrics,{'record_type':'policy_update','update':update_id,'samples':counts,'workers':args.workers,'share_policy_by_type':args.share_policy_by_type,'simulation_clock_rate':args.simulation_clock_rate,'metrics':result}); print('shared_happo_update',update_id,'samples',counts,'checkpoint',latest,flush=True)
   else: print('shared_buffer_waiting',global_buffer.assigned_counts(),'required',minimum,flush=True)
   restarting=[w for w in active if w['episode']<args.episodes_per_worker]
   if restarting: start_workers_parallel(executor,args,restarting)
  return 0 if update_id>=args.updates else 2
 finally:
  # ``cancel_futures`` is unavailable on the older Python used by some
  # training environments. All submitted rollout futures are awaited above,
  # so a plain blocking shutdown is sufficient here.
  executor.shutdown(wait=True)
  for w in workers:
   try: task_command(args,w['id'],'stop')
   except Exception: pass
   w['env'].close()
if __name__=='__main__': raise SystemExit(main())
