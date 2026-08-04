from train.stage_episode_buffer import TaskTrajectoryBuffer

def test_multiple_workers_merge_into_one_ready_buffer():
    shared = TaskTrajectoryBuffer(("recon", "attack"))
    actors = [TaskTrajectoryBuffer(("recon", "attack")) for _ in range(2)]
    for worker, actor in enumerate(actors):
        actor.rows["recon"] = [{"step_id": 0, "task_id": "recon", "done": 1.0}]
        actor.rows["attack"] = [{"step_id": 1, "task_id": "attack", "done": 1.0}]
        shared.merge_completed_from(actor, worker)
    assert shared.assigned_counts() == {"recon": 2, "attack": 2}
    assert shared.ready({"recon": 2, "attack": 2})
    assert {row["worker_id"] for row in shared.rows["recon"]} == {0, 1}
    assert len({row["step_id"] for row in shared.rows["recon"] + shared.rows["attack"]}) == 4
    assert all(not actor.rows["recon"] and not actor.rows["attack"] for actor in actors)
def test_collect_parallel_uses_one_policy_concurrently():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from train.train_recon_attack_parallel import collect_parallel
    seen=[]
    class Collector:
        def collect(self, trainer, n_steps, reset):
            seen.append(id(trainer)); time.sleep(0.2); return id(trainer)
    trainer=object(); workers=[{"id":i,"collector":Collector()} for i in range(2)]
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=collect_parallel(pool,workers,set(),trainer)
    elapsed=time.monotonic()-started
    assert set(results.values()) == {id(trainer)}
    assert set(seen) == {id(trainer)}
    assert elapsed < 0.35, elapsed