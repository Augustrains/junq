from train.train_recon_attack_shared_parallel import _shared_parallel_defaults


def test_shared_parallel_defaults():
    args = _shared_parallel_defaults(["--warlock-ssh-target", "host"])
    assert args[args.index("--workers") + 1] == "4"
    assert args[args.index("--simulation-clock-rate") + 1] == "20"
    assert "--share-policy-by-type" in args
    assert "happo_recon_attack_shared_parallel" in args[args.index("--checkpoint-dir") + 1]


def test_shared_parallel_preserves_overrides():
    args = _shared_parallel_defaults(["--workers", "6", "--simulation-clock-rate", "30"])
    assert args.count("--workers") == 1
    assert args[args.index("--workers") + 1] == "6"
    assert args[args.index("--simulation-clock-rate") + 1] == "30"