"""Tests for the opt-in recon-shared training entry point."""

import unittest

from train.train_recon_shared_attack_parallel_eval import _recon_shared_defaults


class ReconSharedTrainingEntryTest(unittest.TestCase):
    def test_enables_shared_recon_and_separate_checkpoint_directory(self):
        args = _recon_shared_defaults(["--workers", "4"])
        self.assertIn("--shared-recon-actor", args)
        self.assertEqual(args[args.index("--recon-init-policy") + 1], "red_recon_1")
        checkpoint = args[args.index("--checkpoint-dir") + 1]
        self.assertIn("happo_recon_shared_attack_independent_parallel_eval", checkpoint)

    def test_reuses_attack_checkpoint_for_recon_initialization(self):
        args = _recon_shared_defaults(["--attack-init-checkpoint", "old.pt"])
        self.assertEqual(args[args.index("--recon-init-checkpoint") + 1], "old.pt")

    def test_preserves_explicit_checkpoint_directory(self):
        args = _recon_shared_defaults(["--checkpoint-dir", "custom"])
        self.assertEqual(args.count("--checkpoint-dir"), 1)
        self.assertEqual(args[args.index("--checkpoint-dir") + 1], "custom")


if __name__ == "__main__":
    unittest.main()