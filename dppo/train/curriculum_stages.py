"""Curriculum stage definitions for staged bottom-policy training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


BOTTOM_AGENT_TYPES = ("recon", "attack", "landing", "ground")


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    order: int
    scenario_template: str
    scenario_ready: bool
    trainable_agent_types: tuple[str, ...]
    allowed_task_kinds: tuple[str, ...]
    reward_profile: str
    min_samples: Mapping[str, int]
    description: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "order": int(self.order),
            "scenario_template": self.scenario_template,
            "scenario_ready": bool(self.scenario_ready),
            "trainable_agent_types": list(self.trainable_agent_types),
            "allowed_task_kinds": list(self.allowed_task_kinds),
            "reward_profile": self.reward_profile,
            "min_samples": dict(self.min_samples),
            "description": self.description,
        }


CURRICULUM_STAGES = {
    "recon_only": CurriculumStage(
        name="recon_only",
        order=1,
        scenario_template="scenarios/island_assault_min.txt",
        scenario_ready=True,
        trainable_agent_types=("recon",),
        allowed_task_kinds=("RECON", "WAIT"),
        reward_profile="recon_early",
        min_samples={"recon": 2048},
        description="Main scenario; train reconnaissance while all other policies remain frozen.",
    ),
    "recon_attack": CurriculumStage(
        name="recon_attack",
        order=2,
        scenario_template="scenarios/island_assault_min.txt",
        scenario_ready=True,
        trainable_agent_types=("recon", "attack"),
        allowed_task_kinds=("RECON", "ATTACK", "WAIT"),
        reward_profile="recon_attack",
        min_samples={"recon": 2048, "attack": 2048},
        description="Main scenario; jointly train reconnaissance and attack coordination.",
    ),
    "landing": CurriculumStage(
        name="landing",
        order=3,
        scenario_template="scenarios/island_assault_stage_landing.txt",
        scenario_ready=False,
        trainable_agent_types=("recon", "attack", "landing"),
        allowed_task_kinds=("RECON", "ATTACK", "LANDING", "WAIT"),
        reward_profile="landing",
        min_samples={"recon": 1024, "attack": 1024, "landing": 2048},
        description="Landing-ready scenario; add transport navigation and unloading.",
    ),
    "ground": CurriculumStage(
        name="ground",
        order=4,
        scenario_template="scenarios/island_assault_stage_ground.txt",
        scenario_ready=False,
        trainable_agent_types=BOTTOM_AGENT_TYPES,
        allowed_task_kinds=("RECON", "ATTACK", "LANDING", "GROUND", "WAIT"),
        reward_profile="ground",
        min_samples={"recon": 512, "attack": 1024, "landing": 512, "ground": 2048},
        description="Post-unload scenario; train ground combat and capture with air support.",
    ),
    "full": CurriculumStage(
        name="full",
        order=5,
        scenario_template="scenarios/island_assault_min.txt",
        scenario_ready=True,
        trainable_agent_types=BOTTOM_AGENT_TYPES,
        allowed_task_kinds=("RECON", "ATTACK", "LANDING", "GROUND", "WAIT"),
        reward_profile="full",
        min_samples={"recon": 1024, "attack": 1024, "landing": 512, "ground": 1024},
        description="Main scenario end-to-end joint fine-tuning.",
    ),
}


def curriculum_stage_names() -> tuple[str, ...]:
    return tuple(name for name, _stage in sorted(CURRICULUM_STAGES.items(), key=lambda item: item[1].order))


def get_curriculum_stage(name: str) -> CurriculumStage:
    key = str(name or "").strip().lower()
    if key not in CURRICULUM_STAGES:
        raise ValueError("unknown curriculum stage {0!r}; expected one of {1}".format(name, curriculum_stage_names()))
    return CURRICULUM_STAGES[key]
