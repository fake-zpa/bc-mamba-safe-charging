"""Progress tracking utilities for PROJECT_PROGRESS.md management."""
import os
from datetime import datetime
from typing import Dict, Optional


STAGE_NAMES = [
    "Stage 0: Environment Setup",
    "Stage 1: Data Download & Preprocess",
    "Stage 2: Mamba Pretrain",
    "Stage 3: World Model Training",
    "Stage 4: Baseline Offline RL",
    "Stage 5: HM-LatentSafeRL Main Training",
    "Stage 6: Joint Finetune",
    "Stage 7: Evaluation & Ablation",
    "Stage 8: Paper-ready Results",
]

VALID_STATUSES = ["NOT_STARTED", "IN_PROGRESS", "BLOCKED", "DONE", "NEEDS_RETRY"]


def update_progress_file(
    progress_path: str,
    stage_updates: Optional[Dict[str, str]] = None,
    experiment_entry: Optional[str] = None,
    next_action: Optional[str] = None,
    notes: Optional[str] = None,
):
    """Append updates to PROJECT_PROGRESS.md.

    Args:
        progress_path: Path to PROJECT_PROGRESS.md.
        stage_updates: Dict of stage_name -> status.
        experiment_entry: New experiment log entry.
        next_action: Updated next action text.
        notes: Additional notes.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"\n---\n## Update: {timestamp}\n")

    if stage_updates:
        lines.append("### Stage Updates")
        for stage, status in stage_updates.items():
            lines.append(f"- {stage}: **{status}**")
        lines.append("")

    if experiment_entry:
        lines.append("### Experiment Log Entry")
        lines.append(experiment_entry)
        lines.append("")

    if next_action:
        lines.append("### Next Action")
        lines.append(next_action)
        lines.append("")

    if notes:
        lines.append("### Notes")
        lines.append(notes)
        lines.append("")

    with open(progress_path, "a") as f:
        f.write("\n".join(lines))
