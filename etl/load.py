"""LOAD — écriture locale en Parquet (idempotente) puis publication sur le Hub."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def reset_dir(path: Path) -> None:
    """Idempotence : relancer le pipeline remplace la sortie précédente au lieu de l'ajouter."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_shard(rows: list[dict], out_dir: Path, idx: int) -> Path:
    path = out_dir / f"part-{idx:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    return path


def write_quarantine(rejected: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def push_to_hub(out_dir: Path, repo_id: str, token: str | None, card_md: str, tag: str | None,
                commit_message: str) -> str:
    from datasets import load_dataset
    from huggingface_hub import DatasetCard, HfApi

    ds = load_dataset("parquet", data_files=str(out_dir / "*.parquet"), split="train")
    ds.push_to_hub(repo_id, token=token, commit_message=commit_message)
    # push_to_hub a écrit un README avec les métadonnées techniques (configs, dataset_info) :
    # on les CONSERVE et on y ajoute les nôtres, sinon le chargement du dataset peut casser.
    import yaml
    existing = DatasetCard.load(repo_id, repo_type="dataset", token=token).data.to_dict()
    ours = DatasetCard(card_md)
    merged = {**existing, **ours.data.to_dict()}
    final = DatasetCard("---\n" + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True) + "---\n" + ours.text)
    final.push_to_hub(repo_id, token=token, commit_message="Dataset Card générée par le pipeline")
    api = HfApi(token=token)
    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="dataset", tag_message=commit_message, exist_ok=True)
    sha = api.dataset_info(repo_id).sha
    log.info("publié sur https://huggingface.co/datasets/%s (révision %s)", repo_id, sha)
    return sha
