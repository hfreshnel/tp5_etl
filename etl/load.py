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
    """Envoie les fichiers Parquet (dans data/) et la Dataset Card (README.md) en UN seul commit.

    Modifié (TP5) : la version du kit rechargeait les Parquet avec `load_dataset("parquet", ...)`
    puis appelait `Dataset.push_to_hub`. Avec Python 3.14, ce chargement plante dans `dill`
    (version 0.3.8, imposée par datasets 3.6.0, elle-même imposée par le jeu source).
    Les fichiers sont déjà au bon format : on les envoie tels quels avec l'API du Hub.
    Le Hub détecte tout seul les Parquet du dossier data/ (un seul split « train »).
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi(token=token)   # token=None : le Hub utilise le jeton de `hf auth login`
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    shards = sorted(out_dir.glob("*.parquet"))
    nouveaux = {f"data/{p.name}" for p in shards}
    ops = [CommitOperationAdd(path_in_repo=f"data/{p.name}", path_or_fileobj=str(p)) for p in shards]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_md.encode("utf-8")))
    # Idempotence côté Hub : un ancien fichier absent du nouveau run (moins de lots) est supprimé,
    # sinon le dataset publié mélangerait deux runs.
    anciens = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.startswith("data/")]
    ops += [CommitOperationDelete(path_in_repo=f) for f in anciens if f not in nouveaux]
    api.create_commit(repo_id, repo_type="dataset", operations=ops, commit_message=commit_message)

    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="dataset", tag_message=commit_message, exist_ok=True)
    sha = api.dataset_info(repo_id).sha
    log.info("publié sur https://huggingface.co/datasets/%s (révision %s)", repo_id, sha)
    return sha
