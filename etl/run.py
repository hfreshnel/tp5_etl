"""Orchestration du pipeline ETL.

    python -m etl.run --category All_Beauty --limit 200000
    python -m etl.run --category All_Beauty --hub-repo mon-user/amazon-beauty-clean --tag v1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from .card import build_card
from .config import Config
from .extract import extract_batches
from .load import push_to_hub, reset_dir, write_quarantine, write_shard
from .quality import assert_quality, check
from .transform import transform_batch

log = logging.getLogger("etl")


def setup_logging(cfg: Config, run_id: str) -> None:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.log_dir / f"run_{run_id}.log", encoding="utf-8"),
    ])


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def run(cfg: Config) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    setup_logging(cfg, run_id)
    log.info("démarrage run %s : %s", run_id, cfg)
    t0 = time.perf_counter()

    reset_dir(cfg.run_dir)
    quarantine = cfg.quarantine_dir / f"{cfg.category}_{run_id}.jsonl"
    seen: set = set()
    reasons: Counter = Counter()
    n_in = n_rej = 0
    shards = []

    for i, raw in enumerate(extract_batches(cfg)):
        ok, rejected = transform_batch(raw, cfg.min_chars, seen)
        n_in += len(raw); n_rej += len(rejected)
        reasons.update(r["reason"] for r in rejected)
        if rejected:
            write_quarantine(rejected, quarantine)
        if ok:
            shards.append(write_shard(ok, cfg.run_dir, i))
        log.info("lot %d : %d entrées, %d gardées, %d rejetées", i, len(raw), len(ok), len(rejected))

    if not shards:
        raise RuntimeError("aucune ligne valide : vérifier la source et les règles de validation")

    df = pd.concat((pd.read_parquet(p) for p in shards), ignore_index=True)
    report = check(df, n_in, n_rej, cfg.max_reject_rate)
    report.update(run_id=run_id, duration_s=round(time.perf_counter() - t0, 1), reject_reasons=dict(reasons))
    (cfg.run_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("rapport qualité : %s", json.dumps(report["checks"]))
    assert_quality(report)                       # échoue AVANT toute publication

    card = build_card(cfg, report, dict(reasons), git_sha())
    (cfg.run_dir / "README.md").write_text(card, encoding="utf-8")
    if cfg.hub_repo:
        report["hub_revision"] = push_to_hub(cfg.run_dir, cfg.hub_repo, cfg.hf_token, card, cfg.hub_tag,
                                             f"run {run_id} : {report['n_output']:,} lignes")
    log.info("terminé en %.1f s : %s lignes en sortie", report["duration_s"], f"{report['n_output']:,}")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="All_Beauty")
    p.add_argument("--limit", type=int, default=200_000, help="0 = tout le flux")
    p.add_argument("--hub-repo")
    p.add_argument("--tag")
    a = p.parse_args()
    cfg = Config(category=a.category, limit=a.limit or None, hub_repo=a.hub_repo, hub_tag=a.tag)
    try:
        run(cfg)
    except Exception:
        log.exception("échec du pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
