"""EXTRACT — lecture en streaming depuis le Hub, par lots."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from .config import Config

log = logging.getLogger(__name__)
COLUMNS = ["parent_asin", "user_id", "rating", "title", "text", "timestamp",
           "helpful_vote", "verified_purchase"]


def extract_batches(cfg: Config, retries: int = 3) -> Iterator[list[dict]]:
    """Produit des lots de lignes brutes (liste de dicts). Réessaie l'ouverture du flux en cas d'erreur réseau."""
    from datasets import load_dataset  # import tardif : les tests unitaires n'en ont pas besoin

    for attempt in range(1, retries + 1):
        try:
            ds = load_dataset(cfg.source_repo, f"raw_review_{cfg.category}", split="full", streaming=True)
            break
        except Exception as e:  # noqa: BLE001
            log.warning("ouverture du flux impossible (tentative %d/%d) : %s", attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    ds = ds.select_columns(COLUMNS)
    if cfg.limit:
        ds = ds.take(cfg.limit)
    for batch in ds.iter(batch_size=cfg.batch_size):
        # dict de listes -> liste de dicts
        keys = list(batch)
        yield [dict(zip(keys, vals)) for vals in zip(*batch.values())]
