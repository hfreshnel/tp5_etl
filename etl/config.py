from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    source_repo: str = "McAuley-Lab/Amazon-Reviews-2023"
    category: str = "All_Beauty"
    limit: int | None = 200_000                 # None = tout le flux
    batch_size: int = 10_000
    # Seuil baissé de 20 à 10 : à 20, 10 % des avis sont rejetés (des avis courts mais
    # porteurs de sens, comme « Great product » ou « Works well ») et le run dépasse max_reject_rate.
    # À 10, le taux de rejet tombe à environ 4 % (mesuré sur 200 000 avis All_Beauty).
    min_chars: int = 10
    out_dir: Path = Path("data/processed")
    quarantine_dir: Path = Path("data/quarantine")
    log_dir: Path = Path("logs")
    hub_repo: str | None = None                  # ex. "mon-user/amazon-beauty-clean"
    hub_tag: str | None = None                   # ex. "v1.0"
    max_reject_rate: float = 0.05                # au-delà : le run échoue (qualité insuffisante)
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.category
