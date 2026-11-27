"""Contrôles de qualité sur la sortie : contrat de schéma + règles métier.

Retourne un rapport ; `assert_quality` lève une exception si une règle bloquante échoue.
"""
from __future__ import annotations

import pandas as pd

SCHEMA = {
    "parent_asin": "object", "user_id": "object", "rating": "float64", "title": "object",
    "text": "object", "n_words": "int64", "review_date": "object", "year": "int64",
    "helpful_vote": "int64", "verified_purchase": "bool", "label": "object",
    "ratio_majuscules": "float64",               # feature ajoutée (TP5)
}


class QualityError(Exception):
    pass


def check(df: pd.DataFrame, n_input: int, n_rejected: int, max_reject_rate: float) -> dict:
    checks = {}
    missing = set(SCHEMA) - set(df.columns)
    checks["schema_colonnes"] = not missing
    checks["schema_types"] = not missing and all(
        str(df[c].dtype) == t or (t == "object" and str(df[c].dtype) in ("str", "string")) for c, t in SCHEMA.items()
    )
    checks["rating_1_5"] = bool(df["rating"].between(1, 5).all()) if "rating" in df else False
    checks["label_valide"] = bool(df["label"].isin(["neg", "neu", "pos"]).all()) if "label" in df else False
    checks["pas_de_doublon"] = not df.duplicated(["user_id", "parent_asin", "review_date", "text"]).any()
    checks["texte_non_vide"] = bool((df["text"].str.len() > 0).all()) if "text" in df else False
    # Contrôle ajouté (TP5) : un ratio est forcément compris entre 0 et 1. Une valeur hors bornes
    # ou manquante (NaN, que between() traite comme hors bornes) signale un bug dans enrich().
    checks["ratio_majuscules_0_1"] = (
        bool(df["ratio_majuscules"].between(0, 1).all()) if "ratio_majuscules" in df else False
    )
    reject_rate = n_rejected / n_input if n_input else 0.0
    checks["taux_rejet_ok"] = reject_rate <= max_reject_rate
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "n_input": n_input,
        "n_output": int(len(df)),
        "n_rejected": n_rejected,
        "reject_rate": round(reject_rate, 4),
        "label_distribution": df["label"].value_counts(normalize=True).round(4).to_dict() if "label" in df else {},
        "years": [int(df["year"].min()), int(df["year"].max())] if len(df) else [],
    }


def assert_quality(report: dict) -> None:
    failed = [k for k, v in report["checks"].items() if not v]
    if failed:
        raise QualityError(f"contrôles qualité en échec : {failed}")
