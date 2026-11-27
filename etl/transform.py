"""TRANSFORM — validation ligne à ligne, nettoyage, enrichissement, dédoublonnage.

Principe : une ligne invalide n'arrête pas le pipeline ; elle part en QUARANTAINE avec sa raison.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

# Règle « texte_trop_long » : au-delà de cette longueur, l'avis dépasse la limite de tokens
# des modèles de sentiment courants (512 tokens ≈ 2 000 à 3 000 caractères) ; on garde une marge.
# Mesuré sur 200 000 avis All_Beauty : 20 avis concernés (0,01 %).
MAX_CHARS = 5000

# Règle « texte_sans_lettres » : part minimale de lettres dans le texte nettoyé.
# Écarte les avis sans vrai texte (« A+++++++++ », emojis, ponctuation, balises vidéo),
# devenus possibles depuis la baisse de min_chars à 10. Mesuré : 60 avis concernés (0,03 %).
MIN_LETTER_RATIO = 0.5


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", text))).strip()


def validate(row: dict, min_chars: int) -> str | None:
    """Renvoie None si la ligne est valide, sinon la raison du rejet."""
    if not row.get("parent_asin"):
        return "parent_asin_manquant"
    if not row.get("user_id"):
        return "user_id_manquant"
    r = row.get("rating")
    if r is None or not 1 <= r <= 5:
        return "rating_hors_bornes"
    ts = row.get("timestamp")
    if not isinstance(ts, int) or ts <= 0:
        return "timestamp_invalide"
    text = clean_text(row.get("text"))
    if len(text) < min_chars:
        return "texte_trop_court"
    if len(text) > MAX_CHARS:
        return "texte_trop_long"
    # Ici le texte fait au moins min_chars caractères (non vide) : la division est sûre.
    # str.isalpha() compte aussi les lettres accentuées ou non latines (é, ñ, ü…).
    n_lettres = sum(ch.isalpha() for ch in text)
    if n_lettres / len(text) < MIN_LETTER_RATIO:
        return "texte_sans_lettres"
    return None


def uppercase_ratio(text: str) -> float:
    """Part des lettres en majuscules dans le texte, entre 0 et 1 (feature « ratio_majuscules »).

    Mesure si l'auteur « crie » (« GREAT PRODUCT!! ») : un signal d'intensité utile au modèle
    de sentiment, sans être une raison de rejeter l'avis.
    On divise par le nombre de LETTRES et non de caractères : chiffres, espaces et ponctuation
    ne doivent pas faire baisser le ratio.
    """
    lettres = [ch for ch in text if ch.isalpha()]
    if not lettres:
        # Pas de lettre : ratio non défini, on renvoie 0. Cela n'arrive pas dans le pipeline,
        # la règle « texte_sans_lettres » ayant déjà rejeté ces avis, mais enrich() reste sûre.
        return 0.0
    return round(sum(ch.isupper() for ch in lettres) / len(lettres), 4)


def enrich(row: dict) -> dict:
    text = clean_text(row["text"])
    dt = datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc)
    return {
        "parent_asin": row["parent_asin"],
        "user_id": row["user_id"],
        "rating": float(row["rating"]),
        "title": clean_text(row.get("title")),
        "text": text,
        "n_words": len(text.split()),
        "ratio_majuscules": uppercase_ratio(text),
        "review_date": dt.date().isoformat(),
        "year": dt.year,
        "helpful_vote": int(row.get("helpful_vote") or 0),
        "verified_purchase": bool(row.get("verified_purchase")),
        "label": "neg" if row["rating"] <= 2 else ("neu" if row["rating"] == 3 else "pos"),
    }


def transform_batch(rows: list[dict], min_chars: int, seen: set) -> tuple[list[dict], list[dict]]:
    """Renvoie (lignes_ok, lignes_rejetées). `seen` (clés déjà vues) est partagé entre lots pour le dédoublonnage."""
    ok, rejected = [], []
    for row in rows:
        reason = validate(row, min_chars)
        if reason is None:
            key = (row["user_id"], row["parent_asin"], clean_text(row["text"]))
            if key in seen:
                reason = "doublon"
            else:
                seen.add(key)
        if reason:
            rejected.append({"reason": reason, "row": {k: row.get(k) for k in ("parent_asin", "user_id", "rating", "timestamp")}})
        else:
            ok.append(enrich(row))
    return ok, rejected
