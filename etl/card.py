"""Génération automatique de la Dataset Card (métadonnées YAML + documentation Markdown)."""
from __future__ import annotations

from datetime import datetime, timezone

import yaml  # fourni avec huggingface_hub / datasets (PyYAML)

# Seuils des règles ajoutées (TP5), importés pour que la carte reste juste si on les modifie.
from .transform import MAX_CHARS, MIN_LETTER_RATIO


def build_card(cfg, report: dict, reasons: dict[str, int], git_sha: str | None = None) -> str:
    meta = {
        "language": ["en"],
        "license": "other",
        "pretty_name": f"Amazon Reviews 2023 — {cfg.category} (nettoyé)",
        "task_categories": ["text-classification"],
        "size_categories": [_size_category(report["n_output"])],
        "source_datasets": [cfg.source_repo],
        "tags": ["reviews", "sentiment", "etl", "ynov-m1iadata"],
    }
    rejets = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(reasons.items(), key=lambda x: -x[1])) or "| — | 0 |"
    labels = ", ".join(f"{k} : {v:.1%}" for k, v in report["label_distribution"].items())
    checks = "\n".join(f"- [{'x' if v else ' '}] {k}" for k, v in report["checks"].items())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""
# Amazon Reviews 2023 — {cfg.category} (nettoyé)

Jeu de données dérivé de [{cfg.source_repo}](https://huggingface.co/datasets/{cfg.source_repo})
produit par un pipeline ETL (module M1IADATA, Ynov). Généré automatiquement le {now}.

## Contenu
- **Lignes en entrée** : {report['n_input']:,} — **en sortie** : {report['n_output']:,}
- **Période** : {report['years'][0] if report['years'] else '—'} – {report['years'][1] if report['years'] else '—'}
- **Répartition des labels** : {labels}

## Champs
| Champ | Type | Description |
|---|---|---|
| parent_asin | string | Identifiant produit (parent) |
| user_id | string | Identifiant anonymisé de l'auteur |
| rating | float | Note de 1 à 5 |
| title, text | string | Titre et texte nettoyés (HTML retiré, espaces normalisés) |
| n_words | int | Nombre de mots du texte |
| ratio_majuscules | float | Part des lettres en majuscules dans le texte (0 à 1) ; proche de 1 = avis « crié » |
| review_date, year | string, int | Date (UTC) de l'avis |
| helpful_vote | int | Votes « utile » |
| verified_purchase | bool | Achat vérifié |
| label | string | neg (1-2), neu (3), pos (4-5) |

## Traitements
1. Lecture en streaming de `raw_review_{cfg.category}`{f' (limite {cfg.limit:,} lignes)' if cfg.limit else ''}.
2. Validation ligne à ligne ; chaque rejet part en quarantaine avec sa raison :
   - identifiants présents, note entre 1 et 5, horodatage valide ;
   - `texte_trop_court` : texte nettoyé de moins de {cfg.min_chars} caractères ;
   - `texte_trop_long` : texte nettoyé de plus de {MAX_CHARS:,} caractères (au-delà de la limite de tokens des modèles de sentiment courants) ;
   - `texte_sans_lettres` : moins de {MIN_LETTER_RATIO:.0%} de lettres dans le texte (avis réduits à des emojis, de la ponctuation ou des chiffres, ex. « A+++++++++ »).
3. Dédoublonnage sur (user_id, parent_asin, texte nettoyé) : un même avis republié à un autre horodatage est écarté.
4. Nettoyage du texte (HTML retiré, espaces normalisés) et enrichissement : `n_words`, `review_date`, `year`, `label` et `ratio_majuscules`.

### Rejets
| Raison | Lignes |
|---|---|
{rejets}

Taux de rejet : {report['reject_rate']:.2%}

## Contrôles qualité
{checks}

## Limites et usage
- Biais de sélection : les avis Amazon sur-représentent les notes extrêmes (surtout 5).
- Le label est dérivé de la note, pas d'une annotation humaine du texte.
- Respecter la licence et les conditions d'usage du jeu source.

## Reproductibilité
Code du pipeline : révision `{git_sha or 'non renseignée'}`.
"""
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n" + body


def _size_category(n: int) -> str:
    for limit, label in [(1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                         (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M")]:
        if n < limit:
            return label
    return "10M<n<100M"
