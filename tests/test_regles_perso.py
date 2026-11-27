"""Tests des ajouts personnels du TP5 : 2 règles de validation, 1 feature, 1 contrôle qualité.

Fichier séparé : les tests fournis (test_transform.py, test_quality.py) restent intacts.
"""
import pandas as pd
import pytest

from etl.quality import QualityError, assert_quality, check
from etl.transform import MAX_CHARS, enrich, uppercase_ratio, validate

MIN_CHARS = 10  # valeur de Config.min_chars


def raw(**kw):
    """Avis brut valide ; les arguments nommés remplacent des champs."""
    base = {"parent_asin": "B00X", "user_id": "U1", "rating": 4.0, "title": "Top",
            "text": "Un produit vraiment très efficace, je recommande.", "timestamp": 1588687728923,
            "helpful_vote": 2, "verified_purchase": True}
    return {**base, **kw}


# --- Règle 1 : texte_trop_long ------------------------------------------------------------

def test_texte_trop_long_rejete():
    assert validate(raw(text="a" * (MAX_CHARS + 1)), MIN_CHARS) == "texte_trop_long"


def test_texte_a_la_limite_accepte():
    # Cas limite : exactement MAX_CHARS caractères, l'avis est gardé (la règle est « > », pas « >= »).
    assert validate(raw(text="a" * MAX_CHARS), MIN_CHARS) is None


# --- Règle 2 : texte_sans_lettres ---------------------------------------------------------

@pytest.mark.parametrize("texte", [
    "A+++++++++",        # vu dans les données : 1 lettre sur 10 caractères
    "😀😀😀😀😀 !!!!!",   # emojis et ponctuation seulement
    "12345678901",       # que des chiffres
])
def test_texte_sans_lettres_rejete(texte):
    assert validate(raw(text=texte), MIN_CHARS) == "texte_sans_lettres"


@pytest.mark.parametrize("texte", [
    "Très bon produit ! 10/10",   # un peu de chiffres et de ponctuation : toujours gardé
    "Él está muy contento ñ",     # lettres accentuées comptées comme des lettres (isalpha)
])
def test_texte_avec_assez_de_lettres_accepte(texte):
    assert validate(raw(text=texte), MIN_CHARS) is None


# --- Feature : ratio_majuscules -----------------------------------------------------------

@pytest.mark.parametrize("texte, attendu", [
    ("GREAT PRODUCT!!", 1.0),    # avis « crié » ; la ponctuation ne compte pas
    ("Great", 0.2),              # 1 majuscule sur 5 lettres
    ("great product", 0.0),
    ("!!! 123", 0.0),            # aucune lettre : 0 par convention, pas de division par zéro
])
def test_uppercase_ratio(texte, attendu):
    assert uppercase_ratio(texte) == attendu


def test_enrich_ajoute_ratio_majuscules():
    out = enrich(raw(text="SUPER produit"))   # 5 majuscules sur 12 lettres
    assert out["ratio_majuscules"] == round(5 / 12, 4)


# --- Contrôle qualité : ratio_majuscules_0_1 ----------------------------------------------

def make_df(n=3):
    return pd.DataFrame([enrich(raw(parent_asin=f"P{i}", user_id=f"U{i}")) for i in range(n)])


def test_controle_ratio_ok():
    rep = check(make_df(), n_input=3, n_rejected=0, max_reject_rate=0.05)
    assert rep["checks"]["ratio_majuscules_0_1"]
    assert rep["checks"]["schema_types"]      # la colonne est bien en float64


@pytest.mark.parametrize("valeur", [1.5, -0.1, float("nan")])
def test_controle_ratio_hors_bornes_bloque(valeur):
    df = make_df()
    df.loc[0, "ratio_majuscules"] = valeur
    rep = check(df, 3, 0, 0.05)
    assert not rep["checks"]["ratio_majuscules_0_1"]
    with pytest.raises(QualityError):   # le pipeline doit s'arrêter avant publication
        assert_quality(rep)
