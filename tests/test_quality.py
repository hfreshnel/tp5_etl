import pandas as pd
import pytest

from etl.quality import QualityError, assert_quality, check
from etl.transform import enrich


def make_df(n=3, **overrides):
    rows = [enrich({"parent_asin": f"P{i}", "user_id": f"U{i}", "rating": 5.0, "title": "t",
                    "text": f"texte numéro {i} suffisamment long", "timestamp": 1600000000000 + i,
                    "helpful_vote": 0, "verified_purchase": True}) for i in range(n)]
    df = pd.DataFrame(rows)
    for col, val in overrides.items():
        df.loc[0, col] = val
    return df


def test_sortie_conforme():
    rep = check(make_df(), n_input=3, n_rejected=0, max_reject_rate=0.05)
    assert rep["passed"], rep["checks"]
    assert_quality(rep)


def test_rating_hors_bornes_detecte():
    rep = check(make_df(rating=7.0), 3, 0, 0.05)
    assert not rep["checks"]["rating_1_5"]
    with pytest.raises(QualityError):
        assert_quality(rep)


def test_doublon_detecte():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not check(df, 4, 0, 0.05)["checks"]["pas_de_doublon"]


def test_taux_de_rejet_trop_eleve():
    rep = check(make_df(), n_input=100, n_rejected=10, max_reject_rate=0.05)
    assert not rep["checks"]["taux_rejet_ok"] and rep["reject_rate"] == 0.1


def test_colonne_manquante():
    rep = check(make_df().drop(columns=["label"]), 3, 0, 0.05)
    assert not rep["checks"]["schema_colonnes"]
