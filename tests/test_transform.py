import pytest

from etl.transform import clean_text, enrich, transform_batch, validate


def raw(**kw):
    base = {"parent_asin": "B00X", "user_id": "U1", "rating": 4.0, "title": "Top",
            "text": "Un produit vraiment très efficace, je recommande.", "timestamp": 1588687728923,
            "helpful_vote": 2, "verified_purchase": True}
    return {**base, **kw}


@pytest.mark.parametrize("src, attendu", [
    ("Bien<br />vraiment", "Bien vraiment"),
    ("  trop   d'espaces \n", "trop d'espaces"),
    ("Caf&eacute; &amp; th&eacute;", "Café & thé"),
    (None, ""),
])
def test_clean_text(src, attendu):
    assert clean_text(src) == attendu


@pytest.mark.parametrize("modif, raison", [
    ({"rating": 0}, "rating_hors_bornes"),
    ({"rating": None}, "rating_hors_bornes"),
    ({"parent_asin": ""}, "parent_asin_manquant"),
    ({"user_id": None}, "user_id_manquant"),
    ({"timestamp": -1}, "timestamp_invalide"),
    ({"text": "<br />court"}, "texte_trop_court"),
])
def test_validate_rejette(modif, raison):
    assert validate(raw(**modif), min_chars=20) == raison


def test_validate_accepte():
    assert validate(raw(), min_chars=20) is None


def test_enrich_derive_les_champs():
    out = enrich(raw(rating=2.0))
    assert out["label"] == "neg"
    assert out["year"] == 2020 and out["review_date"] == "2020-05-05"
    assert out["n_words"] == 7


def test_transform_batch_dedoublonne_entre_lots():
    seen = set()
    ok1, rej1 = transform_batch([raw(), raw(rating=9)], 20, seen)
    ok2, rej2 = transform_batch([raw()], 20, seen)            # même clé que le 1er lot
    assert len(ok1) == 1 and [r["reason"] for r in rej1] == ["rating_hors_bornes"]
    assert ok2 == [] and rej2[0]["reason"] == "doublon"
