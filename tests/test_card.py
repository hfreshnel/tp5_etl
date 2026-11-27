import yaml

from etl.card import build_card
from etl.config import Config


def test_card_contient_yaml_valide_et_metriques():
    report = {"n_input": 1000, "n_output": 950, "n_rejected": 50, "reject_rate": 0.05,
              "label_distribution": {"pos": 0.8, "neg": 0.15, "neu": 0.05}, "years": [2015, 2023],
              "checks": {"rating_1_5": True, "pas_de_doublon": True}}
    card = build_card(Config(), report, {"texte_trop_court": 40, "doublon": 10}, "abc123")
    front = card.split("---")[1]
    meta = yaml.safe_load(front)
    assert meta["size_categories"] == ["n<1K"]
    assert meta["source_datasets"] == ["McAuley-Lab/Amazon-Reviews-2023"]
    assert "texte_trop_court | 40" in card and "abc123" in card
