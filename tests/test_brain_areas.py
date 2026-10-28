"""Brain-area ontology ids agree with the declared Brodmann area."""

import importlib

from neuralshed import regions
from neuralshed.database.populate import discover_datasets

# Brodmann area -> (HOMBA term it must fall under, UBERON id)
BRODMANN_TERMS = {
    "4": ("HOMBA_10163", "UBERON_0001384"),  # primary motor cortex
    "6": ("HOMBA_10168", "UBERON_0016634"),  # premotor cortex
}


def test_homba_and_uberon_ids_match_the_brodmann_area():
    graph = regions.load()
    mismatches = []
    for dataset_id in discover_datasets():
        metadata = importlib.import_module(
            f"neuralshed.datasets.{dataset_id}.metadata"
        ).DATASET_METADATA
        for subject in metadata["subjects"]:
            for area in subject["brain_areas"]:
                if area.get("brodmann") not in BRODMANN_TERMS:
                    continue
                homba_term, uberon_id = BRODMANN_TERMS[area["brodmann"]]
                lineage = [
                    area["homba_id"],
                    *regions.ancestors(area["homba_id"], graph),
                ]
                if homba_term not in lineage or area["uberon_id"] != uberon_id:
                    mismatches.append((dataset_id, subject["label"], area["raw_label"]))
    assert mismatches == []
