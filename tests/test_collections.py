"""Collection intent and manifest persistence.

Two things are being pinned down here. A collection declares *what it is for* (which event
classes it reads and decodes), and that declaration is checkable against the catalog with no
data drive mounted. And a collection's manifest is a separate, explicitly-built artifact whose
staleness is derived rather than stored.

    conda activate tether && pytest tests/test_collections.py
"""

import pytest

from neuralshed.manifest import BUILDER_VERSION, CollectionManifest, LayoutSpec

CHAIN = {
    "name": "Chain",
    "steps": {
        "CombineStudies": {
            "name": "CombineStudies",
            "chains": {"card_2024": {"name": "Card2024Brain2Text"}},
        }
    },
}


@pytest.fixture
def collections(tmp_path, monkeypatch):
    """A throwaway database with just enough catalog to answer the stream check."""
    from neuralshed.database import db
    from neuralshed.database.schema import Base, Dataset, EventType

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_Session", None)

    engine = db.get_engine()
    Base.metadata.create_all(engine)

    catalog = {
        "card_2024": ["SpikeCounts", "BandPower", "Sentence"],
        "fan_2023": ["SpikeCounts", "Sentence"],  # no band power
        "singer_2025": ["SpikeCounts", "BandPower", "Word"],  # no sentences
        "fan_h1": ["Spikes", "Action"],  # neither
    }
    from sqlalchemy.orm import sessionmaker

    with sessionmaker(bind=engine, future=True)() as s:
        for dataset_id, classes in catalog.items():
            s.add(Dataset(id=dataset_id, study_class=f"{dataset_id}Study"))
            for name in classes:
                s.add(EventType(dataset_id=dataset_id, name=name, event_class=name))
        s.commit()

    from neuralshed.database import collections as module

    return module


def _manifest(*, participant: str = "T15") -> CollectionManifest:
    return CollectionManifest(
        [
            LayoutSpec(
                layout_key=f"{participant}/card/s1/SpikeCounts",
                group_key="card_2024/s1",
                event_type="SpikeCounts",
                kind="data",
                role="input",
                participant=participant,
                study="Card2024Brain2Text",
                member="card_2024",
                groups=("card_2024/s1",),
                channels=256,
                native_rate=50.0,
            )
        ]
    )


# ===========================================================================
# Declared intent
# ===========================================================================


def test_a_collection_records_what_it_is_for(collections):
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts", "BandPower"),
        target_event_types=("Sentence",),
    )
    col = collections.get_collection(collection_name="speech-text")
    assert col["input_event_types"] == ["SpikeCounts", "BandPower"]
    assert col["target_event_types"] == ["Sentence"]


def test_inputs_are_wanted_if_present(collections):
    """fan_2023 has no band power and still belongs in a spike-plus-band-power collection. It
    contributes one input layout instead of two, which is a fact, not an error."""
    report = collections.check_declared_streams(
        dataset_ids=["card_2024", "fan_2023"],
        input_event_types=("SpikeCounts", "BandPower"),
        target_event_types=("Sentence",),
    )
    assert report["inputs"]["fan_2023"] == ["SpikeCounts"]
    assert report["missing_inputs"] == [] and report["missing_targets"] == []

    collections.save_collection(
        collection_name="mixed",
        members=["card_2024", "fan_2023"],
        chain=CHAIN,
        input_event_types=("SpikeCounts", "BandPower"),
        target_event_types=("Sentence",),
    )
    assert collections.get_collection(collection_name="mixed") is not None


def test_targets_are_required_and_refused_at_save_time(collections):
    """singer_2025 has Word but no Sentence, so it would contribute no trials. Caught from the
    catalog while somebody is still assembling the collection, with no data drive involved.
    """
    with pytest.raises(ValueError, match="cannot supply the declared targets"):
        collections.save_collection(
            collection_name="wrong",
            members=["card_2024", "singer_2025"],
            chain=CHAIN,
            input_event_types=("SpikeCounts",),
            target_event_types=("Sentence",),
        )
    assert collections.get_collection(collection_name="wrong") is None


def test_a_member_supplying_no_declared_input_is_reported(collections):
    """fan_h1 stores spike *times* (`Spikes`), not binned counts, so a collection reading
    SpikeCounts cannot read it at all."""
    report = collections.check_declared_streams(
        dataset_ids=["card_2024", "fan_h1"],
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    assert report["missing_inputs"] == ["fan_h1"]
    assert report["missing_targets"] == ["fan_h1"]


def test_an_unknown_dataset_id_raises(collections):
    """Answering "carries nothing" for an id the catalog has never heard of would read exactly
    like a real mismatch, so a typo would look like a wrong choice of target."""
    with pytest.raises(KeyError, match="unknown dataset"):
        collections.check_declared_streams(
            dataset_ids=["card_2024", "card_2025"],
            input_event_types=("SpikeCounts",),
            target_event_types=("Sentence",),
        )


def test_a_typo_in_a_member_is_refused_at_save_time(collections):
    with pytest.raises(KeyError, match="unknown dataset"):
        collections.save_collection(
            collection_name="typo",
            members=["card_2024", "card_2025"],
            chain=CHAIN,
            input_event_types=("SpikeCounts",),
            target_event_types=("Sentence",),
        )
    assert collections.get_collection(collection_name="typo") is None


def test_the_stream_check_can_be_skipped_for_an_undescribed_dataset(collections):
    collections.save_collection(
        collection_name="new",
        members=["card_2024"],
        chain=CHAIN,
        target_event_types=("NotInCatalog",),
        check_streams=False,
    )
    assert collections.get_collection(collection_name="new")["target_event_types"] == [
        "NotInCatalog"
    ]


# ===========================================================================
# Manifest persistence
# ===========================================================================


def test_saving_a_collection_does_not_build_a_manifest(collections):
    """Curating and materializing are different activities. Folding the slow one into the fast
    one is what makes its cost invisible."""
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    assert collections.manifest_status(collection_name="speech-text") == "not_built"


def test_get_manifest_names_the_fix_rather_than_building_one(collections):
    """A silent build here would put the cost back where nobody asked for it."""
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    with pytest.raises(ValueError, match="build_manifest"):
        collections.get_manifest(collection_name="speech-text")


def test_a_stored_manifest_round_trips_through_the_database(collections):
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    manifest = _manifest()
    col = collections.get_collection(collection_name="speech-text")
    manifest.recipe_digest = collections._recipe_digest(col)
    manifest.built_at = "2026-08-05T00:00:00+00:00"
    collections._store_manifest(collection_name="speech-text", manifest=manifest)

    assert collections.manifest_status(collection_name="speech-text") == "built"
    loaded = collections.get_manifest(collection_name="speech-text")
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.by_key["T15/card/s1/SpikeCounts"].channels == 256


def test_a_manifest_goes_stale_when_the_recipe_changes(collections):
    """The recipe digest is the only staleness signal, so a changed chain has to move it."""
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    manifest = _manifest()
    manifest.recipe_digest = collections._recipe_digest(
        collections.get_collection(collection_name="speech-text")
    )
    collections._store_manifest(collection_name="speech-text", manifest=manifest)
    assert collections.manifest_status(collection_name="speech-text") == "built"

    # same members, different query: a different collection in every way that matters
    queried = {
        "name": "Chain",
        "steps": {
            "CombineStudies": {
                "name": "CombineStudies",
                "chains": {
                    "card_2024": {"name": "Card2024Brain2Text", "query": "session < 2"}
                },
            }
        },
    }
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=queried,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    assert (
        collections.manifest_status(collection_name="speech-text") == "not_built"
    )  # replace clears it


def test_declared_streams_are_part_of_the_recipe_digest(collections):
    """Changing what a collection decodes changes what its manifest would resolve to."""
    collections.save_collection(
        collection_name="c",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    base = collections._recipe_digest(collections.get_collection(collection_name="c"))
    collections.save_collection(
        collection_name="c",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts", "BandPower"),
        target_event_types=("Sentence",),
    )
    assert (
        collections._recipe_digest(collections.get_collection(collection_name="c"))
        != base
    )


def test_a_blob_from_a_different_builder_version_reads_as_stale(collections):
    """A stored manifest is a second source of truth. When the code that reads it is not the
    code that wrote it, rebuilding beats trusting."""
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    manifest = _manifest()
    manifest.recipe_digest = collections._recipe_digest(
        collections.get_collection(collection_name="speech-text")
    )
    manifest.builder_version = BUILDER_VERSION + 1
    collections._store_manifest(collection_name="speech-text", manifest=manifest)
    assert collections.manifest_status(collection_name="speech-text") == "stale"


def test_build_manifest_refuses_a_collection_with_no_declared_inputs(collections):
    collections.save_collection(
        collection_name="bare", members=["card_2024"], chain=CHAIN
    )
    with pytest.raises(ValueError, match="declares no input_event_types"):
        collections.build_manifest(collection_name="bare")


def test_list_collections_shows_whether_a_manifest_exists(collections):
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024"],
        chain=CHAIN,
        input_event_types=("SpikeCounts",),
        target_event_types=("Sentence",),
    )
    assert collections.list_collections()["manifest"].tolist() == ["not_built"]
