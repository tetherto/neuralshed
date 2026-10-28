# TODO: Willett 2023 


## phoneme targets

- [ ] Add sentence-level phoneme targets compatible with `card_2024`.
  - Parse `seqClassIDs` and `nSeqElements` from the authors' labeled train/test TFRecords (9,680 trials). Their IDs match `PhonemeVocab.canonical()`: `0` is blank/padding, `1-39` are ARPAbet phonemes, and `40` is `SIL`, represented by `|` in neuralshed.
  - Attach the decoded sequence to `Sentence.phonemes` and use `PhonemeEncoder`; do not emit timed `Phoneme` events because no per-phoneme alignment exists.
  - Generate targets for the remaining 1,220 released sentences with the authors' exact `g2p_en==2.1.0` pipeline, after validating it against the saved TFRecord labels.
  - Document that these are G2P-derived prompt pronunciations, not observed pronunciations. Re-download `derived/tfRecords` first because the current local tree retains only its documentation and generation code.


## per-array dense rows

The corpus-wide design lives in [`others/plan_dense_rows_by_region.md`](../../../../others/plan_dense_rows_by_region.md): `array`, `brain_region`, ontology ids and `hemisphere` on the neural events, a region registry, a catalog without column ranges, and layouts resolved by the consuming model. Its phases 0-1 come first. This dataset's share, once they land:

- [ ] Emit eight neural rows per block, one per array and feature, instead of two `(256, T)` rows.
  - Rows are `_FEATURES` x `utils.ARRAY_ORDER`, each with `n_features=64`, `array` set to the `ARRAY_ORDER` name (equal to `metadata.py` `arrays[].name`), `brain_region` `6v` or `44`, `hemisphere="left"`. The base Study fills the ontology ids from the registry.
  - `_load_binned(timeline, mat_key, array)` indexes the file columns with `utils.ARRAY_CHANNELS[array]` (ascending file columns, the same 64 columns as today's array-order block), through the existing `read_stream` cache so eight rows still cost one file read per session and stream.
  - Emission order `dorsal_6v, ventral_6v, dorsal_44, ventral_44`, so `NeuralFeatureExtractor(aggregation="cat")` with no query reproduces the present `(256, T)` matrix column for column. `CHANNEL_ORDER`, `to_array_order` and `STUDY_ARRAY_SLICES` stay for the source readers and the README grids.
- [ ] Move the 2022.06.14 flag from the timeline dict to the rows: `valid=False` with `"area 44 recordings invalid"` on the two area 44 rows of the nine blocks (18 rows in the corpus), every timeline back to `valid=True`. `utils.INVALID_BLOCKS` keeps the block list, `_load_timeline_events` applies it per row. A `brain_region == '6v'` query then never sees the flag.
- [ ] Catalog and docs. `metadata.py` arrays drop `channel_start` / `channel_stop` and reference the registry labels. README "Study, timelines and events": the eight rows, the `cat` column order, the queries (`brain_region == '6v'` with `aggregation="cat"` gives the 128 area 6v channels). Notebook 02 section 4 loops over the arrays or concatenates them, notebook 03 replaces the `AREA_6V` index with that query.
- [ ] Verification. Per block, the concatenation of the four rows of each feature equals the present matrix. The 6v query returns 128 columns whose file columns are `ARRAY_CHANNELS` of the two 6v arrays. The 18 area 44 rows are the only `valid=False` rows. `validate --check-sessions` (306 timelines, 2,448 dense rows), `run()` on the full corpus (35,148 events), the database repopulated, cold and warm extraction timed on one session.
