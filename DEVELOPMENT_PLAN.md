# Elephant Vocalization Detection — 4-Month Development Plan

**Role of this document:** authoritative engineering plan. Claude (architecture/review), Gemini (implementation), Antigravity (integration/execution), and the human team should all treat this as the current source of truth until it is explicitly revised. Update it — don't fork it — as decisions change.

---

## 1. Current Project Audit

I was given two documents describing the "current" state of the repo, and they disagree with each other on basic facts. I'm not going to paper over that — it's the first real finding.

**The conflict:**

| | Document 1 (role-prompt) | Document 2 (README + Antigravity context) |
|---|---|---|
| Core files | `preprocess.py`, `train_detector.py`, `train_classifier.py` | `elephant_cnn_model.py` (only) — README also describes a *planned* `features.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `prepare_dataset.py` that don't yet exist per the Antigravity context section |
| Dataset layout | `dataset/elephant/{trumpet,roar,rumble}`, `dataset/non_elephant/{animals,human,rain,wind}` | `data/positive/`, `data/negative/`, driven by `train.csv/val.csv/test.csv` |
| Training status | "Best model checkpoint" produced by a working training loop, but *not yet scientifically validated* | Explicitly: only tested on **dummy spectrogram data**, no real training has happened |
| Negative class | Real recordings (rain, wind, human, other animals) | Partly **simulated** — "bandpass-filtered pink noise" standing in for wind/water |

These can't both be accurate descriptions of one repo at one point in time. Two plausible explanations: (a) doc 2 is an aspirational README + a separate context you're about to hand Antigravity for a restructure, written before doc 1's repo existed or was inspected — or (b) doc 1 is describing what the code is *supposed* to do rather than what's been verified to run. Either way, per the ground rule "never fabricate," I won't guess which is real. **Resolving this is Phase 0, Task 1** — someone runs `git log`, `ls -R src/`, and actually opens the files, and reports back.

Everything below is written to be correct *regardless* of which description turns out to be true, and Phase 0 exists specifically to collapse this ambiguity before Phase 1 begins.

### Completed
- A problem statement that is appropriately scoped (binary detection first, no translation/behavioural claims).
- At least one CNN architecture has been written in PyTorch and is syntactically/structurally functional (confirmed by doc 2: it ran on dummy data without crashing).
- Two candidate public datasets have been identified with URLs (LDC2010S05, Dryad African elephant dataset).
- A reasonable class scheme has been proposed (binary detector first, call-type classifier deferred).
- Project boundaries (no IoT, no multimodal, no translation) are clearly and repeatedly stated — good, this will save you from scope creep later.

### Partially Completed
- **Preprocessing pipeline** — described in detail (resample to 22.05kHz or per doc 2's plan, fixed duration, log-Mel) but not confirmed to have run on real audio.
- **Model training** — a training loop may exist in code form, but no real-data training run has been confirmed by either document. Doc 2 is explicit that dummy-data-only is the current truth for at least one model file.
- **Negative/background class** — per doc 2, may be partly synthetic (pink noise) rather than real recorded background sound, which would matter a lot for the model's real-world validity.

### Missing
- Confirmed, inspected, real dataset on disk with known provenance, licensing, and per-clip labels.
- Any real evaluation run (accuracy/precision/recall/F1/confusion matrix on held-out real data).
- Recording-level train/val/test split logic (i.e., proof that clips from the same source recording don't leak across splits).
- Acoustic characteristic extraction (dominant frequency, duration, spectral stats) — described as a goal, not shown as implemented.
- Inference script (upload new audio → prediction).
- Streamlit or any interface.
- `AI_CONTEXT.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md` — none of these exist yet.
- Tests of any kind.

### Needs Verification
- **Which file layout is real** (see conflict table above) — top priority.
- Whether `dataset/` (or `data/`) currently contains real audio files or is still empty/placeholder.
- Dataset licensing for both LDC2010S05 (LDC requires membership/purchase in most cases) and the Dryad set (check the specific CC license on the Dryad page).
- Sample rate consistency across classes (doc 2 flags Dryad at 16kHz vs LDC at 48kHz — mixing them without resampling consistently would quietly corrupt the model).
- Whether "elephant" and "non-elephant" clips were recorded in comparable conditions (if all elephant clips are field recordings with certain background hiss and all negative clips are clean pink noise, the model will learn to detect *recording conditions*, not elephants — this is a classic bioacoustics pitfall).
- Class balance in whatever real data currently exists.
- Whether any clips are duplicated or derived from the same longer original recording (leakage risk).

### Should Be Postponed
- The multi-class call classifier (`train_classifier.py` / Trumpet-Roar-Rumble) — explicitly deferred by both documents until the binary detector is proven.
- Any Streamlit/interface work.
- All Version 2.0 items: noise robustness, distance analysis, infrasound analysis.
- Restructuring the repo into the fuller `src/preprocessing/`, `src/models/`, etc. package layout — not until it's clear the current layout is actually a bottleneck.

---

## 2. Recommended Final Architecture

Keep this decision explicit and don't restructure preemptively: **start from whatever the audit in Phase 0 finds actually exists**, and only introduce the fuller package structure below once a second module (e.g. the acoustic-analysis code) makes a flat `src/` genuinely awkward. Premature restructuring burns time without adding a working result.

Target structure (to grow into, not to build on day one):

```
my_elephant_project/
├── dataset/                     # or data/ — pick ONE name in Phase 0 and stick to it
│   ├── raw/                     # untouched original files, per source, with a manifest
│   └── manifest.csv             # filepath, label, source, duration, sample_rate, split
├── src/
│   ├── preprocessing/           # audio loading, resampling, mel-spectrogram
│   ├── datasets/                # PyTorch Dataset classes
│   ├── models/                  # CNN architecture(s)
│   ├── training/                # training loop, checkpointing
│   ├── evaluation/               # metrics, confusion matrix
│   ├── inference/                # single-file predict script
│   └── analysis/                 # acoustic characteristic extraction
├── tests/
├── configs/                      # e.g. sample_rate, duration, n_mels as YAML, not hardcoded
├── models/                       # saved checkpoints (gitignored)
├── results/                      # metrics, plots, confusion matrices (gitignored or tracked selectively)
├── app.py                        # Streamlit demo (Phase 3+ only)
├── requirements.txt
├── README.md
├── AI_CONTEXT.md
├── ENGINEERING_RULES.md
├── DECISIONS.md
├── EXPERIMENT_LOG.md
└── DEVELOPMENT_PLAN.md           # this file
```

The one non-negotiable addition regardless of current layout: a **manifest file** (`manifest.csv` or similar) that records, per audio clip, its source, original recording ID, label, and split assignment. Without this, recording-level leakage prevention (a repeated explicit requirement) isn't verifiable later.

---

## 3. 4-Month Development Roadmap

| Month | Phase | Version Target | Deliverable |
|---|---|---|---|
| 1 (Weeks 1–2) | Phase 0 — Audit & Reconciliation | — | Confirmed repo state, single source of truth, manifest schema |
| 1 (Weeks 3–4) | Phase 1 — Real Dataset Pipeline | v0.1 – v0.2 | Verified real audio, working preprocessing, recording-level train/val/test split |
| 2 | Phase 2 — Baseline Training & Evaluation | v0.3 – v0.4 | Trained detector on real data, full metrics, confusion matrix |
| 3 | Phase 3 — Inference, Acoustic Analysis, Interface | v0.5 – v0.7 | Predict-on-new-audio script, acoustic characteristics, Streamlit demo |
| 4 | Phase 4 — Experiments, Validation, Documentation | v0.8 – v1.0 | Justified experiments, final report, reproducible results, presentation |

---

## 4. Detailed Phase Breakdown

### PHASE 0 — Audit & Reconciliation
**Objective:** Establish one true description of the repo's current state.

**Why necessary:** The two source documents disagree on what exists. Building on top of an unverified description risks wasted work or, worse, quietly training on synthetic negative data believed to be real.

**Prerequisites:** None — this is the starting point.

**Tasks:**
1. Run `git log --oneline` and `find . -type f -name "*.py"` on the actual repo; list every file that really exists.
2. Open each Python file and note: does it run? Against what data? Real or dummy?
3. Inspect `dataset/` (or `data/`) — is it empty, partially populated, or fully populated? With what?
4. Decide and document one canonical folder name (`dataset/` vs `data/`) and one canonical file layout.
5. Write `AI_CONTEXT.md` summarizing the verified truth, so Gemini/Antigravity don't re-inherit the conflicting docs.

**Expected code/modules:** None new — inspection only, plus documentation.

**Expected experiments:** None.

**Testing requirements:** None yet.

**Deliverables:** `AI_CONTEXT.md`, a short audit note (can be a GitHub issue) stating exactly what exists.

**Definition of Done:** A single person (any team member) can read `AI_CONTEXT.md` and correctly describe the repo without opening it.

**GitHub milestone:** `v0.0-audit`

**Risks:** Team disagreement about which document was "more real" — resolve by trusting the filesystem, not either document.

**Possible improvements:** N/A — deliberately minimal phase.

**What should NOT be done yet:** Any new modeling code, any restructuring, any dataset downloading.

---

### PHASE 1 — Real Dataset Pipeline
**Objective:** Get real, labeled, licensed audio on disk with a verified, leakage-safe split.

**Why necessary:** Every downstream claim (accuracy, generalization) is worthless if built on synthetic negatives or leaking splits.

**Prerequisites:** Phase 0 complete.

**Tasks:**
1. Check current licensing terms for both LDC2010S05 (usually requires LDC membership — verify cost/access before committing) and the Dryad dataset (verify the exact CC license on the Dryad page).
2. Choose one dataset (or a justified real-data source) as primary; document the choice and reasoning in `DECISIONS.md`.
3. Inspect real folder structure, metadata, and per-clip labels — do not assume the categories listed in either document are final until confirmed against the actual files.
4. Source real (not synthetic) negative/background clips wherever possible; if any synthetic negatives remain, label them explicitly as synthetic in the manifest so they can be excluded from evaluation if needed.
5. Build `manifest.csv`: filepath, label, source recording ID, duration, sample rate.
6. Implement/confirm preprocessing: mono conversion, consistent resampling, fixed-duration windowing, log-Mel spectrogram generation.
7. Implement recording-level split (not clip-level) into train/val/test.
8. Check and record class balance.

**Expected code/modules:** `prepare_dataset.py` or equivalent, `preprocess.py` (verified working on real files), split logic with recording-ID awareness.

**Expected experiments:** None yet — this is data engineering, not modeling.

**Testing requirements:** Unit test that split logic never places two clips from the same source recording in different splits. Unit test that preprocessing output shape/sample-rate is consistent across the dataset.

**Deliverables:** `manifest.csv`, verified `train/val/test` clip lists, a short data-quality note.

**Definition of Done:** Running the preprocessing pipeline over the full manifest produces spectrograms for every clip with no errors, and the split-leakage test passes.

**GitHub milestone:** `v0.2-dataset-pipeline`

**Risks:** Chosen dataset turns out to be inaccessible or unlicensed for use — mitigate by checking licensing *before* committing engineering time, and having Dryad as a fallback since it appears more openly accessible.

**Possible improvements:** Automate manifest generation from folder structure via a script rather than manual CSV editing.

**What should NOT be done yet:** Model training, the multi-class classifier, any interface work.

---

### PHASE 2 — Baseline Training & Evaluation
**Objective:** Train the binary detector on real data and evaluate it honestly.

**Why necessary:** This is the core scientific claim of the project — everything else is supporting infrastructure.

**Prerequisites:** Phase 1 complete, manifest and splits verified.

**Tasks:**
1. Connect existing CNN (whichever one Phase 0 confirmed is real) to the real `DataLoader`.
2. Train using cross-entropy loss, track train/val loss and accuracy per epoch.
3. Save best checkpoint by validation metric (not final epoch).
4. Run evaluation on the held-out test set — never touched until this point.
5. Compute accuracy, precision, recall, F1, confusion matrix; add ROC-AUC/PR-AUC if class imbalance warrants it.
6. Log everything to `EXPERIMENT_LOG.md`, including hyperparameters, exact dataset version, and results.

**Expected code/modules:** `train.py`/`train_detector.py`, `evaluate.py`, `model.py` (confirmed architecture).

**Expected experiments:**
- **Experiment A (Baseline CNN):**
  QUESTION: Can the current CNN architecture distinguish elephant vocalizations from background sound on real data, above chance and above a trivial baseline?
  HYPOTHESIS: Yes, given the acoustic distinctiveness of elephant rumbles/trumpets.
  METHOD: Train on train split, tune on val, report test metrics once.
  RESULT: [fill in only after actually running it]
  CONCLUSION: [fill in only after actually running it]

**Testing requirements:** Sanity test that training loss decreases over epochs on a small subset; test that the saved checkpoint loads and produces the same metrics as the training-time best.

**Deliverables:** `best_model.pt` (or `.pth`), `results/confusion_matrix.png`, metrics report, entry in `EXPERIMENT_LOG.md`.

**Definition of Done:** Test-set metrics are computed once, from real data, and documented — not re-run repeatedly to cherry-pick a number.

**GitHub milestone:** `v0.4-baseline-trained`

**Risks:** Model exploits a shortcut (e.g., recording-condition artifacts rather than actual vocalization content) — mitigate by manually listening to a sample of correctly- and incorrectly-classified clips, and checking whether errors cluster by data source.

**Possible improvements:** Simple hyperparameter sweep (learning rate, dropout) if time allows — but only after a working baseline exists.

**What should NOT be done yet:** The multi-class call classifier, noise robustness, distance/infrasound work.

---

### PHASE 3 — Inference, Acoustic Analysis, Interface
**Objective:** Turn the trained model into something a reviewer can actually use and inspect.

**Why necessary:** A checkpoint file alone isn't demonstrable; the problem statement explicitly promises acoustic characteristic analysis, not just a yes/no label.

**Prerequisites:** Phase 2 complete with an acceptable, documented baseline.

**Tasks:**
1. Write a single-file inference script: audio in → preprocessing → model → prediction + confidence.
2. Implement acoustic feature extraction on the same audio: duration, dominant frequency (e.g., via spectral peak), frequency range, basic spectral stats — computed from the real signal, not hardcoded.
3. Generate and save a spectrogram visualization for the input clip.
4. Build a minimal Streamlit interface wrapping the above: upload → predict → show confidence, acoustic characteristics, and spectrogram.

**Expected code/modules:** `predict.py`/`inference/`, `analysis.py`, `app.py`.

**Expected experiments:** None new — this phase is applied engineering on top of Phase 2's model.

**Testing requirements:** Test that inference on a known-elephant clip and a known-non-elephant clip produces sane, differing outputs; test that acoustic feature extraction returns plausible (non-NaN, non-zero-duration) values.

**Deliverables:** Working `app.py`, example output screenshots for the report.

**Definition of Done:** A team member unfamiliar with the code can run `streamlit run app.py`, upload a file, and get a real prediction with real acoustic characteristics.

**GitHub milestone:** `v0.7-interface`

**Risks:** Dominant-frequency extraction is noisy on short/quiet clips — mitigate by validating extracted values against a few manually-inspected spectrograms.

**Possible improvements:** Batch-mode inference for multiple files at once (nice-to-have, not required).

**What should NOT be done yet:** Multi-class classification, any hardware/IoT integration.

---

### PHASE 4 — Experiments, Validation, Documentation
**Objective:** Turn the working baseline into an academically defensible final result.

**Why necessary:** A final-year project needs justified experimental comparisons and honest limitations, not just one trained model.

**Prerequisites:** Phase 3 complete.

**Tasks:**
1. Design and run 2–3 *justified* additional experiments (see Section 7).
2. Write up limitations explicitly: dataset size/provenance, potential recording-condition shortcuts, generalization boundaries.
3. Finalize `README.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md`.
4. Prepare final report and presentation material from real, logged results only.
5. Tag a `v1.0` release on GitHub.

**Expected code/modules:** Any code needed to run Section 7's experiments (e.g., an augmentation flag, an alternate architecture file).

**Expected experiments:** See Section 7 below.

**Testing requirements:** All experiment scripts should be re-runnable from a clean checkout (reproducibility check).

**Deliverables:** Final report, presentation slides, tagged `v1.0` release, full experiment log.

**Definition of Done:** Another student could clone the repo, follow the README, and reproduce the reported test-set metrics within reasonable variance.

**GitHub milestone:** `v1.0-final`

**Risks:** Running out of time to properly document — mitigate by keeping `EXPERIMENT_LOG.md` updated continuously from Phase 2 onward, not written retroactively.

**Possible improvements:** N/A — this phase closes the loop, doesn't open new scope.

**What should NOT be done yet:** Version 2.0 items (below) unless Phase 4 finishes with significant time remaining and the guide agrees to scope expansion.

---

## 5. AI Responsibility Model

- **Claude:** Owns this document and its revisions, architecture decisions, experiment design justification, code review of what Gemini produces, and sanity-checking that no phase's "Definition of Done" is being fudged.
- **Gemini:** Implements individual modules against the interfaces this plan specifies (e.g., "write `evaluate.py` that takes a checkpoint path and manifest split, outputs the metrics listed in Phase 2").
- **Antigravity:** Runs the full pipeline end-to-end, catches cross-file integration bugs, executes training runs, and is the one that should actually confirm "does this run on real data" during Phase 0.
- **GitHub:** Tracks every phase as a milestone, every task as an issue, every module as a feature branch and PR.
- **Human team:** Makes the licensing/dataset choice call, listens to misclassified clips to sanity-check the model isn't cheating, and owns the final report and defense.

---

## 6. Software Engineering Workflow

1. Create a GitHub Issue per task (e.g., "Implement recording-level dataset split").
2. Branch: `feature/recording-level-split` off `develop`.
3. Gemini implements against the interface Claude specified in the issue.
4. Antigravity runs it, confirms it integrates and executes without breaking existing modules.
5. Human team reviews for understanding — not just "does it run" but "do we understand why."
6. PR into `develop`, commit message style `feat: add recording-level dataset splitting`.
7. Merge `develop` → `main` at each phase boundary, tag the version (`v0.2`, `v0.4`, etc.) per the roadmap in Section 3.
8. Keep `main` always in a working, demonstrable state; keep experimental work off `main`.

---

## 7. Research / Experiment Strategy

Only experiments answerable with the actual dataset are included. Each is deferred until Phase 2's baseline exists.

- **Experiment A — Baseline CNN** (Phase 2, required): established above.
- **Experiment B — Mel-spectrogram parameters:** QUESTION: does changing `n_mels`/window size meaningfully change detection performance? Only worth running if Experiment A's baseline is mediocre (e.g., below ~85% F1) — a strong baseline doesn't need this.
- **Experiment C — Data augmentation:** QUESTION: does time/frequency masking or noise injection improve generalization on the test set? Justified if the real dataset is small (likely, given the described scale), since augmentation directly addresses overfitting risk from limited data.
- **Experiment D — Alternative architecture:** Only pursue if there's a specific, stated reason (e.g., a much deeper CNN overfits a small dataset) rather than "try another model for coverage."

Explicitly **not** planned unless Phase 4 finishes early: noise robustness, distance analysis, infrasound-specific analysis — these require additional data/metadata not confirmed to exist yet.

---

## 8. Documentation Strategy

- **`AI_CONTEXT.md`** — the verified, current truth about repo state (owner: updated after every phase, first written in Phase 0).
- **`ENGINEERING_RULES.md`** — the development rules from the original role-prompt (never fabricate metrics, never commit untested code, etc.) — copy them in verbatim as the team's working contract.
- **`DECISIONS.md`** — one entry per significant decision (e.g., "chose Dryad over LDC2010S05 because [reason], on [date]").
- **`EXPERIMENT_LOG.md`** — one entry per experiment run, following the QUESTION → HYPOTHESIS → METHOD → RESULT → CONCLUSION format from Section 7, filled in only with real results.
- **`DEVELOPMENT_PLAN.md`** — this file; revise it, don't fork it, as phases complete or plans change.

---

## 9. What NOT To Build Yet

- Multi-class call classifier (Trumpet/Roar/Rumble) — Phase 2 extension at earliest, only if the binary baseline is solid.
- Noise robustness, distance analysis, infrasound analysis — Version 2.0, contingent on baseline success.
- Any hardware/IoT (Raspberry Pi, GPS, GSM, buzzers, sensors, drones, cameras).
- Multimodal/semantic/LLM-based interpretation of any kind.
- Behavioural or emotional interpretation without labelled behavioural data.
- Full package restructuring (`src/preprocessing/`, `src/models/`, etc.) until a second module actually needs it.
- FastAPI/React interfaces — Streamlit only, and only in Phase 3.

---

## 10. IMMEDIATE NEXT MILESTONE

**Do this one thing next, nothing else:**

Open the actual repository and produce a short, factual inventory: which Python files really exist, whether each one runs, and on what data (real or dummy). Cross-check this against both documents' claims. Write the result into a new `AI_CONTEXT.md` and commit it.

That's it — no training, no new code, no dataset downloading yet. This resolves the one contradiction blocking every later phase, is completable in under a day, and gives you a clean, honest starting point to hand to Gemini and Antigravity.
