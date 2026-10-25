# Web interface

The browser version of the experiment: does the patient or the eager recommender get more sympathy? Everything
lives in this directory; the code outside `web/` is used as-is and is not modified.

## The flow

1. **Researcher**: pick your name (Remy 1–7, Cai 8–14, Dexter 15–21, Wilder 22–28) and the person you are running.
   The screen only shows the persons (finished ones are marked); each person's group, mode order and goals are in
   the PDF linked top right (`public/study-overview.pdf`, rebuilt with `python scripts/make_overview_pdf.py`).
   **Test** runs one conversation with each system (eager and patient, goal "test") instead of three, and saves it to `test-<name>.json`, so you can check
   that saving works on your computer without touching the real data. The screen shows the folder results go to.
2. **Informed consent**, then **participant info** (until the first conversation starts, *Back* returns to the
   previous screen, e.g. after picking the wrong person): age, gender and *How often do you use AI in daily life?* (1–5).
3. **Instructions**.
4. **System A**: three conversations, each with one goal (Q1 area, Q2 price range, Q3 food), then the questionnaire.
5. **System B**: the same with the other mode and the other three goals, then the questionnaire again.
6. **Thank you**.

Group A (persons 1–14) starts with the eager mode, group B (15–28) with the patient mode; person `n + 14` has the
same goals as person `n`. Participants only see "System A" and "System B", never the mode. The design (table,
questionnaire, researchers) is in `chatbot/study.py`.

The questionnaire has ten items on a 1–5 scale: A1–A5 adapted from the GAAIS (A4 and A5 reverse scored) and B1–B5
Godspeed likeability. Answers are stored as given; reverse scoring is left to the analysis.

The progress is kept in the browser's `sessionStorage`, so reloading the page does not lose a participant.

```
web/
├── app.py                      FastAPI app: /api/study, /api/participant, /api/dialog, /api/chat, /api/results
├── chatbot/
│   ├── core_loader.py          makes the existing code importable without torch / transformers
│   ├── session.py              runs DialogManager.step() one turn per request, in eager or patient mode
│   ├── storage.py              stores the results in the core Experiment format (local JSON file)
│   └── study.py                the experiment design: researchers, goals per person, questionnaire
├── public/                     index.html, app.js, styles.css
├── scripts/
│   ├── bundle_core.py          copies src/, the restaurant data, templates and model into _core/ at build time
│   ├── make_overview_pdf.py    builds public/study-overview.pdf (all persons and goals) from study.py
│   └── merge_results.py        merges the researchers' database-<name>.json files into one
└── tests/test_api.py
```

## Stored data

Each system a participant completes becomes one `Experiment` record from `recommender/core/storage` (so a finished
participant has two records with the same `participant_id`), extended with `researcher`, `person`, `group`,
`ai_usage` and `completed_at`. The session has `mode`, `label` (System A/B) and `order`; every trial has the
`duration` in seconds, `nb_prompts` (user messages), `nb_resets` (restarts), `food_type` (food of the last
recommended restaurant), plus `goal_slot`, `goal_value` and `restaurant`. The conversations themselves are not stored. The questionnaire
is answered once per system, so its answers (`[item, score]`, items 1–10 = A1–A5, B1–B5) are on the last trial; the
other trials have an empty questionnaire. The core `EXPERIMENTS_ADAPTER` can read the file as well (it ignores
the extra fields).

It is written to a local JSON file per researcher, `data/experiment/database-<name>.json` (for example
`database-remy.json`), or the same name in the directory set in `EXPERIMENT_DIR`. `data/experiment/` is in
`.gitignore`, so participant data is never committed; the repository is public. The researcher table only knows
about the persons in the files on your own computer.

Every researcher runs the study on their own laptop. Afterwards, send `database-<name>.json` (not the test file) to one person over a closed channel
(Teams, SURFdrive, e-mail; not git), who merges the four files into one:

```sh
python scripts/merge_results.py data/experiment/merged.json database-remy.json database-cai.json database-dexter.json database-wilder.json
```

## Run locally

```sh
cd web
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install fastapi numpy pandas scikit-learn==1.9.1 "levenshtein>=0.27.5,<0.28.0" "pydantic>=2.14.1,<3" uvicorn
uvicorn app:app --reload
```

Open http://localhost:8000 and keep the terminal open during the session. Tests: `pip install pytest httpx` and `pytest tests`.

## Vercel

The Vercel configuration (`[tool.vercel]`, `scripts/bundle_core.py`) still works for the chat, but a Vercel
Function has no persistent disk, so the results file would be lost. On Vercel the app therefore refuses to save
results; run the study locally.

`scikit-learn` is pinned to 1.9.1 because that version pickled the model; change it only after retraining.
