# Web interface

A browser front end for the restaurant recommender: participants first answer a short intake (age, gender,
consent), then chat with the dialog system. Everything lives in this directory; the code outside `web/` is used
as-is and is not modified.

```
web/
├── app.py                 FastAPI app: /api/session, /api/chat, and the static site
├── chatbot/
│   ├── core_loader.py     makes the existing code importable without torch / transformers
│   ├── session.py         runs DialogManager.step() one turn per request
│   └── storage.py         placeholder: plug the data storage in here
├── public/                index.html, app.js, styles.css
├── scripts/bundle_core.py copies src/, the restaurant data, templates and model into _core/ at build time
└── tests/test_api.py
```

## How it works

- **No torch.** The recommender imports the BERT encoders at module level. The web chat only uses the
  logistic-regression classifier (`log_reg_simple_grouped.pkl`) with the Levenshtein fallback, so
  `core_loader.install()` registers small stand-ins for `dialog.core.encoders` and `dialog.core.datasets` before the
  real code is imported. When torch *is* installed (e.g. the poetry environment) the real modules are used.
- **Stateless turns.** Vercel Functions keep no memory between requests, so the dialog state and context are
  returned to the browser after each turn and sent back with the next message.
- **Storage.** `storage.save_participant()` and `storage.save_turn()` are called for the intake and for every turn
  (with act, state and preferences). For now they only log.

## Run locally

```sh
cd web
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install fastapi numpy pandas scikit-learn==1.9.1 "levenshtein>=0.27.5,<0.28.0" uvicorn
uvicorn app:app --reload
```

Open http://localhost:8000. Tests: `pip install pytest httpx` and `pytest tests`.

## Deploy on Vercel

1. Import the GitHub repository in Vercel.
2. Set **Root Directory** to `web` and keep **Include files outside the root directory in the Build Step** enabled.
3. Deploy. Vercel detects FastAPI from `pyproject.toml`, uses Python 3.13 from `.python-version`, and runs
   `scripts/bundle_core.py` as the build step.

`scikit-learn` is pinned to 1.9.1 because that version pickled the model; change it only after retraining.
