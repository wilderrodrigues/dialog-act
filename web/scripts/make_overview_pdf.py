"""Builds public/study-overview.pdf: every researcher's persons with their group, systems, modes and goals.

The researcher screen only shows the persons; this PDF (linked top right) has the full lists. It is generated from
chatbot/study.py, so run it again after changing the design:

    poetry run python web/scripts/make_overview_pdf.py

Needs Microsoft Edge or Google Chrome (headless print to PDF); set BROWSER to the executable to override.
"""
import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB_DIR))

from chatbot import study  # noqa: E402

OUTPUT = WEB_DIR / "public" / "study-overview.pdf"
BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chrome",
)


def browser() -> str:
    for candidate in (os.environ.get("BROWSER"), *BROWSERS):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    raise SystemExit("No Edge or Chrome found; set BROWSER to its executable.")


def system_cell(system: dict) -> str:
    goals = "".join(f"<li>{html.escape(goal['value'])} <span>({goal['slot']})</span></li>" for goal in system["goals"])
    return f"<td><strong>{system['label']}</strong> · {system['mode']}<ol>{goals}</ol></td>"


def page() -> str:
    sections = []
    for researcher, persons in study.RESEARCHERS.items():
        rows = []
        for person in persons:
            first, second = study.plan(person)
            rows.append(f"<tr><td class='person'>{person}</td><td>{study.group_of(person)}</td>"
                        f"{system_cell(first)}{system_cell(second)}</tr>")
        sections.append(
            f"<section><h2>{html.escape(researcher)} <span>persons {persons.start}–{persons.stop - 1}</span></h2>"
            "<table><thead><tr><th>Person</th><th>Group</th><th>First system</th><th>Second system</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<p class='note'>Test run: one conversation with each system (eager and patient), goal “test”; saved in a separate test file.</p>"
            "</section>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Study overview</title><style>
@page {{ size: A4; margin: 14mm; }}
body {{ font: 10.5pt/1.4 "Segoe UI", system-ui, sans-serif; color: #1d2330; margin: 0; }}
h1 {{ font-size: 16pt; margin: 0 0 4px; }}
.intro {{ color: #5d6677; margin: 0 0 14px; }}
section {{ break-inside: avoid; margin-bottom: 18px; }}
h2 {{ font-size: 12.5pt; margin: 0 0 6px; }}
h2 span, td span, .note {{ color: #5d6677; font-weight: 400; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid #d9dde4; }}
th {{ font-size: 9pt; color: #5d6677; }}
td.person {{ font-weight: 700; }}
ol {{ margin: 2px 0 0; padding-left: 18px; }}
.note {{ font-size: 9pt; margin: 4px 0 0; }}
</style></head><body>
<h1>Restaurant recommender study: participants and goals</h1>
<p class="intro">Group A (persons 1–14) starts with the eager mode, group B (persons 15–28) with the patient mode.
Person n + 14 gets the same goals as person n; only the modes are swapped. Participants see “System A” and
“System B”, never the mode. Every system: three conversations with the goals listed in order.</p>
{''.join(sections)}
</body></html>"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "overview.html"
        source.write_text(page(), encoding="utf-8")
        subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUTPUT}", source.as_uri()], check=True, capture_output=True, timeout=120)
    if not OUTPUT.is_file():
        raise SystemExit("The browser did not write the PDF.")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
