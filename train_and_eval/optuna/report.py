"""Self-contained escaped HTML and lossless JSON, in a new timestamped directory."""

import html
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


def render(data):
    esc = lambda value: html.escape(str(value))

    def block(value):
        return (
            "<pre>"
            + esc(
                json.dumps(
                    value, indent=2, ensure_ascii=False, default=str, allow_nan=False
                )
            )
            + "</pre>"
        )

    proposals = data.get("sampling", {}).get("proposals", [])
    cards = ""
    for i, p in enumerate(proposals, 1):
        cards += f"<article><h2>Proposal {i}</h2>" + block(p["params"])
        cards += (
            "<p>Axes outside observed marginal ranges: "
            + esc(", ".join(p["outside_observed_ranges"]) or "none")
            + "</p>"
        )
        cards += (
            "<details><summary>Nearest observations (descriptive, not an explanation of TPE)</summary>"
            + block(p["nearest"])
            + "</details></article>"
        )
    sections = [
        ("Proposals", cards or "<p>No proposal. See status and history checks.</p>"),
        ("History", block(data.get("history", {}))),
        (
            "Sampling",
            block(
                {k: v for k, v in data.get("sampling", {}).items() if k != "proposals"}
            ),
        ),
        ("Paths", block(data.get("paths", []))),
        ("Preparation", block(data.get("preparation", {"status": "not_requested"}))),
        ("Configuration", block(data["config"])),
        ("Provenance", block(data.get("provenance", {}))),
    ]
    tabs = "".join(
        f'<button type="button" data-tab="t{i}">{label}</button>'
        for i, (label, _) in enumerate(sections)
    )
    panes = "".join(
        f'<section id="t{i}" {"hidden" if i else ""}>{body}</section>'
        for i, (_, body) in enumerate(sections)
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optuna propositions</title><style>body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#18273b;background:#f4f7fb}nav{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}button{padding:10px 16px;border:1px solid #96abc1;border-radius:6px;background:white;cursor:pointer}button[aria-selected=true]{background:#173e63;color:white}article,section{background:white;padding:20px;border-radius:10px}section[hidden]{display:none}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;line-height:1.5}summary{cursor:pointer}h1{margin-bottom:8px}</style>
<h1>Optuna propositions</h1>"""
        + "<p>Status: <b>"
        + esc(data["status"])
        + "</b></p>"
        + block(data.get("messages", []))
        + """
<p>Suggestions are candidate experiments, not predictions or guarantees. History uses training seed 1 and final validation results. No training or database changes were performed.</p>
<nav>"""
        + tabs
        + "</nav>"
        + panes
        + """<script>const buttons=document.querySelectorAll('button[data-tab]');buttons.forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==b.dataset.tab);buttons.forEach(x=>x.setAttribute('aria-selected',x===b?'true':'false'));}));buttons[0].setAttribute('aria-selected','true');</script></html>"""
    )


def save(data, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    directory = root / name
    directory.mkdir(exist_ok=False)
    (directory / "report.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (directory / "report.html").write_text(render(data), encoding="utf-8")
    return directory


def update(data, directory):
    """Update only this invocation's newly created report after preparation."""
    for name, content in (
        (
            "report.json",
            json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False)
            + "\n",
        ),
        ("report.html", render(data)),
    ):
        temporary = directory / (name + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(content)
            temporary.replace(directory / name)
        finally:
            temporary.unlink(missing_ok=True)
