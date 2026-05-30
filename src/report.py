"""
Bilingual (Arabic / English) triage reporting.

A MENA-lab deliverable from the roadmap: render a single slide's triage result
as a clean, shareable report in Arabic, English, or both. Pure-Python and
dependency-free (plain text / minimal HTML), so it works anywhere the rest of
the pipeline does.

The wording is deliberately careful: every report repeats that this is decision
support, not a diagnosis, and that a pathologist makes the final call — in both
languages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from src.inference import TriageResult

# Localised strings. Keys mirror the TriageResult fields / priorities.
_STRINGS = {
    "en": {
        "title": "Naseej — Pathology Triage Report",
        "label": "Prediction",
        "prob": "Probability of malignancy",
        "confidence": "Confidence",
        "priority": "Triage priority",
        "disclaimer": "Decision support, not a diagnosis. Naseej re-orders the "
                      "queue and highlights regions of interest; the pathologist "
                      "makes the final call.",
        "untrained": "WARNING: no trained model loaded — this result is a "
                     "placeholder and is not meaningful.",
        "Benign": "Benign",
        "Malignant": "Malignant",
        "URGENT": "URGENT",
        "REVIEW": "REVIEW",
        "ROUTINE": "ROUTINE",
        "dir": "ltr",
    },
    "ar": {
        "title": "نسيج — تقرير فرز الأنسجة المرضية",
        "label": "التنبؤ",
        "prob": "احتمال الورم الخبيث",
        "confidence": "درجة الثقة",
        "priority": "أولوية الفرز",
        "disclaimer": "أداة دعم القرار وليست تشخيصًا. يقوم نسيج بإعادة ترتيب قائمة "
                      "الحالات وإبراز المناطق المهمة، ويبقى القرار النهائي لطبيب "
                      "الأنسجة المرضية.",
        "untrained": "تحذير: لا يوجد نموذج مُدرَّب — هذه النتيجة مبدئية وغير ذات دلالة.",
        "Benign": "حميد",
        "Malignant": "خبيث",
        "URGENT": "عاجل",
        "REVIEW": "مراجعة",
        "ROUTINE": "روتيني",
        "dir": "rtl",
    },
}


def _line(strings: dict, result: "TriageResult") -> list[str]:
    """Shared field rendering for one language."""
    return [
        f"{strings['label']}: {strings[result.label]}",
        f"{strings['prob']}: {result.prob_malignant:.1%}",
        f"{strings['confidence']}: {result.confidence:.1%}",
        f"{strings['priority']}: {strings[result.priority]}",
    ]


def render_text(result: "TriageResult", lang: str = "en") -> str:
    """Render a plain-text report in ``lang`` ('en' or 'ar')."""
    if lang not in _STRINGS:
        raise ValueError(f"Unsupported language {lang!r}; choose 'en' or 'ar'.")
    s = _STRINGS[lang]
    bar = "=" * 48
    parts = [bar, s["title"], bar, *_line(s, result)]
    if not result.trained:
        parts.append(s["untrained"])
    parts += ["-" * 48, s["disclaimer"], bar]
    return "\n".join(parts)


def render_bilingual_text(result: "TriageResult") -> str:
    """English then Arabic, separated by a divider."""
    return render_text(result, "en") + "\n\n" + render_text(result, "ar")


def render_html(result: "TriageResult", lang: str = "bilingual") -> str:
    """Render an HTML report. ``lang`` is 'en', 'ar', or 'bilingual'."""
    color = {"URGENT": "#b00020", "REVIEW": "#c77700", "ROUTINE": "#1b7a3d"}[result.priority]

    def block(code: str) -> str:
        s = _STRINGS[code]
        rows = "".join(
            f"<tr><td style='padding:4px 12px;color:#555'>{k}</td>"
            f"<td style='padding:4px 12px;font-weight:600'>{v}</td></tr>"
            for k, v in [
                (s["label"], s[result.label]),
                (s["prob"], f"{result.prob_malignant:.1%}"),
                (s["confidence"], f"{result.confidence:.1%}"),
                (s["priority"], f"<span style='color:{color}'>{s[result.priority]}</span>"),
            ]
        )
        warn = (f"<p style='color:#b00020;font-size:13px'>{s['untrained']}</p>"
                if not result.trained else "")
        return (
            f"<section dir='{s['dir']}' style='margin:12px 0;font-family:sans-serif'>"
            f"<h3 style='margin:0 0 6px'>{s['title']}</h3>"
            f"<table style='border-collapse:collapse'>{rows}</table>"
            f"{warn}"
            f"<p style='color:#888;font-size:12px;margin-top:8px'>{s['disclaimer']}</p>"
            f"</section>"
        )

    if lang == "bilingual":
        body = block("en") + "<hr style='border:none;border-top:1px solid #ddd'>" + block("ar")
    else:
        body = block(lang)
    return f"<div>{body}</div>"
