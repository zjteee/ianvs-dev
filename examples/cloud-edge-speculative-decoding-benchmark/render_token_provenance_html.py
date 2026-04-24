"""
Render speculative decoding sample outputs into a color-coded HTML report.

This utility reads `specdec_sample_outputs.jsonl` and visualizes the final
completion tokens by provenance:

- draft_accepted: accepted draft tokens
- verifier_bonus: verifier bonus token emitted after a fully accepted block
- verifier_correction: verifier residual correction token after rejection
- rejected_draft: discarded draft suffix after a rejection, rendered in red
"""

import argparse
import html
import json
from collections import Counter


SOURCE_LABELS = {
    "draft_accepted": "Draft Accepted",
    "verifier_bonus": "Verifier Bonus",
    "verifier_correction": "Verifier Correction",
    "rejected_draft": "Rejected Draft",
    "unknown": "Unknown",
}


def _load_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _render_token_span(item):
    token_text = html.escape(item.get("token_text", ""))
    token_id = int(item.get("token_id", 0))
    round_index = int(item.get("round", 0))
    source = str(item.get("source", "unknown"))
    label = SOURCE_LABELS.get(source, source)
    title = html.escape(f"{label} | round={round_index} | token_id={token_id}")
    display_text = token_text if token_text else "∅"
    return f'<span class="token {source}" title="{title}">{display_text}</span>'


def _render_rejected_suffix(items, round_index):
    if not items:
        return ""
    parts = []
    for item in items:
        token_text = html.escape(str(item.get("token_text", "")))
        token_id = int(item.get("token_id", 0))
        title = html.escape(f"Rejected Draft | round={round_index} | token_id={token_id}")
        parts.append(
            f'<span class="token rejected_draft" title="{title}">{token_text if token_text else "∅"}</span>'
        )
    return f'<span class="rejected-group">(</span>{"".join(parts)}<span class="rejected-group">)</span>'


def _render_round_aware_stream(sample, hide_rejected=False):
    provenance = list(sample.get("token_provenance", []) or [])
    round_sequence = list(sample.get("round_sequence", []) or [])
    if not provenance:
        return ""
    grouped = {}
    for item in provenance:
        grouped.setdefault(int(item.get("round", 0) or 0), []).append(item)

    rendered = []
    used_rounds = set()
    for item in round_sequence:
        round_index = int(item.get("round", 0) or 0)
        used_rounds.add(round_index)
        for token in grouped.get(round_index, []):
            rendered.append(_render_token_span(token))
        rejected = list(item.get("rejected_draft_tokens", []) or [])
        if rejected and not hide_rejected:
            rendered.append(_render_rejected_suffix(rejected, round_index))
    for round_index in sorted(set(grouped.keys()) - used_rounds):
        for token in grouped.get(round_index, []):
            rendered.append(_render_token_span(token))
    return "".join(rendered)


def _render_sample(sample, hide_rejected=False):
    provenance = list(sample.get("token_provenance", []) or [])
    source_counts = Counter(item.get("source", "unknown") for item in provenance)
    rejected_count = 0 if hide_rejected else sum(
        len(list(item.get("rejected_draft_tokens", []) or []))
        for item in list(sample.get("round_sequence", []) or [])
    )
    if rejected_count:
        source_counts["rejected_draft"] += rejected_count
    tokens_html = _render_round_aware_stream(sample, hide_rejected=hide_rejected)
    source_summary = " | ".join(
        f"{SOURCE_LABELS.get(source, source)}: {count}"
        for source, count in sorted(source_counts.items())
    )
    prompt = html.escape(str(sample.get("prompt", "")))
    completion = html.escape(str(sample.get("completion", "")))
    request_id = html.escape(str(sample.get("request_id", "unknown")))
    rounds = int(sample.get("rounds", 0) or 0)
    accepted = int(sample.get("accepted_draft_tokens", 0) or 0)
    corrected = int(sample.get("corrected_tokens", 0) or 0)
    total_draft = int(sample.get("total_draft_tokens", 0) or 0)
    return f"""
    <section class="sample-card">
      <h2>{request_id}</h2>
      <div class="meta">rounds={rounds} | accepted_draft_tokens={accepted} | corrected_tokens={corrected} | total_draft_tokens={total_draft}</div>
      <div class="meta">{html.escape(source_summary)}</div>
      <details>
        <summary>Prompt</summary>
        <pre>{prompt}</pre>
      </details>
      <details>
        <summary>Decoded completion</summary>
        <pre>{completion}</pre>
      </details>
      <div class="token-stream">{tokens_html}</div>
    </section>
    """


def build_html(samples, title, hide_rejected=False):
    body = "\n".join(_render_sample(sample, hide_rejected=hide_rejected) for sample in samples)
    rejected_legend = ""
    if not hide_rejected:
        rejected_legend = (
            '<div class="legend-item"><span class="swatch" '
            'style="background: rgba(248, 81, 73, 0.75);"></span>'
            'Rejected Draft (shown in parentheses)</div>'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background: #0f1117;
      color: #e6edf3;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      display: inline-block;
    }}
    .sample-card {{
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 18px;
      background: #161b22;
    }}
    .meta {{
      color: #9da7b3;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #0d1117;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #30363d;
    }}
    .token-stream {{
      margin-top: 12px;
      padding: 12px;
      line-height: 1.9;
      border-radius: 8px;
      background: #0d1117;
      border: 1px solid #30363d;
      white-space: normal;
      word-break: break-word;
    }}
    .token {{
      display: inline;
      padding: 2px 0;
      border-radius: 3px;
    }}
    .draft_accepted {{
      background: rgba(46, 160, 67, 0.28);
      color: #e6ffec;
    }}
    .verifier_bonus {{
      background: rgba(56, 139, 253, 0.28);
      color: #e8f1ff;
    }}
    .verifier_correction {{
      background: rgba(210, 153, 34, 0.32);
      color: #fff4d6;
    }}
    .unknown {{
      background: rgba(139, 148, 158, 0.28);
      color: #f0f6fc;
    }}
    .rejected_draft {{
      background: rgba(248, 81, 73, 0.28);
      color: #ffd7d5;
    }}
    .rejected-group {{
      color: #ff7b72;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Color-coded completion provenance for speculative decoding outputs.</p>
  <div class="legend">
    <div class="legend-item"><span class="swatch" style="background: rgba(46, 160, 67, 0.75);"></span>Draft Accepted</div>
    <div class="legend-item"><span class="swatch" style="background: rgba(56, 139, 253, 0.75);"></span>Verifier Bonus</div>
    <div class="legend-item"><span class="swatch" style="background: rgba(210, 153, 34, 0.75);"></span>Verifier Correction</div>
    {rejected_legend}
  </div>
  {body}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to specdec_sample_outputs.jsonl")
    parser.add_argument("--output", required=True, help="Path to output HTML")
    parser.add_argument("--title", default="Speculative Decoding Token Provenance")
    parser.add_argument(
        "--hide-rejected",
        action="store_true",
        help="Hide rejected draft tokens from the rendered report.",
    )
    args = parser.parse_args()

    samples = _load_samples(args.input)
    html_text = build_html(samples, args.title, hide_rejected=args.hide_rejected)
    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write(html_text)


if __name__ == "__main__":
    main()
