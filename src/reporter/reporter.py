"""
Reporter
========
Consolidates mutation-testing results and produces report.html inside the
TransmutPysparkOutput workdir provided by MutationManager.
"""

import ast
import difflib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..model.test_result import TestResult

if TYPE_CHECKING:
    from src.operators.operator import Mutant

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Source normalisation helpers                                         #
# ------------------------------------------------------------------ #

def _normalise_source(source: str) -> str:
    try:
        return ast.unparse(ast.parse(source))
    except SyntaxError:
        return source


def _normalised_lines(source: str) -> list[str]:
    normalised = _normalise_source(source)
    return [line + "\n" for line in normalised.splitlines()]


def _compute_diff(original_source: str, mutant_source: str, mutant_id: int) -> str:
    original_lines = _normalised_lines(original_source)
    mutant_lines   = _normalised_lines(mutant_source)
    diff_lines = list(difflib.unified_diff(
        original_lines,
        mutant_lines,
        fromfile="original.py",
        tofile=f"mutant_{mutant_id}.py",
        n=4,
    ))
    return "".join(diff_lines)


# ------------------------------------------------------------------ #
# Reporter                                                             #
# ------------------------------------------------------------------ #

@dataclass
class Reporter:
    result_list:      list[TestResult]
    code_original:    str
    mutant_list:      list
    output_dir:       Path                    # ← workdir recebido do MutationManager
    result_calculate: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_result_list()
        self._validate_code_original()
        self._validate_mutant_list()
        self._validate_output_dir()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def calculate(self) -> "Reporter":
        total    = len(self.result_list)
        killed   = sum(1 for r in self.result_list if r.status == "killed")
        survived = sum(1 for r in self.result_list if r.status == "survived")
        timeout  = sum(1 for r in self.result_list if r.status == "timeout")
        error    = sum(1 for r in self.result_list if r.status == "error")
        score    = round(killed / total, 4) if total > 0 else 0.0

        mutant_index = {m.id: m for m in self.mutant_list}
        by_operator: dict[str, dict] = defaultdict(lambda: {
            "killed": 0, "survived": 0, "timeout": 0, "error": 0, "total": 0
        })
        for result in self.result_list:
            mutant = mutant_index.get(result.mutant)
            if mutant:
                op = mutant.operator
                by_operator[op][result.status] += 1
                by_operator[op]["total"] += 1

        self.result_calculate.update({
            "mutation_score":     score,
            "total":              total,
            "killed":             killed,
            "survived":           survived,
            "timeout":            timeout,
            "error":              error,
            "by_operator":        dict(by_operator),
            "diff_original_code": [],
        })
        logger.info(
            f"[Reporter.calculate] MutationScore={score:.2%} | "
            f"killed={killed} survived={survived} timeout={timeout} error={error} total={total}"
        )
        return self

    def make_diff(self) -> "Reporter":
        self._assert_calculated()
        mutant_index = {m.id: m for m in self.mutant_list}
        diffs: list[dict] = []

        for result in self.result_list:
            mutant = mutant_index.get(result.mutant)
            if mutant is None:
                continue
            try:
                mutant_source = Path(mutant.mutant_path).read_text(encoding="utf-8")
            except (FileNotFoundError, OSError) as exc:
                logger.warning(
                    f"[Reporter.make_diff] Could not read '{mutant.mutant_path}': {exc} — skipping."
                )
                continue

            diff_str = _compute_diff(self.code_original, mutant_source, mutant.id)
            diffs.append({
                "key":           f"{mutant.operator.lower()}_{mutant.id}",
                "operator":      mutant.operator,
                "id":            mutant.id,
                "status":        result.status,
                "diff":          diff_str,
                "mutant_source": mutant_source,
                "modified_line": mutant.modified_line,
                "failed_tests":  result.failed_tests,
                "exec_time":     result.execution_time,
            })

        self.result_calculate["diff_original_code"] = diffs
        logger.info(f"[Reporter.make_diff] {len(diffs)} diff(s) generated.")
        return self

    def show_results(self) -> "Reporter":
        """Salva o relatório HTML dentro do workdir (output_dir)."""
        self._assert_calculated()
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"report_{timestamp}.html"
        output_path.write_text(self._build_html(), encoding="utf-8")
        logger.info(f"[Reporter.show_results] Report saved to: {output_path}")
        return self

    # ------------------------------------------------------------------ #
    # HTML builder (inalterado — só _resolve_output_dir foi removido)     #
    # ------------------------------------------------------------------ #

    def _build_html(self) -> str:
        rc    = self.result_calculate
        score = rc.get("mutation_score", 0.0)
        score_pct = f"{score * 100:.1f}%"

        if score >= 0.80:
            score_color = "#22d3a5"
            score_label = "STRONG"
        elif score >= 0.50:
            score_color = "#f59e0b"
            score_label = "MODERATE"
        else:
            score_color = "#f43f5e"
            score_label = "WEAK"

        by_operator  = rc.get("by_operator", {})
        mutant_index = {m.id: m for m in self.mutant_list}
        result_index = {r.mutant: r for r in self.result_list}
        diffs_by_id  = {d["id"]: d for d in rc.get("diff_original_code", [])}

        mutants_by_op: dict[str, list] = defaultdict(list)
        for mutant in self.mutant_list:
            mutants_by_op[mutant.operator].append(mutant)

        total        = rc.get("total", 0)
        killed_pct   = rc.get("killed",   0) / total * 100 if total else 0
        survived_pct = rc.get("survived", 0) / total * 100 if total else 0
        timeout_pct  = rc.get("timeout",  0) / total * 100 if total else 0
        error_pct    = rc.get("error",    0) / total * 100 if total else 0

        op_accordions = ""
        for op_idx, (op_name, op_stats) in enumerate(by_operator.items()):
            op_total    = op_stats.get("total",    0)
            op_killed   = op_stats.get("killed",   0)
            op_survived = op_stats.get("survived", 0)
            op_timeout  = op_stats.get("timeout",  0)
            op_error    = op_stats.get("error",    0)
            op_ratio    = op_killed / op_total if op_total else 0
            op_score    = f"{op_ratio * 100:.0f}%" if op_total else "—"

            if op_total > 0 and op_ratio == 1.0:
                op_bar_color = "#22d3a5"
            elif op_total > 0 and op_ratio >= 0.5:
                op_bar_color = "#f59e0b"
            else:
                op_bar_color = "#f43f5e"

            mutant_rows = ""
            for mutant in mutants_by_op.get(op_name, []):
                result     = result_index.get(mutant.id)
                status     = result.status if result else "unknown"
                diff_d     = diffs_by_id.get(mutant.id, {})
                diff_str   = diff_d.get("diff", "")
                mut_source = diff_d.get("mutant_source", "")
                exec_time  = f"{result.execution_time:.3f}s" if result else "—"
                failed     = result.failed_tests if result and result.failed_tests else []

                diff_html = (
                    self._colorise_diff(self._escape(diff_str))
                    if diff_str.strip()
                    else "<span class='no-diff'>⚠ No diff detected — mutant may be identical to original</span>"
                )
                mutant_source_html = (
                    self._escape(mut_source)
                    if mut_source.strip()
                    else "<span class='no-diff'>Source unavailable</span>"
                )
                failed_html = "".join(
                    f"<span class='failed-test'>✗ {self._escape(t)}</span>"
                    for t in failed
                ) if failed else "<span class='no-failed'>—</span>"

                modified_display = self._escape(str(mutant.modified_line))

                mutant_rows += f"""
                <div class="mutant-card status-{status}" id="mc-{mutant.id}">
                  <button class="mutant-header" onclick="toggleMutant({mutant.id})">
                    <div class="mh-left">
                      <span class="mutant-glyph">{mutant.id:03d}</span>
                      <span class="status-dot dot-{status}"></span>
                      <span class="mutant-line-preview">{modified_display}</span>
                    </div>
                    <div class="mh-right">
                      <span class="badge badge-{status}">{status.upper()}</span>
                      <span class="exec-time">{exec_time}</span>
                      <span class="mchev" id="mchev-{mutant.id}">▶</span>
                    </div>
                  </button>
                  <div class="mutant-body" id="mbody-{mutant.id}">
                    <div class="mutant-body-inner">
                      <div class="failed-section">
                        <div class="section-label">Failed Tests</div>
                        <div class="failed-list">{failed_html}</div>
                      </div>
                      <div class="code-tabs" id="tabs-{mutant.id}">
                        <div class="tab-bar">
                          <button class="tab active" onclick="switchTab({mutant.id}, 'diff', this)">◈ Diff</button>
                          <button class="tab" onclick="switchTab({mutant.id}, 'full', this)">◉ Mutant source</button>
                        </div>
                        <div class="tab-pane active" id="tab-diff-{mutant.id}">
                          <pre class="diff-block">{diff_html}</pre>
                        </div>
                        <div class="tab-pane" id="tab-full-{mutant.id}">
                          <pre class="code-block">{mutant_source_html}</pre>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>"""

            op_bar_width = f"{op_ratio * 100:.1f}%"
            op_accordions += f"""
            <div class="op-section" id="op-{op_idx}">
              <button class="op-header" onclick="toggleOp({op_idx})">
                <div class="op-header-top">
                  <div class="op-hl">
                    <span class="op-name">{self._escape(op_name)}</span>
                    <span class="op-total">{op_total} mutant{"s" if op_total != 1 else ""}</span>
                  </div>
                  <div class="op-hr">
                    <span class="pill pill-killed">{op_killed}K</span>
                    <span class="pill pill-survived">{op_survived}S</span>
                    <span class="pill pill-timeout">{op_timeout}T</span>
                    <span class="pill pill-error">{op_error}E</span>
                    <span class="op-score-val" style="color:{op_bar_color}">{op_score}</span>
                    <span class="op-toggle-chev" id="opchev-{op_idx}">▶</span>
                  </div>
                </div>
                <div class="op-progress">
                  <div class="op-bar" style="width:{op_bar_width};background:{op_bar_color}"></div>
                </div>
              </button>
              <div class="op-body" id="op-body-{op_idx}">
                <div class="mutants-list">{mutant_rows}</div>
              </div>
            </div>"""

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mutation Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:           #0f1117;
      --bg2:          #161820;
      --bg3:          #1c1f2a;
      --panel:        #1a1d27;
      --panel2:       #202330;
      --border:       #2a2d3e;
      --border2:      #363a52;
      --accent:       #7c6af7;
      --accent2:      #5d52c7;
      --text:         #e8eaf0;
      --text2:        #9da3b8;
      --text3:        #5c6285;
      --killed:       #22d3a5;
      --survived:     #f43f5e;
      --timeout:      #f59e0b;
      --error:        #818cf8;
      --mono:         'JetBrains Mono', monospace;
      --sans:         'Syne', sans-serif;
      --r:            8px;
      --r2:           12px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; min-height: 100vh; }}
    .topbar {{ background: var(--panel); border-bottom: 1px solid var(--border); padding: 0 36px; display: flex; align-items: center; justify-content: space-between; height: 56px; position: sticky; top: 0; z-index: 100; backdrop-filter: blur(12px); }}
    .topbar-brand {{ display: flex; align-items: center; gap: 10px; }}
    .brand-icon {{ width: 28px; height: 28px; background: var(--accent); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-family: var(--mono); font-weight: 700; color: #fff; letter-spacing: -.03em; }}
    .brand-title {{ font-size: 15px; font-weight: 700; letter-spacing: -.02em; color: var(--text); }}
    .brand-sub {{ font-family: var(--mono); font-size: 10px; color: var(--text3); letter-spacing: .06em; text-transform: uppercase; }}
    .topbar-ts {{ font-family: var(--mono); font-size: 10px; color: var(--text3); letter-spacing: .05em; }}
    .hero {{ background: var(--bg2); border-bottom: 1px solid var(--border); padding: 40px 36px 32px; display: grid; grid-template-columns: 1fr auto; gap: 32px; align-items: center; }}
    .hero-label {{ font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; color: var(--accent); margin-bottom: 8px; }}
    .hero-score-wrap {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; }}
    .hero-score {{ font-family: var(--mono); font-size: 72px; font-weight: 700; line-height: 1; color: {score_color}; letter-spacing: -.04em; }}
    .hero-strength {{ font-size: 13px; font-weight: 700; letter-spacing: .12em; color: {score_color}; opacity: .7; text-transform: uppercase; align-self: center; padding: 3px 10px; border: 1px solid {score_color}; border-radius: 4px; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0; border: 1px solid var(--border); border-radius: var(--r2); overflow: hidden; background: var(--panel); }}
    .stat-cell {{ padding: 18px 20px; border-right: 1px solid var(--border); }}
    .stat-cell:last-child {{ border-right: none; }}
    .stat-val {{ font-family: var(--mono); font-size: 28px; font-weight: 700; line-height: 1; margin-bottom: 4px; }}
    .stat-label {{ font-size: 10px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--text3); }}
    .sv-total {{ color: var(--text); }} .sv-killed {{ color: var(--killed); }} .sv-survived {{ color: var(--survived); }} .sv-timeout {{ color: var(--timeout); }} .sv-error {{ color: var(--error); }}
    .master-bar-wrap {{ padding: 16px 36px; background: var(--bg2); border-bottom: 1px solid var(--border); }}
    .master-bar-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: .14em; text-transform: uppercase; color: var(--text3); margin-bottom: 6px; }}
    .master-bar {{ height: 6px; background: var(--bg3); border-radius: 99px; overflow: hidden; display: flex; }}
    .bar-seg {{ height: 100%; transition: width .4s ease; }}
    .content {{ max-width: 1200px; margin: 0 auto; padding: 28px 36px 80px; }}
    .content-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }}
    .section-label-lg {{ font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; color: var(--text3); }}
    .expand-all-btn {{ font-family: var(--mono); font-size: 11px; color: var(--accent); background: none; border: 1px solid var(--border2); border-radius: 6px; padding: 4px 12px; cursor: pointer; transition: all .15s; }}
    .expand-all-btn:hover {{ background: var(--border); }}
    .op-section {{ background: var(--panel); border: 1px solid var(--border); border-radius: var(--r2); margin-bottom: 10px; overflow: hidden; }}
    .op-header {{ width: 100%; background: none; border: none; cursor: pointer; padding: 14px 18px 10px; text-align: left; transition: background .1s; }}
    .op-header:hover {{ background: rgba(255,255,255,.02); }}
    .op-header-top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }}
    .op-hl {{ display: flex; align-items: center; gap: 12px; }}
    .op-name {{ font-family: var(--mono); font-size: 13px; font-weight: 700; color: var(--text); letter-spacing: .02em; }}
    .op-total {{ font-size: 11px; color: var(--text3); font-weight: 500; }}
    .op-hr {{ display: flex; align-items: center; gap: 6px; }}
    .pill {{ font-family: var(--mono); font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }}
    .pill-killed {{ background: rgba(34,211,165,.12); color: var(--killed); }} .pill-survived {{ background: rgba(244,63,94,.12); color: var(--survived); }} .pill-timeout {{ background: rgba(245,158,11,.12); color: var(--timeout); }} .pill-error {{ background: rgba(129,140,248,.12); color: var(--error); }}
    .op-score-val {{ font-family: var(--mono); font-size: 15px; font-weight: 700; min-width: 42px; text-align: right; }}
    .op-toggle-chev {{ font-size: 10px; color: var(--text3); transition: transform .2s; margin-left: 4px; }}
    .op-toggle-chev.open {{ transform: rotate(90deg); }}
    .op-progress {{ height: 3px; background: var(--bg3); border-radius: 99px; overflow: hidden; }}
    .op-bar {{ height: 100%; border-radius: 99px; transition: width .4s; }}
    .op-body {{ display: none; border-top: 1px solid var(--border); }}
    .mutants-list {{ padding: 8px; display: flex; flex-direction: column; gap: 4px; }}
    .mutant-card {{ background: var(--bg3); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; transition: border-color .15s; }}
    .mutant-card:hover {{ border-color: var(--border2); }}
    .mutant-card.status-survived {{ border-left: 3px solid var(--survived); }} .mutant-card.status-killed {{ border-left: 3px solid var(--killed); }} .mutant-card.status-timeout {{ border-left: 3px solid var(--timeout); }} .mutant-card.status-error {{ border-left: 3px solid var(--error); }}
    .mutant-header {{ width: 100%; background: none; border: none; cursor: pointer; padding: 10px 14px; display: flex; align-items: center; justify-content: space-between; gap: 12px; text-align: left; transition: background .1s; }}
    .mutant-header:hover {{ background: rgba(255,255,255,.03); }}
    .mh-left {{ display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }}
    .mh-right {{ display: flex; align-items: center; gap: 8px; flex-shrink: 0; }}
    .mutant-glyph {{ font-family: var(--mono); font-size: 10px; color: var(--text3); font-weight: 600; min-width: 30px; flex-shrink: 0; }}
    .status-dot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}
    .dot-killed {{ background: var(--killed); box-shadow: 0 0 6px var(--killed); }} .dot-survived {{ background: var(--survived); box-shadow: 0 0 6px var(--survived); }} .dot-timeout {{ background: var(--timeout); box-shadow: 0 0 6px var(--timeout); }} .dot-error {{ background: var(--error); box-shadow: 0 0 6px var(--error); }} .dot-unknown {{ background: var(--text3); }}
    .mutant-line-preview {{ font-family: var(--mono); font-size: 11px; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }}
    .exec-time {{ font-family: var(--mono); font-size: 10px; color: var(--text3); }}
    .mchev {{ font-size: 9px; color: var(--text3); transition: transform .18s; }}
    .mchev.open {{ transform: rotate(90deg); }}
    .badge {{ font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: .08em; padding: 2px 7px; border-radius: 4px; }}
    .badge-killed {{ background: rgba(34,211,165,.15); color: var(--killed); }} .badge-survived {{ background: rgba(244,63,94,.15); color: var(--survived); }} .badge-timeout {{ background: rgba(245,158,11,.15); color: var(--timeout); }} .badge-error {{ background: rgba(129,140,248,.15); color: var(--error); }} .badge-unknown {{ background: rgba(156,163,175,.1); color: var(--text3); }}
    .mutant-body {{ display: none; border-top: 1px solid var(--border); }}
    .mutant-body-inner {{ padding: 14px; }}
    .failed-section {{ margin-bottom: 14px; }}
    .section-label {{ font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--text3); margin-bottom: 6px; }}
    .failed-list {{ display: flex; flex-wrap: wrap; gap: 4px; }}
    .failed-test {{ font-family: var(--mono); font-size: 11px; background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.2); color: var(--survived); padding: 2px 8px; border-radius: 4px; }}
    .no-failed {{ font-family: var(--mono); font-size: 11px; color: var(--text3); }}
    .code-tabs {{ border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }}
    .tab-bar {{ display: flex; background: var(--bg2); border-bottom: 1px solid var(--border); }}
    .tab {{ font-family: var(--mono); font-size: 11px; font-weight: 500; background: none; border: none; color: var(--text3); padding: 8px 16px; cursor: pointer; border-bottom: 2px solid transparent; transition: all .15s; letter-spacing: .02em; }}
    .tab:hover {{ color: var(--text2); }}
    .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); background: rgba(124,106,247,.06); }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}
    pre.diff-block, pre.code-block {{ font-family: var(--mono); font-size: 12px; line-height: 1.75; background: var(--bg); padding: 14px 16px; overflow-x: auto; white-space: pre; color: var(--text2); max-height: 480px; overflow-y: auto; }}
    .diff-add {{ display: block; color: var(--killed); background: rgba(34,211,165,.07); }}
    .diff-del {{ display: block; color: var(--survived); background: rgba(244,63,94,.07); }}
    .diff-hdr {{ display: block; color: var(--accent); opacity: .6; }}
    .no-diff {{ font-family: var(--mono); font-size: 11px; color: var(--timeout); padding: 12px 0; display: block; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-brand">
      <div class="brand-icon">μ</div>
      <div>
        <div class="brand-title">MutationReport</div>
        <div class="brand-sub">PySpark · DataFrame API</div>
      </div>
    </div>
    <div class="topbar-ts">{timestamp_str}</div>
  </div>
  <div class="hero">
    <div>
      <div class="hero-label">Mutation Score</div>
      <div class="hero-score-wrap">
        <div class="hero-score">{score_pct}</div>
        <div class="hero-strength">{score_label}</div>
      </div>
    </div>
    <div class="stat-grid">
      <div class="stat-cell"><div class="stat-val sv-total">{rc.get('total', 0)}</div><div class="stat-label">Total</div></div>
      <div class="stat-cell"><div class="stat-val sv-killed">{rc.get('killed', 0)}</div><div class="stat-label">Killed</div></div>
      <div class="stat-cell"><div class="stat-val sv-survived">{rc.get('survived', 0)}</div><div class="stat-label">Survived</div></div>
      <div class="stat-cell"><div class="stat-val sv-timeout">{rc.get('timeout', 0)}</div><div class="stat-label">Timeout</div></div>
      <div class="stat-cell"><div class="stat-val sv-error">{rc.get('error', 0)}</div><div class="stat-label">Error</div></div>
    </div>
  </div>
  <div class="master-bar-wrap">
    <div class="master-bar-label">Breakdown</div>
    <div class="master-bar">
      <div class="bar-seg" style="width:{killed_pct:.1f}%;background:var(--killed)"></div>
      <div class="bar-seg" style="width:{survived_pct:.1f}%;background:var(--survived)"></div>
      <div class="bar-seg" style="width:{timeout_pct:.1f}%;background:var(--timeout)"></div>
      <div class="bar-seg" style="width:{error_pct:.1f}%;background:var(--error)"></div>
    </div>
  </div>
  <div class="content">
    <div class="content-header">
      <div class="section-label-lg">Results by Operator</div>
      <button class="expand-all-btn" onclick="expandAll()">Expand all</button>
    </div>
    {op_accordions if op_accordions else "<p style='color:var(--text3);font-family:var(--mono);font-size:12px'>No results available.</p>"}
  </div>
  <script>
    function toggleOp(idx) {{
      const body = document.getElementById('op-body-' + idx);
      const chev = document.getElementById('opchev-' + idx);
      const open = body.style.display === 'block';
      body.style.display = open ? 'none' : 'block';
      chev.classList.toggle('open', !open);
    }}
    function toggleMutant(id) {{
      const body = document.getElementById('mbody-' + id);
      const chev = document.getElementById('mchev-' + id);
      const open = body.style.display === 'block';
      body.style.display = open ? 'none' : 'block';
      chev.classList.toggle('open', !open);
    }}
    function switchTab(id, tab, btn) {{
      ['tab-diff-' + id, 'tab-full-' + id].forEach(pid => {{
        const el = document.getElementById(pid);
        if (el) el.classList.remove('active');
      }});
      const pane = document.getElementById('tab-' + tab + '-' + id);
      if (pane) pane.classList.add('active');
      btn.closest('.tab-bar').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
    }}
    function expandAll() {{
      const btn = document.querySelector('.expand-all-btn');
      const allBodies = document.querySelectorAll('.op-body');
      const allClosed = Array.from(allBodies).some(b => b.style.display !== 'block');
      allBodies.forEach((b, i) => {{
        b.style.display = allClosed ? 'block' : 'none';
        const chev = document.getElementById('opchev-' + i);
        if (chev) chev.classList.toggle('open', allClosed);
      }});
      btn.textContent = allClosed ? 'Collapse all' : 'Expand all';
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      const first = document.getElementById('op-body-0');
      if (first) first.style.display = 'block';
      const firstChev = document.getElementById('opchev-0');
      if (firstChev) firstChev.classList.add('open');
    }});
  </script>
</body>
</html>"""

    # ------------------------------------------------------------------ #
    # HTML helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _colorise_diff(escaped_diff: str) -> str:
        lines  = escaped_diff.splitlines()
        result = []
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                result.append(f"<span class='diff-add'>{line}</span>")
            elif line.startswith("-") and not line.startswith("---"):
                result.append(f"<span class='diff-del'>{line}</span>")
            elif line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                result.append(f"<span class='diff-hdr'>{line}</span>")
            else:
                result.append(line)
        return "\n".join(result)

    # ------------------------------------------------------------------ #
    # Guards & validators                                                  #
    # ------------------------------------------------------------------ #

    def _assert_calculated(self) -> None:
        if "mutation_score" not in self.result_calculate:
            raise RuntimeError(
                "[Reporter] Call calculate() before make_diff() or show_results()."
            )

    def _validate_result_list(self) -> None:
        if not isinstance(self.result_list, list):
            raise TypeError(f"[Reporter] result_list must be a list, got: {type(self.result_list)}")
        invalid = [r for r in self.result_list if not isinstance(r, TestResult)]
        if invalid:
            raise TypeError("[Reporter] All items in result_list must be TestResult instances.")

    def _validate_code_original(self) -> None:
        if not isinstance(self.code_original, str) or not self.code_original.strip():
            raise ValueError("[Reporter] code_original must be a non-empty string.")

    def _validate_mutant_list(self) -> None:
        if not isinstance(self.mutant_list, list):
            raise TypeError(f"[Reporter] mutant_list must be a list, got: {type(self.mutant_list)}")

    def _validate_output_dir(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise TypeError(
                f"[Reporter] output_dir must be a Path instance, got: {type(self.output_dir)}"
            )

    def __repr__(self) -> str:
        score = self.result_calculate.get("mutation_score", "not calculated")
        return (
            f"Reporter(results={len(self.result_list)}, "
            f"mutants={len(self.mutant_list)}, "
            f"output_dir={str(self.output_dir)!r}, "
            f"score={score})"
        )