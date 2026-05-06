"""
Reporter
========
Consolidates mutation-testing results and produces report.html inside the
TransmutPysparkOutput workdir provided by MutationManager.

Estrutura do relatório:
  arquivo fonte (ex: atr.py)
    └── operador (ex: ATR)
          └── mutante 001 — diff + source
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
# Helpers de diff                                                      #
# ------------------------------------------------------------------ #

def _normalise_source(source: str) -> str:
    try:
        return ast.unparse(ast.parse(source))
    except SyntaxError:
        return source


def _normalised_lines(source: str) -> list[str]:
    return [line + "\n" for line in _normalise_source(source).splitlines()]


def _compute_diff(original_source: str, mutant_source: str, mutant_id: int) -> str:
    diff_lines = list(difflib.unified_diff(
        _normalised_lines(original_source),
        _normalised_lines(mutant_source),
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
    output_dir:       Path
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

            # Lê o arquivo original correto de cada mutante para o diff
            try:
                original_source = Path(mutant.original_path).read_text(encoding="utf-8")
            except (FileNotFoundError, OSError):
                original_source = self.code_original  # fallback

            diff_str = _compute_diff(original_source, mutant_source, mutant.id)
            diffs.append({
                "id":            mutant.id,
                "operator":      mutant.operator,
                "source_file":   Path(mutant.original_path).name,
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
        self._assert_calculated()
        output_path = self.output_dir / "report.html"
        html_content = self._build_html()
        output_path.write_text(html_content, encoding="utf-8")
        logger.info(f"[Reporter.show_results] Report saved to: {output_path}")
        return self

    # ------------------------------------------------------------------ #
    # HTML builder — estrutura: arquivo fonte → operador → mutantes       #
    # ------------------------------------------------------------------ #

    def _build_html(self) -> str:
        rc    = self.result_calculate
        score = rc.get("mutation_score", 0.0)
        score_pct = f"{score * 100:.1f}%"

        if score >= 0.80:
            score_color, score_label = "#22d3a5", "STRONG"
        elif score >= 0.50:
            score_color, score_label = "#f59e0b", "MODERATE"
        else:
            score_color, score_label = "#f43f5e", "WEAK"

        result_index = {r.mutant: r for r in self.result_list}
        diffs_by_id  = {d["id"]: d for d in rc.get("diff_original_code", [])}

        total        = rc.get("total", 0)
        killed_pct   = rc.get("killed",   0) / total * 100 if total else 0
        survived_pct = rc.get("survived", 0) / total * 100 if total else 0
        timeout_pct  = rc.get("timeout",  0) / total * 100 if total else 0
        error_pct    = rc.get("error",    0) / total * 100 if total else 0

        # Agrupa: source_file → operator → [mutants]
        by_file: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for mutant in self.mutant_list:
            by_file[Path(mutant.original_path).name][mutant.operator].append(mutant)

        file_sections = ""
        for fi, (source_name, ops) in enumerate(sorted(by_file.items())):
            all_mutants = [m for ms in ops.values() for m in ms]
            f_total     = len(all_mutants)
            f_killed    = sum(1 for m in all_mutants if result_index.get(m.id) and result_index[m.id].status == "killed")
            f_survived  = sum(1 for m in all_mutants if result_index.get(m.id) and result_index[m.id].status == "survived")
            f_timeout   = sum(1 for m in all_mutants if result_index.get(m.id) and result_index[m.id].status == "timeout")
            f_error     = sum(1 for m in all_mutants if result_index.get(m.id) and result_index[m.id].status == "error")
            f_ratio     = f_killed / f_total if f_total else 0
            f_score     = f"{f_ratio * 100:.0f}%"
            f_color     = "#22d3a5" if f_ratio == 1.0 else "#f59e0b" if f_ratio >= 0.5 else "#f43f5e"

            op_sections = ""
            for oi, (op_name, op_mutants) in enumerate(sorted(ops.items())):
                op_total   = len(op_mutants)
                op_killed  = sum(1 for m in op_mutants if result_index.get(m.id) and result_index[m.id].status == "killed")
                op_surv    = sum(1 for m in op_mutants if result_index.get(m.id) and result_index[m.id].status == "survived")
                op_timeout = sum(1 for m in op_mutants if result_index.get(m.id) and result_index[m.id].status == "timeout")
                op_error   = sum(1 for m in op_mutants if result_index.get(m.id) and result_index[m.id].status == "error")
                op_ratio   = op_killed / op_total if op_total else 0
                op_score   = f"{op_ratio * 100:.0f}%"
                op_color   = "#22d3a5" if op_ratio == 1.0 else "#f59e0b" if op_ratio >= 0.5 else "#f43f5e"
                uid        = f"{fi}_{oi}"

                cards = ""
                for mutant in op_mutants:
                    result    = result_index.get(mutant.id)
                    status    = result.status if result else "unknown"
                    d         = diffs_by_id.get(mutant.id, {})
                    diff_str  = d.get("diff", "")
                    mut_src   = d.get("mutant_source", "")
                    exec_time = f"{result.execution_time:.3f}s" if result else "—"
                    failed    = result.failed_tests if result and result.failed_tests else []

                    diff_html = (
                        self._colorise_diff(self._escape(diff_str))
                        if diff_str.strip()
                        else "<span class='no-diff'>⚠ No diff detected</span>"
                    )
                    src_html = (
                        self._escape(mut_src)
                        if mut_src.strip()
                        else "<span class='no-diff'>Source unavailable</span>"
                    )
                    failed_html = "".join(
                        f"<span class='failed-test'>✗ {self._escape(t)}</span>" for t in failed
                    ) if failed else "<span class='no-failed'>—</span>"

                    cards += f"""
                    <div class="mutant-card status-{status}" id="mc-{mutant.id}">
                      <button class="mutant-header" onclick="toggleMutant({mutant.id})">
                        <div class="mh-left">
                          <span class="mutant-glyph">{mutant.id:03d}</span>
                          <span class="status-dot dot-{status}"></span>
                          <span class="mutant-line-preview">{self._escape(str(mutant.modified_line))}</span>
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
                          <div class="code-tabs">
                            <div class="tab-bar">
                              <button class="tab active" onclick="switchTab({mutant.id},'diff',this)">◈ Diff</button>
                              <button class="tab" onclick="switchTab({mutant.id},'full',this)">◉ Mutant source</button>
                            </div>
                            <div class="tab-pane active" id="tab-diff-{mutant.id}">
                              <pre class="diff-block">{diff_html}</pre>
                            </div>
                            <div class="tab-pane" id="tab-full-{mutant.id}">
                              <pre class="code-block">{src_html}</pre>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>"""

                op_sections += f"""
                <div class="op-section" id="op-{uid}">
                  <button class="op-header" onclick="toggleOp('{uid}')">
                    <div class="op-header-top">
                      <div class="op-hl">
                        <span class="op-name">{self._escape(op_name)}</span>
                        <span class="op-total">{op_total} mutant{"s" if op_total != 1 else ""}</span>
                      </div>
                      <div class="op-hr">
                        <span class="pill pill-killed">{op_killed}K</span>
                        <span class="pill pill-survived">{op_surv}S</span>
                        <span class="pill pill-timeout">{op_timeout}T</span>
                        <span class="pill pill-error">{op_error}E</span>
                        <span class="op-score-val" style="color:{op_color}">{op_score}</span>
                        <span class="op-toggle-chev" id="opchev-{uid}">▶</span>
                      </div>
                    </div>
                    <div class="op-progress">
                      <div class="op-bar" style="width:{op_ratio*100:.1f}%;background:{op_color}"></div>
                    </div>
                  </button>
                  <div class="op-body" id="op-body-{uid}">
                    <div class="mutants-list">{cards}</div>
                  </div>
                </div>"""

            file_sections += f"""
            <div class="file-section" id="file-{fi}">
              <button class="file-header" onclick="toggleFile({fi})">
                <div class="op-header-top">
                  <div class="op-hl">
                    <span class="file-icon">📄</span>
                    <span class="file-name">{self._escape(source_name)}</span>
                    <span class="op-total">{f_total} mutant{"s" if f_total != 1 else ""} · {len(ops)} operator{"s" if len(ops) != 1 else ""}</span>
                  </div>
                  <div class="op-hr">
                    <span class="pill pill-killed">{f_killed}K</span>
                    <span class="pill pill-survived">{f_survived}S</span>
                    <span class="pill pill-timeout">{f_timeout}T</span>
                    <span class="pill pill-error">{f_error}E</span>
                    <span class="op-score-val" style="color:{f_color}">{f_score}</span>
                    <span class="op-toggle-chev" id="filechev-{fi}">▶</span>
                  </div>
                </div>
                <div class="op-progress">
                  <div class="op-bar" style="width:{f_ratio*100:.1f}%;background:{f_color}"></div>
                </div>
              </button>
              <div class="file-body" id="file-body-{fi}">
                <div class="file-ops-inner">{op_sections}</div>
              </div>
            </div>"""

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
      --bg:#0f1117;--bg2:#161820;--bg3:#1c1f2a;--panel:#1a1d27;
      --border:#2a2d3e;--border2:#363a52;--accent:#7c6af7;
      --text:#e8eaf0;--text2:#9da3b8;--text3:#5c6285;
      --killed:#22d3a5;--survived:#f43f5e;--timeout:#f59e0b;--error:#818cf8;
      --mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;--r:8px;--r2:12px;
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6;min-height:100vh}}
    .topbar{{background:var(--panel);border-bottom:1px solid var(--border);padding:0 36px;display:flex;align-items:center;justify-content:space-between;height:56px;position:sticky;top:0;z-index:100}}
    .topbar-brand{{display:flex;align-items:center;gap:10px}}
    .brand-icon{{width:28px;height:28px;background:var(--accent);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;font-family:var(--mono);font-weight:700;color:#fff}}
    .brand-title{{font-size:15px;font-weight:700;letter-spacing:-.02em}}
    .brand-sub{{font-family:var(--mono);font-size:10px;color:var(--text3);letter-spacing:.06em;text-transform:uppercase}}
    .topbar-ts{{font-family:var(--mono);font-size:10px;color:var(--text3)}}
    .hero{{background:var(--bg2);border-bottom:1px solid var(--border);padding:40px 36px 32px;display:grid;grid-template-columns:1fr auto;gap:32px;align-items:center}}
    .hero-label{{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}}
    .hero-score-wrap{{display:flex;align-items:baseline;gap:12px;margin-bottom:4px}}
    .hero-score{{font-family:var(--mono);font-size:72px;font-weight:700;line-height:1;color:{score_color};letter-spacing:-.04em}}
    .hero-strength{{font-size:13px;font-weight:700;letter-spacing:.12em;color:{score_color};opacity:.7;text-transform:uppercase;align-self:center;padding:3px 10px;border:1px solid {score_color};border-radius:4px}}
    .stat-grid{{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--border);border-radius:var(--r2);overflow:hidden;background:var(--panel)}}
    .stat-cell{{padding:18px 20px;border-right:1px solid var(--border)}}
    .stat-cell:last-child{{border-right:none}}
    .stat-val{{font-family:var(--mono);font-size:28px;font-weight:700;line-height:1;margin-bottom:4px}}
    .stat-label{{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--text3)}}
    .sv-total{{color:var(--text)}}.sv-killed{{color:var(--killed)}}.sv-survived{{color:var(--survived)}}.sv-timeout{{color:var(--timeout)}}.sv-error{{color:var(--error)}}
    .master-bar-wrap{{padding:16px 36px;background:var(--bg2);border-bottom:1px solid var(--border)}}
    .master-bar-label{{font-family:var(--mono);font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--text3);margin-bottom:6px}}
    .master-bar{{height:6px;background:var(--bg3);border-radius:99px;overflow:hidden;display:flex}}
    .bar-seg{{height:100%}}
    .content{{max-width:1200px;margin:0 auto;padding:28px 36px 80px}}
    .content-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}}
    .section-label-lg{{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:var(--text3)}}
    .expand-all-btn{{font-family:var(--mono);font-size:11px;color:var(--accent);background:none;border:1px solid var(--border2);border-radius:6px;padding:4px 12px;cursor:pointer}}
    .expand-all-btn:hover{{background:var(--border)}}
    .file-section{{background:var(--panel);border:1px solid var(--border);border-radius:var(--r2);margin-bottom:12px;overflow:hidden}}
    .file-header{{width:100%;background:none;border:none;cursor:pointer;padding:16px 18px 12px;text-align:left;transition:background .1s}}
    .file-header:hover{{background:rgba(255,255,255,.02)}}
    .file-icon{{font-size:14px;margin-right:4px}}
    .file-name{{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--text)}}
    .file-body{{display:none;border-top:1px solid var(--border)}}
    .file-ops-inner{{padding:10px;display:flex;flex-direction:column;gap:8px}}
    .op-section{{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}}
    .op-header{{width:100%;background:none;border:none;cursor:pointer;padding:12px 16px 8px;text-align:left;transition:background .1s}}
    .op-header:hover{{background:rgba(255,255,255,.02)}}
    .op-header-top{{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}}
    .op-hl{{display:flex;align-items:center;gap:12px}}
    .op-name{{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--accent);letter-spacing:.06em;text-transform:uppercase}}
    .op-total{{font-size:11px;color:var(--text3)}}
    .op-hr{{display:flex;align-items:center;gap:6px}}
    .pill{{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px}}
    .pill-killed{{background:rgba(34,211,165,.12);color:var(--killed)}}
    .pill-survived{{background:rgba(244,63,94,.12);color:var(--survived)}}
    .pill-timeout{{background:rgba(245,158,11,.12);color:var(--timeout)}}
    .pill-error{{background:rgba(129,140,248,.12);color:var(--error)}}
    .op-score-val{{font-family:var(--mono);font-size:14px;font-weight:700;min-width:38px;text-align:right}}
    .op-toggle-chev{{font-size:10px;color:var(--text3);transition:transform .2s;margin-left:4px}}
    .op-toggle-chev.open{{transform:rotate(90deg)}}
    .op-progress{{height:3px;background:var(--bg);border-radius:99px;overflow:hidden}}
    .op-bar{{height:100%;border-radius:99px}}
    .op-body{{display:none;border-top:1px solid var(--border)}}
    .mutants-list{{padding:8px;display:flex;flex-direction:column;gap:4px}}
    .mutant-card{{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}}
    .mutant-card:hover{{border-color:var(--border2)}}
    .mutant-card.status-survived{{border-left:3px solid var(--survived)}}
    .mutant-card.status-killed{{border-left:3px solid var(--killed)}}
    .mutant-card.status-timeout{{border-left:3px solid var(--timeout)}}
    .mutant-card.status-error{{border-left:3px solid var(--error)}}
    .mutant-header{{width:100%;background:none;border:none;cursor:pointer;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;gap:12px;text-align:left}}
    .mutant-header:hover{{background:rgba(255,255,255,.03)}}
    .mh-left{{display:flex;align-items:center;gap:10px;flex:1;min-width:0}}
    .mh-right{{display:flex;align-items:center;gap:8px;flex-shrink:0}}
    .mutant-glyph{{font-family:var(--mono);font-size:10px;color:var(--text3);font-weight:600;min-width:30px}}
    .status-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
    .dot-killed{{background:var(--killed);box-shadow:0 0 6px var(--killed)}}
    .dot-survived{{background:var(--survived);box-shadow:0 0 6px var(--survived)}}
    .dot-timeout{{background:var(--timeout);box-shadow:0 0 6px var(--timeout)}}
    .dot-error{{background:var(--error);box-shadow:0 0 6px var(--error)}}
    .dot-unknown{{background:var(--text3)}}
    .mutant-line-preview{{font-family:var(--mono);font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}}
    .exec-time{{font-family:var(--mono);font-size:10px;color:var(--text3)}}
    .mchev{{font-size:9px;color:var(--text3);transition:transform .18s}}
    .mchev.open{{transform:rotate(90deg)}}
    .badge{{font-family:var(--mono);font-size:9px;font-weight:700;letter-spacing:.08em;padding:2px 7px;border-radius:4px}}
    .badge-killed{{background:rgba(34,211,165,.15);color:var(--killed)}}
    .badge-survived{{background:rgba(244,63,94,.15);color:var(--survived)}}
    .badge-timeout{{background:rgba(245,158,11,.15);color:var(--timeout)}}
    .badge-error{{background:rgba(129,140,248,.15);color:var(--error)}}
    .badge-unknown{{background:rgba(156,163,175,.1);color:var(--text3)}}
    .mutant-body{{display:none;border-top:1px solid var(--border)}}
    .mutant-body-inner{{padding:14px}}
    .failed-section{{margin-bottom:14px}}
    .section-label{{font-family:var(--mono);font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text3);margin-bottom:6px}}
    .failed-list{{display:flex;flex-wrap:wrap;gap:4px}}
    .failed-test{{font-family:var(--mono);font-size:11px;background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.2);color:var(--survived);padding:2px 8px;border-radius:4px}}
    .no-failed{{font-family:var(--mono);font-size:11px;color:var(--text3)}}
    .code-tabs{{border:1px solid var(--border);border-radius:var(--r);overflow:hidden}}
    .tab-bar{{display:flex;background:var(--bg2);border-bottom:1px solid var(--border)}}
    .tab{{font-family:var(--mono);font-size:11px;font-weight:500;background:none;border:none;color:var(--text3);padding:8px 16px;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}}
    .tab:hover{{color:var(--text2)}}
    .tab.active{{color:var(--accent);border-bottom-color:var(--accent);background:rgba(124,106,247,.06)}}
    .tab-pane{{display:none}}
    .tab-pane.active{{display:block}}
    pre.diff-block,pre.code-block{{font-family:var(--mono);font-size:12px;line-height:1.75;background:var(--bg);padding:14px 16px;overflow-x:auto;white-space:pre;color:var(--text2);max-height:480px;overflow-y:auto}}
    .diff-add{{display:block;color:var(--killed);background:rgba(34,211,165,.07)}}
    .diff-del{{display:block;color:var(--survived);background:rgba(244,63,94,.07)}}
    .diff-hdr{{display:block;color:var(--accent);opacity:.6}}
    .no-diff{{font-family:var(--mono);font-size:11px;color:var(--timeout);padding:12px 0;display:block}}
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
    <div class="topbar-ts">{ts}</div>
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
      <div class="stat-cell"><div class="stat-val sv-total">{rc.get('total',0)}</div><div class="stat-label">Total</div></div>
      <div class="stat-cell"><div class="stat-val sv-killed">{rc.get('killed',0)}</div><div class="stat-label">Killed</div></div>
      <div class="stat-cell"><div class="stat-val sv-survived">{rc.get('survived',0)}</div><div class="stat-label">Survived</div></div>
      <div class="stat-cell"><div class="stat-val sv-timeout">{rc.get('timeout',0)}</div><div class="stat-label">Timeout</div></div>
      <div class="stat-cell"><div class="stat-val sv-error">{rc.get('error',0)}</div><div class="stat-label">Error</div></div>
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
      <div class="section-label-lg">Results by Source File</div>
      <button class="expand-all-btn" onclick="expandAll()">Expand all</button>
    </div>
    {file_sections if file_sections else "<p style='color:var(--text3);font-family:var(--mono);font-size:12px'>No results available.</p>"}
  </div>
  <script>
    function toggleFile(fi) {{
      const b = document.getElementById('file-body-' + fi);
      const c = document.getElementById('filechev-' + fi);
      const open = b.style.display === 'block';
      b.style.display = open ? 'none' : 'block';
      c.classList.toggle('open', !open);
    }}
    function toggleOp(uid) {{
      const b = document.getElementById('op-body-' + uid);
      const c = document.getElementById('opchev-' + uid);
      const open = b.style.display === 'block';
      b.style.display = open ? 'none' : 'block';
      c.classList.toggle('open', !open);
    }}
    function toggleMutant(id) {{
      const b = document.getElementById('mbody-' + id);
      const c = document.getElementById('mchev-' + id);
      const open = b.style.display === 'block';
      b.style.display = open ? 'none' : 'block';
      c.classList.toggle('open', !open);
    }}
    function switchTab(id, tab, btn) {{
      ['tab-diff-'+id,'tab-full-'+id].forEach(p => {{
        const el = document.getElementById(p); if (el) el.classList.remove('active');
      }});
      const pane = document.getElementById('tab-'+tab+'-'+id);
      if (pane) pane.classList.add('active');
      btn.closest('.tab-bar').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
    }}
    function expandAll() {{
      const btn = document.querySelector('.expand-all-btn');
      const all = document.querySelectorAll('.file-body');
      const allClosed = Array.from(all).some(b => b.style.display !== 'block');
      all.forEach((b, i) => {{
        b.style.display = allClosed ? 'block' : 'none';
        const c = document.getElementById('filechev-' + i);
        if (c) c.classList.toggle('open', allClosed);
      }});
      btn.textContent = allClosed ? 'Collapse all' : 'Expand all';
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      const first = document.getElementById('file-body-0');
      if (first) first.style.display = 'block';
      const fc = document.getElementById('filechev-0');
      if (fc) fc.classList.add('open');
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
        lines, result = escaped_diff.splitlines(), []
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
