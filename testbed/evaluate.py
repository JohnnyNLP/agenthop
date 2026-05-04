"""Evaluation aggregation for AgentHop benchmark runs.

Builds a comprehensive model-level report from per-sample trajectories:
  - Accuracy (overall, by cut, position bias, failure attribution)
  - Retrieval (paper recall, section recall hard/soft, conversion rate)
  - Tool usage (distribution, budget split, rejection rate)
  - Cost (token breakdown, USD reconstruction, cache hit rate, wall-time)
  - Errors (API, transient retries, tool-call)

All metrics are aggregate (model level). Per-sample analysis stays in the
individual {sample_id}.json files.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .pricing import lookup_price, compute_cost, ModelPrice
except ImportError:
    from pricing import lookup_price, compute_cost, ModelPrice  # type: ignore


SCHEMA_VERSION = "agenthop-eval-v1"


# ── Section-matching helpers (shared with tools.py semantics) ────────────────

def _normalize_header(h: str) -> str:
    h = h.lower().strip()
    h = re.sub(r"^apdx_[a-z][\d.]*:\s*", "", h)
    h = re.sub(r"^apdx:\s*", "", h)
    h = re.sub(r"^[ivxlc]+-?[a-z]?\s+", "", h)
    h = re.sub(r"^[\d.]+\s*", "", h)
    h = re.sub(r"[:\-–—]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _section_match(query: str, target: str) -> bool:
    """Same fuzzy match used in the read_section tool, header-to-header."""
    if not query or not target:
        return False
    ql, tl = query.lower(), target.lower()
    if ql in tl or tl in ql:
        return True
    qn, tn = _normalize_header(query), _normalize_header(target)
    if qn and tn and (qn in tn or tn in qn):
        return True
    if qn and tn:
        qw, tw = set(qn.split()), set(tn.split())
        if len(qw) >= 2 and tw and len(qw & tw) / max(len(qw), 1) >= 0.6:
            return True
    return False


# ── Small utilities ─────────────────────────────────────────────────────────

def _pct(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "n": 0}
    xs = sorted(xs)
    n = len(xs)
    return {
        "mean": round(statistics.mean(xs), 2),
        "p50": round(xs[n // 2], 2),
        "p90": round(xs[min(int(0.90 * n), n - 1)], 2),
        "p95": round(xs[min(int(0.95 * n), n - 1)], 2),
        "n": n,
    }


def _bucket_accuracy(trajs: list[dict], key_fn) -> dict:
    out: dict[str, dict] = {}
    groups: dict[Any, list[dict]] = defaultdict(list)
    for t in trajs:
        groups[key_fn(t)].append(t)
    for k, ts in sorted(groups.items(), key=lambda kv: str(kv[0])):
        c = sum(1 for t in ts if t["is_correct"])
        out[str(k)] = {
            "n": len(ts),
            "correct": c,
            "accuracy": _pct(c, len(ts)),
        }
    return out


# ── Trajectory iteration helpers ─────────────────────────────────────────────

def _iter_tool_calls(traj: dict):
    """Yield (turn_idx, tool_call_dict) for every tool call in the trajectory."""
    for t in traj.get("turns", []):
        for tc in t.get("tool_calls", []):
            yield t.get("turn", 0), tc


def _iter_tool_results(traj: dict):
    """Yield (turn_idx, tool_result_dict) for every tool result."""
    for t in traj.get("turns", []):
        for r in t.get("tool_results", []):
            yield t.get("turn", 0), r


def _option_letter(idx: int) -> str:
    return chr(ord("A") + idx) if 0 <= idx <= 3 else "?"


# ── Retrieval-metric helpers ─────────────────────────────────────────────────

def _paper_id_to_arxiv(sample: dict) -> dict[str, str]:
    """Map paper_id → arxiv_id from the sample graph."""
    return {pid: n.get("arxivId", "") for pid, n in sample.get("graph", {}).get("nodes", {}).items()}


def _gold_arxiv_set(sample: dict) -> set[str]:
    """Terminal / target gold paper arxiv IDs."""
    return set(sample.get("gold_arxiv_ids", []) or [])


def _recall_labels_by_arxiv(recall_entry: dict) -> dict[str, list[dict]]:
    """Map arxiv_id → list of direct|supporting labels."""
    out: dict[str, list[dict]] = defaultdict(list)
    for lbl in (recall_entry or {}).get("analysis", {}).get("section_recall_labels", []):
        aid = lbl.get("arxiv_id")
        if aid:
            out[aid].append(lbl)
    return out


def _trajectory_read_sections(traj: dict, pid_to_arxiv: dict[str, str]) -> list[tuple[str, str]]:
    """Return (arxiv_id, resolved_section) for every read_section call.

    The agent often invokes read_section with the alphabet alias from
    list_sections (e.g. "(D)" or "D") rather than the descriptive header.
    Matching such aliases against label sections like "IV Method" or
    "5.3.1. Rating Prediction" is brittle---fails on parenthesised aliases
    and false-matches on bare letters. We therefore parse the resolved
    section name from the tool result's `=== Section Name ===` header
    when available, falling back to the raw args section otherwise.
    """
    out = []
    for _, tr in _iter_tool_results(traj):
        if tr.get("name") != "read_section":
            continue
        args = tr.get("args", {})
        pid = args.get("paper_id", "")
        aid = pid_to_arxiv.get(pid, "")
        if not aid:
            continue
        output = tr.get("output", "") or ""
        # Skip rejections (e.g. "REJECTED: Insufficient budget ..."); these reads did not execute.
        if output.startswith("REJECTED") or output.startswith("[ERROR"):
            continue
        sec = ""
        m = re.match(r"^===\s*(.+?)\s*===", output)
        if m:
            sec = m.group(1).strip()
        else:
            sec = args.get("section", "")
        out.append((aid, sec))
    return out


# ── Main entry point ─────────────────────────────────────────────────────────

def build_evaluation(
    trajectories: list[dict],
    samples_by_id: dict[str, dict],
    recall_by_id: dict[str, dict],
    config_dict: dict,
    benchmark_sha: str = "",
    started_at: str = "",
    completed_at: str = "",
) -> dict:
    """Build the comprehensive evaluation report.

    Args:
        trajectories: list of trajectory dicts (from per-sample JSONs).
        samples_by_id: id → sample dict (with distractor_types, gold_arxiv_ids, etc.).
        recall_by_id:  id → recall_labels JSONL entry (analysis.section_recall_labels).
        config_dict:   asdict(HarnessConfig).
        benchmark_sha: sha256 of qa/full.jsonl for provenance.
        started_at, completed_at: ISO 8601 timestamps.

    Returns:
        dict, safe to json.dump.
    """
    n = len(trajectories)
    if n == 0:
        return {"schema_version": SCHEMA_VERSION, "empty": True}

    # ── Accuracy ─────────────────────────────────────────────────────────────
    n_correct = sum(1 for t in trajectories if t.get("is_correct"))
    termination_counts = Counter(t.get("terminated_by", "") for t in trajectories)
    conditional = {}
    for reason in termination_counts:
        bucket = [t for t in trajectories if t.get("terminated_by") == reason]
        if bucket:
            conditional[reason] = {
                "n": len(bucket),
                "correct": sum(1 for t in bucket if t["is_correct"]),
                "accuracy": _pct(sum(1 for t in bucket if t["is_correct"]), len(bucket)),
            }

    # Position bias
    predicted_idx = [t.get("predicted_index", -1) for t in trajectories]
    correct_idx = [t.get("correct_index", -1) for t in trajectories]
    predicted_dist = Counter(_option_letter(i) for i in predicted_idx if 0 <= i <= 3)
    correct_pos_dist = Counter(_option_letter(i) for i in correct_idx if 0 <= i <= 3)
    # Accuracy when correct is at each position
    acc_by_correct_pos: dict[str, float] = {}
    for letter, _ in correct_pos_dist.items():
        bucket = [t for t in trajectories if _option_letter(t.get("correct_index", -1)) == letter]
        acc_by_correct_pos[letter] = _pct(sum(1 for t in bucket if t["is_correct"]), len(bucket))

    # Stratify mt_d2 by bridge_necessary (from GPT-5.4 audit).
    def _bridge_nec_bucket(t):
        s = samples_by_id.get(t["sample_id"], {})
        if s.get("question_type") != "multi-target" or s.get("depth") != 2:
            return None  # skip samples outside mt_d2
        bn = s.get("bridge_necessary")
        if bn is True:
            return "necessary"
        if bn is False:
            return "optional"
        return "unlabeled"
    mt_d2_trajs = [t for t in trajectories if _bridge_nec_bucket(t) is not None]
    by_bridge_necessary = _bucket_accuracy(mt_d2_trajs, _bridge_nec_bucket) if mt_d2_trajs else {}

    accuracy = {
        "overall": _pct(n_correct, n),
        "by_type": _bucket_accuracy(trajectories, lambda t: t.get("question_type") or "?"),
        "by_depth": _bucket_accuracy(trajectories, lambda t: f"d{t.get('depth', 0)}"),
        "by_reasoning": _bucket_accuracy(trajectories, lambda t: t.get("reasoning_type") or "?"),
        "by_venue": _bucket_accuracy(
            trajectories,
            lambda t: (samples_by_id.get(t["sample_id"], {}).get("venue") or "?"),
        ),
        "by_consensus_tier": _bucket_accuracy(
            trajectories,
            lambda t: (samples_by_id.get(t["sample_id"], {}).get("consensus_tier") or "?"),
        ),
        "mt_d2_by_bridge_necessary": by_bridge_necessary,
        "by_filter_agreement": _bucket_accuracy(
            trajectories,
            lambda t: f"agree_{samples_by_id.get(t['sample_id'], {}).get('filter_agreement', '?')}",
        ),
        "conditional_on_termination": conditional,
        "position_bias": {
            "predicted_distribution": {k: predicted_dist.get(k, 0) for k in "ABCD"},
            "correct_position_distribution": {k: correct_pos_dist.get(k, 0) for k in "ABCD"},
            "accuracy_when_correct_at": acc_by_correct_pos,
        },
    }

    # ── Failure attribution ──────────────────────────────────────────────────
    wrong = [t for t in trajectories if not t.get("is_correct")]
    failure_counts = Counter()
    failure_total = 0
    for t in wrong:
        pred = t.get("predicted_index", -1)
        if pred < 0:
            failure_counts["no_answer"] += 1
            failure_total += 1
            continue
        sample = samples_by_id.get(t["sample_id"], {})
        dtypes = sample.get("distractor_types") or []
        if 0 <= pred < len(dtypes):
            ftype = dtypes[pred] if dtypes[pred] != "correct" else "unknown"
            failure_counts[ftype] += 1
        else:
            failure_counts["unknown"] += 1
        failure_total += 1

    failure_attribution = {
        "n_wrong": len(wrong),
        "counts": dict(failure_counts),
        "rates": {k: _pct(v, failure_total) for k, v in failure_counts.items()} if failure_total else {},
    }

    # ── Retrieval ────────────────────────────────────────────────────────────
    paper_hits = section_hits_hard = section_hits_soft = 0
    total_reads = wasted_reads = 0
    min_nav_hits = 0
    conv_num = conv_den = 0  # conversion: correct given reached terminal

    # Stratified navigation tracking
    # For mt: count of distinct target papers reached per sample
    # For st_d2: bridge × target reach pattern, plus whether search_papers was used
    nav_records = []  # list of (question_type, depth, n_targets_reached, n_bridges_reached,
                      #          n_total_targets, n_total_bridges, used_search, is_correct)

    for t in trajectories:
        sid = t["sample_id"]
        sample = samples_by_id.get(sid, {})
        recall = recall_by_id.get(sid, {})
        gold_set = _gold_arxiv_set(sample)
        bridge_set = set(sample.get("bridge_arxiv_ids", []) or [])
        labels_by_aid = _recall_labels_by_arxiv(recall)
        pid2aid = _paper_id_to_arxiv(sample)
        reads = _trajectory_read_sections(t, pid2aid)

        # Tool usage signals for navigation-pattern classification
        tc_names = [tc.get("name") for _, tc in _iter_tool_calls(t)]
        used_search = "search_papers" in tc_names
        if {"get_references", "search_papers"} & set(tc_names):
            min_nav_hits += 1

        reached_targets: set[str] = set()
        reached_bridges: set[str] = set()
        hit_direct = hit_soft = False
        for aid, sec in reads:
            total_reads += 1
            if aid in gold_set:
                reached_targets.add(aid)
                labels = labels_by_aid.get(aid, [])
                for lbl in labels:
                    lbl_sec = lbl.get("section", "")
                    if _section_match(sec, lbl_sec):
                        hit_soft = True
                        if lbl.get("relevance") == "direct":
                            hit_direct = True
                        break
            elif aid in bridge_set:
                reached_bridges.add(aid)
            else:
                wasted_reads += 1

        if reached_targets:
            paper_hits += 1
            conv_den += 1
            if t.get("is_correct"):
                conv_num += 1
        if hit_direct:
            section_hits_hard += 1
        if hit_soft:
            section_hits_soft += 1

        nav_records.append({
            "sample_id": sid,
            "question_type": t.get("question_type") or sample.get("question_type", ""),
            "depth": t.get("depth") or sample.get("depth", 0),
            "n_targets_reached": len(reached_targets),
            "n_bridges_reached": len(reached_bridges),
            "n_total_targets": len(gold_set),
            "n_total_bridges": len(bridge_set),
            "used_search": used_search,
            "is_correct": bool(t.get("is_correct")),
        })

    # ── Stratified navigation patterns ───────────────────────────────────────
    def _pattern_st_d2(r: dict) -> str:
        b = r["n_bridges_reached"] > 0
        tr = r["n_targets_reached"] > 0
        if b and tr:
            return "bridge_and_target"
        if tr and not b:
            return "target_only" + ("_via_search" if r["used_search"] else "_skipped_bridge")
        if b and not tr:
            return "bridge_only"
        return "neither"

    def _mt_target_bucket(r: dict) -> str:
        nreach = r["n_targets_reached"]
        ntotal = r["n_total_targets"]
        if nreach == 0:
            return "reached_0"
        if nreach >= ntotal:
            return "reached_all"
        return f"reached_{nreach}_of_{ntotal}"

    def _mt_d2_bucket(r: dict) -> str:
        """For mt_d2: combine target reach with bridge-path usage.

        Distinguishes proper 2-hop navigation (bridge_then_target) from
        shortcut navigation (search_papers directly to target).
        """
        nreach = r["n_targets_reached"]
        ntotal = r["n_total_targets"]
        bridge_reached = r["n_bridges_reached"] > 0
        if nreach == 0:
            return "reached_0_bridge" if bridge_reached else "reached_0"
        target_label = "reached_all" if nreach >= ntotal else f"reached_{nreach}_of_{ntotal}"
        suffix = "via_bridge" if bridge_reached else ("via_search" if r["used_search"] else "via_seed_cite")
        return f"{target_label}__{suffix}"

    nav_bucket_acc: dict[str, dict] = {}
    def _bucket_add(group: str, key: str, correct: bool):
        g = nav_bucket_acc.setdefault(group, {})
        b = g.setdefault(key, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(correct)

    for r in nav_records:
        qt = r["question_type"]
        d = r["depth"]
        if qt == "multi-target":
            group = f"mt_d{d}"
            bucket = _mt_d2_bucket(r) if d == 2 else _mt_target_bucket(r)
            _bucket_add(group, bucket, r["is_correct"])
        elif qt == "single-target":
            if d == 2:
                _bucket_add("st_d2", _pattern_st_d2(r), r["is_correct"])
            else:
                _bucket_add("st_d1",
                            "reached_target" if r["n_targets_reached"] > 0 else "missed",
                            r["is_correct"])

    # Flatten bucket accounts into summary-friendly dict
    navigation_patterns = {}
    for group, buckets in nav_bucket_acc.items():
        group_total = sum(b["n"] for b in buckets.values())
        navigation_patterns[group] = {
            "total": group_total,
            "buckets": {
                k: {
                    "n": b["n"],
                    "share": _pct(b["n"], group_total),
                    "correct": b["correct"],
                    "accuracy": _pct(b["correct"], b["n"]),
                }
                for k, b in sorted(buckets.items())
            },
        }

    retrieval = {
        "paper_recall": _pct(paper_hits, n),
        "section_recall_hard": _pct(section_hits_hard, n),
        "section_recall_soft": _pct(section_hits_soft, n),
        "conversion_rate": _pct(conv_num, conv_den),
        "conversion_denominator": conv_den,
        "min_navigation_rate": _pct(min_nav_hits, n),
        "wasted_read_rate": _pct(wasted_reads, total_reads),
        "total_read_calls": total_reads,
        "navigation_patterns": navigation_patterns,
    }

    # ── Tool metrics ─────────────────────────────────────────────────────────
    tool_usage_total = Counter()
    tool_budget_used = Counter()  # sum of cost per tool name
    rejected_total = 0
    accepted_total = 0
    turn_counts = [t.get("total_turns", 0) for t in trajectories]
    for t in trajectories:
        for _, tc in _iter_tool_calls(t):
            tool_usage_total[tc.get("name", "?")] += 1
        for _, res in _iter_tool_results(t):
            out = res.get("output", "") or ""
            if out.startswith("REJECTED:"):
                rejected_total += 1
            else:
                accepted_total += 1
                tool_budget_used[res.get("name", "?")] += res.get("cost", 0) or 0
    total_budget_spent = sum(tool_budget_used.values()) or 1  # avoid div/0

    tools_m = {
        "usage_counts": dict(tool_usage_total),
        "usage_per_sample_mean": {k: round(v / n, 2) for k, v in tool_usage_total.items()},
        "budget_spend_by_tool": dict(tool_budget_used),
        "budget_share_by_tool": {
            k: round(v / total_budget_spent, 4) for k, v in tool_budget_used.items()
        },
        "rejection_rate": _pct(rejected_total, rejected_total + accepted_total),
        "turn_stats": _stats(turn_counts),
        "terminated_by": dict(termination_counts),
    }

    # ── Cost ─────────────────────────────────────────────────────────────────
    model = config_dict.get("model", "")
    price = lookup_price(model)
    total_prompt = sum(t.get("total_prompt_tokens", 0) for t in trajectories)
    total_completion = sum(t.get("total_completion_tokens", 0) for t in trajectories)
    total_cr = sum(t.get("cache_read_tokens", 0) for t in trajectories)
    total_cw = sum(t.get("cache_creation_tokens", 0) for t in trajectories)
    total_input = total_prompt + total_cr + total_cw

    cost_breakdown = None
    if price is not None:
        cost_breakdown = compute_cost(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            cache_read_tokens=total_cr,
            cache_creation_tokens=total_cw,
            price=price,
        )

    # Per-turn token trajectory (mean across samples at each turn index)
    max_turn = max((len(t.get("turns", [])) for t in trajectories), default=0)
    trajectory_profile = []
    for k in range(max_turn):
        prompt_at_k = []
        completion_at_k = []
        for t in trajectories:
            if k < len(t.get("turns", [])):
                turn = t["turns"][k]
                prompt_at_k.append(turn.get("prompt_tokens", 0) or 0)
                completion_at_k.append(turn.get("completion_tokens", 0) or 0)
        if prompt_at_k:
            trajectory_profile.append({
                "turn": k,
                "n_samples": len(prompt_at_k),
                "mean_prompt_tokens": round(statistics.mean(prompt_at_k), 1),
                "mean_completion_tokens": round(statistics.mean(completion_at_k), 1),
            })

    wall_times = [t.get("wall_time_s", 0.0) for t in trajectories]
    cost = {
        "pricing_found": price is not None,
        "provider": price.provider if price else None,
        "input_per_m": price.input_per_m if price else None,
        "output_per_m": price.output_per_m if price else None,
        "cache_read_mult": price.cache_read_mult if price else None,
        "cache_write_mult": price.cache_write_mult if price else None,
        "tokens": {
            "regular_input": total_prompt,
            "cache_read": total_cr,
            "cache_creation": total_cw,
            "completion": total_completion,
            "total_input": total_input,
        },
        "per_sample_mean_tokens": {
            "regular_input": round(total_prompt / n, 1),
            "cache_read": round(total_cr / n, 1),
            "cache_creation": round(total_cw / n, 1),
            "completion": round(total_completion / n, 1),
        },
        "cache_hit_rate": _pct(total_cr, total_input),  # read tokens / total input
        "cost_usd": cost_breakdown,
        "wall_time": {
            **_stats(wall_times),
            "total_minutes": round(sum(wall_times) / 60, 2),
        },
        "trajectory_profile": trajectory_profile,
    }

    # ── Errors + sample-level retry telemetry ───────────────────────────────
    samples_with_transient_retries = sorted(
        t["sample_id"] for t in trajectories if (t.get("transient_retry_count") or 0) > 0
    )
    # Sample-level retries (whole-sample reruns on terminated_by == "error")
    retried_success = sorted(
        t["sample_id"] for t in trajectories
        if (t.get("retry_count") or 0) > 0 and t.get("terminated_by") != "error"
    )
    retried_failure = sorted(
        t["sample_id"] for t in trajectories
        if (t.get("retry_count") or 0) > 0 and t.get("terminated_by") == "error"
    )
    retry_histogram = Counter(t.get("retry_count") or 0 for t in trajectories)
    errors = {
        "api_error_total": sum(t.get("api_error_count", 0) for t in trajectories),
        "transient_retry_total": sum(t.get("transient_retry_count", 0) for t in trajectories),
        "tool_call_error_total": sum(t.get("tool_errors", 0) for t in trajectories),
        "samples_with_transient_retries": samples_with_transient_retries,
        # Sample-level retry breakdown (disjoint subsets)
        "retry_count_histogram": {str(k): v for k, v in sorted(retry_histogram.items())},
        "retried_then_succeeded": retried_success,           # rescued samples
        "retried_then_failed": retried_failure,              # unrecoverable
    }

    # ── Fingerprint ─────────────────────────────────────────────────────────
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "backend": config_dict.get("backend"),
        "strategy": config_dict.get("strategy"),
        "started_at": started_at,
        "completed_at": completed_at,
        "benchmark_sha256": benchmark_sha,
    }

    return {
        "fingerprint": fingerprint,
        "config": config_dict,
        "total": n,
        "correct": n_correct,
        "accuracy": accuracy,
        "failure_attribution": failure_attribution,
        "retrieval": retrieval,
        "tools": tools_m,
        "cost": cost,
        "errors": errors,
    }


# ── Loader helpers for disk → dict ──────────────────────────────────────────

def load_recall_labels(agenthop_dir: Path) -> dict[str, dict]:
    path = Path(agenthop_dir) / "audit" / "recall_labels.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def benchmark_sha256(agenthop_dir: Path) -> str:
    qa = Path(agenthop_dir) / "qa" / "full.jsonl"
    if not qa.exists():
        return ""
    h = hashlib.sha256()
    with open(qa, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def trajectory_to_dict(traj) -> dict:
    """Dataclass or already-a-dict → plain dict, without mutating."""
    if isinstance(traj, dict):
        return traj
    return asdict(traj)
