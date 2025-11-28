#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio app: chat + independent assistant-driven chart.
- Chart updates automatically after each assistant answer.
- Chart spec can come from a ```JSON_CHART ... ``` block OR be inferred from NL text.
- Safe: whitelist only allowed metrics and columns; use small in-memory cache.

This patched version fixes a schema-mismatch bug caused by case-sensitive
column matching and improves year-filtering and logging to help debug
why charts might be blank.
"""
import json
import logging
import os
import re
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import gradio as gr

# Logging
LOG = logging.getLogger("bahrain_agent_ui")
logging.basicConfig(level=logging.INFO)

# --- Config / paths ---
DATA_PATH = os.getenv("BAHRAIN_DATA_PATH", "data/bahrain_master")
CHART_CACHE_TTL = int(os.getenv("CHART_CACHE_TTL_SECONDS", "30"))
DEFAULT_USE_LLM = True

# --- Agent imports (keep your existing behavior) ---
from bahrain_agent.agent import BahrainStatsAgent
from bahrain_agent.nlu_router import route_and_answer  # optional LLM-refinement wrapper

agent = BahrainStatsAgent(data_path=DATA_PATH)
LOG.info("Loading BahrainStatsAgent with data path: %s", DATA_PATH)

# --- Safety whitelists & mapping ---
ALLOWED_CHART_TYPES = {"bar", "line"}
ALLOWED_X = {"governorate", "year"}
ALLOWED_Y = {"households", "population", "students", "teachers", "units", "count"}
MAX_GROUPS = 50
MAX_JSON_CHARS = 4000

METRIC_TO_FILES = {
    "households": (["households.parquet", "master.parquet"], ["households.csv", "master.csv"]),
    "population": (["population_density.parquet", "master.parquet"], ["population_density.csv", "master.csv"]),
    "students": (["students.parquet", "master.parquet"], ["students.csv", "master.csv"]),
    "teachers": (["teachers.parquet", "master.parquet"], ["teachers.csv", "master.csv"]),
    "units": (["housing_units.parquet", "master.parquet"], ["housing_units.csv", "master.csv"]),
    "count": (["master.parquet"], ["master.csv"]),
}

# small in-process cache
_chart_cache: Dict[str, Dict] = {}

# ----- helpers to read minimal table safely -----
def _read_table_try(parquets: list, csvs: list) -> pd.DataFrame:
    for p in parquets:
        pth = os.path.join(DATA_PATH, p)
        if os.path.exists(pth):
            try:
                df = pd.read_parquet(pth)
                return df
            except Exception as e:
                LOG.warning("Failed to read parquet %s: %s", pth, e)
    for c in csvs:
        cpth = os.path.join(DATA_PATH, c)
        if os.path.exists(cpth):
            try:
                df = pd.read_csv(cpth)
                return df
            except Exception as e:
                LOG.warning("Failed to read csv %s: %s", cpth, e)
    return pd.DataFrame()

# ----- parse JSON_CHART (strict) -----
def parse_chart_spec_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```JSON_CHART\s*([\s\S]{1,%d}?)\s*```" % MAX_JSON_CHARS, text, flags=re.I)
    candidate = None
    if m:
        candidate = m.group(1).strip()
    else:
        m2 = re.search(r"(\{[\s\S]{10,1000}\})", text)
        if m2:
            candidate = m2.group(1).strip()
    if not candidate:
        return None
    try:
        spec = json.loads(candidate)
    except Exception:
        try:
            spec = json.loads(candidate.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(spec, dict):
        return None
    typ = spec.get("type", "bar")
    if typ not in ALLOWED_CHART_TYPES:
        typ = "bar"
    x = spec.get("x")
    y = spec.get("y")
    if x not in ALLOWED_X or y not in ALLOWED_Y:
        return None
    filters = spec.get("filters", {}) or {}
    limit = spec.get("limit", 20)
    try:
        limit = int(limit)
    except Exception:
        limit = 20
    limit = max(1, min(limit, MAX_GROUPS))
    return {"type": typ, "x": x, "y": y, "filters": filters, "limit": limit}

# ----- light NL intent detector: returns (spec, confidence) -----
def infer_spec_from_text_nl(text: str) -> Tuple[Optional[dict], float]:
    """
    Lightweight heuristic inference from assistant text:
    - looks for metric keywords and x keywords plus years/ranges
    - returns a spec and confidence score (0-1)
    """
    if not text:
        return None, 0.0
    nl = text.lower()
    # metric detection
    metric = None
    for m in ALLOWED_Y:
        if m in nl:
            metric = m
            break
    # x detection (prefer governorate or year)
    x = None
    if "governorate" in nl or "region" in nl or "area" in nl:
        x = "governorate"
    elif "year" in nl or re.search(r"\b20[0-9]{2}\b", nl):
        x = "year"

    if not metric or not x:
        return None, 0.0

    # year or year range detection
    years = re.findall(r"\b(19[7-9]\d|20[0-9]{2})\b", nl)
    filters = {}
    if years:
        yrs = sorted([int(y) for y in years])
        if len(yrs) == 1:
            filters["year"] = yrs[0]
        elif len(yrs) >= 2:
            filters["year"] = {"from": yrs[0], "to": yrs[-1]}

    # confidence heuristic: higher if text explicitly mentions "by governorate" or "for 2022"
    conf = 0.5
    if "by governorate" in nl or "by area" in nl:
        conf += 0.3
    if years:
        conf += 0.15
    conf = min(conf, 0.95)

    spec = {"type": "bar" if x == "governorate" else "line", "x": x, "y": metric, "filters": filters, "limit": 20}
    return spec, conf

# ----- build chart from spec (same safe builder) -----
def build_chart_from_spec(spec: dict):
    if not spec:
        return None
    x = spec["x"]; y = spec["y"]; typ = spec.get("type", "bar"); filters = spec.get("filters", {}) or {}; limit = spec.get("limit", 20)
    parquets, csvs = METRIC_TO_FILES.get(y, (["master.parquet"], ["master.csv"]))
    df = _read_table_try(parquets=parquets, csvs=csvs)
    if df.empty:
        sample = pd.DataFrame({x: ["-"], y: [0]})
        fig = px.bar(sample, x=x, y=y, title=f"{y} by {x} (no data)")
        fig.update_layout(autosize=True, margin=dict(l=30, r=10, t=40, b=40))
        return fig
    # normalize column lookup: map lowercased stripped name -> original
    cols_map = {c.strip().lower(): c for c in df.columns}

    def find_col(wanted):
        # make wanted lower for case-insensitive matching
        lw = (wanted or "").strip().lower()
        # direct exact match
        if lw in cols_map:
            return cols_map[lw]
        # substring match
        for k, real in cols_map.items():
            if lw and (lw in k or k in lw):
                return real
        return None

    x_col = find_col(x); y_col = find_col(y)
    if not x_col or not y_col:
        LOG.info("Schema mismatch: could not find columns for x=%s y=%s in data. Available columns: %s", x, y, list(df.columns))
        sample = pd.DataFrame({x: ["-"], y: [0]})
        fig = px.bar(sample, x=x, y=y, title=f"{y} by {x} (schema mismatch)")
        fig.update_layout(autosize=True, margin=dict(l=30, r=10, t=40, b=40))
        return fig

    sub = df[[x_col, y_col]].copy()
    # coerce types defensively
    sub[x_col] = sub[x_col].astype(str).str.strip()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce").fillna(0)

    # apply year filter defensively: find a year column if filter references year
    year_filter = filters.get("year")
    if year_filter is not None:
        year_col = find_col("year")
        if year_col and year_col in df.columns:
            try:
                if isinstance(year_filter, dict):
                    fr = year_filter.get("from"); to = year_filter.get("to")
                    if fr is not None:
                        sub = sub[df[year_col].astype(float) >= float(fr)]
                    if to is not None:
                        sub = sub[df[year_col].astype(float) <= float(to)]
                else:
                    sub = sub[df[year_col].astype(float) == float(year_filter)]
            except Exception:
                LOG.debug("Year filter could not be applied; continuing without it")
        else:
            LOG.debug("Year filter requested but no year column found in table; skipping year filter")

    try:
        agg = sub.groupby(x_col, dropna=True)[y_col].sum().reset_index()
        agg = agg.sort_values(by=y_col, ascending=False).head(limit)
        agg.columns = [x, y]
    except Exception as e:
        LOG.exception("Aggregation error: %s", e)
        agg = pd.DataFrame({x: ["-"], y: [0]})
    try:
        if typ == "bar":
            fig = px.bar(agg, x=x, y=y, title=f"{y} by {x}")
        else:
            fig = px.line(agg, x=x, y=y, title=f"{y} by {x}")
        fig.update_layout(autosize=True, margin=dict(l=30, r=10, t=46, b=40))
        try:
            if "height" in fig.layout:
                del fig.layout["height"]
        except Exception:
            pass
        LOG.info("Built chart for spec x=%s y=%s (rows=%d)", x, y, len(agg))
        return fig
    except Exception as e:
        LOG.exception("Plot build error: %s", e)
        fallback = pd.DataFrame({x: ["-"], y: [0]})
        fig = px.bar(fallback, x=x, y=y, title=f"{y} by {x} (error)")
        fig.update_layout(autosize=True)
        return fig

# ----- caching wrapper -----
def get_chart_for_spec_cached(spec: dict):
    key = json.dumps(spec, sort_keys=True)
    now = time.time()
    cached = _chart_cache.get(key)
    if cached and (now - cached.get("ts", 0) < CHART_CACHE_TTL):
        return cached["fig"]
    fig = build_chart_from_spec(spec)
    _chart_cache[key] = {"ts": now, "fig": fig}
    return fig

# ----- core submit: returns chart independently based on assistant answer -----
def submit_message(message: str, history: list, use_llm: bool):
    """
    Returns: (history, cleared_textbox, optional_plotly_fig_or_None, last_updated)
    Chart is determined from assistant text via JSON spec or NL inference.
    """
    if not message or not message.strip():
        return history or [], "", None, "Last updated: -"

    history = history or []
    user_msg = {"role": "user", "content": message.strip()}
    history.append(user_msg)

    # get assistant answer
    try:
        if use_llm:
            answer = route_and_answer(agent, message.strip(), use_llm=True)
        else:
            answer = agent.answer_question(message.strip())
    except Exception as e:
        LOG.exception("Error calling agent:")
        answer = f"Error producing answer: {e}"

    assistant_msg = {"role": "assistant", "content": answer}
    history.append(assistant_msg)

    LOG.debug("Assistant answer: %s", answer)

    # 1) strict parse JSON_CHART
    spec = parse_chart_spec_from_text(answer)
    confidence = 1.0 if spec else 0.0

    # 2) if no strict spec, try NL inference
    if not spec:
        inferred, conf = infer_spec_from_text_nl(answer)
        if inferred and conf >= 0.4:  # require minimal confidence
            spec = inferred
            confidence = conf

    # 3) fallback: attempt to infer from user's original question if still no spec
    if not spec:
        inferred_user, conf2 = infer_spec_from_text_nl(message)
        if inferred_user and conf2 >= 0.45:
            spec = inferred_user
            confidence = conf2

    LOG.info("Chart spec chosen: %s (confidence=%.2f)", spec, confidence)

    # 4) If still no spec, don't change chart
    if spec:
        try:
            fig = get_chart_for_spec_cached(spec)
            last_updated = f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} (conf={confidence:.2f})"
        except Exception as e:
            LOG.exception("Chart build failed: %s", e)
            fig = None
            last_updated = "Last updated: -"
    else:
        fig = None
        last_updated = "Last updated: -"

    return history, "", fig, last_updated


def clear_history():
    return []

# ----- Gradio UI (chat + independent chart) -----
with gr.Blocks(title="BH Bahrain Statistical AI Agent") as demo:
    gr.Markdown("## BH Bahrain Statistical AI Agent\nAsk about labour, households, population density, housing, segmentation etc.")
    with gr.Row():
        llm_checkbox = gr.Checkbox(value=DEFAULT_USE_LLM, label="Use LLM (ChatGPT) refinement")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", elem_id="chatbot")
            txt = gr.Textbox(placeholder="Type your question...", show_label=False)
            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Chart (independently driven by assistant answer)")
            chart_plot = gr.Plot(label="Chart")
            last_updated = gr.Text(value="Last updated: -", interactive=False)

    state = gr.State(value=[])

    send_btn.click(fn=submit_message, inputs=[txt, state, llm_checkbox],
                   outputs=[chatbot, txt, chart_plot, last_updated], queue=True).then(
        lambda h: h, inputs=[chatbot], outputs=[state]
    )
    txt.submit(fn=submit_message, inputs=[txt, state, llm_checkbox],
               outputs=[chatbot, txt, chart_plot, last_updated], queue=True).then(
        lambda h: h, inputs=[chatbot], outputs=[state]
    )
    clear_btn.click(fn=clear_history, inputs=None, outputs=[chatbot, state])

    # default chart loaded on app start
    def _default_chart():
        default_spec = {"type": "bar", "x": "governorate", "y": "households", "limit": 20}
        fig = get_chart_for_spec_cached(default_spec)
        return fig, f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    demo.load(fn=_default_chart, inputs=None, outputs=[chart_plot, last_updated])

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
