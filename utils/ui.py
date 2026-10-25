"""utils/ui.py - Streamlit styling helpers (CSS + dashboard cards)."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    color: #fff; padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0 0 .3rem 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.card {
    border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 1rem 1.2rem;
    background: rgba(128,128,128,.07); height: 100%;
}
.card .label {font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
.card .value {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
.card .sub {font-size: .85rem; opacity: .7;}
.pill {display:inline-block; padding: .12rem .7rem; border-radius: 999px; font-size: .82rem;
       font-weight: 600; color: #fff;}
.pill.ok {background:#16a34a;} .pill.warn {background:#d97706;} .pill.bad {background:#dc2626;}
@media (max-width: 700px) {.hero h1 {font-size: 1.4rem;} .card .value {font-size: 1.4rem;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "") -> None:
    """Dashboard card. Only pass trusted, app-generated strings (rendered as HTML)."""
    st.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def pill(text: str, kind: str = "ok") -> str:
    return f'<span class="pill {kind}">{text}</span>'
