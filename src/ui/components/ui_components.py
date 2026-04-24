"""
FinSight-Alpha — Reusable HTML/Streamlit UI components.

All helpers return HTML strings or call st.* directly.
Import selectively in each page.
"""

import streamlit as st
from components.theme import (
    SURFACE, SURFACE2, SURFACE3, BORDER, PRIMARY, PRIMARY_DIM,
    ACCENT, ACCENT_DIM, GREEN, RED, PURPLE, PINK, TEAL,
    TEXT, TEXT_MUTED, TEXT_DIM, MONO, SANS, DISPLAY,
)


# Low-level HTML primitives

def badge_html(label: str, color: str = PRIMARY) -> str:
    # Use double-quotes for the style attribute so the font-family value
    # (which contains single-quoted font names) does not break the HTML.
    return (
        f'<span style="'
        f"background:{color}22; color:{color}; border:1px solid {color}55; "
        f"border-radius:4px; padding:2px 10px; font-size:11px; "
        f"font-family:Space Mono,Courier New,monospace; font-weight:700; "
        f'letter-spacing:0.06em; text-transform:uppercase; margin-right:6px;">'
        f"{label}</span>"
    )


def live_dot_html(color: str = GREEN) -> str:
    return (
        f"<span style='width:7px;height:7px;border-radius:50%;"
        f"background:{color};display:inline-block;"
        f"animation:pulse 2s infinite;margin-right:6px;'></span>"
    )


def section_title(icon: str, title: str, subtitle: str = "") -> None:
    """Renders a styled section heading with optional subtitle."""
    st.markdown(
        f"<div style='margin-bottom:20px;'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
        f"<span style='font-size:20px;'>{icon}</span>"
        f"<span style='font-family:{DISPLAY};font-size:22px;font-weight:700;"
        f"color:{TEXT};'>{title}</span>"
        f"</div>"
        + (f"<p style='color:{TEXT_MUTED};font-size:13px;padding-left:30px;"
           f"margin:0;line-height:1.6;'>{subtitle}</p>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def card_start(border_color: str = BORDER, padding: str = "20px") -> str:
    """Returns opening div HTML for a dark card."""
    return (
        f"<div style='background:{SURFACE};border:1px solid {border_color};"
        f"border-radius:10px;padding:{padding};margin-bottom:0;'>"
    )


def card_end() -> str:
    return "</div>"


def mono_label(text: str, color: str = TEXT_MUTED) -> str:
    return (
        f"<div style='font-family:{MONO};font-size:11px;color:{color};"
        f"letter-spacing:0.08em;margin-bottom:14px;'>{text}</div>"
    )


def kv_row(key: str, value: str, value_color: str = TEXT) -> str:
    return (
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:5px 0;border-bottom:1px solid {BORDER};'>"
        f"<span style='font-size:12px;color:{TEXT_MUTED};'>{key}</span>"
        f"<span style='font-family:{MONO};font-size:11px;color:{value_color};'>{value}</span>"
        f"</div>"
    )


# Streamlit composite widgets

def stat_metric(col, value, label: str, color: str = PRIMARY, prefix: str = "", suffix: str = ""):
    """Renders a coloured big-number metric inside a given st.column."""
    col.markdown(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-radius:10px;padding:18px 14px;text-align:center;'>"
        f"<div style='font-family:{DISPLAY};font-size:30px;font-weight:800;"
        f"color:{color};line-height:1;'>{prefix}{value}{suffix}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:11px;font-family:{MONO};"
        f"margin-top:6px;letter-spacing:0.06em;text-transform:uppercase;'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def hero_banner(
    title_colored: str,
    title_plain: str,
    subtitle: str,
    badges: list[tuple[str, str]],   # [(label, color), ...]
    metrics: list[tuple[str, str, str]],  # [(value, label, color), ...]
    status_text: str = "SYSTEM ONLINE · v2.0.0",
) -> None:
    badges_html = "".join(badge_html(b[0], b[1]) for b in badges)
    metrics_html = "".join(
        f"<div style='background:{SURFACE2};border:1px solid {BORDER};"
        f"border-radius:8px;padding:14px;text-align:center;'>"
        f"<div style='font-family:{DISPLAY};font-size:26px;font-weight:800;color:{m[2]};'>{m[0]}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:10px;font-family:{MONO};"
        f"margin-top:4px;letter-spacing:0.06em;'>{m[1]}</div>"
        f"</div>"
        for m in metrics
    )

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{SURFACE} 0%,{SURFACE3} 100%);
                    border:1px solid {BORDER};border-radius:14px;padding:36px;
                    position:relative;overflow:hidden;margin-bottom:0;'>
          <!-- grid pattern -->
          <div style='position:absolute;inset:0;opacity:0.03;
                      background-image:linear-gradient({PRIMARY} 1px,transparent 1px),
                                       linear-gradient(90deg,{PRIMARY} 1px,transparent 1px);
                      background-size:40px 40px;'></div>
          <div style='position:relative;z-index:1;display:flex;
                      flex-wrap:wrap;gap:24px;align-items:flex-start;'>
            <!-- left copy -->
            <div style='flex:1;min-width:280px;'>
              <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
                {live_dot_html(GREEN)}
                <span style='font-family:{MONO};font-size:10px;color:{TEXT_MUTED};
                             letter-spacing:0.1em;'>{status_text}</span>
              </div>
              <h1 style='font-family:{DISPLAY};font-size:40px;font-weight:800;
                          line-height:1.1;margin-bottom:14px;'>
                <span style='color:{PRIMARY};'>{title_colored}</span>
                <span style='color:{TEXT};'>{title_plain}</span>
              </h1>
              <p style='color:{TEXT_MUTED};font-size:14px;line-height:1.7;
                        max-width:520px;margin-bottom:18px;'>{subtitle}</p>
              <div style='display:flex;flex-wrap:wrap;gap:8px;'>{badges_html}</div>
            </div>
            <!-- right metric grid -->
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;min-width:260px;'>
              {metrics_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_step_card(step: str, name: str, desc: str, tech: str, color: str) -> str:
    return (
        f"<div style='padding:18px 20px;background:{SURFACE2};"
        f"border:1px solid {BORDER};border-radius:8px;'>"
        f"<div style='font-family:{DISPLAY};font-size:26px;font-weight:800;"
        f"color:{color}40;margin-bottom:4px;'>{step}</div>"
        f"<div style='font-family:{DISPLAY};font-size:14px;font-weight:700;"
        f"color:{color};margin-bottom:8px;'>{name}</div>"
        f"<div style='font-size:12px;color:{TEXT_MUTED};line-height:1.6;"
        f"margin-bottom:10px;'>{desc}</div>"
        + badge_html(tech, color)
        + "</div>"
    )


def info_table(headers: list[str], rows: list[list], col_colors: dict = None) -> None:
    """
    Renders a styled HTML table.
    col_colors: {col_index: color_string} for value-based coloring.
    """
    th_style = (
        f"font-family:{MONO};font-size:10px;color:{TEXT_MUTED};"
        f"padding:8px 12px;text-align:left;letter-spacing:0.08em;"
        f"border-bottom:1px solid {BORDER};"
    )
    td_base = f"padding:9px 12px;font-family:{MONO};font-size:11px;border-bottom:1px solid {BORDER};"

    thead = "<tr>" + "".join(f"<th style='{th_style}'>{h}</th>" for h in headers) + "</tr>"
    tbody_rows = []
    for i, row in enumerate(rows):
        row_bg = SURFACE2 if i % 2 == 1 else "transparent"
        cells = []
        for j, cell in enumerate(row):
            color = (col_colors or {}).get(j, TEXT)
            if callable(color):
                color = color(cell)
            cells.append(f"<td style='{td_base}color:{color};background:{row_bg};'>{cell}</td>")
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = (
        f"<div style='overflow-x:auto;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead>{thead}</thead>"
        f"<tbody>{''.join(tbody_rows)}</tbody>"
        f"</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def capability_cards(items: list[dict]) -> None:
    """
    items: [{"icon","title","desc","color"}, ...]
    Renders as a CSS grid of feature cards.
    """
    cards_html = "".join(
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        f"border-left:3px solid {c['color']};border-radius:10px;padding:18px;'>"
        f"<div style='font-size:22px;margin-bottom:8px;'>{c['icon']}</div>"
        f"<div style='font-family:{DISPLAY};font-size:15px;font-weight:700;"
        f"margin-bottom:6px;color:{TEXT};'>{c['title']}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:12px;line-height:1.6;'>{c['desc']}</div>"
        f"</div>"
        for c in items
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
        f"gap:14px;'>{cards_html}</div>",
        unsafe_allow_html=True,
    )


def tech_stack_grid(items: list[tuple[str, str, str]]) -> None:
    """items: [(name, role, color)]"""
    cards_html = "".join(
        f"<div style='padding:10px 12px;background:{SURFACE2};"
        f"border-radius:6px;border:1px solid {BORDER};'>"
        f"<div style='font-family:{MONO};font-size:11px;color:{c};font-weight:700;'>{n}</div>"
        f"<div style='color:{TEXT_DIM};font-size:11px;margin-top:2px;'>{r}</div>"
        f"</div>"
        for n, r, c in items
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
        f"gap:10px;'>{cards_html}</div>",
        unsafe_allow_html=True,
    )