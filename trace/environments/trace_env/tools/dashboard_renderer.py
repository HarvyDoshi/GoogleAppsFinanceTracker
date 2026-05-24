"""
environments/trace_env/tools/dashboard_renderer.py

Dashboard Renderer for Trace.

Converts parsed transaction data → beautiful HTML dashboard.
Works for ALL transaction categories:
  ride | food | shopping | payment | subscription | travel | banking | utility | education | healthcare

Usage:
    from environments.trace_env.tools.dashboard_renderer import render_dashboard
    html = render_dashboard(parsed_result)
    # Save to file or return via a new FastAPI endpoint
"""

from __future__ import annotations
import json
import re
from datetime import datetime


def render_dashboard(parsed_result: dict) -> str:
    """
    Render a full HTML dashboard from parse_transactions_bulk() output.

    Args:
        parsed_result: output of transaction_parser.parse_transactions_bulk()

    Returns:
        Complete HTML string ready to open in browser or serve via FastAPI.
    """
    transactions = parsed_result.get("transactions", [])
    summary = parsed_result.get("summary", {})
    episode = parsed_result.get("episode", {})

    cards_html = "\n".join(_render_card(t, i) for i, t in enumerate(transactions))
    summary_html = _render_summary(summary)
    episode_html = _render_episode_bar(episode)
    chart_data = _build_chart_data(summary)

    sheet_url = parsed_result.get("sheet_url")
    sheets_html = ""
    if sheet_url:
        sheets_html = f"""
  <!-- Sheets banner -->
  <div class="sheets-banner" onclick="window.open('{sheet_url}', '_blank')">
    <div class="sheets-icon">📊</div>
    <div class="sheets-text">
      <div class="sheets-title">Live Google Sheets Ledger Connected</div>
      <div class="sheets-sub">The summary above includes all historical data synced from your spreadsheet.</div>
    </div>
    <div class="sheets-btn">Open Sheet ›</div>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace · Transaction Dashboard</title>
<style>
{_css()}
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header class="header">
    <div class="header-left">
      <div class="logo">🕵️ Trace</div>
      <div class="subtitle">Gmail Transaction Intelligence</div>
    </div>
    <div class="header-right">
      <div class="badge">REAL DATA · LIVE</div>
    </div>
  </header>

  <!-- Episode bar -->
  {episode_html}

  <!-- Summary strip -->
  {summary_html}
  
  {sheets_html}

  <!-- Spend by category chart (simple CSS bars) -->
  {_render_chart(summary)}

  <!-- Transaction cards -->
  <div class="section-title">
    <span>Retrieved Transactions</span>
    <div class="section-line"></div>
    <span class="count">{len(transactions)} emails</span>
  </div>

  <div class="cards">
    {cards_html}
  </div>

  <!-- Analysis footer -->
  {_render_insights(transactions, summary)}

</div>

<script>
{_js(chart_data)}
</script>
</body>
</html>"""


# ── Card renderer ─────────────────────────────────────────────────────────────

def _render_card(t: dict, idx: int) -> str:
    cfg = t["category_config"]
    icon = cfg["icon"]
    color = cfg["color"]
    label = cfg["label"]
    delay = idx * 80

    details_html = _render_details(t)
    amounts_html = _render_amounts(t)
    image_html = _render_image_analysis(t)
    tags_html = _render_tags(t)

    total_display = t["total"] or "—"
    order_display = f'<span class="order-id">#{t["order_id"]}</span>' if t["order_id"] else ""
    reimbursable = '<span class="tag reimb">✓ Reimbursable</span>' if t["reimbursable"] else ""

    return f"""
<div class="card" style="animation-delay:{delay}ms" data-category="{t['category']}">
  <div class="card-header" style="background:{color}">
    <div class="card-header-left">
      <div class="card-icon">{icon}</div>
      <div>
        <div class="card-vendor">{t['vendor']}</div>
        <div class="card-meta">{label} · {t['date'][:22] if t['date'] else '—'}</div>
      </div>
    </div>
    <div class="card-header-right">
      <div class="card-total">{total_display}</div>
      {order_display}
    </div>
  </div>

  <div class="card-subject">{t['subject'] or '(No subject)'}</div>

  <div class="card-body">
    <div class="card-section">
      <div class="section-header">Details</div>
      {details_html}
    </div>
    {amounts_html}
  </div>

  {image_html}

  <div class="card-footer">
    {tags_html}
    {reimbursable}
    <span class="tag pay">{t['payment_method']}</span>
    <span class="tag from">{_clean_email(t['from_email'])}</span>
  </div>
</div>"""


def _render_details(t: dict) -> str:
    details = t.get("details", {})
    category = t["category"]
    rows = []

    if category == "ride":
        for k, v in [("Distance", details.get("distance")), ("Duration", details.get("duration")),
                     ("Plate", details.get("license_plate")), ("From", details.get("from")),
                     ("To", details.get("to"))]:
            if v:
                rows.append(f'<div class="detail-row"><span class="dk">{k}</span><span class="dv">{v}</span></div>')

    elif category == "food":
        if details.get("restaurant"):
            rows.append(f'<div class="detail-row"><span class="dk">Restaurant</span><span class="dv">{details["restaurant"]}</span></div>')
        if details.get("items"):
            items_str = ", ".join(details["items"][:3])
            rows.append(f'<div class="detail-row"><span class="dk">Items</span><span class="dv">{items_str}</span></div>')
        if details.get("delivery_fee"):
            rows.append(f'<div class="detail-row"><span class="dk">Delivery</span><span class="dv">{details["delivery_fee"]}</span></div>')

    elif category == "payment":
        for k, v in [("To", details.get("to")), ("From", details.get("from")),
                     ("UPI Ref", details.get("upi_ref")), ("Status", details.get("status"))]:
            if v:
                rows.append(f'<div class="detail-row"><span class="dk">{k}</span><span class="dv">{v}</span></div>')

    elif category == "shopping":
        if details.get("items"):
            for item in details["items"][:4]:
                rows.append(f'<div class="detail-row"><span class="dk">·</span><span class="dv">{item}</span></div>')
        if details.get("delivery_date"):
            rows.append(f'<div class="detail-row"><span class="dk">Delivery</span><span class="dv">{details["delivery_date"]}</span></div>')

    else:
        # Generic: show snippet
        snippet = t.get("snippet", "")[:150]
        if snippet:
            rows.append(f'<div class="detail-snippet">{snippet}</div>')

    if not rows:
        snippet = t.get("snippet", "")[:150]
        rows.append(f'<div class="detail-snippet">{snippet or "No details extracted."}</div>')

    return "\n".join(rows)


def _render_amounts(t: dict) -> str:
    amounts = t.get("amounts", [])
    if not amounts or len(amounts) <= 1:
        return ""
    items = "".join(f'<span class="amount-chip">{a}</span>' for a in amounts[:6])
    return f'<div class="card-section"><div class="section-header">All Amounts</div><div class="amounts-row">{items}</div></div>'


def _render_image_analysis(t: dict) -> str:
    analyses = t.get("image_analyses", [])
    if not analyses:
        return ""
    items = []
    for a in analyses[:2]:
        summary = a.get("summary", "")[:120]
        doc_type = a.get("doc_type", "unknown")
        items.append(f'<div class="img-analysis"><span class="img-type">{doc_type}</span> {summary}</div>')
    return f'<div class="card-image-analysis"><div class="section-header">🖼 Image Analysis (Ollama)</div>{"".join(items)}</div>'


def _render_tags(t: dict) -> str:
    cfg = t["category_config"]
    return f'<span class="tag cat" style="background:{cfg["color"]}22;color:{cfg["color"]}">{cfg["icon"]} {cfg["label"]}</span>'


def _render_amounts_summary(t: dict) -> str:
    return t.get("total") or (t.get("amounts", ["—"])[0] if t.get("amounts") else "—")


# ── Summary strip ─────────────────────────────────────────────────────────────

def _render_summary(summary: dict) -> str:
    total = summary.get("total_spend", 0)
    count = summary.get("count", 0)
    by_cat = summary.get("by_category", {})
    top_cat = max(by_cat, key=by_cat.get) if by_cat else "—"
    top_spend = by_cat.get(top_cat, 0)

    return f"""
<div class="summary-strip">
  <div class="sum-card total">
    <div class="sum-label">Total Spend</div>
    <div class="sum-value">₹{total:,.2f}</div>
    <div class="sum-sub">Across {count} transactions</div>
  </div>
  <div class="sum-card">
    <div class="sum-label">Transactions</div>
    <div class="sum-value">{count}</div>
    <div class="sum-sub">Gmail emails parsed</div>
  </div>
  <div class="sum-card">
    <div class="sum-label">Top Category</div>
    <div class="sum-value">{top_cat.title()}</div>
    <div class="sum-sub">₹{top_spend:,.2f} spent</div>
  </div>
  <div class="sum-card">
    <div class="sum-label">Categories</div>
    <div class="sum-value">{len(by_cat)}</div>
    <div class="sum-sub">Detected</div>
  </div>
</div>"""


# ── Spend chart ───────────────────────────────────────────────────────────────

def _render_chart(summary: dict) -> str:
    by_cat = summary.get("by_category", {})
    if not by_cat:
        return ""

    total = sum(by_cat.values()) or 1
    from environments.trace_env.tools.transaction_parser import CATEGORY_CONFIG

    bars = ""
    for cat, amount in list(by_cat.items())[:8]:
        cfg = CATEGORY_CONFIG.get(cat, CATEGORY_CONFIG["unknown"])
        pct = (amount / total) * 100
        bars += f"""
      <div class="chart-row">
        <div class="chart-label">{cfg['icon']} {cat.title()}</div>
        <div class="chart-bar-wrap">
          <div class="chart-bar" style="width:{pct:.1f}%;background:{cfg['color']}"></div>
        </div>
        <div class="chart-amount">₹{amount:,.0f}</div>
      </div>"""

    return f"""
<div class="chart-section">
  <div class="section-title"><span>Spend by Category</span><div class="section-line"></div></div>
  <div class="chart">{bars}</div>
</div>"""


# ── Episode bar ───────────────────────────────────────────────────────────────

def _render_episode_bar(episode: dict) -> str:
    if not episode:
        return ""
    steps = ["PLAN", "RETRIEVE", "MEMORIZE", "VERIFY", "ANSWER"]
    current_step = episode.get("step", 1)
    reward = episode.get("reward", 0)

    chips = ""
    for i, s in enumerate(steps):
        if i < current_step:
            cls = "done"
        elif i == current_step:
            cls = "active"
        else:
            cls = ""
        sep = '<span class="step-sep">›</span>' if i < len(steps) - 1 else ""
        chips += f'<div class="step-chip {cls}">{s}</div>{sep}'

    return f"""
<div class="episode-bar">
  <div class="ep-label">Episode · Step {current_step} · Reward {reward:.3f}</div>
  <div class="steps-row">{chips}</div>
</div>"""


# ── Insights panel ────────────────────────────────────────────────────────────

def _render_insights(transactions: list, summary: dict) -> str:
    insights = []

    total = summary.get("total_spend", 0)
    if total > 0:
        insights.append(("💡", "good", f"Total retrievable spend: <strong>₹{total:,.2f}</strong> across {summary.get('count', 0)} transactions."))

    reimb = [t for t in transactions if t.get("reimbursable")]
    if reimb:
        reimb_total = sum(
            float(re.sub(r'[^\d.]', '', t["total"])) for t in reimb if t.get("total")
            and re.sub(r'[^\d.]', '', t["total"]).replace('.', '').isdigit()
        )
        insights.append(("🧾", "good", f"<strong>{len(reimb)} reimbursable</strong> receipts found totaling ₹{reimb_total:,.2f}."))

    by_cat = summary.get("by_category", {})
    if by_cat:
        top = list(by_cat.items())[0]
        insights.append(("📊", "info", f"Highest spending category: <strong>{top[0].title()}</strong> at ₹{top[1]:,.2f}."))

    cash_payments = [t for t in transactions if t.get("payment_method") == "Cash"]
    if cash_payments:
        insights.append(("⚠️", "warn", f"{len(cash_payments)} cash payment(s) detected — no digital trail for these transactions."))

    rows = "".join(f'<div class="insight-row"><div class="insight-icon {s}">{ic}</div><div>{text}</div></div>'
                   for ic, s, text in insights)

    return f"""
<div class="insights-panel">
  <div class="insights-header">🧠 Trace Intelligence Insights</div>
  {rows}
</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

def _css() -> str:
    return """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f0ebe3;
  --card: #ffffff;
  --border: #d4cec7;
  --ink: #1a1a1a;
  --muted: #6b6560;
  --cream: #f0ebe3;
}

body { font-family: 'DM Mono', monospace; background: var(--bg); color: var(--ink); }
.app { max-width: 900px; margin: 0 auto; padding: 24px 20px; }

/* Header */
.header { display: flex; justify-content: space-between; align-items: flex-end;
  border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 20px; }
.logo { font-family: 'Fraunces', serif; font-size: 26px; font-weight: 600; }
.subtitle { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.badge { font-size: 10px; background: var(--ink); color: white; padding: 4px 10px; border-radius: 2px; letter-spacing: 1px; }

/* Episode bar */
.episode-bar { background: var(--card); border: 1px solid var(--border); border-radius: 4px;
  padding: 12px 16px; margin-bottom: 20px; }
.ep-label { font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 10px; }
.steps-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.step-chip { font-size: 10px; padding: 4px 10px; border-radius: 2px; border: 1px solid var(--border); background: var(--cream); }
.step-chip.done { background: #d4f0e0; border-color: #1a7a4a; color: #1a7a4a; }
.step-chip.active { background: #fdf0d5; border-color: #c97b1a; color: #c97b1a; }
.step-sep { color: var(--border); font-size: 12px; }

/* Summary */
.summary-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.sum-card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 14px 16px; }
.sum-card.total { border-top: 3px solid var(--ink); }
.sum-label { font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 6px; }
.sum-value { font-family: 'Fraunces', serif; font-size: 22px; font-weight: 600; line-height: 1; }
.sum-sub { font-size: 10px; color: var(--muted); margin-top: 4px; }

/* Chart */
.chart-section { margin-bottom: 24px; }
.chart { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 16px; }
.chart-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.chart-row:last-child { margin-bottom: 0; }
.chart-label { width: 120px; font-size: 11px; flex-shrink: 0; }
.chart-bar-wrap { flex: 1; height: 8px; background: var(--cream); border-radius: 2px; overflow: hidden; }
.chart-bar { height: 100%; border-radius: 2px; transition: width 1s cubic-bezier(0.4,0,0.2,1); }
.chart-amount { width: 80px; text-align: right; font-size: 11px; font-weight: 500; }

/* Sheets Banner */
.sheets-banner { display: flex; align-items: center; gap: 14px; background: #e8f0fe; border: 1px solid #d2e3fc; border-radius: 4px; padding: 14px 18px; margin-bottom: 24px; cursor: pointer; transition: background 0.2s, transform 0.2s; }
.sheets-banner:hover { background: #d2e3fc; transform: translateY(-1px); }
.sheets-icon { font-size: 24px; }
.sheets-text { flex: 1; }
.sheets-title { font-family: 'Fraunces', serif; font-size: 16px; font-weight: 600; color: #174ea6; }
.sheets-sub { font-size: 11px; color: #1967d2; margin-top: 4px; }
.sheets-btn { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #174ea6; background: rgba(23,78,166,0.1); padding: 6px 12px; border-radius: 2px; }

/* Section title */
.section-title { display: flex; align-items: center; gap: 12px; font-size: 9px;
  text-transform: uppercase; letter-spacing: 2px; color: var(--muted); margin-bottom: 14px; }
.section-line { flex: 1; height: 1px; background: var(--border); }
.count { color: var(--ink); }

/* Cards */
.cards { display: flex; flex-direction: column; gap: 16px; }

.card { background: var(--card); border: 1px solid var(--border); border-radius: 4px;
  overflow: hidden; animation: slideIn 0.4s ease both; }
@keyframes slideIn { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform:translateY(0); } }

.card-header { display: flex; justify-content: space-between; align-items: center;
  padding: 14px 18px; color: white; }
.card-header-left { display: flex; align-items: center; gap: 12px; }
.card-icon { font-size: 22px; }
.card-vendor { font-family: 'Fraunces', serif; font-size: 16px; font-style: italic; }
.card-meta { font-size: 10px; opacity: 0.65; margin-top: 2px; }
.card-total { font-family: 'Fraunces', serif; font-size: 22px; font-weight: 600; text-align: right; }
.order-id { font-size: 10px; opacity: 0.6; text-align: right; display: block; margin-top: 2px; }

.card-subject { padding: 10px 18px; font-size: 11px; border-bottom: 1px solid var(--border);
  background: #fafaf8; color: var(--muted); }

.card-body { display: grid; grid-template-columns: 1fr 1fr; }
.card-section { padding: 14px 18px; border-right: 1px solid var(--border); }
.card-section:last-child { border-right: none; }
.section-header { font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px;
  color: var(--muted); margin-bottom: 10px; }

.detail-row { display: flex; justify-content: space-between; padding: 4px 0;
  border-bottom: 1px dashed var(--border); font-size: 11px; }
.detail-row:last-child { border-bottom: none; }
.dk { color: var(--muted); }
.dv { font-weight: 500; text-align: right; max-width: 60%; }
.detail-snippet { font-size: 11px; color: var(--muted); line-height: 1.5; }

.amounts-row { display: flex; flex-wrap: wrap; gap: 6px; }
.amount-chip { font-size: 10px; background: #f0ebe3; border: 1px solid var(--border);
  padding: 3px 8px; border-radius: 2px; }

.card-image-analysis { padding: 12px 18px; border-top: 1px solid var(--border); background: #fafaf8; }
.img-analysis { font-size: 11px; color: var(--muted); line-height: 1.5; margin-top: 6px; }
.img-type { background: var(--ink); color: white; font-size: 9px; padding: 1px 6px;
  border-radius: 2px; margin-right: 6px; text-transform: uppercase; }

.card-footer { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 18px;
  border-top: 1px solid var(--border); background: #fafaf8; }
.tag { font-size: 9px; padding: 3px 8px; border-radius: 2px; text-transform: uppercase;
  letter-spacing: 0.5px; border: 1px solid var(--border); background: var(--cream); color: var(--muted); }
.tag.reimb { background: #d4f0e0; border-color: #1a7a4a; color: #1a7a4a; }
.tag.pay { background: #fdf0d5; border-color: #c97b1a; color: #c97b1a; }

/* Insights */
.insights-panel { background: var(--card); border: 1px solid var(--border); border-radius: 4px;
  overflow: hidden; margin-top: 24px; }
.insights-header { background: var(--ink); color: white; padding: 12px 18px;
  font-size: 10px; text-transform: uppercase; letter-spacing: 2px; }
.insight-row { display: flex; align-items: flex-start; gap: 10px; padding: 12px 18px;
  border-top: 1px solid var(--border); font-size: 11px; line-height: 1.6; }
.insight-icon { width: 24px; height: 24px; border-radius: 2px; display: flex;
  align-items: center; justify-content: center; font-size: 13px; flex-shrink: 0; }
.insight-icon.good { background: #d4f0e0; }
.insight-icon.warn { background: #fdf0d5; }
.insight-icon.info { background: #e8f0fe; }

@media (max-width: 600px) {
  .summary-strip { grid-template-columns: 1fr 1fr; }
  .card-body { grid-template-columns: 1fr; }
  .card-section { border-right: none; border-bottom: 1px solid var(--border); }
}
"""


# ── JS ────────────────────────────────────────────────────────────────────────

def _js(chart_data: dict) -> str:
    return f"""
// Animate chart bars on load
document.addEventListener('DOMContentLoaded', function() {{
  const bars = document.querySelectorAll('.chart-bar');
  bars.forEach(b => {{
    const w = b.style.width;
    b.style.width = '0%';
    setTimeout(() => {{ b.style.width = w; }}, 300);
  }});
}});
"""


def _build_chart_data(summary: dict) -> dict:
    return summary.get("by_category", {})


def _clean_email(email_str: str) -> str:
    """Extract just the email address from 'Name <email>' format."""
    match = re.search(r'<(.+?)>', email_str)
    if match:
        return match.group(1)
    return email_str[:30]
