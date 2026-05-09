"""
nba_algohub.py — NBA AlgoHub Prop Dashboard
Game-first navigation. Prop overlays. Hot/cold streaks. Overlay finder.

Run: python -m streamlit run nba_algohub.py
"""

import streamlit as st
import json
import os
import requests
from datetime import datetime

st.set_page_config(
    page_title="NBA AlgoHub",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:#06080f; --s1:#0c1018; --s2:#111827; --border:#1c2333;
    --green:#22c55e; --red:#ef4444; --amber:#f59e0b; --blue:#3b82f6;
    --cyan:#06b6d4; --purple:#a855f7; --text:#f1f5f9; --muted:#475569;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text);font-family:'DM Sans',sans-serif;}
.stApp{background:var(--bg);}
.block-container{padding:1rem 1.5rem!important;max-width:100%!important;}

.ah-logo{font-family:'Bebas Neue',sans-serif;font-size:2.2rem;letter-spacing:.15em;color:var(--text);}
.ah-logo span{color:#f59e0b;}
.ah-tagline{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted);letter-spacing:.2em;text-transform:uppercase;}

/* Game buttons */
.game-btn{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:8px 12px;text-align:center;cursor:pointer;}

/* Overlay cards */
.overlay-card{background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:4px;}
.overlay-card.over{border-left:3px solid var(--green)!important;}
.overlay-card.under{border-left:3px solid var(--red)!important;}

/* Player row */
.player-row{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin-bottom:6px;}

/* Prop pill */
.prop-over{background:#052e1633;color:#22c55e;border:1px solid #166534;padding:2px 8px;border-radius:4px;font-family:'DM Mono',monospace;font-size:.75rem;font-weight:600;}
.prop-under{background:#45060633;color:#fca5a5;border:1px solid #7f1d1d;padding:2px 8px;border-radius:4px;font-family:'DM Mono',monospace;font-size:.75rem;font-weight:600;}
.prop-neutral{background:#0f172a;color:#475569;border:1px solid #1e293b;padding:2px 8px;border-radius:4px;font-family:'DM Mono',monospace;font-size:.75rem;}

/* Hot/cold */
.hot{color:#ef4444;font-size:.75rem;}
.cold{color:#60a5fa;font-size:.75rem;}
.neutral-form{color:#475569;font-size:.75rem;}

/* Stat tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--s1)!important;border-radius:8px;padding:4px;gap:2px;}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif!important;font-size:.82rem!important;font-weight:500!important;color:var(--muted)!important;border-radius:6px!important;}
.stTabs [aria-selected="true"]{background:var(--s2)!important;color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    path = "data/nba_today.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("date") == datetime.today().strftime("%Y-%m-%d"):
            return data
        return data  # show even if stale during offseason
    except:
        return None


def form_icon(form):
    return {"hot": "🔥", "cold": "❄️", "neutral": "◆"}.get(form, "◆")


def fmt_odds(odds) -> str:
    """Safely format American odds value."""
    try:
        v = int(odds)
        return f"+{v}" if v > 0 else str(v)
    except:
        return "N/A"


def overlay_color(overlay, direction):
    if overlay is None:
        return "#475569"
    if direction == "Over":
        return "#22c55e" if abs(overlay) >= 3 else "#86efac"
    else:
        return "#ef4444" if abs(overlay) >= 3 else "#fca5a5"


def render_prop_row(prop, show_line=True):
    """Render a single prop stat row."""
    proj      = prop.get("projection", 0)
    s_avg     = prop.get("season_avg", 0)
    r_avg     = prop.get("rolling_avg", 0)
    line      = prop.get("line")
    overlay   = prop.get("overlay")
    direction = prop.get("direction")
    edge_pct  = prop.get("edge_pct")
    over_odds = prop.get("over_odds", -110)
    stat      = prop.get("stat", "")
    form      = prop.get("form", "neutral")

    # Overlay display
    if overlay is not None and line is not None:
        oc = overlay_color(overlay, direction)
        sign = "▲" if direction == "Over" else "▼"
        overlay_str = f'<span style="color:{oc};font-weight:700">{sign}{abs(overlay):.1f}</span>'
        direction_str = f'<span class="prop-{"over" if direction=="Over" else "under"}">{direction} {line}</span>'
        odds_str = fmt_odds(over_odds)
    else:
        overlay_str = '<span style="color:#475569">No line</span>'
        direction_str = '<span class="prop-neutral">—</span>'
        odds_str = "N/A"

    form_str = form_icon(form)

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:5px 8px;background:#0c1018;border-radius:6px;margin-bottom:3px;">
        <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:#94a3b8;min-width:70px">{stat}</span>
        <span style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;color:#f1f5f9;min-width:35px">{proj:.1f}</span>
        <span style="font-size:.7rem;color:#475569;min-width:80px">avg {s_avg:.1f} · L7 {r_avg:.1f} {form_str}</span>
        <span style="min-width:100px">{direction_str}</span>
        <span style="min-width:60px">{overlay_str}</span>
        <span style="font-family:'DM Mono',monospace;font-size:.72rem;color:#f59e0b">{odds_str}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    # Header
    col1, col2 = st.columns([4, 2])
    with col1:
        st.markdown('<div class="ah-logo">NBA ALGO<span>HUB</span></div><div class="ah-tagline">Prop Intelligence · @TheAlgoHub</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:.8rem;color:#475569;padding-top:14px;text-align:right">{datetime.today().strftime("%A, %B %d %Y")}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    data = load_data()

    if not data:
        st.error("No data found. Run `python nba_daily_run.py` first.")
        st.info("This generates today's prop projections and pushes to GitHub.")
        return

    games   = data.get("games", [])
    top_ovr = data.get("top_overlays", [])
    date    = data.get("date", "")

    # Stats bar
    total_players  = sum(len(g["players"]) for g in games)
    total_overlays = len(top_ovr)
    hot_count      = sum(1 for p in top_ovr if p.get("form") == "hot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Date", date)
    c2.metric("🏀 Games", len(games))
    c3.metric("👤 Players", total_players)
    c4.metric("📈 Overlays 1.5+", total_overlays)

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏀 Game Props", "📈 Overlay Finder", "🎰 Parlay Builder"])

    # ── TAB 1: Game Props ──────────────────────────────────────────────────────
    with tab1:
        if not games:
            st.warning("No games today.")
        else:
            # Game selector
            st.markdown("#### TODAY'S SLATE")
            game_cols = st.columns(min(len(games), 6))
            selected_idx = st.session_state.get("nba_game", 0)

            for i, g in enumerate(games):
                with game_cols[i % len(game_cols)]:
                    if st.button(f"{g['away_team']} @ {g['home_team']}", key=f"nbagame_{i}"):
                        st.session_state["nba_game"] = i
                        selected_idx = i

            st.divider()

            g = games[selected_idx]
            st.markdown(f"""
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:.1em;margin-bottom:12px">
                {g['away_team']} @ {g['home_team']}
            </div>
            """, unsafe_allow_html=True)

            # Stat filter
            stat_filter = st.selectbox("Filter by stat", ["All", "Points", "Rebounds", "Assists"], index=0)

            # Color legend
            st.markdown("""
            <div style="display:flex;gap:16px;padding:6px 10px;background:#0c1018;border:1px solid #1c2333;border-radius:6px;margin-bottom:12px;font-size:.7rem;flex-wrap:wrap;">
                <span style="color:#475569">OVERLAY:</span>
                <span><span style="color:#22c55e">▲</span> Over projection</span>
                <span><span style="color:#ef4444">▼</span> Under projection</span>
                <span style="color:#475569;margin-left:8px">FORM:</span>
                <span>🔥 Hot (L7 > avg 15%+)</span>
                <span>❄️ Cold (L7 < avg 15%+)</span>
                <span>◆ Neutral</span>
            </div>
            """, unsafe_allow_html=True)

            # Column headers
            st.markdown("""
            <div style="display:flex;gap:12px;padding:4px 8px;font-family:'DM Mono',monospace;font-size:.65rem;color:#475569;">
                <span style="min-width:70px">STAT</span>
                <span style="min-width:35px">PROJ</span>
                <span style="min-width:80px">AVG · L7</span>
                <span style="min-width:100px">LINE</span>
                <span style="min-width:60px">EDGE</span>
                <span>ODDS</span>
            </div>
            """, unsafe_allow_html=True)

            players = g.get("players", [])
            if not players:
                st.info("No player data for this game.")
            else:
                for player in players:
                    props = player.get("props", [])
                    if stat_filter != "All":
                        props = [p for p in props if p["stat"] == stat_filter]
                    if not props:
                        continue

                    avg_overlay = player.get("avg_overlay", 0)
                    team = player.get("team", "")
                    opp  = player.get("opponent", "")

                    # Only show players with lines or significant projections
                    has_line = any(p.get("line") is not None for p in props)
                    if not has_line and avg_overlay < 0.5:
                        continue

                    st.markdown(f"""
                    <div class="player-row">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <div>
                                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.05rem">{player['player_name']}</span>
                                <span style="font-size:.72rem;color:#475569;margin-left:8px;font-family:'DM Mono',monospace">{team} vs {opp}</span>
                            </div>
                            <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:#f59e0b">avg overlay {avg_overlay:.1f}</span>
                        </div>
                    """, unsafe_allow_html=True)

                    for prop in props:
                        render_prop_row(prop)

                    st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Overlay Finder ──────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 📈 Best Overlays Today")
        st.caption("Players where our projection differs most from the book line — ranked by edge size")

        if not top_ovr:
            st.info("Run `python nba_daily_run.py` to generate overlays. Requires Bovada prop lines.")
        else:
            # Filter controls
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                stat_filt = st.selectbox("Stat", ["All", "Points", "Rebounds", "Assists"], key="ovr_stat")
            with col_f2:
                dir_filt = st.selectbox("Direction", ["All", "Over", "Under"], key="ovr_dir")
            with col_f3:
                form_filt = st.selectbox("Form", ["All", "Hot", "Cold", "Neutral"], key="ovr_form")

            filtered = top_ovr
            if stat_filt != "All":
                filtered = [x for x in filtered if x["stat"] == stat_filt]
            if dir_filt != "All":
                filtered = [x for x in filtered if x.get("direction") == dir_filt]
            if form_filt != "All":
                filtered = [x for x in filtered if x.get("form", "").title() == form_filt]

            if not filtered:
                st.info("No overlays match your filters.")
            else:
                for o in filtered:
                    overlay   = o.get("overlay", 0)
                    direction = o.get("direction", "Over")
                    oc        = overlay_color(overlay, direction)
                    sign      = "▲" if direction == "Over" else "▼"
                    form      = o.get("form", "neutral")
                    form_str  = form_icon(form)
                    line      = o.get("line")
                    odds      = o.get("over_odds") if direction == "Over" else o.get("under_odds", -110)
                    odds_str = fmt_odds(odds)
                    edge_pct  = o.get("edge_pct", 0)

                    st.markdown(f"""
                    <div class="overlay-card {'over' if direction=='Over' else 'under'}">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.05rem">{o['player_name']}</span>
                                <span style="font-size:.72rem;color:#475569;margin-left:8px">{o['team']} vs {o['opponent']} · {o['game']}</span>
                            </div>
                            <div style="text-align:right">
                                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:{oc}">{sign}{abs(overlay):.1f}</span>
                                <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:#f59e0b;margin-left:10px">{odds_str}</span>
                            </div>
                        </div>
                        <div style="margin-top:4px;font-family:'DM Mono',monospace;font-size:.72rem;color:#94a3b8;">
                            {o['stat']} · Proj <strong style="color:#f1f5f9">{o['projection']:.1f}</strong> 
                            vs line <strong style="color:#f1f5f9">{line}</strong> · 
                            avg {o['season_avg']:.1f} · L7 {o['rolling_avg']:.1f} {form_str} ·
                            edge {edge_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 3: Parlay Builder ──────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 🎰 Prop Parlay Builder")
        st.caption("Pick legs from different games")

        if not top_ovr:
            st.info("No overlay data available.")
        else:
            options = [
                f"{o['player_name']} {o['stat']} {o['direction']} {o['line']} ({o['game']})"
                for o in top_ovr if o.get("line") is not None
            ]
            selected = st.multiselect("Add legs", options, max_selections=4)

            if selected:
                legs = [top_ovr[options.index(s)] for s in selected if s in options]

                # Same game check
                games_used = [l["game"] for l in legs]
                if len(games_used) != len(set(games_used)):
                    st.warning("⚠️ Multiple legs from the same game — consider spreading across games")

                for leg in legs:
                    direction = leg.get("direction", "Over")
                    overlay   = leg.get("overlay", 0)
                    oc        = overlay_color(overlay, direction)
                    sign      = "▲" if direction == "Over" else "▼"
                    odds      = leg.get("over_odds") if direction == "Over" else leg.get("under_odds", -110)
                    odds_str = fmt_odds(odds)

                    st.markdown(f"""
                    <div style="background:#0c1018;border:1px solid #1c2333;border-radius:8px;padding:10px 14px;margin-bottom:6px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="font-family:'Bebas Neue',sans-serif;font-size:1rem">{leg['player_name']}</span>
                                <span style="font-size:.72rem;color:#475569;margin-left:8px">{leg['stat']} · {leg['game']}</span>
                            </div>
                            <div>
                                <span class="prop-{'over' if direction=='Over' else 'under'}">{direction} {leg['line']}</span>
                                <span style="color:{oc};font-family:'DM Mono',monospace;margin-left:8px">{sign}{abs(overlay):.1f}</span>
                                <span style="color:#f59e0b;font-family:'DM Mono',monospace;margin-left:8px">{odds_str}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Parlay odds estimate
                if len(legs) >= 2:
                    combined = 1.0
                    for leg in legs:
                        odds_raw = leg.get("over_odds") if leg.get("direction") == "Over" else leg.get("under_odds", -110)
                        if odds_raw and odds_raw > 0:
                            prob = 100 / (odds_raw + 100)
                        elif odds_raw:
                            prob = abs(odds_raw) / (abs(odds_raw) + 100)
                        else:
                            prob = 0.524  # -110 implied
                        combined *= prob

                    impl = (1 / combined) - 1
                    amer = int(impl * 100) if impl >= 1 else int(-100 / impl)
                    odds_disp = f"+{amer:,}" if amer > 0 else str(amer)

                    st.markdown(f"""
                    <div style="background:#052e16;border:1px solid #166534;border-radius:10px;padding:16px;margin-top:12px;text-align:center;">
                        <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:#86efac;letter-spacing:.15em">ESTIMATED PARLAY ODDS</div>
                        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.2rem;color:#22c55e;letter-spacing:.05em">{odds_disp}</div>
                        <div style="font-size:.72rem;color:#4a5568;margin-top:4px">Combined: {combined*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Twitter caption
                    legs_text = "\n".join([
                        f"{l['player_name']} {l['stat']} {l['direction']} {l['line']} ✅"
                        for l in legs
                    ])
                    tweet = f"🏀 NBA PROP PARLAY\n\n{legs_text}\n\n{odds_disp} 🔒 AlgoHub locked in\n\n@TheAlgoHub | #NBAProps"
                    st.markdown("#### Twitter Caption")
                    st.code(tweet, language=None)

    st.divider()
    st.markdown('<div style="text-align:center;font-family:DM Mono,monospace;font-size:.7rem;color:#1c2333">NBA ALGOHUB · @TheAlgoHub · Powered by NBA Stats API</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
