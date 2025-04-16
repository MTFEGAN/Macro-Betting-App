# two_game_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Single‑file Streamlit app with:
#   • Game 1  – Independent Bets
#   • Game 2  – Correlated Bets with trailing stops
# Game 2 now lets the user pick a stop‑multiple  m  ∈ {1,2,3,4,8,10} so that
#       stop_size_bps = m × daily_vol     ⇒     daily_vol = stop_size_bps / m
# The trailing stop equals the full stop_size_bps.
# All charts: Scenario‑1 = red, Scenario‑2 = blue.
# Hit‑Rate and Slugging‑Ratio are shown for Game 2.

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# ──────────────────────────────────── GAME 1 ─────────────────────────────────

def simulate_trading_period_game1(hit_rate, slugging_ratio, max_loss,
                                  trades_per_year, years=5):
    max_win = slugging_ratio * max_loss
    annual_profits, equity_curve, trade_outcomes = [], [], []
    trade_holding_periods, positions_count_curve = [], []
    cumulative = 0.0
    hp = 252.0 / trades_per_year           # constant holding period (days)

    for _ in range(years):
        year_pnl = 0.0
        for _ in range(trades_per_year):
            outcome = np.random.uniform(0, max_win) if np.random.rand() < hit_rate \
                     else np.random.uniform(-max_loss, 0)
            trade_outcomes.append(outcome)
            trade_holding_periods.append(hp)
            positions_count_curve.append(1)
            year_pnl += outcome
            cumulative += outcome
            equity_curve.append(cumulative)
        annual_profits.append(year_pnl)

    # max‑drawdown
    running_max, max_dd = -np.inf, 0.0
    for v in equity_curve:
        running_max = max(running_max, v)
        max_dd = max(max_dd, running_max - v)

    return (annual_profits, max_dd, equity_curve,
            trade_outcomes, trade_holding_periods, positions_count_curve)


def run_monte_carlo_game1(hit_rate, slugging_ratio, max_loss, trades_per_year,
                          num_sim=10_000, years=5, sample_paths=20):
    avg_pnl, max_dd, ann_vol, ann_sr = [], [], [], []
    eq_samples = []

    for i in range(num_sim):
        ann, dd, eq, *_ = simulate_trading_period_game1(
            hit_rate, slugging_ratio, max_loss, trades_per_year, years
        )
        avg_pnl.append(sum(ann) / years)
        max_dd.append(dd)
        sigma = np.std(ann, ddof=1)
        ann_vol.append(sigma)
        ann_sr.append(0.0 if sigma == 0 else np.mean(ann) / sigma)
        if i < sample_paths:
            eq_samples.append(eq)

    return dict(avg_annual_pnl=np.array(avg_pnl),
                max_drawdown=np.array(max_dd),
                portfolio_annual_vols=np.array(ann_vol),
                annual_SR=np.array(ann_sr),
                equity_curves=eq_samples)


def compare_two_scenarios_game1(p1, p2, num_sim=10_000, years=5, sample_paths=20):
    r1 = run_monte_carlo_game1(**p1, num_sim=num_sim, years=years, sample_paths=sample_paths)
    r2 = run_monte_carlo_game1(**p2, num_sim=num_sim, years=years, sample_paths=sample_paths)

    def hist(data1, data2, title, xlabel):
        fig, ax = plt.subplots()
        ax.hist(data1, 50, alpha=.5, color="red", label="Scenario 1")
        ax.hist(data2, 50, alpha=.5, color="blue", label="Scenario 2")
        m1, m2 = np.mean(data1), np.mean(data2)
        ax.axvline(m1, color="red", ls="--", lw=2, label=f"S1 mean {m1:.2f}")
        ax.axvline(m2, color="blue", ls="--", lw=2, label=f"S2 mean {m2:.2f}")
        ax.set(title=title, xlabel=xlabel, ylabel="Freq"); ax.legend()
        st.pyplot(fig)

    hist(r1["avg_annual_pnl"],  r2["avg_annual_pnl"],  "Avg Annual PnL", "PnL")
    hist(r1["max_drawdown"],    r2["max_drawdown"],    "Max Drawdown",   "Drawdown")
    hist(r1["portfolio_annual_vols"], r2["portfolio_annual_vols"],
         "Annual Volatility", "Vol (σ of yearly returns)")
    hist(r1["annual_SR"],       r2["annual_SR"],       "Annual Sharpe",  "Sharpe")

    # equity curves
    for label, curves, color in [("Scenario 1", r1["equity_curves"], "red"),
                                 ("Scenario 2", r2["equity_curves"], "blue")]:
        st.subheader(f"Sample Equity Curves – {label}")
        fig, ax = plt.subplots()
        for ec in curves: ax.plot(ec, color=color, alpha=.7)
        ax.set(xlabel="Trade Step", ylabel="Cumulative PnL", title=label)
        ax.grid(alpha=.3); st.pyplot(fig)

# ──────────────────────────────────── GAME 2 ─────────────────────────────────

def simulate_portfolio_game2(*, num_runs=10_000, years=5, F=5, rho=0.3,
                             stop_size_bps=100, stop_multiple=4,
                             annual_IR=1.0, initial_capital=0,
                             sample_paths=20):
    """Monte‑Carlo portfolio with trailing stops.
       stop_size_bps = m × daily_vol   ⇒  daily_vol = stop_size_bps / m
    """
    trading_days = int(252 * years)
    daily_vol = stop_size_bps / stop_multiple
    daily_drift = (annual_IR / 16) * daily_vol
    trailing_stop = stop_size_bps          # in bps

    p_new_trade = 1 / F
    final, avg_pnl, max_dd = np.zeros(num_runs), np.zeros(num_runs), np.zeros(num_runs)
    port_vol, ann_SR_all = np.zeros(num_runs), []
    eq_samples, pos_samples = [], []
    all_outcomes, all_hold = [], []

    for run in range(num_runs):
        equity = peak = initial_capital
        active, daily_changes = [], []
        if run < sample_paths:
            eq_path = np.zeros(trading_days + 1)
            pos_path = np.zeros(trading_days + 1)

        for day in range(1, trading_days + 1):
            if np.random.rand() < p_new_trade:
                active.append(dict(pnl=0.0, peak=0.0, start=day))

            common = np.random.randn()
            d_change, to_close = 0.0, []
            for i, t in enumerate(active):
                shock = math.sqrt(rho)*common + math.sqrt(1-rho)*np.random.randn()
                new = t['pnl'] + daily_drift + daily_vol*shock
                t['peak'] = max(t['peak'], new)
                if new <= t['peak'] - trailing_stop:
                    outcome = t['peak'] - trailing_stop
                    all_outcomes.append(outcome)
                    all_hold.append(day - t['start'])
                    d_change += outcome - t['pnl']
                    to_close.append(i)
                else:
                    d_change += new - t['pnl']
                    t['pnl'] = new
            for j in reversed(to_close):
                active.pop(j)

            equity += d_change
            daily_changes.append(d_change)
            peak = max(peak, equity)
            max_dd[run] = max(max_dd[run], peak - equity)

            if run < sample_paths:
                eq_path[day] = equity
                pos_path[day] = len(active)

        # close remaining trades
        for t in active:
            all_outcomes.append(t['pnl'])
            all_hold.append(trading_days - t['start'] + 1)
            equity += t['pnl']

        final[run] = equity - initial_capital
        avg_pnl[run] = final[run] / years

        sigma_d = np.std(daily_changes, ddof=1)
        port_vol[run] = sigma_d * np.sqrt(252)

        # per‑year Sharpe
        dc = np.array(daily_changes)
        for y in range(years):
            blk = dc[y*252:(y+1)*252]
            if blk.size:
                s = np.std(blk, ddof=1)
                ann_SR_all.append(0.0 if s == 0 else (np.mean(blk)*252)/(s*np.sqrt(252)))

        if run < sample_paths:
            eq_samples.append(eq_path)
            pos_samples.append(pos_path)

    outcomes = np.array(all_outcomes)
    wins, losses = outcomes[outcomes > 0], outcomes[outcomes < 0]
    hit_rate = np.nan if outcomes.size == 0 else wins.size / outcomes.size
    slug = np.nan
    if wins.size and losses.size:
        slug = np.mean(wins) / np.mean(np.abs(losses))

    return dict(avg_annual_pnl=avg_pnl, max_drawdown=max_dd,
                trade_outcomes=outcomes,
                trade_holding_periods=np.array(all_hold),
                portfolio_annual_vols=port_vol,
                annual_SR=np.array(ann_SR_all),
                equity_curves=eq_samples,
                positions_count_curves=pos_samples,
                hit_rate=hit_rate, slugging_ratio=slug)


def compare_two_scenarios_game2(s1, s2, num_runs=5000, years=5, sample_paths=20):
    r1 = simulate_portfolio_game2(**s1, num_runs=num_runs,
                                  years=years, sample_paths=sample_paths)
    r2 = simulate_portfolio_game2(**s2, num_runs=num_runs,
                                  years=years, sample_paths=sample_paths)

    def hist(d1, d2, title, xlabel):
        fig, ax = plt.subplots()
        ax.hist(d1, 50, alpha=.5, color="red")
        ax.hist(d2, 50, alpha=.5, color="blue")
        m1, m2 = np.mean(d1), np.mean(d2)
        ax.axvline(m1, color="red", ls="--", lw=2, label=f"S1 {m1:.2f}")
        ax.axvline(m2, color="blue", ls="--", lw=2, label=f"S2 {m2:.2f}")
        ax.set(title=title, xlabel=xlabel, ylabel="Freq"); ax.legend()
        st.pyplot(fig)

    hist(r1["avg_annual_pnl"],  r2["avg_annual_pnl"],  "Avg Annual PnL (bps)", "PnL")
    hist(r1["max_drawdown"],    r2["max_drawdown"],    "Max Drawdown (bps)",   "Drawdown")
    hist(r1["trade_outcomes"],  r2["trade_outcomes"],  "Trade‑level PnL (bps)", "PnL")
    hist(r1["trade_holding_periods"], r2["trade_holding_periods"],
         "Holding Period (days)", "Days")
    hist(r1["portfolio_annual_vols"], r2["portfolio_annual_vols"],
         "Annual Volatility (bps)", "Vol")
    hist(r1["annual_SR"],       r2["annual_SR"],       "Annual Sharpe",  "Sharpe")

    # equity curves
    for label, curves, color in [("Scenario 1", r1["equity_curves"], "red"),
                                 ("Scenario 2", r2["equity_curves"], "blue")]:
        st.subheader(f"Sample Equity Curves – {label}")
        fig, ax = plt.subplots()
        for ec in curves: ax.plot(ec, color=color, alpha=.7)
        ax.set(title=label, xlabel="Trading Day", ylabel="Cum PnL (bps)")
        ax.grid(alpha=.3); st.pyplot(fig)

    # positions (first sample)
    for label, pcs, color in [("Scenario 1", r1["positions_count_curves"], "red"),
                              ("Scenario 2", r2["positions_count_curves"], "blue")]:
        if pcs:
            pc = pcs[0]; mean = np.mean(pc)
            st.subheader(f"Active Positions – {label} (sample #1)")
            fig, ax = plt.subplots()
            ax.plot(pc, color=color, alpha=.7)
            ax.axhline(mean, color=color, ls="--", lw=2, label=f"Avg {mean:.2f}")
            ax.set(xlabel="Trading Day", ylabel="# Positions"); ax.legend()
            ax.grid(alpha=.3); st.pyplot(fig)

    # hit rate & slugging
    st.write("---")
    st.write("#### Scenario 1 stats")
    st.write(f"- Hit Rate: **{r1['hit_rate']:.2%}**")
    st.write(f"- Slugging Ratio: **{r1['slugging_ratio']:.3f}**")
    st.write("#### Scenario 2 stats")
    st.write(f"- Hit Rate: **{r2['hit_rate']:.2%}**")
    st.write(f"- Slugging Ratio: **{r2['slugging_ratio']:.3f}**")

# ────────────────────────────── STREAMLIT PAGES ─────────────────────────────

def run_game1_page():
    st.title("Game 1 – Independent Bets")
    st.subheader("Scenario 1")
    hit1 = st.slider("Hit Rate", 0.0, 1.0, .5, .01)
    slug1 = st.number_input("Slugging Ratio", value=1.3)
    loss1 = st.number_input("Max Loss", value=5.0)
    tpy1  = st.number_input("Trades per Year", value=26, step=1)
    st.subheader("Scenario 2")
    hit2 = st.slider("Hit Rate ", 0.0, 1.0, .5, .01, key="h2")
    slug2 = st.number_input("Slugging Ratio ", value=1.3, key="s2")
    loss2 = st.number_input("Max Loss ", value=2.5, key="l2")
    tpy2  = st.number_input("Trades per Year ", value=52, step=1, key="t2")
    sims  = st.number_input("# Simulations", value=5000, step=1000)
    yrs   = st.number_input("Years", value=5, step=1)
    paths = st.number_input("Sample Paths", value=10, step=1)

    if st.button("Run Comparison"):
        compare_two_scenarios_game1(
            dict(hit_rate=hit1, slugging_ratio=slug1, max_loss=loss1, trades_per_year=tpy1),
            dict(hit_rate=hit2, slugging_ratio=slug2, max_loss=loss2, trades_per_year=tpy2),
            num_sim=int(sims), years=int(yrs), sample_paths=int(paths)
        )

def run_game2_page():
    st.title("Game 2 – Correlated Bets with Trailing Stops")
    st.write("Stop size (bps) = **m × daily vol** (choose m).  "
             "Daily drift = (Annual IR / 16) × daily vol.")
    MULTS = [1,2,3,4,8,10]

    st.subheader("Scenario 1")
    F1   = st.number_input("Avg Frequency F", value=5, step=1)
    rho1 = st.slider("Correlation ρ", 0.0, 1.0, .3, .01)
    stop1= st.number_input("Stop Size (bps)", value=100, step=1)
    m1   = st.selectbox("Stop Multiple m", MULTS, index=3)
    ir1  = st.number_input("Annual IR", value=1.0)
    cap1 = st.number_input("Initial Capital", value=0.0)

    st.subheader("Scenario 2")
    F2   = st.number_input("Avg Frequency F ", value=10, step=1)
    rho2 = st.slider("Correlation ρ ", 0.0, 1.0, .2, .01)
    stop2= st.number_input("Stop Size (bps) ", value=200, step=1)
    m2   = st.selectbox("Stop Multiple m ", MULTS, index=5)
    ir2  = st.number_input("Annual IR ", value=0.5)
    cap2 = st.number_input("Initial Capital ", value=0.0)

    sims  = st.number_input("# Simulations", value=5000, step=1000)
    yrs   = st.number_input("Years", value=5, step=1)
    paths = st.number_input("Sample Paths", value=20, step=1)

    if st.button("Run Comparison"):
        compare_two_scenarios_game2(
            dict(F=F1, rho=rho1, stop_size_bps=stop1, stop_multiple=m1,
                 annual_IR=ir1, initial_capital=cap1),
            dict(F=F2, rho=rho2, stop_size_bps=stop2, stop_multiple=m2,
                 annual_IR=ir2, initial_capital=cap2),
            num_runs=int(sims), years=int(yrs), sample_paths=int(paths)
        )

def main():
    st.sidebar.title("Select Game")
    page = st.sidebar.radio("", ["Game 1", "Game 2"])
    run_game1_page() if page == "Game 1" else run_game2_page()

if __name__ == "__main__":
    main()

