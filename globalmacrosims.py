import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

################################################################################
#                             GAME 1 (Independent Bets)                        #
################################################################################

def simulate_trading_period_game1(
    hit_rate: float,
    slugging_ratio: float,
    max_loss: float,
    trades_per_year: int,
    years: int = 5
):
    """
    Single 5-year trading period for Game #1 (Independent Bets).
    Returns:
      annual_profits, max_drawdown, equity_curve,
      trade_outcomes, trade_holding_periods, positions_count_curve
      (though in Game 1 we only use some of these for final charts).
    """
    max_win = slugging_ratio * max_loss
    
    annual_profits = []
    cumulative_profit = 0.0
    equity_curve = []
    trade_outcomes = []
    
    # We'll define a constant holding period for each trade (just for completeness).
    holding_period_for_one_trade = 252.0 / trades_per_year
    trade_holding_periods = []
    
    positions_count_curve = []

    for _ in range(years):
        year_profit = 0.0
        for _ in range(trades_per_year):
            if np.random.rand() < hit_rate:
                outcome = np.random.uniform(0, max_win)
            else:
                outcome = np.random.uniform(-max_loss, 0)
            
            trade_outcomes.append(outcome)
            trade_holding_periods.append(holding_period_for_one_trade)
            positions_count_curve.append(1)
            
            year_profit += outcome
            cumulative_profit += outcome
            equity_curve.append(cumulative_profit)
        
        annual_profits.append(year_profit)
    
    # Compute max drawdown
    running_max = -np.inf
    max_drawdown = 0.0
    for val in equity_curve:
        if val > running_max:
            running_max = val
        dd = running_max - val
        if dd > max_drawdown:
            max_drawdown = dd
    
    return (
        annual_profits,
        max_drawdown,
        equity_curve,
        trade_outcomes,
        trade_holding_periods,
        positions_count_curve
    )

def run_monte_carlo_game1(
    hit_rate: float,
    slugging_ratio: float,
    max_loss: float,
    trades_per_year: int,
    num_simulations: int = 10_000,
    years: int = 5,
    sample_paths: int = 20
):
    """
    Runs multiple 5-year simulations for Game #1 (Independent Bets).
    Collects:
      - avg_annual_pnl
      - max_drawdown
      - annualized portfolio vol (from the 5 annual returns)
      - annualized Sharpe (mean/std of those 5 annual returns)
      - sample equity for up to sample_paths
    """
    all_avg_pnl = np.zeros(num_simulations)
    all_max_dd = np.zeros(num_simulations)
    all_annual_vols = np.zeros(num_simulations)
    all_annual_srs = np.zeros(num_simulations)
    
    equity_curves = []

    for i in range(num_simulations):
        (
            annual_profits,
            max_dd,
            eq_curve,
            trade_outcomes,
            holding_periods,
            pos_curve
        ) = simulate_trading_period_game1(
            hit_rate, slugging_ratio, max_loss, trades_per_year, years
        )
        
        total_5yr = sum(annual_profits)
        avg_annual = total_5yr / years
        
        all_avg_pnl[i] = avg_annual
        all_max_dd[i] = max_dd
        
        # annual vol from 5 data points
        ann_std = np.std(annual_profits, ddof=1)
        if ann_std == 0:
            sr = 0.0
        else:
            sr = (np.mean(annual_profits) / ann_std)
        all_annual_vols[i] = ann_std
        all_annual_srs[i] = sr
        
        if i < sample_paths:
            equity_curves.append(eq_curve)
    
    return {
        "avg_annual_pnl": all_avg_pnl,
        "max_drawdown": all_max_dd,
        "portfolio_annual_vols": all_annual_vols,
        "annual_SR": all_annual_srs,
        "equity_curves": equity_curves
    }

def compare_two_scenarios_game1(
    scenario1_params: dict,
    scenario2_params: dict,
    num_simulations: int = 10000,
    years: int = 5,
    sample_paths: int = 20
):
    """
    Compare two parameter sets for Game #1.
    We only plot:
      1) Avg Annual PnL
      2) Max Drawdown
      3) Annual Vol
      4) Annual Sharpe
    Then separate equity curves for scenario #1, scenario #2.
    """
    results1 = run_monte_carlo_game1(
        scenario1_params["hit_rate"],
        scenario1_params["slugging_ratio"],
        scenario1_params["max_loss"],
        scenario1_params["trades_per_year"],
        num_simulations,
        years,
        sample_paths
    )
    results2 = run_monte_carlo_game1(
        scenario2_params["hit_rate"],
        scenario2_params["slugging_ratio"],
        scenario2_params["max_loss"],
        scenario2_params["trades_per_year"],
        num_simulations,
        years,
        sample_paths
    )
    
    # 1) Avg Annual PnL
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["avg_annual_pnl"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["avg_annual_pnl"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["avg_annual_pnl"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["avg_annual_pnl"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Avg Annual PnL")
    ax.set_xlabel("Average Annual PnL")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    # 2) Max Drawdown
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["max_drawdown"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["max_drawdown"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    
    ax.hist(results2["max_drawdown"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["max_drawdown"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Max Drawdown")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    # 3) Annual Vol
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["portfolio_annual_vols"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["portfolio_annual_vols"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    
    ax.hist(results2["portfolio_annual_vols"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["portfolio_annual_vols"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Annual Volatility")
    ax.set_xlabel("Annual Vol (std of yearly returns)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    # 4) Annual Sharpe
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["annual_SR"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["annual_SR"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    
    ax.hist(results2["annual_SR"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["annual_SR"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Annual Sharpe Ratios")
    ax.set_xlabel("Annual Sharpe")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    # Separate Equity Curves for Scenario #1
    st.subheader("Sample Equity Curves - Scenario #1")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ec in results1["equity_curves"]:
        ax.plot(ec, color="red", alpha=0.7)
    ax.set_title("Scenario #1 Equity Curves")
    ax.set_xlabel("Trade Step")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Separate Equity Curves for Scenario #2
    st.subheader("Sample Equity Curves - Scenario #2")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ec in results2["equity_curves"]:
        ax.plot(ec, color="blue", alpha=0.7)
    ax.set_title("Scenario #2 Equity Curves")
    ax.set_xlabel("Trade Step")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


################################################################################
#                        GAME 2 (Market Correlation, Updated)                  #
################################################################################

def simulate_portfolio_game2(
    num_runs=10000, years=5, F=5, rho=0.3, 
    stop_size_bps=100, annual_IR=1.0, 
    initial_capital=0, sample_paths=20
):
    """
    Game #2: Bets with Market Correlation & trailing stops.
    - daily_vol = stop_size_bps / 4
    - daily_drift = (annual_IR / 16) * daily_vol
    - trailing stop = 4 * daily_vol = stop_size_bps
    Returns dict including 'hit_rate' and 'slugging_ratio'.
    """
    trading_days = int(252 * years)
    final_pnl = np.zeros(num_runs)
    avg_annual_pnl = np.zeros(num_runs)
    max_drawdown = np.zeros(num_runs)
    equity_curves = []
    positions_count_curves = []
    all_trade_outcomes = []
    all_holding_periods = []
    portfolio_annual_vols = np.zeros(num_runs)
    annual_SR_all = []
    
    # compute daily vol/drift from user inputs
    daily_vol = stop_size_bps / 4.0
    daily_drift = (annual_IR / 16.0) * daily_vol
    
    p_new_trade = 1.0 / F
    
    for run in range(num_runs):
        equity = initial_capital
        peak_equity = initial_capital
        max_dd_run = 0.0
        active_trades = []
        trade_outcomes_run = []
        daily_changes = []
        
        if run < sample_paths:
            equity_path = np.zeros(trading_days + 1)
            positions_path = np.zeros(trading_days + 1)
            equity_path[0] = equity
            positions_path[0] = 0
        
        for day in range(1, trading_days + 1):
            # Possibly start new trade
            if np.random.rand() < p_new_trade:
                active_trades.append({'pnl': 0.0, 'peak': 0.0, 'start_day': day})
            
            common_shock = np.random.randn()
            daily_portfolio_change = 0.0
            trades_to_close = []
            
            for i, trade in enumerate(active_trades):
                prev_pnl = trade['pnl']
                idio_shock = np.random.randn()
                shock = math.sqrt(rho)*common_shock + math.sqrt(1-rho)*idio_shock
                
                new_pnl = prev_pnl + daily_drift + (daily_vol * shock)
                
                if new_pnl > trade['peak']:
                    trade['peak'] = new_pnl
                
                stop_level = trade['peak'] - (4.0 * daily_vol)  # trailing stop
                if new_pnl <= stop_level:
                    final_pnl_trade = stop_level
                    trade_outcomes_run.append(final_pnl_trade)
                    holding_period = day - trade['start_day']
                    all_holding_periods.append(holding_period)
                    daily_portfolio_change += (final_pnl_trade - prev_pnl)
                    trades_to_close.append(i)
                else:
                    trade['pnl'] = new_pnl
                    daily_portfolio_change += (new_pnl - prev_pnl)
            
            # Remove trades that got stopped out
            for j in reversed(trades_to_close):
                active_trades.pop(j)
            
            equity += daily_portfolio_change
            daily_changes.append(daily_portfolio_change)
            
            if equity > peak_equity:
                peak_equity = equity
            dd = peak_equity - equity
            if dd > max_dd_run:
                max_dd_run = dd
            
            if run < sample_paths:
                equity_path[day] = equity
                positions_path[day] = len(active_trades)
        
        # force-close remaining trades
        for trade in active_trades:
            trade_outcomes_run.append(trade['pnl'])
            holding_period = trading_days - trade['start_day'] + 1
            all_holding_periods.append(holding_period)
            equity += trade['pnl']
        
        final_pnl_run = equity - initial_capital
        final_pnl[run] = final_pnl_run
        avg_annual_pnl[run] = final_pnl_run / years
        max_drawdown[run] = max_dd_run
        
        all_trade_outcomes.extend(trade_outcomes_run)
        
        # portfolio annual vol
        if len(daily_changes) > 0:
            daily_vol_ = np.std(daily_changes, ddof=1)
            annual_vol = daily_vol_ * np.sqrt(252)
        else:
            annual_vol = 0.0
        portfolio_annual_vols[run] = annual_vol
        
        # Sharpe per-year
        daily_changes_arr = np.array(daily_changes)
        days_per_year = 252
        sr_list_this_run = []
        for y in range(years):
            start_idx = y * days_per_year
            end_idx = (y+1)*days_per_year
            if end_idx <= len(daily_changes_arr):
                block = daily_changes_arr[start_idx:end_idx]
                if len(block) > 0:
                    md = np.mean(block)
                    sd = np.std(block, ddof=1)
                    sr = (md * 252)/(sd*np.sqrt(252)) if sd>0 else 0.0
                    sr_list_this_run.append(sr)
        annual_SR_all.extend(sr_list_this_run)
        
        if run < sample_paths:
            equity_curves.append(equity_path)
            positions_count_curves.append(positions_path)
    
    trade_outcomes_arr = np.array(all_trade_outcomes)
    total_trades = len(trade_outcomes_arr)
    if total_trades > 0:
        hit_rate = np.mean(trade_outcomes_arr > 0)
        wins = trade_outcomes_arr[trade_outcomes_arr > 0]
        losses = trade_outcomes_arr[trade_outcomes_arr < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else np.nan
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else np.nan
        if avg_loss and avg_loss != 0 and not np.isnan(avg_loss):
            slugging_ratio = avg_win / avg_loss
        else:
            slugging_ratio = np.nan
    else:
        hit_rate = np.nan
        slugging_ratio = np.nan

    return {
        'avg_annual_pnl': avg_annual_pnl,
        'max_drawdown': max_drawdown,
        'trade_outcomes': trade_outcomes_arr,
        'trade_holding_periods': np.array(all_holding_periods),
        'portfolio_annual_vols': portfolio_annual_vols,
        'annual_SR': np.array(annual_SR_all),
        'equity_curves': equity_curves,
        'positions_count_curves': positions_count_curves,
        'hit_rate': hit_rate,
        'slugging_ratio': slugging_ratio
    }

def compare_two_scenarios_game2(
    scenario1_params: dict,
    scenario2_params: dict,
    num_runs: int = 5000,
    years: int = 5,
    sample_paths: int = 20
):
    """
    Compare two parameter sets for Game #2, now also displaying
    'hit_rate' and 'slugging_ratio' for each scenario.
    """
    results1 = simulate_portfolio_game2(
        num_runs=num_runs,
        years=years,
        F=scenario1_params["F"],
        rho=scenario1_params["rho"],
        stop_size_bps=scenario1_params["stop_size_bps"],
        annual_IR=scenario1_params["annual_IR"],
        initial_capital=scenario1_params.get("initial_capital", 0),
        sample_paths=sample_paths
    )
    results2 = simulate_portfolio_game2(
        num_runs=num_runs,
        years=years,
        F=scenario2_params["F"],
        rho=scenario2_params["rho"],
        stop_size_bps=scenario2_params["stop_size_bps"],
        annual_IR=scenario2_params["annual_IR"],
        initial_capital=scenario2_params.get("initial_capital", 0),
        sample_paths=sample_paths
    )
    
    # 1) Avg Annual PnL
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["avg_annual_pnl"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["avg_annual_pnl"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["avg_annual_pnl"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["avg_annual_pnl"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Avg Annual PnL")
    ax.set_xlabel("Average Annual PnL (bps)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 2) Max Drawdown
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["max_drawdown"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["max_drawdown"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["max_drawdown"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["max_drawdown"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Max Drawdown")
    ax.set_xlabel("Max Drawdown (bps)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 3) Trade-Level PnL
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["trade_outcomes"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["trade_outcomes"]) if len(results1["trade_outcomes"])>0 else np.nan
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["trade_outcomes"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["trade_outcomes"]) if len(results2["trade_outcomes"])>0 else np.nan
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Trade-Level PnL")
    ax.set_xlabel("Trade PnL (bps)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    # 4) Holding Periods
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["trade_holding_periods"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["trade_holding_periods"]) if len(results1["trade_holding_periods"])>0 else np.nan
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["trade_holding_periods"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["trade_holding_periods"]) if len(results2["trade_holding_periods"])>0 else np.nan
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Holding Periods")
    ax.set_xlabel("Holding Period (days)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 5) Annual Vol
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["portfolio_annual_vols"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["portfolio_annual_vols"])
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["portfolio_annual_vols"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["portfolio_annual_vols"])
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Annualized Volatility")
    ax.set_xlabel("Annual Vol (bps)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # 6) Annual Sharpe
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(results1["annual_SR"], bins=50, alpha=0.5, color="red", label="Scenario 1")
    mean1 = np.mean(results1["annual_SR"]) if len(results1["annual_SR"])>0 else np.nan
    ax.axvline(mean1, color="red", linestyle='--', linewidth=2, label=f"S1 Mean: {mean1:.2f}")
    ax.hist(results2["annual_SR"], bins=50, alpha=0.5, color="blue", label="Scenario 2")
    mean2 = np.mean(results2["annual_SR"]) if len(results2["annual_SR"])>0 else np.nan
    ax.axvline(mean2, color="blue", linestyle='--', linewidth=2, label=f"S2 Mean: {mean2:.2f}")
    ax.set_title("Comparison of Annual Sharpe Ratios")
    ax.set_xlabel("Annual Sharpe")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # Scenario #1 equity curves
    st.subheader("Sample Equity Curves - Scenario #1")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ec in results1["equity_curves"]:
        ax.plot(ec, color="red", alpha=0.7)
    ax.set_title("Scenario #1 Equity Curves")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative PnL (bps)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Scenario #2 equity curves
    st.subheader("Sample Equity Curves - Scenario #2")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ec in results2["equity_curves"]:
        ax.plot(ec, color="blue", alpha=0.7)
    ax.set_title("Scenario #2 Equity Curves")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative PnL (bps)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Positions Count - Scenario #1
    st.subheader("Active Positions Over Time - Scenario #1 (First Sample Only)")
    if len(results1["positions_count_curves"]) > 0:
        pc_s1 = results1["positions_count_curves"][0]
        mean_s1 = np.mean(pc_s1)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(pc_s1, color="red", alpha=0.7, label="Positions")
        ax.axhline(mean_s1, color="red", linestyle='--', linewidth=2, label=f"Avg = {mean_s1:.2f}")
        ax.set_title("Scenario #1 Positions (Sample #1)")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Number of Active Positions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Positions Count - Scenario #2
    st.subheader("Active Positions Over Time - Scenario #2 (First Sample Only)")
    if len(results2["positions_count_curves"]) > 0:
        pc_s2 = results2["positions_count_curves"][0]
        mean_s2 = np.mean(pc_s2)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(pc_s2, color="blue", alpha=0.7, label="Positions")
        ax.axhline(mean_s2, color="blue", linestyle='--', linewidth=2, label=f"Avg = {mean_s2:.2f}")
        ax.set_title("Scenario #2 Positions (Sample #1)")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Number of Active Positions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # --- DISPLAY HIT RATE & SLUGGING RATIO ---
    st.write("---")
    st.write("### Scenario #1: Hit Rate and Slugging Ratio")
    st.write(f"- **Hit Rate**: {results1['hit_rate']:.2%}")
    st.write(f"- **Slugging Ratio** (avg win / avg loss): {results1['slugging_ratio']:.3f}")
    
    st.write("### Scenario #2: Hit Rate and Slugging Ratio")
    st.write(f"- **Hit Rate**: {results2['hit_rate']:.2%}")
    st.write(f"- **Slugging Ratio** (avg win / avg loss): {results2['slugging_ratio']:.3f}")


################################################################################
#                               STREAMLIT PAGES                                #
################################################################################

def run_game1_page():
    st.title("Game 1: Independent Bets")
    st.write("Enter parameters for **two scenarios** below, then click 'Run Comparison'.")

    st.subheader("Scenario #1 Parameters")
    hit_rate_1 = st.slider("Hit Rate (Scenario 1)", 0.0, 1.0, 0.50, 0.01)
    slug_1 = st.number_input("Slugging Ratio (Scenario 1)", value=1.3, format="%.3f")
    max_loss_1 = st.number_input("Max Loss (Scenario 1)", value=5.0, format="%.3f")
    tpy_1 = st.number_input("Trades per Year (Scenario 1)", value=26, step=1)

    st.subheader("Scenario #2 Parameters")
    hit_rate_2 = st.slider("Hit Rate (Scenario 2)", 0.0, 1.0, 0.50, 0.01)
    slug_2 = st.number_input("Slugging Ratio (Scenario 2)", value=1.3, format="%.3f")
    max_loss_2 = st.number_input("Max Loss (Scenario 2)", value=2.5, format="%.3f")
    tpy_2 = st.number_input("Trades per Year (Scenario 2)", value=52, step=1)

    num_sims = st.number_input("Number of Simulations", value=5000, step=1000)
    years = st.number_input("Number of Years", value=5, step=1)
    sample_paths = st.number_input("Sample Paths to Plot", value=10, step=1)

    if st.button("Run Comparison"):
        scenario1 = {
            "hit_rate": hit_rate_1,
            "slugging_ratio": slug_1,
            "max_loss": max_loss_1,
            "trades_per_year": int(tpy_1),
        }
        scenario2 = {
            "hit_rate": hit_rate_2,
            "slugging_ratio": slug_2,
            "max_loss": max_loss_2,
            "trades_per_year": int(tpy_2),
        }
        compare_two_scenarios_game1(
            scenario1, scenario2,
            num_simulations=int(num_sims),
            years=int(years),
            sample_paths=int(sample_paths)
        )

def run_game2_page():
    st.title("Game 2: Bets with Market Correlation")
    st.write("Stop size (bps) => daily_vol = stop_size_bps/4, trailing stop = 4 * daily_vol.")
    st.write("Annual IR => daily_drift = (annual_IR/16)*daily_vol.")

    st.subheader("Scenario #1 Parameters")
    F_1 = st.number_input("Average Frequency (F) [Scenario 1]", value=5, step=1)
    rho_1 = st.slider("Correlation (rho) [Scenario 1]", 0.0, 1.0, 0.3, 0.01)
    stop_bps_1 = st.number_input("Stop Size (bps) [Scenario 1]", value=100, step=1)
    ir_1 = st.number_input("Annual IR (Scenario 1)", value=1.0, format="%.3f")
    initcap_1 = st.number_input("Initial Capital [Scenario 1]", value=0.0, format="%.2f")

    st.subheader("Scenario #2 Parameters")
    F_2 = st.number_input("Average Frequency (F) [Scenario 2]", value=10, step=1)
    rho_2 = st.slider("Correlation (rho) [Scenario 2]", 0.0, 1.0, 0.2, 0.01)
    stop_bps_2 = st.number_input("Stop Size (bps) [Scenario 2]", value=200, step=1)
    ir_2 = st.number_input("Annual IR (Scenario 2)", value=0.5, format="%.3f")
    initcap_2 = st.number_input("Initial Capital [Scenario 2]", value=0.0, format="%.2f")

    num_sims = st.number_input("Number of Simulations", value=5000, step=1000)
    years = st.number_input("Number of Years", value=5, step=1)
    sample_paths = st.number_input("Sample Paths to Plot", value=20, step=1)

    if st.button("Run Comparison"):
        scenario1 = {
            "F": F_1,
            "rho": rho_1,
            "stop_size_bps": stop_bps_1,
            "annual_IR": ir_1,
            "initial_capital": initcap_1
        }
        scenario2 = {
            "F": F_2,
            "rho": rho_2,
            "stop_size_bps": stop_bps_2,
            "annual_IR": ir_2,
            "initial_capital": initcap_2
        }
        compare_two_scenarios_game2(
            scenario1, scenario2,
            num_runs=int(num_sims),
            years=int(years),
            sample_paths=int(sample_paths)
        )

def main():
    st.sidebar.title("Select a Game")
    page_choice = st.sidebar.radio("Go to:", ["Game 1", "Game 2"])
    
    if page_choice == "Game 1":
        run_game1_page()
    else:
        run_game2_page()

if __name__ == "__main__":
    main()
