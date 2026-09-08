from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from itertools import product
from wfo_slicer import generate_wfo_windows

def calculate_window_metrics(pnl_series, inventory_series, fills_data, initial_cash=100_000.0):
    """
    Computes all standard risk, inventory, and performance metrics for a slice.
    """
    final_net_pnl = float(pnl_series[-1]) if len(pnl_series) > 0 else 0.0
    portfolio_value = initial_cash + final_net_pnl
    
    # Calculate Drawdown
    wealth_curve = initial_cash + pnl_series
    peak = np.maximum.accumulate(wealth_curve)
    drawdowns = wealth_curve - peak
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0  # Non-positive float
    
    # Inventory metrics
    abs_inv = np.abs(inventory_series)
    avg_abs_inv = float(np.mean(abs_inv)) if len(abs_inv) > 0 else 0.0
    max_abs_inv = float(np.max(abs_inv)) if len(abs_inv) > 0 else 0.0
    ending_inv = float(inventory_series[-1]) if len(inventory_series) > 0 else 0.0
    
    # Score requested by teammate
    risk_adjusted_score = final_net_pnl + (0.5 * max_drawdown) - (100.0 * avg_abs_inv)
    
    total_fills = fills_data['total_fills']
    bid_fills = fills_data['bid_fills']
    ask_fills = fills_data['ask_fills']
    fees_paid = fills_data['fees_paid']
    total_quotes = fills_data['total_quotes']
    fill_rate = (total_fills / total_quotes) if total_quotes > 0 else 0.0

    return {
        "final_net_pnl": final_net_pnl,
        "max_drawdown": max_drawdown,
        "avg_abs_inventory": avg_abs_inv,
        "max_abs_inventory": max_abs_inv,
        "ending_inventory": ending_inv,
        "risk_adjusted_score": risk_adjusted_score,
        "total_fills": total_fills,
        "bid_fills": bid_fills,
        "ask_fills": ask_fills,
        "fees_paid": fees_paid,
        "fill_rate": fill_rate,
        "final_portfolio_value": portfolio_value
    }

def run_wfo_experiment(
    parquet_path,
    model_name,
    model_version,
    model_simulation_func,
    param_grid,
    fee_rate=0.0002,
    starting_cash=100_000.0
):
    """
    Executes Walk-Forward Optimization across the dataset according to team specifications.
    """
    wfo_records = []
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    cumulative_oos_pnl = 0.0
    current_cash = starting_cash
    
    for window in generate_wfo_windows(parquet_path):
        wfo_step = window['wfo_step']
        df_train = window['df_train']
        df_test = window['df_test']
        
        # --- PHASE 1: In-Sample Grid Search (24-Hour Train) ---
        best_score = -np.inf
        best_params = None
        best_train_metrics = None
        
        for params in tqdm(param_combinations, desc="Testing Parameters", leave=False):
            pnl_arr, inv_arr, fills = model_simulation_func(
                df=df_train,
                params=params,
                fee_rate=fee_rate,
                starting_cash=starting_cash,
                starting_inv=0.0
            )
            metrics = calculate_window_metrics(pnl_arr, inv_arr, fills, initial_cash=starting_cash)
            
            if metrics["risk_adjusted_score"] > best_score:
                best_score = metrics["risk_adjusted_score"]
                best_params = params
                best_train_metrics = metrics
        
        # --- PHASE 2: Out-of-Sample Evaluation (6-Hour Test) ---
        # Freeze best_params and execute strictly on unseen df_test
        test_pnl_arr, test_inv_arr, test_fills = model_simulation_func(
            df=df_test,
            params=best_params,
            fee_rate=fee_rate,
            starting_cash=current_cash,
            starting_inv=0.0
        )
        test_metrics = calculate_window_metrics(test_pnl_arr, test_inv_arr, test_fills, initial_cash=current_cash)
        
        window_pnl = test_metrics["final_net_pnl"]
        cumulative_oos_pnl += window_pnl
        current_cash += window_pnl
        
        # Build test window record
        record = {
            "model_name": model_name,
            "model_version": model_version,
            "wfo_step": wfo_step,
            "train_start_datetime": str(window['train_start_datetime']),
            "train_end_datetime": str(window['train_end_datetime']),
            "test_start_datetime": str(window['test_start_datetime']),
            "test_end_datetime": str(window['test_end_datetime']),
            "train_rows": window['train_rows'],
            "test_rows": window['test_rows'],
            "selected_parameters": json.dumps(best_params),
            "train_final_net_pnl": best_train_metrics["final_net_pnl"],
            "train_max_drawdown": best_train_metrics["max_drawdown"],
            "train_total_fills": best_train_metrics["total_fills"],
            "train_average_abs_inventory": best_train_metrics["avg_abs_inventory"],
            "train_risk_adjusted_score": best_train_metrics["risk_adjusted_score"],
            "test_final_net_pnl": cumulative_oos_pnl,
            "test_window_pnl": window_pnl,
            "test_max_drawdown": test_metrics["max_drawdown"],
            "test_fees_paid": test_metrics["fees_paid"],
            "test_total_fills": test_metrics["total_fills"],
            "test_bid_fills": test_metrics["bid_fills"],
            "test_ask_fills": test_metrics["ask_fills"],
            "test_average_abs_inventory": test_metrics["avg_abs_inventory"],
            "test_max_abs_inventory": test_metrics["max_abs_inventory"],
            "test_ending_inventory": test_metrics["ending_inventory"],
            "test_fill_rate": test_metrics["fill_rate"],
            "test_final_portfolio_value": test_metrics["final_portfolio_value"],
            "test_risk_adjusted_score": test_metrics["risk_adjusted_score"]
        }
        wfo_records.append(record)
        print(f"Step {wfo_step} complete. Window PnL: ${window_pnl:,.2f} | Score: {test_metrics['risk_adjusted_score']:.2f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(wfo_records)
    
    # Compute aggregates across all out-of-sample windows
    print("\nDEBUG - Available columns:", results_df.columns.tolist())
    aggregate_summary = generate_aggregate_summary(results_df)
    
    return results_df, aggregate_summary

def generate_aggregate_summary(results_df):
    """
    Computes required aggregate benchmarks for model-to-model comparison.
    """
    total_windows = len(results_df)
    positive_windows = int((results_df['test_window_pnl'] > 0).sum())
    
    param_counts = results_df['selected_parameters'].value_counts()
    most_frequent_params = param_counts.index[0] if not param_counts.empty else "N/A"

    return {
        "number_of_wfo_test_windows": total_windows,
        "total_out_of_sample_pnl": float(results_df['test_window_pnl'].sum()),
        "final_portfolio_value": float(results_df['test_final_portfolio_value'].iloc[-1]),
        "worst_test_window_drawdown": float(results_df['test_max_drawdown'].min()),
        "average_test_window_drawdown": float(results_df['test_max_drawdown'].mean()),
        "total_fees": float(results_df['test_fees_paid'].sum()),
        "total_fills": int(results_df['test_total_fills'].sum()),
        "average_absolute_inventory": float(results_df['test_average_abs_inventory'].mean()),
        "average_risk_adjusted_score": float(results_df['test_risk_adjusted_score'].mean()),
        "positive_test_windows": positive_windows,
        "positive_window_percentage": float((positive_windows / total_windows) * 100) if total_windows > 0 else 0.0,
        "final_ending_inventory": float(results_df['test_ending_inventory'].iloc[-1]),
        "most_frequently_selected_parameters": most_frequent_params
    }