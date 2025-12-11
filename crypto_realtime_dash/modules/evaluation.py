"""
Evaluation Module - RMSE, MAE, MAPE, Directional Accuracy
SINTA 1 Bitcoin Forecasting System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        RMSE value
    """
    if len(actual) == 0 or len(predicted) == 0:
        return np.nan
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MAE value
    """
    if len(actual) == 0 or len(predicted) == 0:
        return np.nan
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    return np.mean(np.abs(actual - predicted))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MAPE value (as percentage)
    """
    if len(actual) == 0 or len(predicted) == 0:
        return np.nan
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.nan
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Directional Accuracy.
    Measures how often the model correctly predicts the direction of price movement.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        DA value (as percentage)
    """
    if len(actual) < 2 or len(predicted) < 2:
        return np.nan
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate direction of change
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    
    # Count correct predictions
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)
    
    return (correct / total) * 100 if total > 0 else 0


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        SMAPE value (as percentage)
    """
    if len(actual) == 0 or len(predicted) == 0:
        return np.nan
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    
    if not mask.any():
        return np.nan
    
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'RMSE': calculate_rmse(actual, predicted),
        'MAE': calculate_mae(actual, predicted),
        'MAPE': calculate_mape(actual, predicted),
        'SMAPE': calculate_smape(actual, predicted),
        'DA': calculate_directional_accuracy(actual, predicted)
    }


def compare_models(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        model_results: Dictionary mapping model name to result dict
    
    Returns:
        DataFrame with comparison
    """
    comparisons = []
    
    for model_name, result in model_results.items():
        if 'actual' not in result or 'predictions' not in result:
            continue
        
        actual = result['actual']
        predicted = result['predictions']
        
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        metrics = calculate_all_metrics(actual, predicted)
        metrics['Model'] = model_name
        
        # Add AIC/BIC if available
        if 'aic' in result:
            metrics['AIC'] = result['aic']
        if 'bic' in result:
            metrics['BIC'] = result['bic']
        
        comparisons.append(metrics)
    
    if not comparisons:
        return pd.DataFrame()
    
    df = pd.DataFrame(comparisons)
    
    # Reorder columns
    cols = ['Model', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'DA', 'AIC', 'BIC']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Sort by RMSE
    df = df.sort_values('RMSE').reset_index(drop=True)
    
    return df


def generate_evaluation_report(model_results: Dict[str, Dict]) -> str:
    """
    Generate text evaluation report.
    
    Args:
        model_results: Dictionary with model results
    
    Returns:
        Formatted report string
    """
    comparison = compare_models(model_results)
    
    if comparison.empty:
        return "No model results to evaluate."
    
    report = []
    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Best model
    best = comparison.iloc[0]
    report.append(f"BEST MODEL: {best['Model']}")
    report.append(f"  RMSE: {best['RMSE']:,.2f}")
    report.append(f"  MAE:  {best['MAE']:,.2f}")
    report.append(f"  MAPE: {best['MAPE']:.2f}%")
    if 'DA' in best and not np.isnan(best['DA']):
        report.append(f"  DA:   {best['DA']:.1f}%")
    report.append("")
    
    # Comparison table
    report.append("-" * 60)
    report.append(f"{'Model':<15} {'RMSE':>12} {'MAE':>12} {'MAPE':>10}")
    report.append("-" * 60)
    
    for _, row in comparison.iterrows():
        report.append(
            f"{row['Model']:<15} {row['RMSE']:>12,.2f} {row['MAE']:>12,.2f} {row['MAPE']:>9.2f}%"
        )
    
    report.append("-" * 60)
    
    # ARIMAX vs ARIMA comparison
    if 'ARIMAX' in comparison['Model'].values and 'ARIMA' in comparison['Model'].values:
        arima_rmse = comparison[comparison['Model'] == 'ARIMA']['RMSE'].values[0]
        arimax_rmse = comparison[comparison['Model'] == 'ARIMAX']['RMSE'].values[0]
        
        improvement = (arima_rmse - arimax_rmse) / arima_rmse * 100
        
        report.append("")
        if improvement > 0:
            report.append(f"[+] ARIMAX improves upon ARIMA by {improvement:.1f}%")
            report.append("    Sentiment and indicators contribute to better predictions.")
        else:
            report.append(f"[-] ARIMA outperforms ARIMAX by {-improvement:.1f}%")
            report.append("    Exogenous features may add noise in current market conditions.")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Testing Evaluation Module...")
    print("=" * 50)
    
    # Sample data
    np.random.seed(42)
    actual = np.array([100, 102, 105, 103, 108, 110, 112])
    predicted = np.array([101, 103, 104, 105, 107, 111, 113])
    
    print("\nMetrics:")
    metrics = calculate_all_metrics(actual, predicted)
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}")
    
    # Test comparison
    print("\nModel Comparison:")
    results = {
        'ARIMA': {'actual': actual, 'predictions': predicted, 'aic': 100, 'bic': 110},
        'ARIMAX': {'actual': actual, 'predictions': predicted * 0.98, 'aic': 95, 'bic': 105}
    }
    
    comparison = compare_models(results)
    print(comparison)
