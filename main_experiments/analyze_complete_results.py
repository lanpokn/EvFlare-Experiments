#!/usr/bin/env python3
"""
Analyze complete evaluation results and generate comprehensive statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results(csv_file):
    """Analyze complete evaluation results."""
    
    # Read results
    df = pd.read_csv(csv_file)
    
    print("="*80)
    print("COMPLETE EVALUATION ANALYSIS - ALL 50 H5 PAIRS")
    print("="*80)
    
    print(f"Total samples evaluated: {len(df)}")
    # Check if 'failed' column exists
    has_failed_col = 'failed' in df.columns
    if has_failed_col:
        failed_count = df['failed'].sum()
        success_rate = (len(df) - failed_count) / len(df) * 100
    else:
        success_rate = 100.0  # Assume all successful if no failed column
    print(f"Success rate: {success_rate:.1f}%")
    print()
    
    # Filter successful results - assume all successful if no failed column
    if has_failed_col:
        success_df = df[df['failed'] == False]
    else:
        success_df = df.copy()  # All are successful
    
    if len(success_df) == 0:
        print("No successful evaluations found!")
        return
    
    print(f"Successful evaluations: {len(success_df)}")
    print()
    
    # Calculate statistics for each metric
    metrics = ['chamfer_distance', 'gaussian_distance', 'event_count_ratio', 'temporal_overlap']
    
    print("DETAILED METRIC STATISTICS:")
    print("="*80)
    
    for metric in metrics:
        if metric in success_df.columns:
            values = success_df[metric].dropna()
            
            if len(values) > 0:
                print(f"\n{metric.upper()}:")
                print(f"  Count:    {len(values)}")
                print(f"  Mean:     {values.mean():.6f}")
                print(f"  Median:   {values.median():.6f}")
                print(f"  Std Dev:  {values.std():.6f}")
                print(f"  Min:      {values.min():.6f}")
                print(f"  Max:      {values.max():.6f}")
                print(f"  25th %:   {values.quantile(0.25):.6f}")
                print(f"  75th %:   {values.quantile(0.75):.6f}")
    
    # Additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS:")
    print("="*80)
    
    # Event counts
    est_counts = success_df['est_event_count']
    gt_counts = success_df['gt_event_count']
    
    print(f"\nEVENT COUNTS:")
    print(f"  Flare files (estimated):")
    print(f"    Mean: {est_counts.mean():,.0f} events")
    print(f"    Range: {est_counts.min():,} - {est_counts.max():,}")
    print(f"  Ground truth files:")
    print(f"    Mean: {gt_counts.mean():,.0f} events") 
    print(f"    Range: {gt_counts.min():,} - {gt_counts.max():,}")
    
    # Performance stats
    if 'total_time_s' in success_df.columns:
        times = success_df['total_time_s']
        print(f"\nPROCESSING TIME:")
        print(f"  Mean time per pair: {times.mean():.2f}s")
        print(f"  Total processing time: {times.sum():.2f}s ({times.sum()/3600:.2f}h)")
        print(f"  Fastest pair: {times.min():.2f}s")
        print(f"  Slowest pair: {times.max():.2f}s")
    
    # Sample with best/worst performance
    if 'chamfer_distance' in success_df.columns:
        chamfer_values = success_df['chamfer_distance'].dropna()
        if len(chamfer_values) > 0:
            best_idx = chamfer_values.idxmin()
            worst_idx = chamfer_values.idxmax()
            
            print(f"\nPERFORMANCE EXTREMES:")
            print(f"  Best Chamfer Distance: {chamfer_values.min():.6f}")
            print(f"    Sample: {success_df.loc[best_idx, 'sample_id']}")
            print(f"  Worst Chamfer Distance: {chamfer_values.max():.6f}")
            print(f"    Sample: {success_df.loc[worst_idx, 'sample_id']}")
    
    # Save enhanced summary
    output_file = Path(csv_file).parent / "complete_analysis_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write("COMPLETE H5 PAIRS EVALUATION - FINAL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: 50 paired H5 files\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write(f"Total successful evaluations: {len(success_df)}\n\n")
        
        f.write("METRIC AVERAGES:\n")
        f.write("-"*40 + "\n")
        for metric in metrics:
            if metric in success_df.columns:
                values = success_df[metric].dropna()
                if len(values) > 0:
                    f.write(f"{metric}: {values.mean():.6f} ± {values.std():.6f}\n")
        
        f.write(f"\nEVENT COUNT AVERAGES:\n")
        f.write(f"  Flare events: {est_counts.mean():,.0f}\n")
        f.write(f"  Ground truth events: {gt_counts.mean():,.0f}\n")
        
        if 'total_time_s' in success_df.columns:
            f.write(f"\nPROCESSING PERFORMANCE:\n")
            f.write(f"  Average time per pair: {times.mean():.2f}s\n")
            f.write(f"  Total time: {times.sum():.2f}s ({times.sum()/3600:.2f}h)\n")
    
    print(f"\nDetailed analysis saved to: {output_file}")
    
    return success_df

def main():
    csv_file = "results/complete/simu_pairs_evaluation_results.csv"
    
    if not Path(csv_file).exists():
        print(f"ERROR: Results file not found: {csv_file}")
        return
    
    results_df = analyze_results(csv_file)
    
    # Also save CSV with averages row
    if results_df is not None:
        # Calculate averages for numeric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        averages = results_df[numeric_cols].mean()
        
        # Create a summary row
        avg_row = pd.DataFrame([averages], index=['AVERAGE'])
        avg_row['sample_id'] = 'AVERAGE'
        avg_row['flare_file'] = 'N/A'
        avg_row['gt_file'] = 'N/A'
        
        # Append to original data
        enhanced_df = pd.concat([results_df, avg_row], ignore_index=False)
        
        # Save enhanced CSV
        output_csv = Path(csv_file).parent / "simu_pairs_results_with_averages.csv"
        enhanced_df.to_csv(output_csv, index=False)
        
        print(f"\nEnhanced CSV with averages saved to: {output_csv}")

if __name__ == "__main__":
    main()