
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize Performance Results")
    parser.add_argument("--input", default="bin/results.csv", help="Path to results.csv")
    parser.add_argument("--output", default="performance_plot.png", help="Output image file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        # Fallback to checking the current directory
        if os.path.exists("results.csv"):
             print("Found results.csv in current directory, using that instead.")
             args.input = "results.csv"
        else:
            return

    print(f"Reading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Basic Validation
    required_cols = {'Version', 'N', 'GFLOPS/s'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    # Setup the plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Create the plot
    # Using a marker for each point to make it clearer
    sns.lineplot(
        data=df,
        x="N",
        y="GFLOPS/s",
        hue="Version",
        style="Version",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="viridis" # Good for distinct colors
    )

    plt.title("Performance Comparison (GFLOPS/s vs N)", fontsize=18, pad=20)
    plt.xlabel("Problem Size (N)", fontsize=14)
    plt.ylabel("Performance (GFLOPS/s)", fontsize=14)
    
    # Adjust x-axis to log scale if N varies by orders of magnitude, 
    # but linear might be fine for small ranges. 
    # Let's check the data range. N=[128, 256, 512, 1024, 2048]. 
    # Linear scale is okay but log2 might be better for power of 2 steps.
    plt.xscale('log', base=2)
    
    # Show values on plot points (optional but helpful)
    # Using matplotlib directly for annotations could get crowded, so skipping for now 
    # unless requested.

    plt.legend(title="Version", fontsize=12, title_fontsize=12)
    plt.tight_layout()

    print(f"Saving plot to {args.output}...")
    plt.savefig(args.output, dpi=300)
    print("Done!")

if __name__ == "__main__":
    main()
