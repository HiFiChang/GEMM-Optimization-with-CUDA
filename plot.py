
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize Performance Results")
    parser.add_argument("--input", default="results.csv", help="Path to results.csv")
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
    
    # Plot GFLOPS/s
    plt.figure(figsize=(24, 8))
    sns.lineplot(
        data=df,
        x="N",
        y="GFLOPS/s",
        hue="Version",
        style="Version",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="viridis"
    )

    plt.title("Performance Comparison (GFLOPS/s vs N)", fontsize=28, pad=20)
    plt.xlabel("Problem Size (N)", fontsize=28)
    plt.ylabel("Performance (GFLOPS/s)", fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xscale('log', base=2)
    plt.legend(title="Version", fontsize=16, title_fontsize=28)
    plt.tight_layout()

    print(f"Saving performance plot to {args.output}...")
    plt.savefig(args.output, dpi=300)
    
    plt.figure(figsize=(24, 8))
    sns.lineplot(
        data=df,
        x="N",
        y="GB/s",
        hue="Version",
        style="Version",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="viridis"
    )

    plt.title("Bandwidth Comparison (GB/s vs N)", fontsize=28, pad=20)
    plt.xlabel("Problem Size (N)", fontsize=28)
    plt.ylabel("Bandwidth (GB/s)", fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xscale('log', base=2)
    plt.legend(title="Version", fontsize=16, title_fontsize=28)
    plt.tight_layout()
    
    bw_output = args.output.replace(".png", "_bandwidth.png")
    if bw_output == args.output:
        bw_output = "bandwidth_plot.png"
        
    print(f"Saving bandwidth plot to {bw_output}...")
    plt.savefig(bw_output, dpi=300)

    print("Done!")

if __name__ == "__main__":
    main()
