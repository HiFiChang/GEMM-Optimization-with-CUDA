import subprocess
import os
import csv
import re

# Configuration
NVCC_PATH = "/usr/local/cuda-12.0/bin/nvcc"
VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]
N_VALUES = [128, 256, 512, 1024, 2048]
OUTPUT_DIR = "bin"
CSV_FILE = os.path.join("results.csv")

def compile_code(version, n):
    src_file = f"{version}.cu"
    exe_file = os.path.join(OUTPUT_DIR, f"{version}_{n}")
    cmd = [NVCC_PATH, "-O3", f"-DMY_N={n}", src_file, "-o", exe_file]
    print(f"Compiling {version} with N={n}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed for {version} N={n}:")
        print(result.stderr)
        return None
    return exe_file

def run_code(exe_file):
    print(f"Running {exe_file}...")
    try:
        result = subprocess.run([exe_file], capture_output=True, text=True, timeout=6000)
        if result.returncode != 0:
            print(f"Execution failed for {exe_file}:")
            print(result.stderr)
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Execution timed out for {exe_file}")
        return None

def parse_output(output):
    if not output:
        return None
    # Look for GFLOPS/s=...
    match = re.search(r"GFLOPS/s=([0-9.]+)", output)
    if match:
        return float(match.group(1))
    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize CSV
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Version", "N", "GFLOPS/s"])

    for n in N_VALUES:
        for version in VERSIONS:
            exe_file = compile_code(version, n)
            if exe_file:
                output = run_code(exe_file)
                gflops = parse_output(output)
                if gflops is not None:
                    print(f"Result: {version} N={n} -> {gflops} GFLOPS/s")
                    # Append to CSV
                    with open(CSV_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([version, n, gflops])
                else:
                    print(f"Failed to parse output for {version} N={n}")
            else:
                print(f"Skipping run for {version} N={n} due to compilation failure")

    print(f"Results saved to {CSV_FILE}")

if __name__ == "__main__":
    main()
