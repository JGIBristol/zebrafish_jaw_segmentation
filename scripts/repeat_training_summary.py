"""
Once we've trained lots of models and run the inference with all of them,
read the tables of metrics from the logs and build a table.

Then print some summary stats
"""
import io
import os
import pathlib
import argparse
import pandas as pd


def extract_table_from_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the start of the markdown table
    for i, line in enumerate(lines):
        if line.strip().startswith("| label"):
            table_start = i
            break
    else:
        raise ValueError(f"No table found in {filepath}")

    # Extract table lines and read using StringIO
    table_lines = lines[table_start:]
    table_str = ''.join(table_lines)
    df = pd.read_csv(io.StringIO(table_str), sep='|', engine='python', skipinitialspace=True)

    # Clean up DataFrame
    # df = df.drop(columns=[''])  # Drop empty column caused by leading/trailing '|'
    df.columns = [col.strip() for col in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.set_index("label")
    df = df.apply(pd.to_numeric, errors='ignore')

    return df

def main():
    """
    Read all the markdown tables, convert them to dataframes, do some checks
    and print a summary table.
    """
    log_dir = pathlib.Path("logs/")

    dfs = []
    for file in log_dir.glob("*inference.log"):
        df = extract_table_from_file(file)
        dfs.append(df)

    # Ensure felix, harry, tahlia are the same in all files
    ref_df = dfs[0].loc[["felix", "harry", "tahlia"]]
    for i, df in enumerate(dfs[1:], start=1):
        if not df.loc[["felix", "harry", "tahlia"]].equals(ref_df):
            raise AssertionError(f"Mismatch in felix/harry/tahlia values in file: {files[i]}")

    # Create the final combined DataFrame
    final_df = ref_df.copy()
    for i, df in enumerate(dfs):
        inference_row = df.loc["inference"].copy()
        inference_row.name = f"inference_{i+1}"
        final_df = pd.concat([final_df, pd.DataFrame([inference_row])])

    print(final_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and summarize training metrics from model logs in logs/"
    )

    main(**vars(parser.parse_args()))