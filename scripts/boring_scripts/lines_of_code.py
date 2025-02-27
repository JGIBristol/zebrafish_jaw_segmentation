"""
Fun pointless script that plots the number of lines of code in the repository over time.

Creates lines_of_code.png in the script output directory.

"""

import datetime
import subprocess

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

from fishjaw.util import files


def _get_commit_time(commit: str) -> datetime.datetime:
    """
    Get the time at which a commit was made (UTC)

    """
    commit_info = subprocess.run(
        ["git", "cat-file", "commit", commit], capture_output=True, check=True
    ).stdout.decode("utf-8")
    for line in commit_info.split("\n"):
        if line.startswith("committer"):
            time = line.split(" ")[-2]
            return datetime.datetime.fromtimestamp(int(time), tz=datetime.timezone.utc)
    raise ValueError(f"Could not find time for commit {commit}")


def _list_files(commit: str) -> list[str]:
    """
    List all *.py, .yml or .md files in the repository at a given commit

    """
    all_files = (
        subprocess.run(
            ["git", "ls-tree", "--name-only", "-r", commit],
            capture_output=True,
            check=True,
        )
        .stdout.decode("utf-8")
        .split("\n")
    )
    return [file for file in all_files if file.endswith((".py", ".yml", ".md", ".sh"))]


def _get_total_lines(commit: str, file_paths: list[str]) -> int:
    """
    Get the number of lines in a file at a specific commit.
    """
    total_lines = 0
    for file_path in file_paths:
        file_content = subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{file_path}"],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        total_lines += len(file_content.split("\n"))

    return total_lines


def main():
    """
    Plot the number of lines of code in the repository over time.

    """
    out_file = files.script_out_dir() / "lines_of_code.png"

    commits = (
        subprocess.run(["git", "rev-list", "main"], capture_output=True, check=True)
        .stdout.decode("utf-8")
        .split("\n")
    )[
        :-1  # The last element is an empty string
    ]

    dates = []
    loc = []
    for commit in tqdm(commits):
        date_time = _get_commit_time(commit)

        # Find the length of all python files in the repository at this commit
        python_files = _list_files(commit)
        total_len = _get_total_lines(commit, python_files)

        dates.append(date_time)
        loc.append(total_len)

    # I've decided I only care about entries after a certain date
    # since this repo was inactive for a while
    keep = np.array(
        [
            date > datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc)
            for date in dates
        ]
    )
    dates = np.array(dates)[keep]
    loc = np.array(loc)[keep]

    # Plot
    fig, axis = plt.subplots()
    axis.fill_between(dates, loc, color="plum")

    axis.set_xlabel("Date")
    axis.set_ylabel("LOC")
    axis.set_title("Lines in *.py/md/yml/sh files over time")

    axis.format_xdata = mdates.DateFormatter("%Y-%m-%d")

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(out_file)


if __name__ == "__main__":
    main()
