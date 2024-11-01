"""
Fun pointless script that plots the number of lines of code in the repository over time.

Creates lines_of_code.png in the script output directory.

"""

import datetime
import subprocess

from tqdm import tqdm
import matplotlib.pyplot as plt

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


def _list_python_files(commit: str) -> list[str]:
    """
    List all *.py files in the repository at a given commit

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
    return [file for file in all_files if file.endswith(".py")]


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
    commits = (
        subprocess.run(["git", "rev-list", "main"], capture_output=True, check=True)
        .stdout.decode("utf-8")
        .split("\n")
    )[
        :-1  # The last element is an empty string
    ]

    for commit in tqdm(commits):
        date_time = _get_commit_time(commit)

        # Find all python files for each
        python_files = _list_python_files(commit)
        total_len = _get_total_lines(commit, python_files)

        # Count the lines of code
    # Plot


if __name__ == "__main__":
    main()
