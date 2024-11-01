"""
Fun pointless script that plots the number of lines of code in the repository over time.

Creates lines_of_code.png in the script output directory.

"""

import datetime
import subprocess

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

    for commit in commits:
        time = _get_commit_time(commit)

    # Find the date
    # Find all python files for each
    # Count the lines of code
    # Plot


if __name__ == "__main__":
    main()
