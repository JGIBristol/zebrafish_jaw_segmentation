"""
Fun pointless script that plots the number of lines of code in the repository over time.

Creates lines_of_code.png in the script output directory.

"""

import subprocess

from fishjaw.util import files


def main():
    """
    Plot the number of lines of code in the repository over time.

    """
    commits = (
        subprocess.run(["git", "rev-list", "main"], capture_output=True, check=True)
        .stdout.decode("utf-8")
        .split("\n")
    )[:-1]  # The last element is an empty string

    # Get the hashes from main
    # Find the date
    # Find all python files for each
    # Count the lines of code
    # Plot


if __name__ == "__main__":
    main()
