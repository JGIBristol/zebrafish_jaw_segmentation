"""
Copy TIFF files from the DATABASE/ on the RDSF to a subfolder in Felix + Rich's directory.

Just so I don't break anything in the original data, I've been working off TIFF files in
1Felix and Rich make models/wahabs_scans/
This script:
 1. Copies the existing 3D TIFF files from the database to this directory
 2. Converts the remaining 2D TIFFs to 3D, renames them from the old_n to n numbering scheme
    and saves them here

"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(help=__doc__)
    parser.parse_args()
