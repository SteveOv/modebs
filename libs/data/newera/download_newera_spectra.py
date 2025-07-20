#!/usr/bin/env python3
""" Script to download the full Z=0 spectra files from the NewEra archive. """
from inspect import getsourcefile
from pathlib import Path
import re
from hashlib import md5
import pandas as pd
import requests

this_dir = Path(getsourcefile(lambda:0)).parent

list_file = this_dir / "list_of_available_NewEra_models.txt"
save_dir = this_dir / "../../../.cache/.newera_spectra/"

# Don't use the header row in the file as the names have spaces
flist = pd.read_csv(list_file, header=0, skiprows=1, sep=r"\s+",
                    names=["indx", "filename", "checksum", "filesize", "url"], index_col="indx")

# Filter the list on the basic spectra based on [M/H]=0
fname_match = re.compile(r"lte(?:[\d]*)-(?:[\d.]*)-0.0.PHOENIX(?:\.*)")
sol_flist = flist[flist["filename"].apply(lambda v: fname_match.search(v) is not None)].iterrows()

for ctr, (ix, row) in enumerate(sol_flist, start=1):
    filename = save_dir / row["filename"]
    is_good_file = False        # pylint: disable=invalid-name

    print(f"{ix} ({ctr}): {filename.name} ", end="")
    if filename.exists():
        file_md5 = md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                file_md5.update(chunk)
        is_good_file = file_md5.hexdigest() == row["checksum"]

    if is_good_file:
        print("exists and the checksums match.")
    else:
        print(f"to get from {row['url']} ", end="")
        file_md5 = md5()
        with requests.get(row['url'], stream=True, timeout=60) as response:
            response.raise_for_status()  # Check for request errors

            # Open a local file with (over)write-binary mode
            with open(filename, 'wb') as file:
                for chunk_ix, chunk in enumerate(response.iter_content(chunk_size=8192)):
                    file.write(chunk)
                    file_md5.update(chunk)
                    if chunk_ix % 1000 == 0:
                        print(".", end="")

        if file_md5.hexdigest() == row["checksum"]:
            print(" saved (checksums match).")
        else:
            print(" saved (\033[93m\033[1mchecksums differ\033[0m)")
