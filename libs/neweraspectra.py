""" Functions for interacting with prebuilt NewEra model atmosphere spectra. """
from pathlib import Path
from io import StringIO
import requests

import numpy as np

#
#   These handle LowRes NewEra text files
#
def iterate_text_file(filename: (str|Path), encoding: str="utf8", skip: int=0, step: int=1):
    """
    Creates a generator, iterating over the lines in the passed text file.
    The skip argument indicates to skip the first n lines.
    The step argument indicates to yield every nth line after any initial skip.
    """
    with open(filename, "r", encoding=encoding) as f:
        for ix, line in enumerate(f):
            if ix >= skip and (ix-skip) % step == 0:
                yield line.rstrip()

def iterate_spectra_file(filename: (str|Path), skip: int=0):
    """
    Creates a generator which iterates over a PHOENIX NewEra low res spectra text file
    two lines at a time, yielding the fields line and flux line separately.
    """
    field_line, flux_line = None, None
    for count, line in enumerate(iterate_text_file(filename, "utf8", skip=skip)):
        if count % 2 == 0:
            field_line = line
        else:
            flux_line = line
            yield field_line, flux_line

def parse_lr_spectra_file(filename: (str|Path)):
    """
    Will parse the requested text based low res NewEra spectra file into a
    structured numpy array.
    """
    # Open the file to work out how many spectra
    spec_count = sum(1 for _ in iterate_text_file(filename, step=2))

    # Mappings for the columns in text file to named fields in a numpy structured array
    fields_index = { "teff": 12, "logg": 13, "mass": 27,
                    "lam_steps": 8, "lam_from": 9, "lam_to": 10, "lam_step": 11 }
    fields_names = list(fields_index.keys())
    fields_cols = [fields_index[n] for n in fields_names]
    fields_dtype = [(n, int if n == "lam_steps" else float) for n in fields_names]

    # Parse the text file (with fields and fluxes on separate contiguous lines) into a table
    table = np.empty((spec_count, ), dtype=fields_dtype + [("flux", object)])
    row_num = 0
    for field_line, flux_line in iterate_spectra_file(filename):
        if len(field_line) and len(flux_line):
            fields = np.loadtxt(StringIO(field_line), dtype=fields_dtype, usecols=fields_cols)
            fluxes = np.loadtxt(StringIO(flux_line), dtype=np.dtype(float), unpack=True)

            # Reduce the resolution of the spectrum by a factor of 1000 (from nm to um ramge)
            lambdas = np.linspace(fields["lam_from"], fields["lam_to"], len(fluxes))    # nm
            low_cents = np.arange(1e3, len(lambdas), 1e3, dtype=int)[:-1]
            low_flux = np.array([np.sum(fluxes[cent-499:cent+499]) for cent in low_cents])  # bins
            #low_flux = np.array([np.sum(fluxes[cent]) for cent in low_cents])           # not-bins

            # Revise the metadata to reflect the reduction in spectral resolution and change
            # the lambdas from nm to um (rescaling leaves the step size scalar value unchanged).
            fields["lam_steps"] = len(low_flux)
            fields["lam_from"] = lambdas[low_cents[0]] / 1e3                            # nm to um
            fields["lam_to"] = lambdas[low_cents[-1]] / 1e3                             # nm to um

            # Store this row
            for fn in fields_names:
                table[row_num][fn] = fields[fn]
            table[row_num]["flux"] = low_flux
            row_num += 1
    return table

#
#   These handle NewEra h5 files
#
def make_newera_filename(teff: (int|float), logg: float, zscale: float=0, alpha_scale: float=0):
    """
    Generate the (LTE) NewEra HSR filename for given criteria.
    """
    if zscale == +0: # Ensure zero Z is prefixed with -
        zscale = -0.0

    if alpha_scale == 0.0:
        job_name = f"lte{teff:0=5.0f}{-logg:3.2f}{zscale:0=+4.1f}"
    else:
        job_name = f"lte{teff:0=5.0f}{-logg:3.2f}{zscale:0=+4.1f}.alpha={alpha_scale:0=+3.1f}"
    return f"{job_name}.PHOENIX-NewEra-ACES-COND-2023.HSR.h5"

def download_newera_file(filename: (str|Path), save_dir: Path):
    """
    Will download the requested file from the NewEra archive and save to the requested
    save directory, having first checked that the target does not already exist.
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    if not save_dir.exists():
        raise ValueError("save_dir does not exist")
    if not save_dir.is_dir():
        raise ValueError("save_dir is not a directory")

    if not (save_dir / filename).exists():
        url = f"https://www.fdr.uni-hamburg.de/record/16738/files/{filename}?download=1"
        print(f"Downloading {filename} from {url}")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()  # Check for request errors

            # Open a local file with write-binary mode
            with open(save_dir / filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f'Saved to {save_dir / filename}')
    else:
        print(f"File {save_dir / filename} exists.")
