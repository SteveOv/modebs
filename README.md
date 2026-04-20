# Mass characterisation of Detached Eclipsing Binary Stars

A tool for characterising even mo'dEBs.

## Setup of runtime environment
This code base was developed within the context of a Python3 virtual environment which
supports Python 3.9-3.12, numpy, scipy, uncertainties, matplotlib,
astropy, lightkurve, dust_extinction, mocpy, emcee, corner, mariadb and the custom
[ebopmaven](https://github.com/SteveOv/ebop_maven),
[sed fit](https://github.com/SteveOv/sed_fit) &
[deblib](https://github.com/SteveOv/deblib) packages upon which the code is dependent.
The dependencies are documented in the [requirements.txt](../main/requirements.txt)
file.

Having first cloned this GitHub repo, open a Terminal at the root of the local repo
and run the following commands. First to create and activate the venv;

```sh
$ python -m venv .modebs
$ source .modebs/bin/activate
```
Then run the following to set up the required packages:
```sh
$ pip install -r requirements.txt
```
You may need to install the jupyter kernel in the new venv:
```sh
$ ipython kernel install --user --name=.modebs
```

#### MariaDB and the data aceess layers
The MariaDB requirement is a dependency of the MariaDbTableDal data access layer class.
This is one of a number of options for storing target working data as it is populated and
updated by the various stages of the pipeline. The other Dal classes, the QTableDal and
QTableFileDal, store the working data in astropy QTables, with the latter saving this to
a file as updates are made. The significance of the MariaDbTableDal is that, by using a
database table as its underlying storage mechanism, it can safely support multiple
concurrent client processes. This QTable Dals are suited to small datasets and with a
single running client, whereas the MariaDBTableDal is intended for large datasets and
scaling out the fitting processes with the use of multiple concurrent clients.

Clearly there is a dependency of a working installation of MariaDB, either locally or
accessible over a network connection. Installation instructions can be found
[here](https://mariadb.com/docs/server/mariadb-quickstart-guides/installing-mariadb-server-guide).
There may be client side dependencies, depending on the operating system in use.
Documentation covering the setup of the python MariaDB connector and configuring database
connections can be found [here](https://mariadb.com/docs/connectors/connectors-quickstart-guides/connector-python-guide).

The `mariadb` requirement may be omitted (commented out in requirements.txt) if you
do not intend on using the MariaDbTableDal.

#### JKTEBOP
These codes have a dependency on the JKTEBOP tool for generating and fitting lightcurves. The
installation media and build instructions can be found
[here](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html). The `JKTEBOP_DIR` environment
variable is used to locate the executable at runtime.
```sh
$ export JKTEBOP_DIR=~/jktebop/
```
Set this to match the location where JKTEBOP has been set up.

#### MIST Isochrones
MIST pre-build model grids are required to support stellar MASS fitting.
The instructions on downloading these data are documented in the
[readme.txt](../main/libs/data/mist/MIST_v1.2_vvcrit0.4_basic_isos/readme.txt) file.

#### Alternative, conda virtual environment
To set up an `modebs` conda environment, from the root of the local repo run the
following command;
```sh
$ conda env create -f environment.yaml
```
You will need to activate the environment whenever you wish to run any of these modules.
Use the following command;
```sh
$ conda activate modebs
```
The conda environment sets the JKTEBOP_DIR environment variable to ~/jktebop/.
