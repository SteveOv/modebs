# Mass characterisation of Detached Eclipsing Binary Stars

A tool for characterising even mo'dEBs.

## Setup of runtime environment
This code base was developed within the context of a Python3 virtual environment which
supports Python 3.9-3.12, Scikit-Learn, TensorFlow, Keras, numpy, matplotlib, astropy,
lightkurve, emcee, and the custom [ebopmaven](https://github.com/SteveOv/ebop_maven),
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

#### JKTEBOP
These codes have a dependency on the JKTEBOP tool for generating and fitting lightcurves. The
installation media and build instructions can be found
[here](https://www.astro.keele.ac.uk/jkt/codes/jktebop.html). The `JKTEBOP_DIR` environment
variable is used to locate the executable at runtime.
```sh
$ export JKTEBOP_DIR=~/jktebop/
```
Set this to match the location where JKTEBOP has been set up.

#### Alternative, conda virtual environment
To set up an `platodebcat` conda environment, from the root of the local repo run the
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
