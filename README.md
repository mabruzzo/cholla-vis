# Overview

This is a heavily work in progress repository.

Some brief history:
- this originally started out as a public repository that provided routines that primarily aided with visualization of Cholla datasets (I was using it early during debugging). I originally thought that it might be use
- at some point, I started to push analysis scripts to this repository

## Organization

At a high-level, this repository can be divided into a python package, ``cholla_vis`` and analysis scripts/notebooks. The analysis scripts/notebooks directly use the ``cholla_vis`` package.

Let's go through the contents:
- **pyproject.toml** describes the ``cholla_vis`` package (and any requirements) and the actual package is defined within **src/cholla_vis**
- **analysis-scripts** holds some analysis scripts (more information is provided below about each of the scripts)
- **analysis-notebooks** holds some analysis notebooks (I plan to add more as I continue cleaning up my existing notebooks)
- **figures** holds some some figures produced from the analysis scripts
- **unsorted-notebooks** holds some older notebooks that I haven't looked at in quite a while (_I need to go through these_)

## How to run the analysis scripts (and reproduce plots)

> [!WARNING]  
> Currently things aren't very reproducible. I've got a lot of intermediate data files that the scripts are reading from. I need to organize those data files on the CRC so that it is easy for anyone in our research group to run the logic.

Before you can run the scripts, you need to:
1. Clone the repository
2. Set up configuration, with paths to all of the appropriate data products _(I still need to take care of this!!)_
3. install the ``cholla_vis`` python package

### Manually installing the ``cholla_vis`` package

The minimum required python version is recorded in **pyproject.toml** under the ``project.requires-python`` field.

To manually install ``cholla_vis``, you would simply invoke

```sh
$ python -m pip install .
```

from the root of the repository. If your system has multiple versions of python, you can provide the full path to your desired python installation. I recommend that you do this using a virtual environment.

### A convenient automatic workflow with `uv`

While manually managing your python environment will absolutely work, I wanted to mention that I've recently been working with [`uv`](https://docs.astral.sh/uv/), and it simplifies the workflow quite nicely.

If you are unfamiliar with `uv`, let me provide a quick overview. After the tool is installed, you can get started by

- invoking `uv run python <path/to/script.py>` from anywhere in this repository to run a given script

- invoking `uv run --with jupyter jupyter lab` from within this repository to launch jupyter notebooks

While this might not seem like anything special, the main point is that you don't have to manually install/reinstall any packages or anything. In more detail, every time you invoke `uv` from within this project's directory, it automatically
- sets up an appropriate python environment. Using information from the ``pyproject.toml`` file, it will then
  - locate a compatible python version on your machine. If you don't have a compatible python version, I think it automatically install a compatible python version (there's a chance you may need to manually tell to download a new python version)
  - set up a virtual environment with that version of python (I'm pretty sure the environment is saved in ``.venv`` at the root of the repository) that includes an installed cpy of the ``cholla_vis`` package and all required dependencies.
- importantly ``uv`` also has a robust caching scheme. It will reuse a venv that it previously constructed, unless it detected some kind of changes.


## Code Comments

Things are a little messy and complicated right now.

## analysis-scripts

At the time of running, 2 primary logic files are scripts in the analysis-scripts directory.
- Originally these scripts were developed in Notebooks and a while back I had started to migrate them over (from Notebooks). So I made sure I could get them to run as scripts. They are packed with a ridiculous amount of logic, and I've started to flesh things out, but we need a lot more clarifying comments.
- The remaining analysis logic is still defined in Notebooks

## Describing Scripts By the Figures they make

For the most part, it's more helpful to talk through figures and reference the scripts that were used to generate them (but there are other useful tools as well)

When calling theses script, you should pass the `--conf <path/to/config.toml>`, where `<path/to/config.toml>` holds the path to a configuration file (that specifies the paths to the sim directory, the intermediate-data directory, and the processed-data directory).

The **sample-conf-files** holds examples of what these configuration files might look like.
You can directly use **./sample-conf-files/crc.toml** if you want to invoke the scripts on Pitt's CRC cluster.

### fluxes

the **figures/fluxes** directory was all created by **analysis-scripts/fluxes.py**. These show outflow rates from various simulations.
- the **figures/fluxes/cmp_** shows loading factors
- the **figures/fluxes/deriv_** shows the values written in terms of $`\odot{M}`$, $`\odot{p}`$, and `$\odot{E}`$
- the **figures/fluxes/fluxvals_** show the absolute flux values

Within a directory, the individual files are named according to the template: `dur{duration}_{time}.png`
- the first number in the template is the duration (in Myr) that we averaged over
- the second number in the template the time (in Myr) at which the measurement was actually made

We definitely to play around with the scaling.
The contents of **analysis-scripts/fluxes.py** also need to be cleaned and documented more.


### scale-height

the **figures/fluxes** directory was all created by **analysis-scripts/fluxes.py**. These show outflow rates from various simulations.
- the **figures/fluxes/cmp_** shows loading factors
- the **figures/fluxes/deriv_** shows the values written in terms of $`\odot{M}`$, $`\odot{p}`$, and `$\odot{E}`$
- the **figures/fluxes/fluxvals_** show the absolute flux values

Within a directory, the individual files are named according to the template: `dur{duration}_{time}.png`
- the first number in the template is the duration (in Myr) that we averaged over
- the second number in the template the time (in Myr) at which the measurement was actually made

We definitely to play around with the scaling.
The contents of **analysis-scripts/fluxes.py** also need to be cleaned and documented more.

I also need to do a much better job documenting the origin of the input data

### scale-heights

These plots were generated with the analysis-scripts/scale-height.py file.

They show the scale-height and the average-density for all the simulations over time.

**TODO(what is the height associated with the density?)**

## Other Scripts

The repository includes some other useful scripts.

### Plotting Slices and Figures

The `analysis-scripts/plot_2d_output.py` script generates images of slice-outputs and projection outputs that were previously written to disk by Cholla.
This script works by using certain hard-coded plotting presets for slices and projections.

You can see the available presets by invoking:

```sh
python <path/to/analysis-scripts/plot_2d_output.py> show-presets
```

To actually make a plot you would invoke something like:

```sh
python <path/to/analysis-scripts/plot_2d_output.py> plot \
    --load-dir=./catfiles \
    --save-dir=./my-image-test \
    --no-distributed-load \
    --kind=slice \
    --snaps 150 \
    --quan temperature, ndens, phat
```

On the CRC, I confirmed that the above command will work from within the
   /ix/eschneider/projects/galactic-center/sims/708cube_GasStaticG-1Einj
directory (it writes outputs to the `./my-image-test` directory)

There are a few extra things to consider:
- the `./catfiles` directory is a directory of output files to plot.
  - In this case the, the files were previously concatenated (using scripts
    provided by the main Cholla directory). Because they were concatenated, we need to
    pass the `--no-distributed-load` flag
  - in principle, we could also load distributed output files. In that case, you need
    to pass the `--distributed-load` (and remove the `--no-distributed-load` flag)
- `--kind` must specify `proj` or `slice`
- `--quan` must specify one or more of the known presets
- `--snaps` must specify the index of a single snapshot or a range of snapshots using
  python slice-notation (e.g. `150:163:4` specifies `150`, `154`, `158`, `162`)

You can also optionally pass `--ncores` to specify how to parallelize operations on a single CPU node.

For more info, invoke 

```sh
python <path/to/analysis-scripts/plot_2d_output.py> plot --help
```

Unfortunately, the plotting presets are currently hardcoded (but that shouldn't be
too hard to change)

### Concatenation

We provide a legacy concatenation script. The following snippet shows how you might
launch it:

```sh
python <path/to/concat_snaps.py> \
    --snaps <snap-range> \
    --source-directory <path/to/unconcatenated> \
    --output-directory <path/to/output-dir> \
    --cholla-src-dir <path/to/cholla/repository> \
    --proc-per-snap=12
```

### Parse SNe History

We provide a script to generate the CSV file containing the SNe history for our simulations by parsing the logs produced while running the simulations.
This script is called ``analysis-scripts/gen_SNe_tables.py``

Use the ``--help`` argument to find out more.


## TODO

I need to do a much better job documenting the origin of the input data
