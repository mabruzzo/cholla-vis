## Overview

This is a heavily work in progress repository.

Some context: it started out as a public repository that originally provided routines that primarily aided with visualization.


### Reproducibility

Currently things aren't very reproducible. I've got a lot of intermediate data files that the scripts are reading from. I need to organize those data files on the CRC so that it is easy for anyone in our research group to run the logic.

### Code Comments

Things are a little messy and complicated right now.

## How things are run

As I noted up above, things won't necessarily run very well since you need access to all of the temporary data.

When you do want to run things, it's important that you install the ``cholla_vis`` scripts.

## analysis-scripts

At the time of running, 2 primary logic files are scripts in the analysis-scripts directory.
- Originally these scripts were developed in Notebooks and a while back I had started to migrate them over (from Notebooks). So I made sure I could get them to run as scripts. They are packed with a ridiculous amount of logic, and I've started to flesh things out, but we need a lot more clarifying comments.
- The remaining analysis logic is still defined in Notebooks

Anyways, it may be more helpful to talk through figures and reference the scripts that were used to generate the figures...

## figures

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
TODO(what is the height associated with the density?)


I also need to do a much better job documenting the origin of the input data
