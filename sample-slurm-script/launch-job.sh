#!/bin/sh
#SBATCH --ntasks=37
#SBATCH --nodes=1
#SBATCH -J analyze
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=smp
#SBATCH --constraint=amd,genoa

# This is a template for a slurm job that generates profile data
# - ideally, this script would be a little more portable. But that's very much a job
#   for the future...

# Step 1: Setup LMOD Environment
# (obviously, you may need to adjust this logic)
module purge
module load anaconda3/2023.09-0-python_3.11.5
module load openmpi/5.0.6

# Step 2: activate the appropriate python environment
# (again this will change based on your particular case)
source activate /ihome/eschneider/mwa25/h2p-envs/py312

# Step 3:
# - specify the path to the cholla-vis directory and your configuration file
CHOLLA_VIS_DIR="/ix/eschneider/mabruzzo/packages/cholla-vis/"
CONF_FILE="/ix/eschneider/mabruzzo/packages/cholla-vis/sample-conf-files/crc.toml"

# Step 4: specify a few other choices

# PROFILE_KIND is the kind of profile
# - you can query the available kinds of profiles
PROFILE_KIND="z_fluxes"
# N_WORK_GROUP specifies the number of snapshots that should be processed in parallel.
# -> a value of 1 means that all MPI ranks will (try) to work together on making a
#    a single profile
# -> a value of 3 means that MPI ranks will be divided up to work on generating 3
#    profiles at a time.
# -> Be aware, only so many ranks can effectively collaborate (work is fundamentally
#    divided by spatial partitions, so if the simulation was run with 12 MPI processes,
#    that's effectively the largest division we can have)
N_WORK_GROUPS=3

# A few remaining notes:
# - its unclear whether the `--mca opal_warn_on_missing_libcuda 0` group of args is
#   actually needed. Historically I needed it on the CRC, but I think that was because
#   I had a pretty funky MPI/python setup. Now that I reconstructed my environment
#   (after the CRC updated the Operating system), I doubt its needed
# - we can specify multiple simulation names
# - if we don't use the slice-syntax to specify the snapshot range, then the script
#   will try to process **EVERY** available snapshot

# the following command tells bash to print out subsequent commands it invokes
set -x

mpirun --mca opal_warn_on_missing_libcuda 0 -np ${SLURM_NTASKS} python \
    ${CHOLLA_VIS_DIR}/analysis-scripts/make_profile.py \
    make \
    --conf=${CONF_FILE} \
    --profile-kind=${PROFILE_KIND} \
    --n-work-groups=${N_WORK_GROUPS} \
    --dynamic-balance \
    --sim 708cube_GasStaticG-1Einj,800:881:5
