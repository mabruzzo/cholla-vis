from functools import partial
import os.path

import h5py
import numpy as np

def _path_iter(prefix):
    # we can probably improve this by looking at the nprocs attribute of the hdf5 file
    assert(prefix[-3:] == '.h5')
    i = 0
    while True:
        path = f'{prefix}.{i}'
        if not os.path.isfile(path):
            if i == 0:
                raise RuntimeError(f"no file found called {path!r}")
            break
        yield path
        i += 1

def _copy_header_attrs(hdr_dest, hdr_src, attr_l):
    for i, attr in enumerate(attr_l):
        if attr in hdr_dest:
            if attr in attr_l[:i]:
                raise ValueError(f"'{attr}' was specified multiple times")
            else:
                raise ValueError(f"'{attr}' already exists in f_out")
        hdr_dest[attr] = hdr_src[attr]
        
def _write_concat_h5file(n, dnamein, dnameout, wrapped = None,
                         out_fname_suffix = None, **kwargs):
    assert wrapped is not None
    assert out_fname_suffix is not None
    
    hdr, dsets = wrapped(n = n, dnamein = dnamein, **kwargs)
    
    fname = f'{dnameout}{n}_{out_fname_suffix}'

    with h5py.File(fname, 'w') as f:
        for k,v in hdr.items():
            f.attrs[k] = v
        for k,v in dsets.items():
            f.create_dataset(k, data=v)
    return fname

def concat_particles(n, dnamein, preserve_density_arr = False):
    """
    Adapted from Orlando

    Reads files of form dnamein{n}_particles.h5.{rank}, looping over rank.

    Parameters
    ----------
    n: integer
        output number of file
    dnamein: string
        directory name of input files, should include '/' at end or leave blank for current directory
    preserve_density_arr: bool
        whether to concatenate the density array or ignore it entirely
        
    Returns
    -------
    hdr_out: dict
        contains header information
    dset: dict
        contains concatenated datasets
    """

    hdr_out = {}
    particle_props = {}
    n_total_particles = 0

    # loop over files for a given output time
    for i, path in enumerate(_path_iter(f"{dnamein}{n}_particles.h5")):

        if not os.path.isfile(path):
            raise RuntimeError(f"no file found called {path}")

        # open the input file for reading
        with h5py.File(path, 'r') as filein:

            # read in the header data from the input file
            head = filein.attrs

            # if it's the first input file, write the header attributes
            # and create the datasets in the output file
            if (i == 0):
                nx, ny, nz = head['dims'][...]
                _copy_header_attrs(
                    hdr_dest = hdr_out, hdr_src = head,
                    attr_l = ['n_step', 'dims', 'gamma', 't', 'dt',
                              'velocity_unit', 'length_unit', 'mass_unit', 
                              'density_unit']
                )

                if preserve_density_arr:
                    density = np.zeros((nx, ny, nz))

            # write data from individual processor file to
            # correct location in concatenated file
            nxl, nyl, nzl = head['dims_local'][:]
            xs, ys, zs = head['offset'][:]

            n_particle_local = head['n_particles_local']
            
            # loop through particle attributes
            for h5_field in filein.keys():
                if h5_field == 'density':
                    continue # handled separately
                elif filein[h5_field].shape != (n_particle_local,):
                    raise RuntimeError(
                        f"the '{h5_field}' dataset has an unexpected shape. "
                        "Does this dataset not hold particle attributes?"
                    )

                if h5_field not in particle_props:
                    assert n_total_particles == 0
                    particle_props[h5_field] = []    
                particle_props[h5_field].append(filein[h5_field][...])

            n_total_particles += n_particle_local
            if preserve_density_arr:
                density[xs:xs+nxl, ys:ys+nyl, zs:zs+nzl] += filein['density']

    dset = {}

    # write out the new datasets
    for (k,v) in particle_props.items():
        dset[k] = np.concatenate(v)

    if preserve_density_arr:
        dset['density'] = density

    assert len(hdr_out) > 0
    assert len(dset) > 0

    hdr_out['n_total_particles'] = n_total_particles

    return hdr_out, dset
    
write_concat_particles = partial(_write_concat_h5file, wrapped = concat_particles,
                                 out_fname_suffix = 'particles.h5')
write_concat_particles.__doc__ = """
    Adapted from Orlando

    Reads files of form dnamein{n}_particles.h5.{rank}, looping over rank, outputting to file 
    dnameout{n}_slice.h5.

    Parameters
    ----------
    n: integer
        output number of file
    dnamein: string
        directory name of input files, should include '/' at end or leave blank for current directory
    dnameout: string
        directory name of output files, should include '/' at end or leave blank for current directory
    preserve_density_arr: bool
        whether to concatenate the density array or ignore it entirely
    """

def _get_field_list_slice(keys):
    # assume full 3D
    key_set = set()
    for k in keys:
        if k[-3:] not in ['_xy', '_yz', '_xz']:
            raise RuntimeError(f'unexpected field: {f}')
        field_type = k[:-3]
        if field_type not in key_set:
            key_set.add(field_type)
    return list(key_set)

def concat_slice(n,dnamein):
    """
    Inherited from Alwin

    Reads files of form dnamein{n}_slice.h5.{rank}, looping over rank, outputting to file 
    dnameout{n}_slice.h5.

    Parameters
    ----------
    n: integer
        output number of file
    dnamein: string
        directory name of input files, should include '/' at end or leave blank for current directory

    Returns
    -------
    hdr_out: dict
        contains header information
    dset: dict
        contains concatenated datasets

    Note
    ----
    There's lots of room for improvement.
    - For starters, we can probably infer the number of slices from the 'nprocs' attribute
    - we can also probably write directly to the h5 to save ram (probably not relevant)
    - think about how we can optionally execute with mpi4py
    """
    hdr_out = {}

    # bound is used to track the left boundary (start with large vals)
    bounds = np.array([np.inf, np.inf, np.inf])

    field_list = [] # filled up during first loop

    # loop over files for a given output time
    for i, path in enumerate(_path_iter(f"{dnamein}{n}_slice.h5")):

        if not os.path.isfile(path):
            raise RuntimeError(f"no file found called {path}")
        
        # open the input file for reading
        with h5py.File(path, 'r') as filein:

            # read in the header data from the input file
            head = filein.attrs

            # if it's the first input file, write the header attributes
            # and create the datasets in the output file
            if (i == 0):
                nx, ny, nz = head['dims'][:]
                _copy_header_attrs(
                    hdr_dest = hdr_out, hdr_src = head,
                    attr_l = ['n_step', 'dims', 'gamma', 't', 'dt',
                              'velocity_unit', 'length_unit', 'mass_unit', 
                              'density_unit', 'dx', 'domain']
                )

                field_list += _get_field_list_slice(filein.keys())

                xy_map = dict((f, np.zeros((nx,ny))) for f in field_list)
                xz_map = dict((f, np.zeros((nx,nz))) for f in field_list)
                yz_map = dict((f, np.zeros((ny,nz))) for f in field_list)

            # sanity check!
            if len(field_list) == 0:
                raise RuntimeError(
                    f"Something is horribly wrong! While reading file {i} "
                    f"(at {path}) we noticed there aren't any fields"
                )
            
            # make sure to do this next operation from each file
            bounds = np.minimum(bounds, filein.attrs['bounds'][:])

            # write data from individual processor file to
            # correct location in concatenated file
            nxl, nyl, nzl = head['dims_local'][:]
            xs, ys, zs = head['offset'][:]

            for f in field_list:
                xy_map[f][xs:xs+nxl,ys:ys+nyl] += filein[f'{f}_xy']
                xz_map[f][xs:xs+nxl,zs:zs+nzl] += filein[f'{f}_xz']
                yz_map[f][ys:ys+nyl,zs:zs+nzl] += filein[f'{f}_yz']

    dset = {}
    for f in field_list:
        dset[f'{f}_xy'] = xy_map[f]
        dset[f'{f}_xz'] = xz_map[f]
        dset[f'{f}_yz'] = yz_map[f]

    hdr_out['bounds'] = bounds

    return hdr_out, dset
    
write_concat_slice = partial(_write_concat_h5file, wrapped = concat_slice,
                             out_fname_suffix = 'slice.h5')
write_concat_slice.__doc__ = """
    Inherited from Alwin

    Reads files of form dnamein{n}_slice.h5.{rank}, looping over rank, outputting to file 
    dnameout{n}_slice.h5.

    Parameters
    ----------
    n: integer
        output number of file
    dnamein: string
        directory name of input files, should include '/' at end or leave blank for current directory
    dnameout: string
        directory name of output files, should include '/' at end or leave blank for current directory
    """


def load_2D_h5(path):
    hdr_out = {}
    with h5py.File(path, 'r') as filein:
        # read in the header data from the input file
        head = filein.attrs

        _copy_header_attrs(
            hdr_dest = hdr_out, hdr_src = head,
            attr_l = ['n_step', 'dims', 'gamma', 't', 'dt',
                      'velocity_unit', 'length_unit', 'mass_unit', 
                      'density_unit', 'dx', 'domain', 'bounds']
        )

        data = {}
        for field in filein.keys():
            #print(field)
            data[field] = filein[field][...]
    return hdr_out, data
