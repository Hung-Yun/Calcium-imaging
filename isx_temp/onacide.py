# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:59:37 2020

@author: Hung-Yun Lu
"""

from isx._internal import ensure_list
from .io import Movie, CellSet, EventSet, export_movie_to_tiff, export_movie_to_nwb

import os
import math
import yaml
import h5py
import subprocess

import numpy as np


def _get_memmap_name(file_names):
    """ Return the name of a memmap file created by save_memmap in caiman, based on the name of an .isxd file, or a list of .isxd files. """

    file_names = ensure_list(file_names)

    root_dir, fname = os.path.split(file_names[0])
    base_name, ext = os.path.splitext(fname)

    num_rows = 0
    num_cols = 0
    num_frames = 0
    for fname in file_names:
        mov = Movie.read(fname)
        ti = mov.timing
        sp = mov.spacing

        num_rows, num_cols = sp.num_pixels
        num_frames += ti.num_samples - len(ti.dropped)

        del mov

    byte_order = 'C'
    mm_name = '{}_d1_{}_d2_{}_d3_1_order_{}_frames_{}_.mmap'.format(base_name, num_rows, num_cols, byte_order, num_frames)

    return mm_name

def _export_movie_to_tiff(isxd_movie_files, overwrite=False, output_dir=None):
    """ Export one or more movie files to .tiff.

    Arguments
    ---------
    isxd_movie_files : str OR list<str>
        Path to an .isxd movie file, or a list of .isxd movie files.
    overwrite : bool
        Overwrite the file if it already exists.

    Returns
    --------
    file_name : list<str>
        The file names of the created .tiff files.
    """

    isxd_movie_files = ensure_list(isxd_movie_files)

    # compute the number of frames in each movie
    num_rows = -1
    num_cols = -1
    num_frames = 0
    for mov_file in isxd_movie_files:
        mov = Movie.read(mov_file)
        ti = mov.timing
        sp = mov.spacing

        num_rows, num_cols = sp.num_pixels
        num_frames += ti.num_samples - len(ti.dropped)

    # write a tiff file, use the name of the first movie as the tiff file base name
    root_dir, fname = os.path.split(isxd_movie_files[0])
    base_name, ext = os.path.splitext(fname)
    if output_dir is None:
        output_dir = root_dir

    tiff_file_first = os.path.join(output_dir, '{}.tiff'.format(base_name))

    # Each tiff can only store 65535 frames
    num_tiffs = math.ceil(num_frames / 65535)

    # any extra tiff files are named based on the order of magnitude of the num_frames
    # to match IDPS naming scheme
    # e.g. if exporting movie.isxd with 70005 frames, first file is movie.tiff,
    #      second file is movie_00001.tiff
    tiff_files = [tiff_file_first]
    width = math.floor(math.log10(num_frames - 1)) + 1 if num_frames > 10 else 1

    for i in range(1, num_tiffs):
        suffix = "_" + str(i).zfill(width)

        tiff_file = os.path.join(output_dir, '{}.tiff'.format(base_name + suffix))
        tiff_files.append(tiff_file)

    # Remove output files if overwrite is True, otherwise throw an error
    for tiff_file in tiff_files:
        if os.path.exists(tiff_file):
            if overwrite:
                os.remove(tiff_file)
            else:
                raise ValueError(tiff_file + " already exists but overwrite is set to False. Set to True to overwrite.")

    export_movie_to_tiff(isxd_movie_files, tiff_file_first)

    return tiff_files, num_frames, num_rows, num_cols


def _reshape_A(num_rows, num_cols, A):
    """ Turn the sparse scipy.csc_matrix A into a dense matrix with shape (num_rows, num_cols, num_cells) """

    Adense = np.array(A.todense())
    npx, ncells = Adense.shape
    if npx != num_rows * num_cols:
        raise ValueError('A.shape[0] must be equal to num_rows*num_cols')

    Adense = Adense.reshape([num_cols, num_rows, ncells])
    Ars = np.zeros([num_rows, num_cols, ncells])
    for k in range(ncells):
        Ars[:, :, k] = Adense[:, :, k].T
    del Adense
    Ars[np.isnan(Ars)] = 0
    return Ars.astype('float32')


def _turn_into_array(val):
    """ Turn val into a numpy array with two elements, if it is not already. """
    if val is not None:
        if hasattr(val, '__iter__'):
            val = np.array(val)
        else:
            val = np.array([val, val])
    return val


def run_onacide(input_movie_files, output_cell_set_files, output_events_files,
                num_processes=1, overwrite_tiff=False,
                K=20, rf=[25, 25], stride=6,
                gSiz=13, gSig=5,
                min_pnr=5, min_corr=0.8,
                ssub_B=1,
                min_SNR=5, rval_thr=0.85,
                decay_time=0.400, event_threshold=0.025,
                merge_threshold=0.8,
                output_dir=None):
    """ Run the CaImAn CNMFe algorithm on an input movie file.

    Arguments
    ---------
    input_movie_files : list<str>
        Path to an .isxd movie file, or a list of paths to .isxd movie files that are a part of a Series.
    output_cell_set_files : list<str>
        The path to a cell set .isxd file that will be written with the identified traces and footprints, or if
        there are multiple input movie files, a list of output cell set file paths.
    output_events_files : list<str>
        The path to an events .isxd file that will be written with the deconvolved spikes of each neuron, or if there
        are multiple input movie files, a list of output event file paths.
    num_processes : int
        The number of processes to run in parallel. The more parallel processes, the more memory that is used.
    overwite_tiff : bool
        If the tiff file already exists, delete it and create a new one if True.
    rf : array-like
        An array [half-width, half-height] that specifies the size of a patch.
    stride : int
        The amount of overlap in pixels between patches.
    K : int
        The maximum number of cells per patch.
    gSiz : int
        The expected diameter of a neuron in pixels.
    gSig : int
        The standard deviation a high pass Gaussian filter applied to the movie prior to seed pixel search, roughly
        equal to the half-size of the neuron in pixels.
    min_pnr : float
        The minimum peak-to-noise ratio that is taken into account when searching for seed pixels.
    min_corr : float
        The minimum pixel correlation that is taken into account when searching for seed pixels.
    min_SNR : float
        Cells with an signal-to-noise (SNR) less than this are rejected.
    rval_thr : float
        Cells with a spatial correlation of greater than this are accepted.
    decay_time : float
        The expected decay time of a calcium event in seconds.
    ssub_B : int
        The spatial downsampling factor used on the background term.
    event_threshold : float
        Spikes with an amplitude of less than this are not written to the events file.
    merge_threshold : float
        Cells that are spatially close with a temporal correlation of greater than merge_threshold are automatically merged.
    output_dir : str
        Directory that .yaml, .tiff and .mmap files are written to. Defaults to movie directory.
    """

    input_movie_files = ensure_list(input_movie_files)
    output_cell_set_files = ensure_list(output_cell_set_files)
    output_events_files = ensure_list(output_events_files)

    if len(input_movie_files) != len(output_cell_set_files) or len(input_movie_files) != len(output_events_files):
        raise ValueError('The number of input movie files must match the number of output cell set and event set files.')

    # check extension for input movies
    base_name = None
    for input_movie_file in input_movie_files:
        base_dir, input_movie_file_name = os.path.split(input_movie_file)
        mov_base, mov_ext = os.path.splitext(input_movie_file_name)
        if base_name is None:
            base_name = mov_base
        if not mov_ext.endswith('isxd'):
            raise ValueError('Input_movie_file must be an .isxd movie.')

    # remove cell set and events files if they already exist
    for output_cell_set_file, output_events_file in zip(output_cell_set_files, output_events_files):
        if os.path.exists(output_cell_set_file):
            os.remove(output_cell_set_file)
        if os.path.exists(output_events_file):
            os.remove(output_events_file)

    # read frame rates of movies, assert that they are equal
    frame_periods = list()
    for input_movie_file in input_movie_files:
        mov = Movie.read(input_movie_file)
        frame_periods.append(mov.timing.period.to_msecs())
        del mov
    frame_periods = np.array(frame_periods)
    if not np.sum(np.diff(frame_periods)) == 0:
        raise ValueError('Frame rates must be same for all input movies.')

    frame_rate = 1 / (frame_periods[0]*1e-3)

    # determine output directory
    in_movie_dir, _ = os.path.split(input_movie_files[0])
    if output_dir is None:
        output_dir = in_movie_dir

    if not os.path.exists(output_dir):
        raise FileNotFoundError('Missing output directory {}'.format(output_dir))
    elif not os.path.isdir(output_dir):
        raise NotADirectoryError('output_dir is not a directory: {}'.format(output_dir))

    # write tiff file
    print('Exporting .isxd to tiff file(s)...')
    tiff_files, num_frames, num_rows, num_cols = _export_movie_to_tiff(input_movie_files, overwrite=overwrite_tiff,
                                                                        output_dir=output_dir)

    for tiff_file in tiff_files:
        print('Wrote .tiff file to: {}'.format(tiff_file))

    # write parameters to a yaml file
    out_params = dict()
    out_params['num_processes'] = num_processes
    out_params['K'] = K
    out_params['rf'] = rf
    out_params['stride'] = stride
    out_params['gSiz'] = gSiz
    out_params['gSig'] = gSig
    out_params['min_pnr'] = min_pnr
    out_params['min_corr'] = min_corr
    out_params['ssub_B'] = ssub_B
    out_params['min_SNR'] = min_SNR
    out_params['rval_thr'] = rval_thr
    out_params['decay_time'] = decay_time
    out_params['merge_threshold'] = merge_threshold
    out_params['num_frames'] = num_frames
    out_params['num_rows'] = num_rows
    out_params['num_cols'] = num_cols
    out_params['frame_rate'] = float(frame_rate)

    param_file = os.path.join(output_dir, 'caiman_params.yaml')
    with open(param_file, 'w') as f:
        yaml.dump(out_params, f)

    # specify output file name
    out_file = os.path.join(output_dir, '{}_caiman_output.h5'.format(base_name))

    onacide_wrapper_commands = ['python', '-m', 'isx_onacide_wrapper.runner', '--input_files']
    for tiff_file in tiff_files:
        onacide_wrapper_commands.append(tiff_file)
    onacide_wrapper_commands += ['--params_file', param_file, '--output_file', out_file]

    proc = subprocess.run(onacide_wrapper_commands)

    # check for the output file
    if not os.path.exists(out_file):
        raise FileNotFoundError('No CaImAn output file found, check the output for errors.')

    # save traces, footprints, and events to output files
    _save_cnmfe(out_file, input_movie_files, output_cell_set_files, output_events_files, event_threshold=event_threshold)


def _save_cnmfe(output_hdf_file, input_movie_files, output_cell_set_files, output_events_files, event_threshold=0.025):
    """ Save the essential components of a CNMF object to cell set and events files.

    Arguments
    ---------
    output_hdf_file : str
        The path to an hdf5 file written with isx_cnmfe_wrapper.
    input_movie_files : list<str>
        Path to an .isxd movie file, or a list of paths to .isxd movie files.
    output_cell_set_files : list<str>
        The path to a cell set .isxd file that will be written with the identified traces and footprints, or a list
        of output cell set files.
    output_events_files : list<str>
        The path to an events .isxd file that will be written with the deconvolved spikes of each neuron, or a list of
        output event files.
    event_threshold : float
        Spikes with an amplitude of less than this are not written to the events file.
    """

    input_movie_files = ensure_list(input_movie_files)
    output_cell_set_files = ensure_list(output_cell_set_files)
    output_events_files = ensure_list(output_events_files)

    with h5py.File(output_hdf_file, 'r') as hf:
        for val in ['A', 'C', 'S']:
            if val not in hf.keys():
                raise ValueError('Missing cell footprints {} in {}'.format(val, output_hdf_file))
        A = np.array(hf['A'])
        C = np.array(hf['C'])
        S = np.array(hf['S'])

    ncells = A.shape[-1]
    if C.shape[0] != ncells:
        raise ValueError('Malformed C, A.shape={}, C.shape={}, S.shape={}'.format(str(A.shape), str(C.shape), str(S.shape)))
    if S.shape[0] != ncells:
        raise ValueError('Malformed S, A.shape={}, C.shape={}, S.shape={}'.format(str(A.shape), str(C.shape), str(S.shape)))
    if S.shape[1] != C.shape[1]:
        raise ValueError('Malformed S and C, A.shape={}, C.shape={}, S.shape={}'.format(str(A.shape), str(C.shape), str(S.shape)))

    # determine the indices used for each cell set
    cs_indices = list()
    last_index = 0
    for ifile, cfile, efile in zip(input_movie_files, output_cell_set_files, output_events_files):

        # open the movie to get timing and spacing info
        mov = Movie.read(ifile)
        nframes = mov.timing.num_samples - len(mov.timing.dropped)
        del mov
        sidx = last_index
        eidx = sidx + nframes
        cs_indices.append(np.arange(sidx, eidx))
        last_index = eidx

    for ifile, cfile, efile, cs_idx in zip(input_movie_files, output_cell_set_files, output_events_files, cs_indices):

        # open the movie to get timing and spacing info
        mov = Movie.read(ifile)

        # write the cell set file
        if os.path.exists(cfile):
            os.remove(cfile)

        cs = CellSet.write(cfile, timing=mov.timing, spacing=mov.spacing)

        num_digits = len(str(ncells))
        fmt_string = 'C{{}}'.format(':0{}d'.format(num_digits - 1))

        cell_names = [fmt_string.format(k) for k in range(ncells)]
        for k in range(ncells):
            trc = C[k, cs_idx]
            trc -= trc.min()
            cs.set_cell_data(k, A[:, :, k].astype('float32'), trc, cell_names[k])
        for k in range(ncells):
            cs.set_cell_status(k, 'accepted')

        # write the events file
        if os.path.exists(efile):
            os.remove(efile)

        events = EventSet.write(efile, timing=mov.timing, cell_names=cell_names)
        offsets = np.array([x.to_usecs() for x in cs.timing.get_offsets_since_start()], dtype='uint64')

        # there may be dropped frames in the movie, the offsets array has to be adjusted to accommodate this
        offsets_i = np.ones(cs.timing.num_samples, dtype='bool')
        offsets_i[cs.timing.dropped] = False
        offsets_without_dropped_frames = offsets[offsets_i]

        for k in range(ncells):
            s = S[k, cs_idx]
            s /= s.max()
            nz = s > event_threshold
            event_amps = s[nz]
            event_offsets = offsets_without_dropped_frames[nz]
            events.set_cell_data(k, event_offsets, event_amps)

        del mov
        del cs
        del events
