"""
    This module contains an example for running CNMFe through CaImAn.
"""

import os
import isx
import isx.cnmfe


def main():
    """
    Run the CaImAn CNMFe example.
    """

    # Define the recording path to the cell set and event data, as done in the standard workflow example
    data_dir = os.path.join('/path/to/demo/dataset', 'S1prism_AAV1_demo_v2_data')

    # Pick out a movie from a particular session.
    mov_file = os.path.join(data_dir, 'recording_20160616_104600-PP-PP-BP.isxd')

    # spatially downsample the movie 2X, making sure to set fix_defective_pixels to False because the movie has already
    # been preprocessed, and we don't want to median filter the movie a second time.
    pp_mov_file = os.path.join(data_dir, 'recording_20160616_104600-PP-PP-BP-PP.isxd')
    if not os.path.exists(pp_mov_file):
        isx.preprocess(mov_file, pp_mov_file, temporal_downsample_factor=1,
                       spatial_downsample_factor=2, fix_defective_pixels=False)

    # specify the output cell set and events files
    cellset_file = os.path.join(data_dir, 'recording_20160616_104600-PP-PP-BP-PP-CNMFE.isxd')
    events_file = os.path.join(data_dir, 'recording_20160616_104600-PP-PP-BP-PP-CNMFE-ED.isxd')

    # remove output files if they already exist
    if os.path.exists(cellset_file):
        os.remove(cellset_file)
    if os.path.exists(events_file):
        os.remove(events_file)

    # run CNMFe using just a single process
    isx.cnmfe.run_cnmfe(pp_mov_file, cellset_file, events_file, num_processes=1, K=20, rf=[25, 25], stride=6,
                        gSig=5, gSiz=10, min_pnr=30, min_corr=0.9, event_threshold=0.1)

    # Next - import the output cell set and events files underneath the preprocessed movie to visualize the cells and
    # traces


if __name__ == '__main__':
    main()
