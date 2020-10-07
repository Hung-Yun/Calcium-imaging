"""
This module contains an example that demonstrates the standard workflow
using two series of four movies.
"""

import os
import isx


def main():
    """
    Runs the standard workflow example from the demo data.
    """

    # Define the recording names of the two series.
    data_dir = os.path.join('path_to_demo_data', 'S1prism_AAV1_demo_with_LR', 'S1prism_AAV1_demo_v2_data')
    series_rec_names = {
            'day_1' :
            [
                'recording_20160613_105808',
                'recording_20160613_110507',
                'recording_20160613_111207',
                'recording_20160613_111907',
            ],
            'day_4' :
            [
                'recording_20160616_102500',
                'recording_20160616_103200',
                'recording_20160616_103900',
                'recording_20160616_104600',
            ],
    }

    # Process each series all the way to event detection.
    output_dir = os.path.join(data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    proc_movie_files = []
    proc_cs_files = []
    for series_name, rec_names in series_rec_names.items():

        # Generate the recording file paths.
        # Here, we start with pre-processed '*-PP-PP.isxd' files, but you would
        # typically start with recording .xml files instead.
        rec_files = [os.path.join(data_dir, name + '-PP-PP.isxd') for name in rec_names]
        #rec_files = [os.path.join(data_dir, name + '.xml') for name in rec_names]

        # Preprocess the recordings by spatially downsampling by a factor of 2.
        # In the example data set, this has already been performed, but we do it
        # again for demonstration purposes and to speed everything up here.
        pp_files = isx.make_output_file_paths(rec_files, output_dir, 'PP')
        isx.preprocess(rec_files, pp_files, spatial_downsample_factor=2)

        # Perform spatial bandpass filtering with defaults.
        bp_files = isx.make_output_file_paths(pp_files, output_dir, 'BP')
        isx.spatial_filter(pp_files, bp_files, low_cutoff=0.005, high_cutoff=0.500)

        # Motion correct the movies using the mean projection as a reference frame.
        mean_proj_file = os.path.join(output_dir, '{}-mean_image.isxd'.format(series_name))
        isx.project_movie(bp_files, mean_proj_file, stat_type='mean')
        mc_files = isx.make_output_file_paths(bp_files, output_dir, 'MC')
        translation_files = isx.make_output_file_paths(mc_files, output_dir, 'translations', 'csv')
        crop_rect_file = os.path.join(output_dir, '{}-crop_rect.csv'.format(series_name))
        isx.motion_correct(
                bp_files, mc_files, max_translation=20, reference_file_name=mean_proj_file,
                low_bandpass_cutoff=None, high_bandpass_cutoff=None,
                output_translation_files=translation_files, output_crop_rect_file=crop_rect_file)

        # Run DF/F on the motion corrected movies.
        dff_files = isx.make_output_file_paths(mc_files, output_dir, 'DFF')
        isx.dff(mc_files, dff_files, f0_type='mean')

        # Run PCA-ICA on the DF/F movies.
        # Note that you will have to manually determine the number of cells, which we
        # determined here as 180.
        # Increase the block_size to increase speed at the expense of more memory usage.
        ic_files = isx.make_output_file_paths(dff_files, output_dir, 'PCA_ICA')
        num_cells = 180
        isx.pca_ica(dff_files, ic_files, num_cells, int(1.15 * num_cells), block_size=1000)

        # Run event detection on the PCA-ICA cell sets.
        event_files = isx.make_output_file_paths(ic_files, output_dir, 'ED')
        isx.event_detection(ic_files, event_files, threshold=5)

        # Automatically accept and reject cells based on their cell metrics
        # Only accept cells that have a nonzero event rate, an SNR greater
        # than 3, and only one connected component after thresholding
        auto_ar_filters = [('SNR', '>', 3), ('Event Rate', '>', 0), ('# Comps', '=', 1)]
        isx.auto_accept_reject(ic_files, event_files, filters=auto_ar_filters)

        # Store the processed movies and cell sets for longitudinal registration.
        proc_movie_files += dff_files
        proc_cs_files += ic_files

    # Perform longitudinal registration on the processed movies and cell sets
    # to align the two days of data.
    lr_cs_files = isx.make_output_file_paths(proc_cs_files, output_dir, 'LR')
    lr_movie_files = isx.make_output_file_paths(proc_movie_files, output_dir, 'LR')
    lr_csv_file = os.path.join(output_dir, 'LR.csv')
    isx.longitudinal_registration(
            proc_cs_files, lr_cs_files, input_movie_files=proc_movie_files,
            output_movie_files=lr_movie_files, csv_file=lr_csv_file, accepted_cells_only=True)

    # Then run event detection and automatically classify the cells based on their
    # cell metrics.
    lr_event_files = isx.make_output_file_paths(lr_cs_files, output_dir, 'ED')
    isx.event_detection(lr_cs_files, lr_event_files, threshold=5)
    auto_ar_filters = [('SNR', '>', 3), ('Event Rate', '>', 0), ('# Comps', '=', 1)]
    isx.auto_accept_reject(lr_cs_files, lr_event_files, filters=auto_ar_filters)

    # Finally, export the registered movies, cell sets, and event sets to non-native
    # formats (TIFF and CSV).
    tiff_movie_file = os.path.join(output_dir, 'DFF-LR.tif')
    isx.export_movie_to_tiff(lr_movie_files, tiff_movie_file, write_invalid_frames=True)

    tiff_image_file = os.path.join(output_dir, 'DFF-PCA_ICA-LR.tif')
    csv_trace_file = os.path.join(output_dir, 'DFF-PCA_ICA-LR.csv')
    isx.export_cell_set_to_csv_tiff(lr_cs_files, csv_trace_file, tiff_image_file, time_ref='start')

    csv_event_file = os.path.join(output_dir, 'DFF-PCA_ICA-LR-ED.csv')
    isx.export_event_set_to_csv(lr_event_files, csv_event_file, time_ref='start')


if __name__ == '__main__':
    main()
