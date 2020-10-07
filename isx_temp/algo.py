"""
The algo module deals with running algorithms on movies,
cell sets, and event sets.
"""

import os
import ctypes

import numpy as np

import isx._internal


def preprocess(
        input_movie_files, output_movie_files,
        temporal_downsample_factor=1, spatial_downsample_factor=1,
        crop_rect=None, fix_defective_pixels=True, trim_early_frames=True):
    """
    Preprocess movies, optionally spatially and temporally downsampling and cropping.

    For more details see :ref:`preprocessing`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths to the input movies.
    output_movie_files : list<str>
        The file paths to write the preprocessed output movies to.
        This must be the same length as input_movie_files.
    temporal_downsample_factor : int >= 1
        The factor that determines how much the movie is temporally downsampled.
    spatial_downsample_factor : int >= 1
        The factor that determines how much the movie is spatially downsampled.
    crop_rect : 4-tuple<int>
        A list of 4 pixel locations that determines the crop rectangle: [top, left, bottom, right].
    fix_defective_pixels : bool
        If True, then check for defective pixels and correct them.
    trim_early_frames : bool
        If True, then remove early frames that are usually dark or dim.
    """
    if crop_rect is None:
        crop_rect = (-1, -1, -1, -1)
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    isx._internal.c_api.isx_preprocess_movie(
            num_files, in_arr, out_arr, temporal_downsample_factor, spatial_downsample_factor,
            crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3], fix_defective_pixels, trim_early_frames)


def de_interleave(input_movie_files, output_movie_files, in_efocus_values):
    """
    De-interleave multiplane movies.

    For more details see :ref:`deInterleave`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths to the input movies.
        All files should have the same efocus values and same number of planes.
    output_movie_files : list<str>
        The file paths to write the de-interleaved output movies to.
        This must be the length of input_movie_files * the number of planes.
        The sequence of every number of planes elements must match the sequence of efocus values.
        E.g: [in_1, in_2], [efocus1, efocus2] -> [out_1_efocus1, out_1_efocus2, out_2_efocus1, out_2_efocus2]
    in_efocus_values : list<int>
        The efocus value for each planes.
        This must in range 0 <= efocus <= 1000.
    """
    efocus_arr = isx._internal.list_to_ctypes_array(in_efocus_values, ctypes.c_uint16)
    num_planes = len(in_efocus_values)
    num_in_files, in_arr = isx._internal.check_input_files(input_movie_files)
    num_output_files, out_arr = isx._internal.check_input_files(output_movie_files)

    if num_output_files != num_in_files * num_planes:
        raise ValueError('Number of output files must match the number of input files times the number of planes.')

    isx._internal.c_api.isx_deinterleave_movie(num_in_files, num_planes, efocus_arr, in_arr, out_arr)


def motion_correct(
        input_movie_files, output_movie_files, max_translation=20,
        low_bandpass_cutoff=0.004, high_bandpass_cutoff=0.016, roi=None,
        reference_segment_index=0, reference_frame_index=0, reference_file_name='',
        global_registration_weight=1.0, output_translation_files=None,
        output_crop_rect_file=None):
    """
    Motion correct movies to a reference frame.

    For more details see :ref:`motionCorrection`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to motion correct.
    output_movie_files : list<str>
        The file paths of the output movies.
        This must be the same length as input_movie_files.
    max_translation : int > 0
        The maximum translation allowed by motion correction in pixels.
    low_bandpass_cutoff : float > 0
        If not None, then the low cutoff of the spatial filter applied to each frame prior to motion estimation.
    high_bandpass_cutoff : float > 0
        If not None, then the high cutoff for a spatial filter applied to each frame prior to motion estimation.
    roi : Nx2 array-like
        If not None, each row is a vertex of the ROI to use for motion estimation.
        Otherwise, use the entire frame.
    reference_segment_index : int > 0
        If a reference frame is to be specified, this parameter indicates the index of the movie whose frame will
        be utilized, with respect to input_movie_files.
        If only one movie is specified to be motion corrected, this parameter must be 0.
    reference_frame_index : int > 0
        Use this parameter to specify the index of the reference frame to be used, with respect to reference_segment_index.
        If reference_file_name is specified, this parameter, as well as reference_segment_index, is ignored.
    reference_file_name : str
        If an external reference frame is to be used, this parameter should be set to path of the .isxd file
        that contains the reference image.
    global_registration_weight : 0.05 <= float <= 1
        When this is set to 1, only the reference frame is used for motion estimation.
        When this is less than 1, the previous frame is also used for motion estimation.
        The closer this value is to 0, the more the previous frame is used and the less
        the reference frame is used.
    output_translation_files : list<str>
        A list of file names to write the X and Y translations to.
        Must be either None, in which case no files are written, or a list of valid file names equal
        in length to the number of input and output file names.
        The output translations are written into a .csv file with three columns.
        The first two columns, "translationX" and "translationY", store the X and Y translations from
        each frame to the reference frame respectively.
        The third column contains the time of the frame since the beginning of the movie.
        The first row stores the column names as a header.
        Each subsequent row contains the X translation, Y translation, and time offset for that frame.
    output_crop_rect_file : str
        The path to a file that will contain the crop rectangle applied to the input movies to generate the output
        movies.
        The format of the crop rectangle is a comma separated list: x,y,width,height.
    """
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)

    use_low = int(low_bandpass_cutoff is not None)
    use_high = int(high_bandpass_cutoff is not None)

    if use_low == 0:
        low_bandpass_cutoff = 0.0
    if use_high == 0:
        high_bandpass_cutoff = 1.0

    # The first two elements tell the C layer the number of ROIs, then the
    # number of vertices in the first ROI.
    if roi is not None:
        roi_np = isx._internal.convert_to_nx2_numpy_array(roi, np.int, 'roi')
        roi_arr = isx._internal.list_to_ctypes_array([1, roi_np.shape[0]] + list(roi_np.ravel()), ctypes.c_int)
    else:
        roi_arr = isx._internal.list_to_ctypes_array([0], ctypes.c_int)

    if reference_file_name is None:
        ref_file_name = ''
    else:
        ref_file_name = reference_file_name

    out_trans_arr = isx._internal.list_to_ctypes_array([''], ctypes.c_char_p)
    write_output_translations = int(output_translation_files is not None)
    if write_output_translations:
        out_trans_files = isx._internal.ensure_list(output_translation_files)
        assert len(out_trans_files) == num_files, "Number of output translation files must match number of input movies ({} != {})".format(len(out_trans_files), len(in_arr))
        out_trans_arr = isx._internal.list_to_ctypes_array(out_trans_files, ctypes.c_char_p)

    write_crop_rect = int(output_crop_rect_file is not None)
    if not write_crop_rect:
        output_crop_rect_file = ''

    isx._internal.c_api.isx_motion_correct_movie(
            num_files, in_arr, out_arr, max_translation,
            use_low, low_bandpass_cutoff, use_high, high_bandpass_cutoff,
            roi_arr, reference_segment_index, reference_frame_index,
            ref_file_name.encode('utf-8'), global_registration_weight,
            write_output_translations, out_trans_arr,
            write_crop_rect, output_crop_rect_file.encode('utf-8'))


def pca_ica(
        input_movie_files, output_cell_set_files, num_pcs, num_ics, unmix_type='spatial',
        ica_temporal_weight=0, max_iterations=100, convergence_threshold=1e-5, block_size=1000):
    """
    Run PCA-ICA cell identification on movies.

    For more details see :ref:`PCA_ICA`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to run PCA-ICA on.
    output_cell_set_files : list<str>
        The paths of the output cell set files. Must be same length as input_movie_files.
    num_pcs : int > 0
        The number of principal components (PCs) to estimate.
    num_ics : int > 0
        The number of independent components (ICs) to estimate. Must be >= num_pcs.
    unmix_type : {'temporal', 'spatial', 'both'}
        The unmixing type or dimension.
    ica_temporal_weight : 0 <= float <= 1
        The temporal weighting factor used for ICA.
    max_iterations : int > 0
        The maximum number of iterations for ICA.
    convergence_threshold : float > 0
        The convergence threshold for ICA.
    block_size : int > 0
        The size of the blocks for the PCA step. The larger the block size, the more memory that will be used.

    Returns
    -------
    bool
        True if PCA-ICA converged, False otherwise.
    """
    unmix_type_int = isx._internal.lookup_enum('unmix_type', isx._internal.ICA_UNMIX_FROM_STRING, unmix_type)
    if ica_temporal_weight < 0 or ica_temporal_weight > 1:
        raise ValueError("ica_temporal_weight must be between zero and one")

    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_cell_set_files)
    converged = ctypes.c_int()
    isx._internal.c_api.isx_pca_ica_movie(
            num_files, in_arr, out_arr, num_pcs, num_ics, unmix_type_int, ica_temporal_weight,
            max_iterations, convergence_threshold, block_size, ctypes.byref(converged), 0)

    return converged.value > 0


def spatial_filter(
        input_movie_files, output_movie_files, low_cutoff=0.005, high_cutoff=0.500,
        retain_mean=False, subtract_global_minimum=True):
    """
    Apply spatial bandpass filtering to each frame of one or more movies.

    For more details see :ref:`spatialBandpassFilter`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to filter.
    output_movie_files : list<str>
        The file paths of the output movies. Must be the same length as input_movie_files.
    low_cutoff : float > 0
        If not None, then the low cutoff for the spatial filter.
    high_cutoff : float > 0
        If not None, then the high cutoff for the spatial filter.
    retain_mean : bool
        If True, retain the mean pixel intensity for each frame (the DC component).
    subtract_global_minimum : bool
        If True, compute the minimum pixel intensity across all movies, and subtract this
        after frame-by-frame mean subtraction.
        By doing this, all pixel intensities will stay positive valued, and integer-valued
        movies can stay that way.
    """
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    use_low = int(low_cutoff is not None)
    use_high = int(high_cutoff is not None)
    isx._internal.c_api.isx_spatial_band_pass_movie(
            num_files, in_arr, out_arr, use_low, low_cutoff, use_high, high_cutoff,
            int(retain_mean), int(subtract_global_minimum))


def dff(input_movie_files, output_movie_files, f0_type='mean'):
    """
    Compute DF/F movies, where each output pixel value represents a relative change
    from a baseline.

    For more details see :ref:`DFF`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the input movies.
    output_movie_files : list<str>
        The file paths of the output movies.
    f0_type : {'mean', 'min}
        The reference image or baseline image used to compute DF/F.
    """
    f0_type_int = isx._internal.lookup_enum('f0_type', isx._internal.DFF_F0_FROM_STRING, f0_type)
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    isx._internal.c_api.isx_delta_f_over_f(num_files, in_arr, out_arr, f0_type_int)


def project_movie(input_movie_files, output_image_file, stat_type='mean'):
    """
    Project movies to a single statistic image.

    For more details see :ref:`movieProjection`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to project.
    output_image_file : str
        The file path of the output image.
    stat_type: {'mean', 'min', 'max'}
        The type of statistic to compute.
    """
    stat_type_int = isx._internal.lookup_enum('stat_type', isx._internal.PROJECTION_FROM_STRING, stat_type)
    num_files, in_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_project_movie(num_files, in_arr, output_image_file.encode('utf-8'), stat_type_int)


def event_detection(
        input_cell_set_files, output_event_set_files, threshold=5, tau=0.2,
        event_time_ref='beginning', ignore_negative_transients=True, accepted_cells_only=False):
    """
    Perform event detection on cell sets.

    For more details see :ref:`eventDetection`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to perform event detection on.
    output_event_set_files : list<str>
        The file paths of the output event sets.
    threshold : float > 0
        The threshold in median-absolute-deviations that the trace has to cross to be considered an event.
    tau : float > 0
        The minimum time in seconds that an event has to last in order to be considered.
    event_time_ref : {'maximum', 'beginning', 'mid_rise'}
        The temporal reference that defines the event time.
    ignore_negative_transients : bool
        Whether or not to ignore negative events.
    accepted_cells_only : bool
        If True, detect events only for accepted cells.
    """
    event_time_ref_int = isx._internal.lookup_enum('event_time_ref', isx._internal.EVENT_REF_FROM_STRING, event_time_ref)
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_cell_set_files, output_event_set_files)
    isx._internal.c_api.isx_event_detection(
            num_files, in_arr, out_arr, threshold, tau, event_time_ref_int,
            int(ignore_negative_transients), int(accepted_cells_only))


def trim_movie(input_movie_file, output_movie_file, crop_segments, keep_start_time=False):
    """
    Trim frames from a movie to produce a new movie.

    For more details see :ref:`trimMovie`.

    Arguments
    ---------
    input_movie_file : str
        The file path of the movie.
    output_movie_file : str
        The file path of the trimmed movie.
    crop_segments : Nx2 array-like
        A numpy array of shape (num_segments, 2), where each row contains the start and
        end indices of frames that will be cropped out of the movie. Or a list like:
        [(start_index1, end_index1), (start_index2, end_index2), ...].
    keep_start_time : bool
        If true, keep the start time of the movie, even if some of its initial frames are to be trimmed.
    """
    num_files, in_arr, out_arr = isx._internal.check_input_and_output_files(input_movie_file, output_movie_file)
    if num_files != 1:
        raise TypeError("Only one movie can be specified.")

    crop_segs = isx._internal.convert_to_nx2_numpy_array(crop_segments, np.int, 'crop_segments')
    indices_arr = isx._internal.list_to_ctypes_array([crop_segs.shape[0]] + list(crop_segs.ravel()), ctypes.c_int)

    isx._internal.c_api.isx_temporal_crop_movie(1, in_arr, out_arr, indices_arr, keep_start_time)


def apply_cell_set(input_movie_files, input_cell_set_file, output_cell_set_files, threshold):
    """
    Apply the images of a cell set to movies, producing a new cell sets.

    For more details see :ref:`applyContours`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to apply the cell set to.
    input_cell_set_file : list<str>
        The file path of the cell set to apply.
    output_cell_set_files : list<str>
        The file paths of the output cell sets that will contain the images and new traces.
    threshold : 0 >= float >= 1
        A threshold that will be applied to each footprint prior to application.
        This indicates the fraction of the maximum image value that will be used as the
        absolute threshold.
    """
    num_movies, in_movie_arr, out_cs_arr = isx._internal.check_input_and_output_files(input_movie_files, output_cell_set_files)
    num_cs_in, in_cs_arr = isx._internal.check_input_files(input_cell_set_file)
    if num_cs_in != 1:
        raise TypeError("Only one input cell set can be specified.")
    isx._internal.c_api.isx_apply_cell_set(num_movies, in_movie_arr, out_cs_arr, in_cs_arr[0], threshold)


def longitudinal_registration(
        input_cell_set_files, output_cell_set_files, input_movie_files=[], output_movie_files=[],
        csv_file='', min_correlation=0.5, accepted_cells_only=False,
        transform_csv_file='', crop_csv_file=''):
    """
    Run longitudinal registration on multiple cell sets.

    Optionally, also register the corresponding movies the cell sets were derived from.

    For more details see :ref:`LongitudinalRegistration`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to register.
    output_cell_set_files : list<str>
        The file paths of the output cell sets.
    input_movie_files : list<str>
        The file paths of the associated input movies (optional).
    output_movie_files: list<str>
        The file paths of the output movies (optional)
    csv_file : str
        The path of the output CSV file to be written (optional).
    min_correlation : 0 >= float >= 1
        The minimum correlation between cells to be considered a match.
    accepted_cells_only : bool
        Whether or not to use accepted cells from the input cell sets only, or to use both accepted and undecided cells.
    transform_csv_file : str
        The file path of the CSV file to store the affine transform parameters
        from the reference cellset to each cellset.
        Each row represents an input cell set and contains the values in the
        2x3 affine transform matrix in a row-wise order.
        I.e. if we use a_{i,j} to represent the values in the 2x2 upper left
        submatrix and t_{i} to represent the translations, the values are
        written in the order: a_{0,0}, a_{0,1}, t_{0}, a_{1,0}, a_{1,1}, t_{1}.
    crop_csv_file : str
        The file path of the CSV file to store the crop rectangle applied after
        transforming the cellsets and movies.
        The format of the crop rectangle is a comma separated list: x,y,width,height.
    """
    num_cell_files, in_cell_arr, out_cell_arr = isx._internal.check_input_and_output_files(input_cell_set_files, output_cell_set_files)
    num_movie_files, in_movie_arr, out_movie_arr = isx._internal.check_input_and_output_files(input_movie_files, output_movie_files)
    if (num_movie_files > 0) and (num_movie_files != num_cell_files):
        raise ValueError("If specified, the number of movies must be the same as the number of cell sets.")
    isx._internal.c_api.isx_longitudinal_registration(num_cell_files, in_cell_arr, out_cell_arr, in_movie_arr, out_movie_arr, csv_file.encode('utf-8'), min_correlation, int(not accepted_cells_only), int(num_movie_files > 0), transform_csv_file.encode('utf-8'), crop_csv_file.encode('utf-8'))


def auto_accept_reject(input_cell_set_files, input_event_set_files, filters=None):
    """
    Automatically classify cell statuses as accepted or rejected.

    For more details see :ref:`autoAcceptReject`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to classify.
    input_event_set_files : list<str>
        The file paths of the event sets to use for classification.
    filters : list<3-tuple>
        Each element describes a filter as (<statistic>, <operator>, <value>).
        The statistic must be one of {'# Comps', 'Cell Size', 'SNR', 'Event Rate'}.
        The operator must be one of {'<', '=', '>'}.
        The value is a floating point number.
    """
    num_cell_sets, in_cell_arr = isx._internal.check_input_files(input_cell_set_files)
    num_event_sets, in_event_arr = isx._internal.check_input_files(input_event_set_files)

    statistics = []
    operators = []
    values = []
    num_filters = 0
    if filters is not None:
        if isinstance(filters, list):
            statistics, operators, values = map(list, zip(*filters))
            num_filters = len(filters)
        else:
            raise TypeError('Filters must be contained in a list.')

    in_statistics = isx._internal.list_to_ctypes_array(statistics, ctypes.c_char_p)
    in_operators = isx._internal.list_to_ctypes_array(operators, ctypes.c_char_p)
    in_values = isx._internal.list_to_ctypes_array(values, ctypes.c_double)

    isx._internal.c_api.isx_classify_cell_status(
            num_cell_sets, in_cell_arr, num_event_sets, in_event_arr,
            num_filters, in_statistics, in_operators, in_values,
            0, isx._internal.SizeTPtr())


def cell_metrics(input_cell_set_files, input_event_set_files, output_metrics_files):
    """
    Compute cell metrics for a given cell set and events combination.

    For more details see :ref:`cellMetrics`.

    Arguments
    ---------
    input_cell_set_files : list<str>
        One or more input cell sets.
    input_event_set_files : list<str>
        One or more events files associated with the input cell sets.
    output_metrics_files : list<str>
        One or more .csv files that will be written which contain cell metrics.
    """
    num_cs_in, in_cs_arr = isx._internal.check_input_files(input_cell_set_files)
    num_events_in, in_events_arr, out_arr = isx._internal.check_input_and_output_files(input_event_set_files, output_metrics_files)
    if num_cs_in != num_events_in:
        raise TypeError("The number of cell sets and events must be the same.")
    isx._internal.c_api.isx_compute_cell_metrics(num_cs_in, in_cs_arr, in_events_arr, out_arr)


def export_cell_contours(input_cell_set_file, output_json_file, threshold=0.0, rectify_first=True):
    """
    Export cell contours to a JSON file.

    If a cell image has multiple components the contour for each component is exported in a separate array.

    These are the contours calculated from preprocessed cell images as described in :ref:`cellMetrics`.

    Arguments
    ---------
    input_movie_file : str
        The file path of a cell set.
    output_json_file : str
        The file path to the output JSON file to be written.
    threshold : 0 >= float >= 1
        The threshold to apply to the footprint before computing the contour, specified as a
        fraction of the maximum pixel intensity.
    rectify_first : bool
        Whether or not to rectify the image (remove negative components) prior to computing the threshold.
    """
    num_cs_in, in_cs_arr, out_js_arr = isx._internal.check_input_and_output_files(input_cell_set_file, output_json_file)
    if num_cs_in != 1:
        raise TypeError("Only one input cell set can be specified.")
    isx._internal.c_api.isx_export_cell_contours(num_cs_in, in_cs_arr, out_js_arr, threshold, int(rectify_first))


def multiplane_registration(
        input_cell_set_files,
        output_cell_set_file,
        min_spat_correlation=0.5,
        temp_correlation_thresh=0.99,
        accepted_cells_only=False):
    """
    Identify unique signals in 4D imaging data using longitudinal registration
    of spatial footprints and temporal correlation of activity.

    :param input_cell_set_files: (list <str>) the file paths of the cell sets from de-interleaved multiplane movies.
    :param output_cell_set_file: (str) the file path of the output cell set of multiplane registration.
    :param min_spat_correlation: (0 <= float <= 1) the minimum spatial overlap between cells to be considered a match.
    :param temp_correlation_thresh: (0 <= float <= 1) the percentile of the comparison distribution below which
                                    activity correlations are considered from distinct signals
    :param accepted_cells_only: (bool) whether or not to include only accepted cells from the input cell sets.
    """
    if not 0 <= min_spat_correlation <= 1:
        raise TypeError("Spatial correlation must be between 0 and 1.")
    if not 0 <= temp_correlation_thresh <= 1:
        raise TypeError("Temporal correlation threshold must be between 0 and 1.")
    num_cs_in, in_cs_arr, out_cs_arr = isx._internal.check_input_and_output_files(input_cell_set_files, output_cell_set_file, True)
    isx._internal.c_api.isx_multiplane_registration(
        num_cs_in,
        in_cs_arr,
        out_cs_arr,
        min_spat_correlation,
        temp_correlation_thresh,
        accepted_cells_only
    )
