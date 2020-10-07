"""
The behavior of this may change in the future, so we recommend that you
do not use it and we will not support it.
"""

import os
import ctypes
import atexit
import json
import warnings

import numpy as np

# Load the C library.
# For Windows we must temporarily change directory to load the
# C library. We change back afterwards.
_this_dir = os.path.dirname(os.path.realpath(__file__))
_lib_dir = os.path.join(_this_dir, '..')
_is_windows = os.name == 'nt'
if _is_windows:
    _isx_lib_name = 'isxpublicapi'
    _cwd = os.getcwd()
    os.chdir(_lib_dir)
else:
    _isx_lib_name = os.path.join(_lib_dir, 'libisxpublicapi.so')

c_api = ctypes.CDLL(_isx_lib_name)

if _is_windows:
    os.chdir(_cwd)

# Define utility functions for interaction with C library.


def validate_ptr(ptr):
    if not ptr:
        raise RuntimeError('Underlying pointer is null. Try using the read or write function instead of the constructor.')


def list_to_ctypes_array(input_list, input_type):
    """ Convert a list of a certain type for ctypes so it can be passed as a pointer.
    """
    if not isinstance(input_list, list):
        raise TypeError('Input must be contained in a list.')

    array = (input_type * len(input_list))()
    for i, s in enumerate(input_list):
        if input_type is ctypes.c_char_p:
            array[i] = s.encode('utf-8')
        else:
            array[i] = s

    return array


def numpy_array_to_ctypes_array(numpy_array, element_type):
    ctypes_array = (element_type * numpy_array.size)()
    for i in range(numpy_array.size):
        ctypes_array[i] = numpy_array[i]
    return ctypes_array


def ctypes_ptr_to_list(ptr_to_element0, num_elements):
    py_list = []
    for i in range(num_elements):
        py_list.append(ptr_to_element0[i])
    return py_list


def convert_to_1d_numpy_array(input_array, dtype, name):
    try:
        array = np.array(input_array, dtype=dtype)
        assert array.ndim == 1
    except Exception:
        raise TypeError('{} must be an 1D array-like'.format(name))
    return array


def convert_to_nx2_numpy_array(input_array, dtype, name):
    try:
        array = np.array(input_array, dtype=dtype).reshape((-1, 2))
        assert (array.ndim == 2) and (array.shape[1] == 2)
    except Exception:
        raise TypeError('{} must be an Nx2 array-like'.format(name))
    return array


def ensure_list(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    return inputs


def check_input_files(input_file_paths):
    input_file_paths_arr = ensure_list(input_file_paths)
    num_files = len(input_file_paths_arr)
    in_arr = list_to_ctypes_array(input_file_paths_arr, ctypes.c_char_p)
    return num_files, in_arr


def check_input_and_output_files(input_file_paths, output_file_paths, allow_single_output=False):
    num_input_files, in_arr = check_input_files(input_file_paths)
    output_file_paths_arr = ensure_list(output_file_paths)
    if not allow_single_output:
        if num_input_files != len(output_file_paths_arr):
            raise ValueError('Number of input files must match the number of output files.')
    out_arr = list_to_ctypes_array(output_file_paths_arr, ctypes.c_char_p)
    return num_input_files, in_arr, out_arr


def _standard_errcheck(return_code, func, args=None):
    """ The standard function to use for errcheck for CDLL functions.
    """
    if return_code != 0:
        error_message = c_api.isx_get_last_exception_string().decode()
        raise Exception("Error calling C library function {}.\n{}".format(func.__name__, error_message))
    return args


def get_mode_from_read_only(read_only):
    if read_only:
        return 'r'
    else:
        return 'w'


def get_acquisition_info(ptr, get_info_func, get_info_size_func):
    validate_ptr(ptr)
    info_size = ctypes.c_size_t(0)
    get_info_size_func(ptr, ctypes.byref(info_size))
    info_size = info_size.value

    info_str = ctypes.create_string_buffer(info_size)
    get_info_func(ptr, info_str, info_size)
    info_str = info_str.value.decode('utf-8')
    return json.loads(info_str)


def ndarray_as_type(array, dtype):
    """ Convert a numpy ndarray using the astype method but with a warning.
    """
    if array.dtype != dtype:
        warnings.warn('Converting from {} to {}.'.format(array.dtype, dtype))
        return array.astype(dtype)
    return array


# Define maps to/from strings to enum values.

def _reverse_dictionary(dictionary):
    return {v: k for k, v in dictionary.items()}


def lookup_enum(enum_name, enum_dict, key):
    try:
        return enum_dict[key]
    except KeyError:
        raise ValueError("Unknown {} '{}'. Options are {}.".format(enum_name, key, ', '.join(enum_dict.keys())))


DATA_TYPE_FROM_NUMPY = {
        np.uint16: c_api.isx_get_data_type_u16(),
        np.float32: c_api.isx_get_data_type_f32(),
}
DATA_TYPE_TO_NUMPY = _reverse_dictionary(DATA_TYPE_FROM_NUMPY)

CELL_STATUS_FROM_STRING = {
        'accepted': c_api.isx_get_cell_status_accepted(),
        'undecided': c_api.isx_get_cell_status_undecided(),
        'rejected': c_api.isx_get_cell_status_rejected(),
}
CELL_STATUS_TO_STRING = _reverse_dictionary(CELL_STATUS_FROM_STRING)

ICA_UNMIX_FROM_STRING = {
        'temporal': c_api.isx_get_ica_unmix_type_temporal(),
        'spatial': c_api.isx_get_ica_unmix_type_spatial(),
        'both': c_api.isx_get_ica_unmix_type_both(),
}
ICA_UNMIX_TO_STRING = _reverse_dictionary(ICA_UNMIX_FROM_STRING)

DFF_F0_FROM_STRING = {
        'mean': c_api.isx_get_dff_image_type_mean(),
        'min': c_api.isx_get_dff_image_type_min(),
}
DFF_F0_TO_STRING = _reverse_dictionary(DFF_F0_FROM_STRING)

PROJECTION_FROM_STRING = {
        'mean': c_api.isx_get_projection_type_mean(),
        'min': c_api.isx_get_projection_type_min(),
        'max': c_api.isx_get_projection_type_max(),
}
PROJECTION_TO_STRING = _reverse_dictionary(PROJECTION_FROM_STRING)

EVENT_REF_FROM_STRING = {
        'maximum': c_api.isx_get_event_time_reference_maximum(),
        'beginning': c_api.isx_get_event_time_reference_beginning(),
        'mid_rise': c_api.isx_get_event_time_reference_mid_rise(),
}
EVENT_REF_TO_STRING = _reverse_dictionary(EVENT_REF_FROM_STRING)

TIME_REF_FROM_STRING = {
        'start': c_api.isx_get_time_reference_start(),
        'unix': c_api.isx_get_time_reference_unix(),
}
TIME_REF_TO_STRING = _reverse_dictionary(TIME_REF_FROM_STRING)


# Common types
CharPtrPtr = ctypes.POINTER(ctypes.c_char_p)
IntPtr = ctypes.POINTER(ctypes.c_int)
FloatPtr = ctypes.POINTER(ctypes.c_float)
DoublePtr = ctypes.POINTER(ctypes.c_double)
UInt16Ptr = ctypes.POINTER(ctypes.c_uint16)
Int64Ptr = ctypes.POINTER(ctypes.c_int64)
UInt64Ptr = ctypes.POINTER(ctypes.c_uint64)
SizeTPtr = ctypes.POINTER(ctypes.c_size_t)


# Common structs

class IsxRatio(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int64),
                ("den", ctypes.c_int64)]

    def __eq__(self, other):
        return (self.num == other.num) and (self.den == other.den)

    def __repr__(self):
        return 'IsxRatio({}, {})'.format(self.num, self.den)


class IsxTime(ctypes.Structure):
    _fields_ = [("secs_since_epoch", IsxRatio),
                ("utc_offset", ctypes.c_int32)]

    def __eq__(self, other):
        return ((self.secs_since_epoch == other.secs_since_epoch) and
                (self.utc_offset == other.utc_offset))

    def __repr__(self):
        return 'IsxTime({}, {})'.format(self.secs_since_epoch, self.utc_offset)


class IsxIndexRange(ctypes.Structure):
    _fields_ = [("first", ctypes.c_size_t),
                ("last", ctypes.c_size_t)]


class IsxTimingInfo(ctypes.Structure):
    _fields_ = [("start", IsxTime),
                ("step", IsxRatio),
                ("num_samples", ctypes.c_size_t),
                ("dropped", SizeTPtr),
                ("num_dropped", ctypes.c_size_t),
                ("cropped_first", SizeTPtr),
                ("cropped_last", SizeTPtr),
                ("num_cropped", ctypes.c_size_t)]

    def __repr__(self):
        return 'IsxTimingInfo({}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
                self.num_samples, self.step, self.start, self.num_dropped, self.dropped, self.num_cropped, self.cropped_first, self.cropped_last)


class IsxSpacingInfo(ctypes.Structure):
    _fields_ = [("num_cols", ctypes.c_size_t),
                ("num_rows", ctypes.c_size_t),
                ("pixel_width", IsxRatio),
                ("pixel_height", IsxRatio),
                ("left", IsxRatio),
                ("top", IsxRatio)]

    def __eq__(self, other):
        return ((self.num_cols == other.num_cols) and
                (self.num_rows == other.num_rows) and
                (self.pixel_width == other.pixel_width) and
                (self.pixel_height == other.pixel_height) and
                (self.left == other.left) and
                (self.top == other.top))

    def __repr__(self):
        return 'IsxSpacingInfo({}, {}, {}, {}, {}, {})'.format(self.num_cols, self.num_rows, self.pixel_width, self.pixel_height, self.left, self.top)

    @classmethod
    def from_num_pixels(cls, num_pixels):
        return cls(
                num_pixels[1], num_pixels[0],
                IsxRatio(3, 1), IsxRatio(3, 1),
                IsxRatio(0, 1), IsxRatio(0, 1))


# Movie struct and methods

class IsxMovie(ctypes.Structure):
    _fields_ = [("id", ctypes.c_size_t),
                ("timing", IsxTimingInfo),
                ("spacing", IsxSpacingInfo),
                ("data_type", ctypes.c_int),
                ("read_only", ctypes.c_bool),
                ("file_path", ctypes.c_char_p)]
IsxMoviePtr = ctypes.POINTER(IsxMovie)

c_api.isx_read_movie.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(IsxMoviePtr)]
c_api.isx_read_movie.errcheck = _standard_errcheck

c_api.isx_write_movie.argtypes = [
    ctypes.c_char_p,
    IsxTimingInfo,
    IsxSpacingInfo,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(IsxMoviePtr)]
c_api.isx_write_movie.errcheck = _standard_errcheck

c_api.isx_movie_get_frame_data_u16.argtypes = [
    IsxMoviePtr,
    ctypes.c_size_t,
    UInt16Ptr]
c_api.isx_movie_get_frame_data_u16.errcheck = _standard_errcheck

c_api.isx_movie_get_frame_data_f32.argtypes = [
    IsxMoviePtr,
    ctypes.c_size_t,
    FloatPtr]
c_api.isx_movie_get_frame_data_f32.errcheck = _standard_errcheck

c_api.isx_movie_write_frame_u16.argtypes = [
    IsxMoviePtr,
    ctypes.c_size_t,
    UInt16Ptr]
c_api.isx_movie_write_frame_u16.errcheck = _standard_errcheck

c_api.isx_movie_write_frame_f32.argtypes = [
    IsxMoviePtr,
    ctypes.c_size_t,
    FloatPtr]
c_api.isx_movie_write_frame_f32.errcheck = _standard_errcheck

c_api.isx_movie_flush.argtypes = [
    IsxMoviePtr]
c_api.isx_movie_flush.errcheck = _standard_errcheck

c_api.isx_movie_delete.argtypes = [
    IsxMoviePtr]
c_api.isx_movie_delete.errcheck = _standard_errcheck

c_api.isx_movie_get_acquisition_info_size.argtypes = [
    IsxMoviePtr,
    ctypes.POINTER(ctypes.c_size_t)]
c_api.isx_movie_get_acquisition_info_size.errcheck = _standard_errcheck

c_api.isx_movie_get_acquisition_info.argtypes = [
    IsxMoviePtr,
    ctypes.c_char_p,
    ctypes.c_size_t]
c_api.isx_movie_get_acquisition_info.errcheck = _standard_errcheck


# CellSet struct and methods

class IsxCellSet(ctypes.Structure):
    _fields_ = [("id", ctypes.c_size_t),
                ("timing", IsxTimingInfo),
                ("spacing", IsxSpacingInfo),
                ("num_cells", ctypes.c_size_t),
                ("roi_set", ctypes.c_bool),
                ("read_only", ctypes.c_bool),
                ("file_path", ctypes.c_char_p)]
IsxCellSetPtr = ctypes.POINTER(IsxCellSet)

c_api.isx_read_cell_set.argtypes = [
    ctypes.c_char_p,
    ctypes.c_bool,
    ctypes.POINTER(IsxCellSetPtr)]
c_api.isx_read_cell_set.errcheck = _standard_errcheck

c_api.isx_write_cell_set.argtypes = [
    ctypes.c_char_p,
    IsxTimingInfo,
    IsxSpacingInfo,
    ctypes.c_bool,
    ctypes.POINTER(IsxCellSetPtr)]
c_api.isx_write_cell_set.errcheck = _standard_errcheck

c_api.isx_cell_set_get_name.argtypes = [
    IsxCellSetPtr,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p]
c_api.isx_cell_set_get_name.errcheck = _standard_errcheck

c_api.isx_cell_set_get_status.argtypes = [
    IsxCellSetPtr,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int)]
c_api.isx_cell_set_get_status.errcheck = _standard_errcheck

c_api.isx_cell_set_set_status.argtypes = [
    IsxCellSetPtr,
    ctypes.c_size_t,
    ctypes.c_int]
c_api.isx_cell_set_set_status.errcheck = _standard_errcheck

c_api.isx_cell_set_get_trace.argtypes = [
    IsxCellSetPtr,
    ctypes.c_size_t,
    FloatPtr]
c_api.isx_cell_set_get_trace.errcheck = _standard_errcheck

c_api.isx_cell_set_get_image.argtypes = [
    IsxCellSetPtr,
    ctypes.c_size_t,
    FloatPtr]
c_api.isx_cell_set_get_image.errcheck = _standard_errcheck

c_api.isx_cell_set_write_image_trace.argtypes = [
    IsxCellSetPtr,
    ctypes.c_size_t,
    FloatPtr,
    FloatPtr,
    ctypes.c_char_p]
c_api.isx_cell_set_write_image_trace.errcheck = _standard_errcheck

c_api.isx_cell_set_flush.argtypes = [
    IsxCellSetPtr]
c_api.isx_cell_set_flush.errcheck = _standard_errcheck

c_api.isx_cell_set_delete.argtypes = [
    IsxCellSetPtr]
c_api.isx_cell_set_delete.errcheck = _standard_errcheck

c_api.isx_cell_set_get_acquisition_info_size.argtypes = [
    IsxCellSetPtr,
    ctypes.POINTER(ctypes.c_size_t)]
c_api.isx_cell_set_get_acquisition_info_size.errcheck = _standard_errcheck

c_api.isx_cell_set_get_acquisition_info.argtypes = [
    IsxCellSetPtr,
    ctypes.c_char_p,
    ctypes.c_size_t]
c_api.isx_cell_set_get_acquisition_info.errcheck = _standard_errcheck


# Events struct and methods.

class IsxEvents(ctypes.Structure):
    _fields_ = [("id", ctypes.c_size_t),
                ("timing", IsxTimingInfo),
                ("num_cells", ctypes.c_size_t),
                ("read_only", ctypes.c_bool),
                ("file_path", ctypes.c_char_p)]
IsxEventsPtr = ctypes.POINTER(IsxEvents)

c_api.isx_read_events.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(IsxEventsPtr)]
c_api.isx_read_events.errcheck = _standard_errcheck

c_api.isx_write_events.argtypes = [
    ctypes.c_char_p,
    IsxTimingInfo,
    CharPtrPtr,
    ctypes.c_size_t,
    ctypes.POINTER(IsxEventsPtr)]
c_api.isx_write_events.errcheck = _standard_errcheck

c_api.isx_events_write_cell.argtypes = [
    IsxEventsPtr,
    ctypes.c_size_t,
    ctypes.c_size_t,
    UInt64Ptr,
    FloatPtr]
c_api.isx_events_write_cell.errcheck = _standard_errcheck

# gets number of events in event file (for efficient memory allocation)
c_api.isx_events_get_cell_count.argtypes = [
    IsxEventsPtr,
    ctypes.c_char_p,
    SizeTPtr]
c_api.isx_events_get_cell_count.errcheck = _standard_errcheck

c_api.isx_events_get_cell_name.argtypes = [
    IsxEventsPtr,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p]
c_api.isx_events_get_cell_name.errcheck = _standard_errcheck

c_api.isx_events_get_cell.argtypes = [
    IsxEventsPtr,
    ctypes.c_char_p,
    UInt64Ptr,
    FloatPtr]
c_api.isx_events_get_cell.errcheck = _standard_errcheck

c_api.isx_events_flush.argtypes = [
    IsxEventsPtr]
c_api.isx_events_flush.errcheck = _standard_errcheck

c_api.isx_events_delete.argtypes = [
    IsxEventsPtr]
c_api.isx_events_delete.errcheck = _standard_errcheck

c_api.isx_events_get_acquisition_info_size.argtypes = [
    IsxEventsPtr,
    ctypes.POINTER(ctypes.c_size_t)]
c_api.isx_events_get_acquisition_info_size.errcheck = _standard_errcheck

c_api.isx_events_get_acquisition_info.argtypes = [
    IsxEventsPtr,
    ctypes.c_char_p,
    ctypes.c_size_t]
c_api.isx_events_get_acquisition_info.errcheck = _standard_errcheck


# Core enums which do not return an error code, but directly return their values.

c_api.isx_get_data_type_u16.argtypes = []
c_api.isx_get_data_type_f32.argtypes = []

c_api.isx_get_cell_status_accepted.argtypes = []
c_api.isx_get_cell_status_undecided.argtypes = []
c_api.isx_get_cell_status_rejected.argtypes = []

# Gets the message associated with the last error.
# This returns that message directly as a char *, because it is assumed it can
# not error.
c_api.isx_get_last_exception_string.argtypes = []
c_api.isx_get_last_exception_string.restype = ctypes.c_char_p

# Other core functions

c_api.isx_initialize.argtypes = []
c_api.isx_initialize.errcheck = _standard_errcheck

c_api.isx_shutdown.argtypes = []
c_api.isx_shutdown.errcheck = _standard_errcheck

c_api.isx_export_movie_nwb.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p]
c_api.isx_export_movie_nwb.errcheck = _standard_errcheck

c_api.isx_export_movie_tiff.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_char_p]
c_api.isx_export_movie_tiff.errcheck = _standard_errcheck

c_api.isx_export_cell_set.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_bool]
c_api.isx_export_cell_set.errcheck = _standard_errcheck

c_api.isx_export_event_set.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_char_p,
    ctypes.c_int]
c_api.isx_export_event_set.errcheck = _standard_errcheck


# Algo enums which do not return an error code, but their value directly.

c_api.isx_get_ica_unmix_type_temporal.argtypes = []
c_api.isx_get_ica_unmix_type_spatial.argtypes = []
c_api.isx_get_ica_unmix_type_both.argtypes = []

c_api.isx_get_dff_image_type_mean.argtypes = []
c_api.isx_get_dff_image_type_min.argtypes = []

c_api.isx_get_projection_type_mean.argtypes = []
c_api.isx_get_projection_type_min.argtypes = []
c_api.isx_get_projection_type_max.argtypes = []

c_api.isx_get_event_time_reference_maximum.argtypes = []
c_api.isx_get_event_time_reference_beginning.argtypes = []
c_api.isx_get_event_time_reference_mid_rise.argtypes = []

# Version numbers

c_api.isx_get_core_version_major.argtypes = []
c_api.isx_get_core_version_minor.argtypes = []
c_api.isx_get_core_version_patch.argtypes = []
c_api.isx_get_core_version_build.argtypes = []


def get_core_version():
    return [c_api.isx_get_core_version_major(),
            c_api.isx_get_core_version_minor(),
            c_api.isx_get_core_version_patch(),
            c_api.isx_get_core_version_build()]


# Algo functions

c_api.isx_preprocess_movie.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_bool]
c_api.isx_preprocess_movie.errcheck = _standard_errcheck

c_api.isx_deinterleave_movie.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    UInt16Ptr,
    CharPtrPtr,
    CharPtrPtr]
c_api.isx_deinterleave_movie.errcheck = _standard_errcheck

c_api.isx_motion_correct_movie.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    IntPtr,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_float,
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_int,
    ctypes.c_char_p]
c_api.isx_motion_correct_movie.errcheck = _standard_errcheck

c_api.isx_pca_ica_movie.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    IntPtr]
c_api.isx_pca_ica_movie.errcheck = _standard_errcheck

c_api.isx_spatial_band_pass_movie.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int]
c_api.isx_spatial_band_pass_movie.errcheck = _standard_errcheck

c_api.isx_delta_f_over_f.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_int]
c_api.isx_delta_f_over_f.errcheck = _standard_errcheck

c_api.isx_project_movie.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_char_p,
    ctypes.c_int]
c_api.isx_project_movie.errcheck = _standard_errcheck

c_api.isx_event_detection.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int]
c_api.isx_event_detection.errcheck = _standard_errcheck

c_api.isx_temporal_crop_movie.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    IntPtr,
    ctypes.c_bool]
c_api.isx_temporal_crop_movie.errcheck = _standard_errcheck

c_api.isx_compute_cell_metrics.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    CharPtrPtr]
c_api.isx_compute_cell_metrics.errcheck = _standard_errcheck

c_api.isx_apply_cell_set.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_char_p,
    ctypes.c_double]
c_api.isx_apply_cell_set.errcheck = _standard_errcheck

c_api.isx_export_cell_contours.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_double,
    ctypes.c_int]
c_api.isx_export_cell_contours.errcheck = _standard_errcheck

c_api.isx_longitudinal_registration.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_char_p,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int]
c_api.isx_longitudinal_registration.errcheck = _standard_errcheck

c_api.isx_multiplane_registration.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_bool]
c_api.isx_multiplane_registration.errcheck = _standard_errcheck

c_api.isx_classify_cell_status.argtypes = [
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_int,
    CharPtrPtr,
    ctypes.c_int,
    CharPtrPtr,
    CharPtrPtr,
    DoublePtr,
    ctypes.c_int,
    SizeTPtr]
c_api.isx_classify_cell_status.errcheck = _standard_errcheck

c_api.isx_movie_verify_deinterleave.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint16,
    IntPtr]
c_api.isx_movie_verify_deinterleave.errcheck = _standard_errcheck

def initialize():
    c_api.isx_initialize()


def shutdown():
    c_api.isx_shutdown()


# Initialize the API so the client does not have to, then register shutdown on exit.
initialize()
atexit.register(shutdown)
