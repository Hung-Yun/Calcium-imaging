"""
The io module deals with data input and output.

This includes reading from and writing to supported file formats for
movies, images, cell sets and event sets.
"""

import os
import ctypes
import textwrap

import numpy as np

import isx._internal
import isx.core


class Movie(object):
    """
    A movie contains a number of frames with timing and spacing information.

    It is always backed by a file, which can be read or written using this class.
    See :ref:`importMovie` for details on what formats are supported for read.
    Only the native `.isxd` format is supported for write.

    Examples
    --------
    Read an existing movie and get its first frame as a numpy array.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> frame_data = movie.get_frame_data(0)

    Write a 400x300 movie with 200 random frames of float32 values.

    >>> timing = isx.Timing(num_samples=200)
    >>> spacing = isx.Spacing(num_pixels=(300, 400))
    >>> movie = isx.Movie.write('movie-400x300x200.isxd', timing, spacing, numpy.float32)
    >>> for i in range(timing.num_samples):
    >>>     movie.set_frame_data(i, numpy.random.random(spacing.num_pixels).astype(numpy.float32))
    >>> movie.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the frames.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in each frame.
    data_type : {numpy.uint16, numpy.float32}
        The data type of each pixel.
    """

    def __init__(self):
        self._ptr = isx._internal.IsxMoviePtr()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def spacing(self):
        return isx.core.Spacing._from_impl(self._ptr.contents.spacing) if self._ptr else None

    @property
    def data_type(self):
        return isx._internal.DATA_TYPE_TO_NUMPY[self._ptr.contents.data_type] if self._ptr else None

    @classmethod
    def read(cls, file_path):
        """
        Open an existing movie from a file for reading.

        This is a light weight operation that simply reads the meta-data from the movie,
        and does not read any frame data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.

        Returns
        -------
        :class:`isx.Movie`
            The movie that was read. Meta-data is immediately available.
            Frames must be read using :func:`isx.Movie.get_frame`.
        """
        movie = cls()
        isx._internal.c_api.isx_read_movie(file_path.encode('utf-8'), ctypes.byref(movie._ptr))
        return movie

    @classmethod
    def write(cls, file_path, timing, spacing, data_type):
        """
        Open a new movie to a file for writing.

        This is a light weight operation. It does not write any frame data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : :class:`isx.Timing`
            The timing of the movie to write.
        spacing : :class:`isx.Spacing`
            The spacing of the movie to write.
        data_type : {numpy.uint16, numpy.float32}
            The data type of each pixel.

        Returns
        -------
        :class:`isx.Movie`
            The empty movie that was written.
            Frame data must be written with :func:`isx.Movie.set_frame_data`.
        """
        movie = cls()
        data_type_int = isx._internal.lookup_enum('data_type', isx._internal.DATA_TYPE_FROM_NUMPY, data_type)
        isx._internal.c_api.isx_write_movie(file_path.encode('utf-8'), timing._impl, spacing._impl, data_type_int, False, ctypes.byref(movie._ptr))
        return movie

    def get_frame_data(self, index):
        """
        Get a frame from the movie by index.

        Arguments
        ---------
        index : int >= 0
            The index of the frame. If this is out of range, this should error.

        Returns
        -------
        :class:`numpy.ndarray`
            The retrieved frame data.
        """
        isx._internal.validate_ptr(self._ptr)

        shape = self.spacing.num_pixels
        f = np.zeros([np.prod(shape)], dtype=self.data_type)

        if self.data_type == np.uint16:
            f_p = f.ctypes.data_as(isx._internal.UInt16Ptr)
            isx._internal.c_api.isx_movie_get_frame_data_u16(self._ptr, index, f_p)
        elif self.data_type == np.float32:
            f_p = f.ctypes.data_as(isx._internal.FloatPtr)
            isx._internal.c_api.isx_movie_get_frame_data_f32(self._ptr, index, f_p)
        else:
            raise RuntimeError('Cannot read from movie with datatype: {}'.format(str(self.data_type)))

        return f.reshape(shape)

    def set_frame_data(self, index, frame):
        """
        Set frame data in a writable movie.

        Frames must be set in increasing order, otherwise this will error.

        Arguments
        ---------
        index : int >= 0
            The index of the frame.
        frame : :class:`numpy.ndarray`
            The frame data.
        """
        isx._internal.validate_ptr(self._ptr)

        if self.mode != 'w':
            raise ValueError('Cannot set frame data if movie is read-only.')

        if not isinstance(frame, np.ndarray):
            raise TypeError('Frame must be a numpy array')

        if frame.shape != self.spacing.num_pixels:
            raise ValueError('Cannot set frame with different shape than movie')

        f_flat = isx._internal.ndarray_as_type(frame, np.dtype(self.data_type)).ravel()

        if self.data_type == np.uint16:
            FrameType = ctypes.c_uint16 * np.prod(frame.shape)
            c_frame = FrameType(*f_flat)
            isx._internal.c_api.isx_movie_write_frame_u16(self._ptr, index, c_frame)
        elif self.data_type == np.float32:
            FrameType = ctypes.c_float * np.prod(frame.shape)
            c_frame = FrameType(*f_flat)
            isx._internal.c_api.isx_movie_write_frame_f32(self._ptr, index, c_frame)
        else:
            raise RuntimeError('Cannot write frames for movie with datatype: {}'.format(str(self.data_type)))

    def flush(self):
        """
        Flush all meta-data and frame data to the file.

        This should be called after setting all frames of a movie opened with :func:`isx.Movie.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_movie_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_movie_get_acquisition_info,
                isx._internal.c_api.isx_movie_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_movie_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        Movie
            file_path: {}
            mode: {}
            timing: {}
            spacing: {}
            data_type: {}\
        """.format(self.file_path, self.mode, self.timing, self.spacing, self.data_type))


class Image(object):
    """
    An image is effectively a movie with one frame and no timing.

    It is always backed by a file, which can be read or written using this class.
    See :ref:`importMovie` for details on what formats are supported for read.
    Only the native `.isxd` format is supported for write.

    Examples
    --------
    Read an existing image and get its data.

    >>> image = isx.Image.read('recording_20160613_105808-PP-PP-BP-Mean Image.isxd')
    >>> image_data = image.get_data()

    Calculate the minimum image from an existing movie and write it.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> min_image = 4095 * numpy.ones(movie.spacing.num_pixels, dtype=movie.data_type)
    >>> for i in range(movie.timing.num_samples):
    >>>     min_image = numpy.minimum(min_image, movie.get_frame_data(i))
    >>> isx.Image.write('recording_20160613_105808-PP-PP-min.isxd', movie.spacing, movie.data_type, min_image)

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in the image.
    data_type : {numpy.uint16, numpy.float32}
        The data type of each pixel.
    """

    def __init__(self):
        self._impl = isx.Movie()
        self._data = None

    @property
    def file_path(self):
        return self._impl.file_path

    @property
    def mode(self):
        return self._impl.mode

    @property
    def spacing(self):
        return self._impl.spacing

    @property
    def data_type(self):
        return self._impl.data_type

    @classmethod
    def read(cls, file_path):
        """
        Read an existing image from a file.

        Arguments
        ---------
        file_path : str
            The path of the image file to read.

        Returns
        -------
        :class:`isx.Image`
            The image that was read.
        """
        self = cls()
        self._impl = isx.Movie.read(file_path)
        if self._impl.timing.num_samples > 1:
            raise AttributeError('File has more than one frame. Use isx.Movie.read instead.')
        self._data = self._impl.get_frame_data(0)
        return self

    @classmethod
    def write(cls, file_path, spacing, data_type, data):
        """
        Write an image to a file.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        spacing : :class:`isx.Spacing`
            The spacing of the image to write.
        data_type : {numpy.uint16, numpy.float32}
            The data type of each pixel.
        data : :class:`numpy.array`
            The 2D array of data to write.

        Returns
        -------
        :class:`isx.Image`
            The image that was written.
        """
        self = cls()
        self._impl = isx.Movie.write(file_path, isx.Timing(num_samples=1), spacing, data_type)
        self._data = isx._internal.ndarray_as_type(data, np.dtype(data_type))
        self._impl.set_frame_data(0, self._data)
        self._impl.flush()
        return self

    def get_data(self):
        """
        Get the data stored in the image.

        Returns
        -------
        :class:`numpy.ndarray`
            The image data.
        """
        return self._data

    def __str__(self):
        return textwrap.dedent("""\
        Image
            file_path: {}
            mode: {}
            spacing: {}
            data_type: {}\
        """.format(self.file_path, self.mode, self.spacing, self.data_type))


class CellSet(object):
    """
    A cell set contains the image and trace data associated with components in
    a movie, such as cells or regions of interest.

    It is always backed by a file in the native `.isxd` format.

    Examples
    --------
    Read an existing cell set from a file and get the image and trace data of
    the first cell.

    >>> cell_set = isx.CellSet.read('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA.isxd')
    >>> image_0 = cell_set.get_cell_image_data(0)
    >>> trace_0 = cell_set.get_cell_trace_data(0)

    Write a new cell set to a file with the same timing and spacing as an
    existing movie, with 3 random cell images and traces.

    >>> movie = isx.Movie.read('recording_20160613_105808-PP-PP.isxd')
    >>> cell_set = isx.CellSet.write('cell_set.isxd', movie.timing, movie.spacing)
    >>> for i in range(3):
    >>>     image = numpy.random.random(cell_set.spacing.num_pixels).astype(numpy.float32)
    >>>     trace = numpy.random.random(cell_set.timing.num_samples).astype(numpy.float32)
    >>>     cell_set.set_cell_data(i, image, trace, 'C{}'.format(i))
    >>> cell_set.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the samples in each cell trace.
    spacing : :class:`isx.Spacing`
        The spacing of the pixels in each cell image.
    num_cells : int
        The number of cells or components.
    """

    _MAX_CELL_NAME_SIZE = 256

    def __init__(self):
        self._ptr = isx._internal.IsxCellSetPtr()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def spacing(self):
        return isx.core.Spacing._from_impl(self._ptr.contents.spacing) if self._ptr else None

    @property
    def num_cells(self):
        return self._ptr.contents.num_cells if self._ptr else None

    @classmethod
    def read(cls, file_path, read_only=True):
        """
        Open an existing cell set from a file for reading.

        This is a light weight operation that simply reads the meta-data from the cell set,
        and does not read any image or trace data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.
        read_only : bool
            If true, only allow meta-data and data to be read, otherwise allow some meta-data
            to be written (e.g. cell status).

        Returns
        -------
        :class:`isx.CellSet`
            The cell set that was read. Meta-data is immediately available.
            Image and trace data must be read using :func:`isx.CellSet.get_cell_image_data`
            and :func:`isx.CellSet.get_cell_trace_data` respectively.
        """
        cell_set = cls()
        isx._internal.c_api.isx_read_cell_set(file_path.encode('utf-8'), read_only, ctypes.byref(cell_set._ptr))
        return cell_set

    @classmethod
    def write(cls, file_path, timing, spacing):
        """
        Open a new cell set to a file for writing.

        This is a light weight operation. It does not write any image or trace data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : :class:`isx.Timing`
            The timing of the cell set to write. Typically this comes from the movie this
            is derived from.
        spacing : :class:`isx.Spacing`
            The spacing of the movie to write. Typically this comes from the movie this is
            derived from.

        Returns
        -------
        :class:`isx.CellSet`
            The empty cell set that was written.
            Image and trace data must be written with :func:`isx.CellSet.set_cell_data`.
        """
        if not isinstance(timing, isx.core.Timing):
            raise TypeError('timing must be a Timing object')

        if not isinstance(spacing, isx.core.Spacing):
            raise ValueError('spacing must be a Spacing object')

        cell_set = cls()
        isx._internal.c_api.isx_write_cell_set(
                file_path.encode('utf-8'), timing._impl, spacing._impl, False, ctypes.byref(cell_set._ptr))
        return cell_set

    def get_cell_name(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        str
            The name of the indexed cell.
        """
        isx._internal.validate_ptr(self._ptr)
        result = ctypes.create_string_buffer(CellSet._MAX_CELL_NAME_SIZE)
        isx._internal.c_api.isx_cell_set_get_name(self._ptr, index, CellSet._MAX_CELL_NAME_SIZE, result)
        return result.value.decode('utf-8')

    def get_cell_status(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        {'accepted', 'undecided', 'rejected'}
            The status of the indexed cell as a string.
        """
        isx._internal.validate_ptr(self._ptr)
        status_int = ctypes.c_int(0)
        isx._internal.c_api.isx_cell_set_get_status(self._ptr, index, ctypes.byref(status_int))
        return isx._internal.CELL_STATUS_TO_STRING[status_int.value]

    def set_cell_status(self, index, status):
        """
        Set the status of cell. This will also flush the file.

        .. warning:: As this flushes the file, only use this after all cells have been
                     written using :func:`isx.CellSet.set_cell_data`.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.
        status : {'accepted', 'undecided', 'rejected'}
            The desired status of the indexed cell.
        """
        isx._internal.validate_ptr(self._ptr)
        if self.mode != 'w':
            raise RuntimeError('Cannot set cell status in read-only mode')
        status_int = isx._internal.lookup_enum('cell_status', isx._internal.CELL_STATUS_FROM_STRING, status)
        isx._internal.c_api.isx_cell_set_set_status(self._ptr, index, status_int)

    def get_cell_trace_data(self, index):
        """
        Get the trace data associated with a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        :class:`numpy.ndarray`
            The trace data in a 1D array.
        """
        isx._internal.validate_ptr(self._ptr)
        trace = np.zeros([self.timing.num_samples], dtype=np.float32)
        trace_p = trace.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_cell_set_get_trace(self._ptr, index, trace_p)
        return trace

    def get_cell_image_data(self, index):
        """
        Get the image data associated with a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        :class:`numpy.ndarray`
            The image data in a 2D array.
        """
        isx._internal.validate_ptr(self._ptr)
        f = np.zeros([np.prod(self.spacing.num_pixels)], dtype=np.float32)
        f_p = f.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_cell_set_get_image(self._ptr, index, f_p)
        return f.reshape(self.spacing.num_pixels)

    def set_cell_data(self, index, image, trace, name):
        """
        Set the image and trace data of a cell.

        Cells must be set in increasing order, otherwise this will error.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.
        image : :class:`numpy.ndarray`
            The image data in a 2D array.
        trace : :class:`numpy.ndarray`
            The trace data in a 1D array.
        name : str
            The name of the cell.
        """
        isx._internal.validate_ptr(self._ptr)

        if self.mode != 'w':
            raise RuntimeError('Cannot set cell data in read-only mode')

        if name is None:
            name = 'C{}'.format(index)

        im = isx._internal.ndarray_as_type(image.reshape(np.prod(self.spacing.num_pixels)), np.dtype(np.float32))
        im_p = im.ctypes.data_as(isx._internal.FloatPtr)
        tr = isx._internal.ndarray_as_type(trace, np.dtype(np.float32))
        tr_p = tr.ctypes.data_as(isx._internal.FloatPtr)
        isx._internal.c_api.isx_cell_set_write_image_trace(self._ptr, index, im_p, tr_p, name.encode('utf-8'))

    def flush(self):
        """
        Flush all meta-data and cell data to the file.

        This should be called after setting all cell data of a cell set opened with :func:`isx.CellSet.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_cell_set_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_cell_set_get_acquisition_info,
                isx._internal.c_api.isx_cell_set_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_cell_set_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        CellSet
            file_path: {}
            mode: {}
            timing: {}
            spacing: {}
            num_cells: {}\
        """.format(self.file_path, self.mode, self.timing, self.spacing, self.num_cells))


class EventSet(object):
    """
    An event set contains the event data of a number of components or cells.

    It is typically derived from a cell set after applying an event detection
    algorithm.
    Each event of a cell is comprised of a time stamp offset and a value or amplitude.

    Examples
    --------
    Read an existing event set from a file and get the event data associated with the
    first cell.

    >>> event_set = isx.EventSet.read('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA-ED.isxd')
    >>> [offsets, amplitudes] = event_set.get_cell_data(0)

    Write a new event set to a file by applying a threshold to the traces of an existing
    cell set.

    >>> cell_set = isx.CellSet.read('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA.isxd')
    >>> cell_names = ['C{}'.format(c) for c in range(cell_set.num_cells)]
    >>> event_set = isx.EventSet.write('recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA-custom_ED.isxd', cell_set.timing, cell_names)
    >>> offsets = numpy.array([x.to_usecs() for x in cell_set.timing.get_offsets_since_start()], numpy.uint64)
    >>> for c in range(cell_set.num_cells):
    >>>     trace = cell_set.get_cell_trace_data(c)
    >>>     above_thresh = trace > 500
    >>>     event_set.set_cell_data(c, offsets[above_thresh], trace[above_thresh])
    >>> event_set.flush()

    Attributes
    ----------
    file_path : str
        The path of the file that stores this.
    mode : {'r', 'w'}
        The mode the file was opened with.
    timing : :class:`isx.Timing`
        The timing of the samples in each event trace.
    num_cells : int
        The number of cells or components.
    """

    def __init__(self):
        self._ptr = isx._internal.IsxEventsPtr()

    @property
    def file_path(self):
        return self._ptr.contents.file_path.decode() if self._ptr else None

    @property
    def mode(self):
        return isx._internal.get_mode_from_read_only(self._ptr.contents.read_only) if self._ptr else None

    @property
    def timing(self):
        return isx.core.Timing._from_impl(self._ptr.contents.timing) if self._ptr else None

    @property
    def num_cells(self):
        return self._ptr.contents.num_cells if self._ptr else None

    @classmethod
    def read(cls, file_path):
        """
        Open an existing event set from a file for reading.

        This is a light weight operation that simply reads the meta-data from the event set,
        and does not read any event data.

        Arguments
        ---------
        file_path : str
            The path of the file to read.

        Returns
        -------
        :class:`isx.EventSet`
            The event set that was read. Meta-data is immediately available.
            Event data must be read using :func:`isx.EventSet.get_cell_data`.
        """
        event_set = cls()
        isx._internal.c_api.isx_read_events(file_path.encode('utf-8'), ctypes.byref(event_set._ptr))
        return event_set

    @classmethod
    def write(cls, file_path, timing, cell_names):
        """
        Open a new event set to a file for writing.

        This is a light weight operation. It does not write any event data immediately.

        Arguments
        ---------
        file_path : str
            The path of the file to write. If it already exists, this will error.
        timing : isx.Timing
            The timing of the event set to write. Typically this comes from the cell set this
            is derived from.
        cell_names : list<str>
            The names of the cells that will be written. Typically these come from the cell set
            this is derived from.

        Returns
        -------
        :class:`isx.EventSet`
            The empty event set that was written.
            Image and trace data must be written with :func:`isx.EventSet.set_cell_data`.
        """
        if not isinstance(timing, isx.core.Timing):
            raise TypeError('timing must be a Timing object')

        num_cells = len(cell_names)
        if num_cells <= 0:
            raise ValueError('cell_names must not be empty')

        cell_names_c = isx._internal.list_to_ctypes_array(cell_names, ctypes.c_char_p)
        event_set = cls()
        isx._internal.c_api.isx_write_events(file_path.encode('utf-8'), timing._impl, cell_names_c, num_cells, ctypes.byref(event_set._ptr))
        return event_set

    def get_cell_name(self, index):
        """
        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        str
            The name of the indexed cell.
        """
        isx._internal.validate_ptr(self._ptr)
        result = ctypes.create_string_buffer(CellSet._MAX_CELL_NAME_SIZE)
        isx._internal.c_api.isx_events_get_cell_name(self._ptr, index, CellSet._MAX_CELL_NAME_SIZE, result)
        return result.value.decode('utf-8')

    def get_cell_data(self, index):
        """
        Get the event data associated with a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.

        Returns
        -------
        offsets : :class:`numpy.ndarray`
            The 1D array of time stamps offsets from the start in microseconds.
        amplitudes : :class:`numpy.ndarray`
            The 1D array of event amplitudes.
        """
        isx._internal.validate_ptr(self._ptr)

        cell_name = self.get_cell_name(index)

        num_events = ctypes.c_size_t(0)
        isx._internal.c_api.isx_events_get_cell_count(self._ptr, cell_name.encode('utf-8'), ctypes.byref(num_events))
        num_events = num_events.value

        f = np.zeros([np.prod(num_events)], dtype=np.float32)
        f_p = f.ctypes.data_as(isx._internal.FloatPtr)

        usecs = np.zeros([np.prod(num_events)], dtype=np.uint64)
        usecs_p = usecs.ctypes.data_as(isx._internal.UInt64Ptr)

        isx._internal.c_api.isx_events_get_cell(self._ptr, cell_name.encode('utf-8'), usecs_p, f_p)

        return usecs, f

    def set_cell_data(self, index, offsets, amplitudes):
        """
        Set the event data of a cell.

        Arguments
        ---------
        index : int >= 0
            The index of a cell.
        offsets : :class:`numpy.ndarray`
            The 1D array of time stamps offsets from the start in microseconds.
        amplitudes : :class:`numpy.ndarray`
            The 1D array of event amplitudes.
        """
        isx._internal.validate_ptr(self._ptr)

        if len(offsets) != len(amplitudes):
            raise TypeError("Number of events must be the same as the number of timestamps.")

        amps = isx._internal.ndarray_as_type(amplitudes, np.dtype(np.float32))
        offs = isx._internal.ndarray_as_type(offsets, np.dtype(np.uint64))
        f_p = amps.ctypes.data_as(isx._internal.FloatPtr)
        usecs_p = offs.ctypes.data_as(isx._internal.UInt64Ptr)
        isx._internal.c_api.isx_events_write_cell(self._ptr, index, len(offs), usecs_p, f_p)

    def flush(self):
        """
        Flush all meta-data and cell data to the file.

        This should be called after setting all cell data of an event set opened with :func:`isx.EventSet.write`.
        """
        isx._internal.validate_ptr(self._ptr)
        isx._internal.c_api.isx_events_flush(self._ptr)

    def get_acquisition_info(self):
        """
        Get information about acquisition that may be stored in some files,
        such as nVista 3 movies and data derived from those.

        Returns
        -------
        dict
            A dictionary likely parsed from JSON that maps from string keys to variant values.
        """
        return isx._internal.get_acquisition_info(
                self._ptr,
                isx._internal.c_api.isx_events_get_acquisition_info,
                isx._internal.c_api.isx_events_get_acquisition_info_size);

    def __del__(self):
        if self._ptr:
            isx._internal.c_api.isx_events_delete(self._ptr)

    def __str__(self):
        return textwrap.dedent("""\
        EventSet
            file_path: {}
            mode: {}
            timing: {}
            num_cells: {}\
        """.format(self.file_path, self.mode, self.timing, self.num_cells))


def export_movie_to_tiff(input_movie_files, output_tiff_file, write_invalid_frames=False):
    """
    Export movies to a TIFF file.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to be exported.
    output_tiff_file : str
        The path of the TIFF file to be written.
    write_invalid_frames : bool
        If True, write invalid (dropped and cropped) frames as zero,
        otherwise, do not write them at all.
    """
    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_export_movie_tiff(num_movies, in_movie_arr, output_tiff_file.encode('utf-8'), write_invalid_frames)


def export_movie_to_nwb(
        input_movie_files, output_nwb_file,
        identifier='', session_description='', comments='',
        description='', experiment_description='', experimenter='',
        institution='', lab='', session_id=''):
    """
    Export movies to an HDF5-based neurodata without borders (NWB) file.

    For more details see :ref:`exportMovie`.

    Arguments
    ---------
    input_movie_files : list<str>
        The file paths of the movies to be exported.
    output_nwb_file : str
        The path of the NWB file to be written.
    identifier : str
        An identifier for the file according to the NWB spec.
    session_description : str
        A session description for the file according to the NWB spec.
    comments : str
        Comments on the recording session.
    description : str
        Description for the file according to the NWB spec.
    experiment_description : str
        Details about the experiment.
    experimenter : str
        The person who recorded the data.
    institution : str
        The place where the recording was performed.
    lab : str
        The lab where the recording was performed.
    session_id : str
        A unique session identifier for the recording.
    """
    num_movies, in_movie_arr = isx._internal.check_input_files(input_movie_files)
    isx._internal.c_api.isx_export_movie_nwb(
            num_movies, in_movie_arr, output_nwb_file.encode('utf-8'),
            identifier.encode('utf-8'), session_description.encode('utf-8'),
            comments.encode('utf-8'), description.encode('utf-8'),
            experiment_description.encode('utf-8'), experimenter.encode('utf-8'),
            institution.encode('utf-8'), lab.encode('utf-8'), session_id.encode('utf-8'))


def export_cell_set_to_csv_tiff(input_cell_set_files, output_csv_file, output_tiff_file, time_ref='start', output_props_file=''):
    """
    Export cell sets to a CSV file with trace data and TIFF files with image data.

    For more details see :ref:`exportCellsAndStuff`.

    Unlike the desktop application, this will only produce a TIFF cell map image file
    and not a PNG file too.

    Arguments
    ---------
    input_cell_set_files : list<str>
        The file paths of the cell sets to export.
    output_csv_file : str
        The path of the CSV file to write.
    output_tiff_file : str
        The base name of the TIFF files to write.
    time_ref : {'start', 'unix'}
        The time reference for the CSV time stamps.
        If 'start' is used, the time stamps represent the seconds since the start of the cell set.
        If 'unix' is used, the time stamps represents the second since the Unix epoch.
    output_props_file : str
        The path of the properties CSV file to write.
    """
    num_cell_sets, in_cell_sets = isx._internal.check_input_files(input_cell_set_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_cell_set(
            num_cell_sets, in_cell_sets, output_csv_file.encode('utf-8'),
            output_tiff_file.encode('utf-8'), time_ref_int, False, output_props_file.encode('utf-8'))


def export_event_set_to_csv(input_event_set_files, output_csv_file, time_ref='start', output_props_file=''):
    """
    Export event sets to a CSV file.

    For more details see :ref:`exportCellsAndStuff`.

    Arguments
    ---------
    input_event_set_files : list<str>
        The file paths of the cell sets to export.
    output_csv_file : str
        The path of the CSV file to write.
    time_ref : {'start', 'unix'}
        The time reference for the CSV time stamps.
        If 'start' is used, the time stamps represent the seconds since the start of the cell set.
        If 'unix' is used, the time stamps represents the second since the Unix epoch.
    output_props_file : str
        The path of the properties CSV file to write.
    """
    num_event_sets, in_event_sets = isx._internal.check_input_files(input_event_set_files)
    time_ref_int = isx._internal.lookup_enum('time_ref', isx._internal.TIME_REF_FROM_STRING, time_ref)
    isx._internal.c_api.isx_export_event_set(
            num_event_sets, in_event_sets, output_csv_file.encode('utf-8'), time_ref_int,
            output_props_file.encode('utf-8'))
