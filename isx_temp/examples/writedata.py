"""
This module contains an example that demonstrates how to write data
to the native .isxd format.
"""

import os
import isx
import numpy as np
import scipy.signal as spsig

# In order to use matplotlib with anaconda and the isx module, we have to switch to
# the TkAgg backend, there can be issues with Qt library conflicts otherwise.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    """
    Writes data to the .isxd format in a custom processing workflow.
    """

    # We will use the first motion corrected movie of the first day as our
    # starting point and will processed that in a custom way that is not
    # possible using the standard algorithms.
    data_dir = os.path.join('path_to_demo_data', 'S1prism_AAV1_demo_with_LR', 'S1prism_AAV1_demo_v2_data')
    data_set_base = os.path.join(data_dir, 'recording_20160613_105808-PP-PP-BP-MC')
    out_dir = os.path.join(data_dir, 'custom')
    os.makedirs(out_dir, exist_ok=True)

    # Read all the frames into a matrix.
    movie = isx.Movie.read(data_set_base + '.isxd')
    num_pixels = movie.spacing.num_pixels
    num_samples = movie.timing.num_samples
    movie_data = np.zeros(list(num_pixels) + [num_samples], np.uint16)
    for i in range(num_samples):
        movie_data[:, :, i] = movie.get_frame_data(i)

    # Compute the median frame as a baseline, write it to file, then plot it.
    # Note that the image data is immediately available without a flush.
    median_image = isx.Image.write(
            os.path.join(out_dir, 'movie-median.isxd'), movie.spacing, movie.data_type,
            np.median(movie_data, axis=2).astype(np.float32))
    median_image_data = median_image.get_data()
    plt.imshow(median_image_data)
    plt.show()

    # Perform the DF/F operation using the median frame.
    # Note that we do not perform any checks for divide by zero, etc.
    # We also store all the movie data for later use.
    dff_movie_file = os.path.join(out_dir, 'dff.isxd')
    dff_movie = isx.Movie.write(dff_movie_file, movie.timing, movie.spacing, np.float32)
    dff_movie_data = np.zeros(list(num_pixels) + [num_samples], dff_movie.data_type)
    for i in range(num_samples):
        dff_movie_data[:, :, i] = (movie.get_frame_data(i) - median_image_data) / median_image_data
        dff_movie.set_frame_data(i, dff_movie_data[:, :, i])
    dff_movie.flush()

    # Free up some memory by deleting the movie data, as we do not need it anymore.
    del movie_data

    # You can only read frames after reading the movie.
    dff_movie = isx.Movie.read(dff_movie_file)
    dff_frame0 = dff_movie.get_frame_data(0)
    plt.imshow(dff_frame0)
    plt.show()

    # Next we threshold and normalize the cell images from an existing
    # cell set and apply them to the movie to get a new one.
    orig_cell_set = isx.CellSet.read(data_set_base + '-DFF-PCA-ICA.isxd')
    num_cells = orig_cell_set.num_cells
    total_num_pixels = np.prod(num_pixels)

    cell_set_file = os.path.join(out_dir, 'cell_set.isxd')
    cell_set = isx.CellSet.write(cell_set_file, dff_movie.timing, orig_cell_set.spacing)
    for i in range(num_cells):
        image = orig_cell_set.get_cell_image_data(i)
        image[image < 0] = 0
        image /= image.sum()
        trace = image.ravel().dot(dff_movie_data.reshape([total_num_pixels, num_samples]))
        cell_set.set_cell_data(i, image, trace, 'C{:03}'.format(i))

    # Clone the status from the original cell set.
    # Note that we can only set the cell statuses after all cell
    # data has been written.
    for i in range(num_cells):
        cell_set.set_cell_status(i, orig_cell_set.get_cell_status(i))

    # Free up the DF/F movie data as we do not need it anymore.
    del dff_movie_data

    # Compare the original and new images.
    cell_set = isx.CellSet.read(cell_set_file)
    f, (orig_ax, new_ax) = plt.subplots(1, 2)
    orig_ax.imshow(orig_cell_set.get_cell_image_data(0))
    new_ax.imshow(cell_set.get_cell_image_data(0))
    plt.show()

    # Compare the original and new traces.
    f, (orig_ax, new_ax) = plt.subplots(2, 1, sharex=True)
    time_stamps = [offset.secs_float for offset in cell_set.timing.get_offsets_since_start()]
    orig_ax.plot(time_stamps, orig_cell_set.get_cell_trace_data(0))
    orig_ax.set_ylabel('IC Trace Value')
    new_ax.plot(time_stamps, cell_set.get_cell_trace_data(0))
    new_ax.set_ylabel('New Trace Value')
    new_ax.set_xlabel('Time since Start (s)')
    plt.show()

    # Do some custom event detection by finding the peaks above a threshold.
    cell_names = [cell_set.get_cell_name(i) for i in range(num_cells)]
    event_set_file = os.path.join(out_dir, 'event_set.isxd')
    event_set = isx.EventSet.write(event_set_file, cell_set.timing, cell_names)
    offsets = np.array([x.to_usecs() for x in cell_set.timing.get_offsets_since_start()], np.uint64)
    for i in range(num_cells):
        trace = cell_set.get_cell_trace_data(i)
        peak_inds = spsig.find_peaks_cwt(trace, [1])
        peak_inds = peak_inds[trace[peak_inds] > 0.01]
        event_set.set_cell_data(i, offsets[peak_inds], trace[peak_inds])
    event_set.flush()

    # Read the event set and plot the first events with the corresponding trace.
    event_set = isx.EventSet.read(event_set_file)
    plt.plot(time_stamps, cell_set.get_cell_trace_data(0), label='Trace')
    cell0_offsets, cell0_amps = event_set.get_cell_data(0)
    plt.plot(cell0_offsets * 1e-6, cell0_amps, 'r.', label='Events')
    plt.xlabel('Time since Start (s)')
    plt.ylabel('Trace/Event Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
