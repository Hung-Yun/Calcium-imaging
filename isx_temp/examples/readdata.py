"""
This module contains an example that demonstrates how to read data
stored in supported file formats.
"""

import os
import isx

# In order to use matplotlib with anaconda and the isx module, we have to switch to
# the TkAgg backend, there can be issues with Qt library conflicts otherwise.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    """
    Reads the demo data and visualizes it.
    """

    # We will use the first DF/F movie of the first day as an example
    # for reading processed data.
    data_dir = os.path.join('path_to_demo_data', 'S1prism_AAV1_demo_with_LR', 'S1prism_AAV1_demo_v2_data')
    data_set_base = os.path.join(data_dir, 'recording_20160613_105808-PP-PP-BP-MC-DFF')

    # Read the DF/F movie and print it to see some basic information.
    movie = isx.Movie.read(data_set_base + '.isxd')
    print(movie)

    # Get the first frame and show it.
    frame0 = movie.get_frame_data(0)
    plt.imshow(frame0)
    plt.show()

    # Read the corresponding cell set extracted using PCA-ICA and print it.
    cell_set = isx.CellSet.read(data_set_base + '-PCA-ICA.isxd')
    print(cell_set)

    # Get the image of the first cell and show it.
    cell0_image = cell_set.get_cell_image_data(0)
    plt.imshow(cell0_image)
    plt.show()

    # Get the trace of the first cell and plot it with time stamps.
    time_stamps = [offset.secs_float for offset in cell_set.timing.get_offsets_since_start()]
    cell0_trace = cell_set.get_cell_trace_data(0)
    plt.plot(time_stamps, cell0_trace)
    plt.xlabel('Time since Start (s)')
    plt.ylabel('Trace Value')
    plt.show()

    # Read the corresponding event and print it.
    event_set = isx.EventSet.read(data_set_base + '-PCA-ICA-ED.isxd')
    print(event_set)

    # Read the events of the first cell and plot them with the trace.
    cell0_event_usecs, cell0_event_amps = event_set.get_cell_data(0)
    plt.plot(time_stamps, cell0_trace, label='Trace')
    plt.plot(cell0_event_usecs * 1e-6, cell0_event_amps, 'r.', label='Events')
    plt.xlabel('Time since Start (s)')
    plt.ylabel('Trace/Event Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
