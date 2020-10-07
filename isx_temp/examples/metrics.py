"""
    This module contains some functions for computing and plotting metrics.
"""

import os
import isx

import numpy as np
import pandas as pd

# In order to use matplotlib with anaconda and the isx module, we have to switch to
# the TkAgg backend, there can be issues with Qt library conflicts otherwise.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():
    """
    Run the metrics example.
    """

    # Define the recording path to the cell set and event data, as done in the standard workflow example
    data_dir = os.path.join('path_to_demo_data', 'S1prism_AAV1_demo_with_LR', 'S1prism_AAV1_demo_v2_data')

    # Pick out a cell set file from a particular session.
    cellset_file = os.path.join(data_dir, 'recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA.isxd')

    # Specify the associated events file.
    events_file = os.path.join(data_dir, 'recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA-ED.isxd')

    # Specify the output .csv file that the metrics will be written to.
    metrics_file = os.path.join(data_dir, 'recording_20160613_105808-PP-PP-BP-MC-DFF-PCA-ICA-METRICS.csv')

    # Run the cell metrics calculation and write to the output file.
    if not os.path.exists(metrics_file):
        isx.cell_metrics(cellset_file, events_file, metrics_file)

    # Read the .csv file as a pandas DataFrame.
    metrics_df = pd.read_csv(metrics_file)

    # Open the cell set and get the names of accepted cells.
    cellset = isx.CellSet.read(cellset_file)
    accepted_cells = [cellset.get_cell_name(k)
                      for k in range(cellset.num_cells)
                      if cellset.get_cell_status(k) == 'accepted']
    del cellset

    # Create a logical index that only references accepted cells with reasonable SNRs.
    accept_i = metrics_df.cellName.isin(accepted_cells) & (metrics_df.snr > 3)

    # Make a scatter plot of cells at the centroids where the color indicates
    # event rate and size indicates signal-to-noise ratio.
    circle_size = metrics_df[accept_i]['snr'] * 2
    metrics_df[accept_i].plot.scatter(x='largestComponentCenterInPixelsX',
                                      y='largestComponentCenterInPixelsY',
                                      c='eventRate', s=circle_size,
                                      colormap=plt.cm.magma)

    # Create histograms of SNR, decay time, and event rate that show accepted
    # vs rejected cells.
    plt.figure()

    # Enumerate over the three columns of interest and make histograms that
    # show the accepted cells in one color, and rejected cells in another color.
    for k,col in enumerate(['snr', 'decayMedian', 'eventRate']):

        ax = plt.subplot(1, 3, k+1)
        n, bins, patches = plt.hist(metrics_df[col][accept_i], bins=15, facecolor='#d69aff')
        plt.hist(metrics_df[col][~accept_i], bins=bins, facecolor='k', alpha=0.5)
        plt.legend(['Accepted', 'Rejected'])
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.show()


if __name__ == '__main__':
    main()
