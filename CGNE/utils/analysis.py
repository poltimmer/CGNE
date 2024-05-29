import numpy as np
import ot


def get_wasserstein_distances(rho, gt_area, gt_boundary_length, area, boundary_length, num_slices=16):
    """Given arrays for the conditioning variable, ground truth and rollout area and boundary length,
    compute the expected Wasserstein distance between the ground truth and rollout distributions."""

    # Compute Wasserstein distances
    var_min = np.min(rho)
    var_max = np.max(rho)

    # Create equal-width bins for the variable between its min and max
    var_bins = np.linspace(var_min, var_max, num=num_slices + 1)

    # Bin the variable
    var_bin_indices = np.digitize(rho, var_bins) - 1  # Get bin indices for each 'rho'

    # Initialize a list to store Wasserstein distances
    wasserstein_distances = []

    # Compute Wasserstein distances for each bin
    for bin_index in range(num_slices):
        # Select the rows for the current bin
        bin_mask = (var_bin_indices == bin_index)

        if np.any(bin_mask):
            # Extract GT and Rollout distributions for the current bin
            gt_distribution = np.stack((gt_area[bin_mask], gt_boundary_length[bin_mask]), axis=-1)
            rollout_distribution = np.stack((area[bin_mask], boundary_length[bin_mask]), axis=-1)

            # Uniform distribution weights for Wasserstein distance calculation
            gt_weights = np.ones((gt_distribution.shape[0],)) / gt_distribution.shape[0]
            rollout_weights = np.ones((rollout_distribution.shape[0],)) / rollout_distribution.shape[0]

            # Compute the cost matrix between the two distributions
            cost_matrix = ot.dist(gt_distribution, rollout_distribution, metric='euclidean')

            # Compute the Wasserstein distance using the EMD solver
            wasserstein_dist = ot.emd2(gt_weights, rollout_weights, cost_matrix)
            wasserstein_distances.append(wasserstein_dist)
        else:
            print(f"No data in bin {bin_index}.")

    return wasserstein_distances, var_bins
