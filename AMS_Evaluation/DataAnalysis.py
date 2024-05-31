from Plot_Methods.plot_standards import *
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import matplotlib.colors


def threshold_otsu(x, *args, **kwargs) -> float or None:
    # hist, bins, range=None
    """Find the threshold value for a bimodal histogram using the Otsu method.

  If you have a distribution that is bimodal (AKA with two peaks, with a valley
  between them), then you can use this to find the location of that valley, that
  splits the distribution into two.

  From the SciKit Image threshold_otsu implementation:
  https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
  """
    if np.NaN in x or pd.NA in x:
        return None
    counts, bin_edges = np.histogram(x, *args, **kwargs)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    with np.errstate(divide='ignore', invalid='ignore'):
        mean1 = np.cumsum(counts * bin_centers) / weight1
        mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


def interpolate_map_data(x, y, z, max_distance=np.inf):
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    xi = np.linspace(x_min - (x_max - x_min) * 0.01, x_max + (x_max - x_min) * 0.01,
                     min(len(set(x)) * 5, 10000))
    yi = np.linspace(y_min - (y_max - y_min) * 0.01, y_max + (y_max - y_min) * 0.01,
                     min(len(set(y)) * 5, 10000))

    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    xy = [[x[i], y[i]] for i in range(len(x))]
    tree = cKDTree(xy)
    xyi = np.meshgrid(xi, yi)
    xyi = _ndim_coords_from_arrays(tuple(xyi), ndim=2)
    dists, indexes = tree.query(xyi)

    zi[dists >= max_distance] = np.nan
    return xi, yi, zi
