import numpy as np

# Paste the printed assignment matrix for the BIG matrix here
_BIG1_LOOKUP = np.array([
     [ 35,  37,  41,  47,  55,  64,  62,  52,  44,  38,  34],
     [ 31,  33,  39,  45,  53,  63,  60,  50,  42,  36,  32],
     [ 25,  27,  29,  43,  51,  61,  58,  48,  40,  30,  28],
     [ 17,  19,  21,  23,  49,  59,  56,  46,  26,  24,  22],
     [  7,   9,  11,  13,  15,  57,  54,  20,  18,  16,  14],
     [120, 118, 116, 114, 112,  79,  12,  10,   8,   6,   4],
     [110, 108, 106, 104,  70,  77, 121, 123, 125, 127,   2],
     [102, 100,  98,  78,  68,  75, 119, 111, 113, 115, 117],
     [ 96,  94,  84,  76,  66,  73,  85,  91, 105, 107, 109],
     [ 92,  88,  82,  74,  65,  71,  83,  89,  95, 101, 103],
     [ 90,  86,  80,  72,  67,  69,  81,  87,  93,  97,  99]
])

# Paste the printed assignment matrix for the BIG matrix here
_SMALL2_LOOKUP = np.array([
     [ 35,  37,  41,  47,  55,  64,  62,  52,  44,  38,  34],
     [ 31,  33,  39,  45,  53,  63,  60,  50,  42,  36,  32],
     [ 25,  27,  29,  43,  51,  61,  58,  48,  40,  30,  28],
     [ 17,  19,  21,  23,  49,  59,  56,  46,  26,  24,  22],
     [  7,   9,  11,  13,  15,  57,  54,  20,  18,  16,  14],
     [120, 118, 116, 114, 112,  12,  10,   8,   6,   4,   2],
     [110, 108, 106, 104,  70,  77, 119, 121, 123, 125, 127],
     [102, 100,  98,  78,  68,  75,  85, 111, 113, 115, 117],
     [ 96,  94,  84,  76,  66,  73,  83,  91, 105, 107, 109],
     [ 92,  88,  82,  74,  65,  71,  81,  89,  95, 101, 103],
     [ 90,  86,  80,  72,  67,  69,  79,  87,  93,  97,  99]
])


def reconstruct_matrix(channel_array, lookup=_BIG1_LOOKUP, for_imshow=True):
    """
    Convert a 128-channel 1D array into a reconstructed 11x11 matrix
    using the precomputed BIG matrix lookup table.

    Parameters
    ----------
    channel_array : array-like, length >= max channel number
    lookup : array-like, optional - channel assignment
    for_imshow : bool, optional - whether to return a matrix with orientation for imshow
    Returns
    -------
    11x11 numpy array
    """
    channel_array = np.asarray(channel_array)

    # lookup table contains channel numbers starting at 1
    if for_imshow:
     return channel_array[lookup - 1]
    else:
     return channel_array[lookup.T[::-1, :] - 1]


print(reconstruct_matrix(np.arange(128)))
print(reconstruct_matrix(np.arange(128), for_imshow=False))

# Orientation_logic:
# With for_imshow=True the composed signal equals the real orientation when used like ax.imshow(signal)
# With for_imshow=False the signal is compatible with the Analyzer class readout orientation logic
# (in readout step the signal is assigned like signal[lookup[::1, :].flatten()-1].reshape(analyzer.diode_dimension) and
# can then be used with the image generation of the analyzer class (to consider diode size and spacing))