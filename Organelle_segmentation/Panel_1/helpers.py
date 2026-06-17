import tqdm
import urllib
import zipfile
import requests
from pathlib import Path
import skimage
import numpy as np
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev

###############################################################################

def download(url, fname = None):
  """
  Download the file at the given url

  Parameters
  ----------
  url : string
    The url of the file to download
  fname : pathlike, default=None
    The on-disc name of the downloaded file (if None, derived from the url)

  Returns
  -------
  The disc path to the downloaded artifact
  """

  # prepare the file name
  url = urllib.parse.urlparse(url) # parse the url to extract the file name
  p_url = Path(url.path)
  if fname is None: fname = Path(p_url.name)
  else: fname = Path(fname).with_suffix(p_url.suffix)

  # destroy the existing file prior to download if any
  fname.unlink(missing_ok = True)

  # download the file
  with open(fname, 'wb') as f: # open on-disc destination file for writing
    with requests.get(url.geturl(), stream=True) as r: # issue GET request
      r.raise_for_status()
      total = int(r.headers.get('content-length', 0)) # read total byte size
      params = { 'desc': url.geturl()
               , 'total': total
               , 'miniters': 1
               , 'unit': 'B'
               , 'unit_scale': True
               , 'unit_divisor': 1024 }
      with tqdm.tqdm(**params) as pb: # progress bar setup
        for chunk in r.iter_content(chunk_size=8192): # go through result chunks
          pb.update(len(chunk)) # update progress bar
          f.write(chunk) # write file content
  return fname

def download_and_extract(url, dest_dir = './data', fname = None):
  """
  Download and extract the file at the given url.

  Parameters
  ----------
  url : string
    The url of the file to download
  dest_dir : pathlike, default='./data'
    The path to the destination folder for download and extraction

  Returns
  -------
  None
  """
  # create destination directory if necessary, and prepare destination file name
  dest_dir = Path(dest_dir)
  dest_dir.mkdir(parents = True, exist_ok = True)
  fpath = Path(urllib.parse.urlparse(url).path)
  if fname is None: fname = fpath.name
  # download dataset
  dl = download(url, dest_dir / fname)
  suffixes = Path(dl).suffixes
  # extract dataset
  with zipfile.ZipFile(dl, 'r') as zf:
    zf.extractall(path = dest_dir)

# BBBC010_v1_foreground_eachworm dataset
#
# Sources:
# - here is where the dataset comes from:
#   https://bbbc.broadinstitute.org/BBBC010
# - here is the download link:
#   https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip
#
# You can get a copy of the dataset running the following:
# $ wget https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip
# $ unzip BBBC010_v1_foreground_eachworm.zip

_dataset_url = 'https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip'

################################################################################

def get_dataset(url = _dataset_url, dest_dir = './data'):
  """
  Download and extract the BBBC010_v1_foreground_eachworm dataset.

  Parameters
  ----------
  url : string
    The url for the BBBC010_v1_foreground_eachworm dataset download
  dest_dir : pathlike, default='./data'
    The path to the destination folder for download and extraction

  Returns
  -------
  None
  """
  download_and_extract(url, dest_dir)

###############################################################################

def rgb2grey(rgb, cr = 0.2989, cg = 0.5870, cb = 0.1140):
  """
  Turn an rgb array into a greyscale array using the following reduction:
    grey = cr * r + cg * g + cb * b

  Parameters
  ----------
  rgb : 3-d numpy array
    A 2-d image with 3 colour channels, red, green and blue
  cr : float, default=0.2989
    The red coefficient
  cg : float, default=0.5870
    The green coefficient
  cb : float, default=0.1140
    The blue coefficient

  Returns
  -------
  2-d numpy array
    The greyscale image
  """
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  return cr * r + cg * g + cb * b

def find_longest_contour(mask, normalise_coord=False):
  """
  Find the longest of all object contours present in the input `mask`

  Parameters
  ----------
  img : 2-d numpy array
    The image with masked objects (if it has a 3rd dimension, it is assumed to
    contain the r, g and b channels, and will be first converted to a greyscale
    image)
  normalise_coord: bool, default=False
    optionally normalise coordinates

  Returns
  -------
  2-d numpy array
    the longest contour as sequence of x, y coordinates in a column stacked
    array
  """
  # force the image to grayscale
  if len(mask.shape) == 3: # (lines, columns, number of channels)
    mask = rgb2grey(mask)
  # extract the contours from the now grayscale image
  contours = skimage.measure.find_contours(mask, 0.8)
  # sort the contours by length
  contours = sorted(contours, key=lambda x: len(x), reverse=True)
  # isolate the longest contour (first in the sorted list)
  x, y = contours[0][:, 0], contours[0][:, 1]
  # optionally normalise the coordinates in the countour
  if normalise_coord:
    x = x - np.min(x)
    x = x / np.max(x)
    y = y - np.min(y)
    y = y / np.max(y)
  # return the contour as a pair of lists of x and y coordinates
  return np.column_stack([x, y])

def contour_spline_resample(contour, n_samples, sparsity=1):
  """
  Return a resampled spline interpolation of a provided contour

  Parameters
  ----------
  contour : 2-d numpy array
    A sequence of x, y coordinates defining contour points
  n_samples : int
    The number of points to sample on the spline
  sparsity : int, default=1
    The distance (in number of gaps) to the next point to consider in the
    original contour (i.e. whether to consider every point, every other point,
    every 3 points... One would consider non-1 sparsity to avoid artifacts due
    to high point count contours over low pixel resolution images, with contours
    effectively curving around individual pixel edges)

  Returns
  -------
  2-d numpy array
    The spline-resampled contour with n_samples points as a sequence of x, y
    coordinates
  """
  # Force sparsity to be at least one
  sparsity = max(1, sparsity)
  # prepare the spline interpolation of the given contour
  sparse_contour = contour[::sparsity]
  tck, u = splprep( sparse_contour.T # reshaped contour
                  , s = 0 # XXX
                  , k = min(3, len(sparse_contour) - 1)
                  , per = True # closed contour (periodic spline)
                  )
  # how many times to sample the spline
  # last parameter is how dense is our spline, how many points.
  new_u = np.linspace(u.min(), u.max(), n_samples)
  # evaluate and return the sampled spline
  return np.column_stack(splev(new_u, tck))