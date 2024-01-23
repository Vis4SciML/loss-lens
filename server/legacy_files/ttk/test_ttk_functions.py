# Notes
# - accept loss landscape as a list of lists
# - return csv files provided by paraview


# load the parameters from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--loss-landscape-file", default="../ttk/loss_landscapes_npy/pretrained_convection_u0sin(x)_nu0.0_beta1.0_rho0.0_Nf100_50,50,50,50,1_L1.0_source0_seed0_dim2_points1600_random.npy", help="input npy file")
parser.add_argument("--output-path", default=None, help="output file name (no extension)")
args = parser.parse_args()


# check output path
if args.output_path is None:
	args.output_path = args.loss_landscape_file.replace('.npy','')
	args.output_path = args.output_path.replace('loss_landscapes_npy','paraview_files')


### load loss_landscape from a file
import numpy as np
loss_landscape = np.load(args.loss_landscape_file)
loss_landscape = loss_landscape.reshape(-1, 1)
loss_landscape = loss_landscape.tolist()


### functions taking matrix as input
from ttk_functions import compute_persistence_barcode, compute_merge_tree, compute_merge_tree_planar

# compute persistence barcode
persistence_barcode = compute_persistence_barcode(loss_landscape, output_path=args.output_path)

# compute merge tree
merge_tree = compute_merge_tree(loss_landscape, output_path=args.output_path)

# compute merge tree (planar version)
merge_tree = compute_merge_tree_planar(loss_landscape, output_path=args.output_path)

