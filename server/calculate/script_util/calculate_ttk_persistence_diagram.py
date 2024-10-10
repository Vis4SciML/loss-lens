#!/Applications/ParaView-5.11.0.app/Contents/bin/pvpython
import os
import sys
import argparse


# ----------------------------------------------------------------
# parse command-line arguments
# ----------------------------------------------------------------

# load the parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--ttk-plugin', default="/Applications/ParaView-5.11.1.app/Contents/Plugins/TopologyToolKit.so", help='Path to TTK Plugin')
parser.add_argument("--input-file", default="../ttk/output_state_from_ttk/source/MNIST_CNN_all_loss_mnist_training_3d_contour.vti", help="input vti file")
parser.add_argument("--output-file", default="../ttk/output_csv_from_ttk/MNIST_PersistantDiagram_Training/MNIST_CNN_all_loss_mnist_training_3d_contour.csv", help="output csv file")
parser.add_argument("--persistence-threshold", type=float, default=0, help="Threshold for simplification by persistence (use --threshold-is-absolute if passing a scalar value.")
parser.add_argument("--threshold-is-absolute", action="store_true", help="Is the threshold an absolute scalar value or a fraction (0 - 1) of the function range.")
args = parser.parse_args()


# check output folder
output_folder = os.path.dirname(args.output_file)
if not os.path.exists(output_folder) and len(output_folder):
    os.makedirs(output_folder)


# ----------------------------------------------------------------
# paraview imports
# ----------------------------------------------------------------

# state file generated using paraview version 5.11.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()



# ----------------------------------------------------------------
# load plugins
# ----------------------------------------------------------------

# load ttk plugin
LoadPlugin(args.ttk_plugin, remote=False, ns=globals())



# ----------------------------------------------------------------
# load the data
# ----------------------------------------------------------------

loss_landscape = None

if args.input_file.endswith('.vti'):

    # create a new 'XML Image Data Reader'
    loss_landscape = XMLImageDataReader(registrationName='loss_landscape', FileName=[args.input_file])
    loss_landscape.CellArrayStatus = ['Cell']
    loss_landscape.PointArrayStatus = ['Loss']
    loss_landscape.TimeArray = 'None'

elif args.input_file.endswith('.vtu'):

    # create a new 'XML Unstructured Grid Reader'
    loss_landscape = XMLUnstructuredGridReader(registrationName='loss_landscape', FileName=[args.input_file])
    loss_landscape.CellArrayStatus = ['Cell']
    loss_landscape.PointArrayStatus = ['Loss']
    loss_landscape.TimeArray = 'None'

else:
    
    raise ValueError("VTK file format cannot be determined, please provide a .vti or vtu file")



# ----------------------------------------------------------------
# simplify by persistence
# ----------------------------------------------------------------

# create a new 'TTK TopologicalSimplificationByPersistence'
tTKTopologicalSimplificationByPersistence1 = TTKTopologicalSimplificationByPersistence(registrationName='TTKTopologicalSimplificationByPersistence1', Input=loss_landscape)
tTKTopologicalSimplificationByPersistence1.InputArray = ['POINTS', 'Loss']
tTKTopologicalSimplificationByPersistence1.PairType = 'Minimum-Saddle'
tTKTopologicalSimplificationByPersistence1.PersistenceThreshold = args.persistence_threshold
tTKTopologicalSimplificationByPersistence1.ThresholdIsAbsolute  = args.threshold_is_absolute



# ----------------------------------------------------------------
# compute persistence diagram (PD)
# ----------------------------------------------------------------

# create a new 'TTK PersistenceDiagram'
tTKPersistenceDiagram1 = TTKPersistenceDiagram(registrationName='TTKPersistenceDiagram1', Input=tTKTopologicalSimplificationByPersistence1)
tTKPersistenceDiagram1.ScalarField = ['POINTS', 'Loss']
tTKPersistenceDiagram1.InputOffsetField = ['POINTS', 'Loss']




# ----------------------------------------------------------------
# save persistence diagram (PD)
# ----------------------------------------------------------------

# save source to CSV file
SaveData(args.output_file, tTKPersistenceDiagram1, Precision=8)
       
# display progress
print(f"[+] {args.output_file}")



# ----------------------------------------------------------------
# close paraview
# ----------------------------------------------------------------

import sys
sys.exit(0)





