#!/Applications/ParaView-5.11.0.app/Contents/bin/pvpython
import os
import sys
import argparse


# ----------------------------------------------------------------
# parse command-line arguments
# ----------------------------------------------------------------

# load the parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--ttk-plugin', default="/Applications/ParaView-5.11.0.app/Contents/Plugins/TopologyToolKit.so", help='Path to TTK Plugin')
parser.add_argument("--input-file", default="../ttk/output_state_from_ttk/source/MNIST_CNN_all_loss_mnist_training_3d_contour.vti", help="input vti file")
parser.add_argument("--output-file", default="../ttk/output_csv_from_ttk/MNIST_PersistantDiagram_Training/MNIST_CNN_all_loss_mnist_training_3d_contour.csv", help="output csv file")
args = parser.parse_args()


# check output folder
if not os.path.exists(os.path.dirname(args.output_file)):
    os.makedirs(os.path.dirname(args.output_file))


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
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Image Data Reader'
loss_landscape = XMLImageDataReader(registrationName='loss_landscape', FileName=[args.input_file])
loss_landscape.CellArrayStatus = ['Cell']
loss_landscape.PointArrayStatus = ['Loss']
loss_landscape.TimeArray = 'None'

# create a new 'TTK PersistenceDiagram'
tTKPersistenceDiagram1 = TTKPersistenceDiagram(registrationName='TTKPersistenceDiagram1', Input=loss_landscape)
tTKPersistenceDiagram1.ScalarField = ['POINTS', 'Loss']
tTKPersistenceDiagram1.InputOffsetField = ['POINTS', 'Loss']




# ----------------------------------------------------------------
# save merge tree (MT)
# ----------------------------------------------------------------

# save source to CSV file
SaveData(args.output_file, tTKPersistenceDiagram1, Precision=5)
       
# display progress
print(f"[+] {args.output_file}")



# ----------------------------------------------------------------
# close paraview
# ----------------------------------------------------------------

import sys
sys.exit(0)





