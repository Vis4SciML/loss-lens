#!/bin/bash

# Ensure the directory exists and contains .npy files
if [ ! -d "../data/loss_landscape_files" ]; then
    echo "Directory ../data/loss_landscape_files does not exist. Please create it and add .npy files."
    exit 1
fi

# Check if there are any .npy files in the directory
if [ -z "$(ls -A ../data/loss_landscape_files/*.npy 2>/dev/null)" ]; then
    echo "No .npy files found in ../data/loss_landscape_files. Please add .npy files."
    exit 1
fi

# Process each .npy file
ls ../data/loss_landscape_files/*.npy | while read F ; do
    _cmd="python turn_landscape_to_csv.py --loss-coords-file='../data/loss_landscape_coordinates/loss_coordinates_2d_21s_441p.npy' --loss-values-file='${F}' --output-folder='../data/paraview_files' --vtk-format=vtu --graph-kwargs=aknn --persistence-threshold=0.0"
    echo "$" $_cmd
    eval $_cmd
done