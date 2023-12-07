#!/bin/bash

# loop over numpy files and process landscapes
ls loss_landscapes_npy/*.npy | while read F ; do
    _cmd="python test_ttk_functions.py --loss-landscape-file='$F'"
    echo "$" $_cmd
    eval $_cmd
done
