for i in ../ttk/output_csv_from_ttk/MNIST_MergeTree3D_Training/*.csv; do 
    python3 load_from_csv_mergetree_training.py --device=nersc --file=$i;
done
