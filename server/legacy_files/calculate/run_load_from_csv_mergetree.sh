for i in ../ttk/output_csv_from_ttk/MNIST_MergeTree3D/*.csv; do 
    python3 load_from_csv_mergetree.py --file=$i;
done
