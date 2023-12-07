for i in ../ttk/output_csv_from_ttk/MNIST_PersistantDiagram/*.csv; do 
    python3 load_from_csv_persistantdiagram.py --device=nersc --file=$i;
done
