for i in ../ttk/output_csv_from_ttk/MNIST_PersistantDiagram_Training/*.csv; do 
    python3 load_from_csv_persistantdiagram_training.py --device=nersc --file=$i;
done
