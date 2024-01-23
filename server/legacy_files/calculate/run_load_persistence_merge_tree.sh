for i in ../ttk/output_csv_from_ttk/MNIST_PersistantDiagram/*.csv; do 
    python3 load_from_csv_persistantdiagram_training.py --device=local --file=$i;
done

for i in ../ttk/output_csv_from_ttk/MNIST_PersistantDiagram_Training/*.csv; do 
    python3 load_from_csv_persistantdiagram_training.py --device=local --file=$i;
done

for i in ../ttk/output_csv_from_ttk/MNIST_MergeTree3D_Training/*.csv; do 
    python3 load_from_csv_mergetree_training.py --device=local --file=$i;
done

for i in ../ttk/output_csv_from_ttk/MNIST_MergeTree3D/*.csv; do 
    python3 load_from_csv_mergetree.py --device=local --file=$i;
done
