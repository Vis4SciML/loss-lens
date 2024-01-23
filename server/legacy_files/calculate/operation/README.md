# Back-End Operations Instructions
This is an instruction to introduce all the operation functions in the whole pipeline. For different model structures, there might have slightly differences. If the users do need to apply different model structures into this tool such as a model structure that `torchvision` does not accept, the users might need to define a new function by slightly changing some APIs based on current functions. In this instruction, we will introduce the basic functions for 2-Layer MLP model structure. You can find more details for the `Torchvision` and `VIT` models in `functions.py`.

## Model Euclidean Distance Similarity
Model Euclidean Distance Similarity is defined in the `calculate_model_euclidean_distance_similarity` under `functions.py` in the current folder.

```python
def calculate_model_similarity_global_structure(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.linear_1.weight.data.reshape(-1),model.linear_2.weight.data.reshape(-1))))
    
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for i in range(len(model_list)):
        euclidean_distance_row = []
        for j in range(len(model_list)):
            euclidean_distance_row.append((modelTensor_list[i] - modelTensor_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    euclidean_distance_matrix = np.array(euclidean_distance_column)
    
    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)

    # calculate the global structure of the models based on the MDS
    condensed_distance_matrix = pdist(X_transformed)
    Z = hierarchy.linkage(condensed_distance_matrix, 'single')
    T = to_tree(Z, rd=False)
    figure  = to_newick(T, ascii_lowercase)

    return X_transformed, figure
```

## Model CKA Similarity
Model CKA Similarity is defined in the `calculate_model_cka_similarity` under `functions.py` in the current folder.

```python
def calculate_model_cka_similarity_global_structure(dataset, subdataset_list, IN_DIM, OUT_DIM, reshape_x, reshape_y):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.linear_1.weight.data.reshape(-1),model.linear_2.weight.data.reshape(-1))))
    
    # initialize the CKA class
    np_cka = CKA()
    
    # calculate the linear CKA similarity between models
    linear_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        linear_cka_similarity_row = []
        for j in range(len(model_list)):
            linear_cka_similarity_row.append(np_cka.linear_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        linear_cka_similarity_column.append(linear_cka_similarity_row)
    
    kernel_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        kernel_cka_similarity_row = []
        for j in range(len(model_list)):
            kernel_cka_similarity_row.append(np_cka.kernel_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        kernel_cka_similarity_column.append(kernel_cka_similarity_row)
    
    # generate the distance matrix for linear CKA
    linear_cka_matrix = 1 - np.array(linear_cka_similarity_column)
    
    # generate the distance matrix for RBF kernel CKA
    kernel_cka_matrix = 1 - np.array(kernel_cka_similarity_column)
    
    # calculate the MDS of the linear CKA distance matrix for models similarity
    linear_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    linear_cka_embedding = linear_cka_mds.fit_transform(linear_cka_matrix)

    # calculate the global structure of the models based on the linear CKA MDS
    linear_cka_condensed_distance_matrix = pdist(linear_cka_embedding)
    linear_cka_Z = hierarchy.linkage(linear_cka_condensed_distance_matrix, 'single')
    linear_cka_T = to_tree(linear_cka_Z, rd=False)
    linear_cka_figure  = to_newick(linear_cka_T, ascii_lowercase)

    # calculate the MDS of the RBF kernel CKA distance matrix for models similarity
    kernel_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    kernel_cka_embedding = kernel_cka_mds.fit_transform(kernel_cka_matrix)

    # calculate the global structure of the models based on the RBF kernel CKA MDS
    kernel_cka_condensed_distance_matrix = pdist(kernel_cka_embedding)
    kernel_cka_Z = hierarchy.linkage(kernel_cka_condensed_distance_matrix, 'single')
    kernel_cka_T = to_tree(kernel_cka_Z, rd=False)
    kernel_cka_figure  = to_newick(kernel_cka_T, ascii_lowercase)

    return linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure
```

## Layer CKA Similarity
Layer CKA Similarity is defined in the `calculate_model_layer_torch_cka_similarity` under `functions.py` in the current folder.

```python
def calculate_model_layer_torch_cka_similarity(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # prepare the dataloader
    mnist_original = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
    mnist_original_train_loader = torch.utils.data.DataLoader(mnist_original, batch_size=1, shuffle=False)
    x, y = iter(mnist_original_train_loader).__next__()
    mnist_test = []
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x[i], y[i]])
    x_c = load('../data/MNIST_C/brightness/train_images.npy')
    y_c = load('../data/MNIST_C/brightness/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x_c)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/canny_edges/train_images.npy')
    y_c = load('../data/MNIST_C/canny_edges/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x_c)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/dotted_line/train_images.npy')
    y_c = load('../data/MNIST_C/dotted_line/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/fog/train_images.npy')
    y_c = load('../data/MNIST_C/fog/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/glass_blur/train_images.npy')
    y_c = load('../data/MNIST_C/glass_blur/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/impulse_noise/train_images.npy')
    y_c = load('../data/MNIST_C/impulse_noise/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/motion_blur/train_images.npy')
    y_c = load('../data/MNIST_C/motion_blur/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/rotate/train_images.npy')
    y_c = load('../data/MNIST_C/rotate/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/scale/train_images.npy')
    y_c = load('../data/MNIST_C/scale/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/shear/train_images.npy')
    y_c = load('../data/MNIST_C/shear/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/shot_noise/train_images.npy')
    y_c = load('../data/MNIST_C/shot_noise/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/spatter/train_images.npy')
    y_c = load('../data/MNIST_C/spatter/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/stripe/train_images.npy')
    y_c = load('../data/MNIST_C/stripe/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/translate/train_images.npy')
    y_c = load('../data/MNIST_C/translate/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/zigzag/train_images.npy')
    y_c = load('../data/MNIST_C/zigzag/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    # define testing data loader
    dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=MNIST_BATCH_SIZE_LAYER_CKA, shuffle=False)
    
    # calculate the CKA distance for models using torch CKA
    cka_result = []
    for i in range(len(model_list)):
        cka_result_row = []
        for j in range(len(model_list)):
            cka = torch_cka.CKA(model_list[i], model_list[j], device=DEVICE)
            cka.compare(dataloader)
            results = cka.export()
            cka_result_row.append(results)
        cka_result.append(cka_result_row)

    return cka_result
```

## Loss Landscapes with Random Projection
Besides the hessian parametric loss landscapes, we also provide one way to generate the classic loss landscapes for different models and compare the similarity among them using random projection method. The method to generate the classic loss landscapes has already merged into the main calculation function. It is calculated by `calculate_loss_landscapes_random_projection`.

```python
def calculate_loss_landscapes_random_projection(dataset, subdataset_list, criterion, x, y, IN_DIM, OUT_DIM, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    max_loss_value_list = []
    min_loss_value_list = []
    model_info_list = []
    
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    
    # calculate the first original sub-dataset and get one random projection
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    # calculate the loss landscape in 2 dimensions for the rest of the models
    for i in range(len(model_list)):
        # calculate the loss landscape in 2 dimensions for the model
        loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)
        model_info_list.append(model_info_this)

        # first array corresponds to which row, and the latter corresponds to which colresnet50umn
        max_loss_this = np.where(loss_data_fin_this == np.max(loss_data_fin_this))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_this == np.min(loss_data_fin_this))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])
        
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list
```

All the results will be saved in the MongoDB and all the generated binary files used for ttk will be saved under the `ttk/input_binary_for_ttk` folder. The loss landscapes information stored in the MongoDB can be requested by the Front-End. In the Front-End, this information can be plotted as several 2D and 3D loss landscapes plots.

## Model Evaluation
Model Evaluation calculations is defined in the `calculate_model_information` under `functions.py` in the current folder.

```python
def calculate_model_information(dataset, subdataset_list, i, x, y, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    model = model_list[i]
    model.eval()

    # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
    preds = model(x)
    # get the accuracy
    accuracy = Accuracy(task="multiclass", average='macro', num_classes=10)
    model_accuracy = accuracy(preds, y)
    # get the recall
    recall = Recall(task="multiclass", average='macro', num_classes=10)
    model_recall = recall(preds, y)
    # get the precision
    precision = Precision(task="multiclass", average='macro', num_classes=10)
    model_precision = precision(preds, y)
    # get the f1 score
    f1 = F1Score(task="multiclass", num_classes=10)
    model_f1 = f1(preds, y)
    # get the confusion matrix
    confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=10)
    model_confusionMatrix = confusionMatrix(preds, y)

    return model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix
```

## Hessian Top Eigenvalues
Hessian Top Eigenvalues calculations is defined in the `calculate_top_eigenvalues_hessian` under `functions.py` in the current folder.

```python
def calculate_top_eigenvalues_hessian(dataset, subdataset_list, criterion, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    top_eigenvalues_list = []
    for j in range(len(model_list)):
        # get one model
        model = model_list[j]
        model.eval()

        # get the training dataset for the model
        x, y = get_mnist_for_one_model(subdataset_list[j])

        if FLAG == True:
            model.cuda()
            x = x.cuda()
            y = y.cuda()
        
        # calculate hessian and top eigenvalues
        hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        top_eigenvalues_list.append(top_eigenvalues)
    return top_eigenvalues_list
```

## Model Detailed Similarity
Model Detailed Similarity calculations is defined in the `calculate_detailed_similarity_from_loss_landscapes_models` under `functions.py` in the current folder.

```python
def calculate_detailed_similarity_from_loss_landscapes_models(selected_model_list_all, i, j):
    selected_models_one = copy.deepcopy(selected_model_list_all[i])
    selected_models_two = copy.deepcopy(selected_model_list_all[j])
    this_whole_model_list = []
    for m in range(len(selected_models_one)):
        this_whole_model_list.append(selected_models_one[m])
    for n in range(len(selected_models_two)):
        this_whole_model_list.append(selected_models_two[n])
        
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for p in range(len(this_whole_model_list)):
        euclidean_distance_row = []
        for q in range(len(this_whole_model_list)):
            euclidean_distance_row.append(torch.tensor(np.array(this_whole_model_list[p]) - np.array(this_whole_model_list[q])).pow(2).sum().sqrt())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    euclidean_distance_matrix = np.array(euclidean_distance_column)
    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)
    
    return X_transformed
```

## Prediction Distribution
Prediction Distribution calculations is defined in the `calculate_prediction_distribution` under `functions.py` in the current folder.

```python
def calculate_prediction_distribution(model_set_one, model_set_two, distribution_test_loader, MAX_NUM):
    # prepare the result set
    correct_correct = set()
    correct_wrong = set()
    wrong_correct = set()
    wrong_wrong = set()

    # create the iterator for the testing dataset
    distribution_test_loader_iter = iter(distribution_test_loader)

    print("Length of the testing dataset: ", len(distribution_test_loader))

    # calculate the prediction distribution result for each testing image
    for i in range(len(distribution_test_loader)):
        distribution_x, distribution_y = distribution_test_loader_iter.__next__()
        # predict all the results of the first model set using voting
        result_one_list = []
        for j in range(len(model_set_one)):
            model = model_set_one[j]
            model.eval()
            prediction = model(distribution_x).reshape(1,10)
            result = torch.argmax(prediction, dim=1).item()
            result_one_list.append(result)
        # calculate the voting result for the first model set
        voting_result_one = most_frequent(result_one_list)
        # predict all the results of the second model set using voting
        result_two_list = []
        for k in range(len(model_set_two)):
            model = model_set_two[k]
            model.eval()
            prediction = model(distribution_x).reshape(1,10)
            result = torch.argmax(prediction, dim=1).item()
            result_two_list.append(result)
        # calculate the voting result for the second model set
        voting_result_two = most_frequent(result_two_list)
        # get the correct label
        label_y = distribution_y.item()

        # calculate the prediction distribution result for one testing image
        if voting_result_one == voting_result_two:
            if voting_result_one == label_y:
                if len(correct_correct) < MAX_NUM:
                    correct_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        else:
            if voting_result_one == label_y:
                if len(correct_wrong) < MAX_NUM:
                    correct_wrong.add(i)
            elif voting_result_two == label_y:
                if len(wrong_correct) < MAX_NUM:
                    wrong_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        
        # check if the result sets are full
        if len(correct_correct) >= MAX_NUM and len(correct_wrong) >= MAX_NUM and len(wrong_correct) >= MAX_NUM and len(wrong_wrong) >= MAX_NUM:
            break

    # return the result sets as lists
    return list(correct_correct), list(correct_wrong), list(wrong_correct), list(wrong_wrong)
```
