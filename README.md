********* Running N-Layer Neural Network ************
1. Open nn-start.py or nn-start.ipynb files
2. Modify model parameters:
  - 'dataset': set path to dataset. For now, dataset have to be 1 file with 1 target column
     (but equal or more labels, or text or integers)
  - 'target_column': name of target column in dataset
  - 'lambda': set regularization lambda value to "penalize" overfitting.
  - 'learning_rate': set the learning rate for gradient descent update steps.
  - 'iterations': set the number of training iterations
  - 'test_split': set percent of test split, which will be slice from loaded dataset (after shuffling).
  - 'hidden_layers': set the structure for hidden layers of the model.
  
     Format: list of integers. Each number - the amount of neurons per layer. Several numbers - several hidden layers.
     
     For example, input '[25]' will create:
     - #1 Input Layer (number of neurons == number of feauters/columns (excluding target column)) ===>>>
     - #2 Hidden Layer (one hidden layer with 25 neurons).  Fully connected to 1# Input layer
     - #3 Output Layer (number of neurons, equals the number of classes in target column). Fully connected to 2# Hidden layer
     
     For example, input '[25, 10]' will create:
     - #1 Input Layer (number of neurons == number of feauters/columns (excluding target column)) ===>>>
     - #2 Hidden Layer (first hidden layer with 25 neurons).  Fully connected to 1# Input layer
     - #3 Hidden Layer (second hidden layer with 10 neurons).  Fully connected to 2# Hidden layer
     - #4 Output Layer (number of neurons, equals the number of classes in target column). Fully connected to 3# Hidden layer
  -  'print_split': while training proccess, it will print training results each "print_split" iteration.
  -  'func_type': type of activation function to use. Option (for now): 'sigmoid', 'tanh'.
3. Run file.

********* Information ************

Implementation of algorythm completed based on numpy vectorization of forward and back propagations.
