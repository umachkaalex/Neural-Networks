from nn_functions import *
np.random.seed(1)

model_params = dict({'dataset': './datasets/titanic/train.csv',  # path to dataset
                     'lambda': 0.1,                              # regularisation lambda value
                     'learning_rate': 0.1,                       # learning rate
                     'iterations': 100000,                       # number of training iterations
                     'test_split': 0.2,                          # percent of test split
                     'hidden_layers': [25, 10],                  # model for hidden layers
                     'print_split': 10000,                       # each "print_split" print training results
                     'target_column': 'target',                  # name of target column in dataset
                     'func_type': 'sigmoid'})                    # activation function to use (sigmoid, tanh)
# create model
model = VectorNN(model_params)
# train model and get trained weights
weights = model.run()