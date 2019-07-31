import itertools

HYPERPARAM_DICT = {
    "BATCH_SIZE": [16, 32],
    "N_EPOCHS": [50, 100, 500],
    "LEARNING_RATE": [1e-4, 1e-5, 1e-6],
    "WEIGHT_DECAY": [1e-5, 1e-6],
    "CLIPPING_THRESHOLD": [5],
    "VALIDATION_SIZE": [0.2],
    "TEST_SIZE": [0.1],
    "DROP_LAST": [True, False],
}


def get_hyperparam_list(hyperparameter_dict):
    """
    input: ?
    output: a list of dictionaries, each containing the hyperparameter name and value pairs
    """
    var_names = hyperparameter_dict.keys()
    hyperparameter_list = []
    for combination in list(itertools.product(*hyperparameter_dict.values())):
        hyperparameter_list.append(dict(zip(var_names, combination)))
    return hyperparameter_list
