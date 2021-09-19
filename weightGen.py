# Used to generate random weights and biases prior to the network having undergone any training

import numpy as np
import json

# np.random.seed(1)
def generate_weights(layer_dims, save, seed=None):
    if seed:
        np.random.seed(seed)
    weight_dimensions = [(i, o) for i, o in zip(layer_dims[1:], layer_dims[:-1])]
    weights = [json.dumps((np.random.standard_normal(dim)/dim[1]**.5).tolist()) for dim in weight_dimensions]
    # weights = {'syn_weights_1': json.dumps((np.random.standard_normal((18, 28 * 28))/(28 * 28)**.5).tolist()),
    #            'syn_weights_2': json.dumps((np.random.standard_normal((18, 18))/18**.5).tolist()),
    #            'syn_weights_3': json.dumps((np.random.standard_normal((10, 18))/18**.5).tolist())
    # }
    weight_dict = {'syn_weight_' + str(i + 1): w for i, w in zip(list(range(len(weights))), weights)}

    with open(r'saves\%s\weights.json' % save, 'w') as weight_file:
        json.dump(weight_dict, weight_file)

def generate_biases(layer_dims, save):
    # biases = {
    #     'biases_1': json.dumps((np.zeros((18, ))).tolist()),
    #     'biases_2': json.dumps((np.zeros((18, ))).tolist()),
    #     'biases_3': json.dumps((np.zeros((10, ))).tolist())
    # }
    biases_dict = {'syn_weight_' + str(i + 1): json.dumps((np.zeros((b, ))).tolist()) for i, b in zip(list(range(len(layer_dims))), layer_dims[1:])}

    with open(r'saves\%s\biases.json' % save, 'w') as biases_file:
        json.dump(biases_dict, biases_file)

