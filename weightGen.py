# Used to generate random weights and biases prior to the network having undergone any training

import numpy as np
import json

# np.random.seed(1)

weights = {'syn_weights_1': json.dumps((np.random.standard_normal((18, 28 * 28))/(28 * 28)**.5).tolist()),
           'syn_weights_2': json.dumps((np.random.standard_normal((18, 18))/18**.5).tolist()),
           'syn_weights_3': json.dumps((np.random.standard_normal((10, 18))/18**.5).tolist())
}

biases = {
    'biases_1': json.dumps((np.zeros((18, ))).tolist()),
    'biases_2': json.dumps((np.zeros((18, ))).tolist()),
    'biases_3': json.dumps((np.zeros((10, ))).tolist())
}

with open('weights.json', 'w') as weight_file:
    json.dump(weights, weight_file)

with open('biases.json', 'w') as biases_file:
    json.dump(biases, biases_file)

