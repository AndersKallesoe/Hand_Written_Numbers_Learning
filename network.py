import numpy as np
import json
from mlxtend.data import loadlocal_mnist


class Network:
    def __init__(self):
        self.images, self.labels = loadlocal_mnist(
            images_path=r"training_data\train-images.idx3-ubyte",
            labels_path=r"training_data\train-labels.idx1-ubyte"
        )
        self.test_images, self.test_labels = loadlocal_mnist(
            images_path=r"test_data\t10k-images.idx3-ubyte",
            labels_path=r"test_data\t10k-labels.idx1-ubyte"
        )

        self.input_layer = np.zeros((28 * 28, 1))
        self.expected_output = np.zeros((10, 1))

        weights_file = open(r'weights_trained.json')
        weights_dict = json.load(weights_file)
        self.weights = [np.array(json.loads(w)) for w in weights_dict.values()]

        biases_file = open(r'biases_trained.json')
        biases_dict = json.load(biases_file)
        self.biases = [np.array(json.loads(b)) for b in biases_dict.values()]

        self.errors = 0

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def sig_deriv(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def compute_output(self, input):
        a = input
        a_list = [a]
        z_list = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_list.append(z)
            a = self.sigmoid(z)
            a_list.append(a)
        return a, z_list, a_list

    def compute_cost(self, output):
        return (output - self.expected_output) ** 2

    def back_prop(self, start, stop):
        num_of_inputs = stop - start
        weight_adjustments = [np.zeros(np.shape(w)) for w in self.weights]
        biases_adjustments = [np.zeros(np.shape(b)) for b in self.biases]
        for i in range(start, stop + 1):
            #print('now starting training for input', i + 1)
            w_adj = [np.zeros(np.shape(w)) for w in self.weights]
            b_adj = [np.zeros(np.shape(b)) for b in self.biases]
            expected_output = np.zeros(10,)
            expected_output[self.labels[i]] = 1.0
            a, z_list, a_list = self.compute_output(self.images[i])
            if np.argmax(a) == self.labels[i]:
                pass
                #print('correctly classified a', self.labels[i])
            else:
                self.errors += 1
                #print('Error! Classified a ', self.labels[i], 'as a ', np.argmax(a))
            # chain rule:
            z_error = np.multiply(a - expected_output, self.sig_deriv(z_list[-1]))
            b_adj[-1] = z_error
            w_adj[-1] = np.outer(a_list[-2], z_error).T
            for l in range(2, 4):
                z = z_list[-l]
                z_error = self.sig_deriv(z) * (self.weights[-l+1].T @ z_error)
                b_adj[-l] = z_error
                w_adj[-l] = np.outer(a_list[-l-1], z_error).T
            weight_adjustments = [(w - adj) for w, adj in list(zip(weight_adjustments, w_adj))]
            biases_adjustments = [(b - adj) for b, adj in list(zip(biases_adjustments, b_adj))]
        self.weights = [w + (adj/(num_of_inputs*100)) for w, adj in list(zip(self.weights, weight_adjustments))]
        self.biases = [b + (adj/(num_of_inputs*100)) for b, adj in list(zip(self.biases, biases_adjustments))]
        self.write_settings_to_file()

    def write_settings_to_file(self):
        weights = {
            'syn_weights_1': json.dumps(self.weights[0].tolist()),
            'syn_weights_2': json.dumps(self.weights[1].tolist()),
            'syn_weights_3': json.dumps(self.weights[2].tolist())
        }

        biases = {
            'biases_1': json.dumps(self.biases[0].tolist()),
            'biases_2': json.dumps(self.biases[1].tolist()),
            'biases_3': json.dumps(self.biases[2].tolist())
        }

        print('starting writing weights and biases to files')
        with open('weights_trained.json', 'w') as weight_file:
            json.dump(weights, weight_file)
            weight_file.close()

        with open('biases_trained.json', 'w') as biases_file:
            json.dump(biases, biases_file)
            biases_file.close()

        print('Weights and biases written to file.')

    def train_network(self, no_of_batches):
        self.errors = 0
        batch_size = int(len(self.images) / no_of_batches)
        self.shuffle_data()

        data_size = len(self.labels)
        max_index = batch_size - 1
        for i in range(0, len(self.labels), batch_size):
            self.back_prop(i, max_index)
            max_index += batch_size
            if max_index > (data_size - 1):
                max_index = data_size - 1
        with open('errors.txt', 'a+') as errors_file:
            errors_file.write('\n{} errors and {} correct classifications'
                              .format(self.errors, len(self.labels) - self.errors))
            errors_file.close()
        print('Training completed!', self.errors, 'errors and',
              len(self.labels) - self.errors, 'correct classifications')

    def shuffle_data(self):
        training_data = list(zip(self.images, self.labels))
        np.random.shuffle(training_data)
        self.images, self.labels = zip(*training_data)

    def test_network(self):
        errors = 0
        for i in range(0, len(self.test_labels)):
            print(np.shape(self.test_labels[i]))
            a, _, _ = self.compute_output(self.test_images[i])
            if np.argmax(a) == self.labels[i]:
                print('correctly classified a', self.labels[i])
            else:
                errors += 1
                print('Error! Classified a ', self.labels[i], 'as a ', np.argmax(a))
        print('Test completed!', errors, 'errors and', len(self.test_labels) - errors, 'correct classifications!')
        print('Error rate:', (errors / len(self.test_labels) * 100))

network = Network()
network.test_network()