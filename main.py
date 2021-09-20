from network import Network
from drawer import Draw
from weightGen import generate_biases, generate_weights
import sys
import re
import os.path as op
import os


def train_network(network, n, no_of_batches):
    for i in range(n):
        with open('errors.txt', 'a+') as errors_file:
            errors_file.write('\n')
            errors_file.close()
        print('starting training session %d with' % (i + 1), no_of_batches, 'batches')
        network.train_network(no_of_batches)
        network.test_network()


def new_network(save, layer_dims, learning_rate=100, seed=None):
    os.makedirs(fr'saves\{save}')
    with open(fr'saves\{save}\specs.txt', 'w') as spec_file:
        spec_file.write('Dimensions: ' + str(layer_dims))
    generate_weights(layer_dims, save, seed)
    generate_biases(layer_dims, save)
    network = Network(learning_rate, save)
    return network


def load_network(save, learning_rate):
    return Network(learning_rate, save)


def listen(network):
    while True:
        for line in sys.stdin:
            args = [l.strip() for l in line.split(' ')]
            if args[0] in 'l_rate':
                network.set_learning_rate(int(args[1]))
            if args[0] in 'train':
                train_network(network, int(args[1]), int(args[2]))
            if args[0] in 'test':
                network.test_network()
            if args[0] in 'draw':
                draw = Draw(network)
                draw.start()
            if args[0] in 'quit':
                exit()


def main():
    while True:
        for line in sys.stdin:
            args = [l.strip() for l in line.split(' ')]
            if args[0] in 'create':
                if op.exists(fr'saves\{args[1]}'):
                    print(args)
                    print("Save % already exists! Choose different name" % args[1])
                    continue
                network = new_network(args[1], [784] + [int(d) for d in re.findall(r'\d+', args[2])] + [10], learning_rate=100)
            if args[0] in 'load':
                if not op.exists(fr'saves\{args[1]}'):
                    print("Requested Save %s doesn't exist!" % args[1])
                    continue
                network = load_network(args[1], 100)
            break
        break
    listen(network)


if __name__ == '__main__':
    main()
    # if sys.argv[1] in 'train':
    #     train_network(Network(int(sys.argv[4])), int(sys.argv[2]), int(sys.argv[3]))
    # if sys.argv[1] in 'test':
    #     Network(1).test_network()
    # if sys.argv[1] in 'draw':
    #     draw = Draw(Network(1))
    #     draw.start()

