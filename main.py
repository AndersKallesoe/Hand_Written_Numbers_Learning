from network import Network
from drawer import Draw
import sys


def train_network(network, n, no_of_batches):
    for i in range(n):
        with open('errors.txt', 'a+') as errors_file:
            errors_file.write('\n')
            errors_file.close()
        print('starting training session %d with' % (i + 1), no_of_batches, 'batches')
        network.train_network(no_of_batches)
        network.test_network()


if __name__ == '__main__':
    if sys.argv[1] in 'train':
        train_network(Network(int(sys.argv[4])), int(sys.argv[2]), int(sys.argv[3]))
    if sys.argv[1] in 'test':
        Network(1).test_network()
    if sys.argv[1] in 'draw':
        draw = Draw(Network(1))
        draw.start()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
