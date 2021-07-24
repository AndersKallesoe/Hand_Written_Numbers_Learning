from network import Network
network = Network()
for n in range(1, 50):
    with open('errors.txt', 'a+') as errors_file:
        errors_file.write('\n')
        errors_file.close()
    print('starting training with ', 600, 'batches')
    network.train_network(600)