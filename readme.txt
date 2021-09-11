This project has been made using the numpy, json, cv2 and mlxtend libraries.

It is an implementation of a classic neural network that can recognize hand-written digits from 0 to 9.
The code of the actual network is in the network.py file. the weightGen.py file can generate initial weights and biases.
The drawer.py file contains code for rendering a graphical user interface for drawing a number and submitting it to the
network to make a guess on which digit it is.
The whole project can be run via the main.py file.

The main.py file takes parameters: 
	"train [n] [b] [l]" -> perform n training sessions on the network splitting the training data into b randomized
	                       batches for each training session with a learning rate 1/l.
	"test" 		        -> test how the network performs on the test data
	"draw"		        -> render graphical user interface for drawing a number.
			               right-clicking resets the canvas and space-bar submits the picture to the network
			               and prints the network's guess on the input to the terminal.

TODO:
	- Add command-line functionality for choosing which weight and bias files to read from and write to
	- Add possibility of training with dynamic learning rate
	- Add functionality to log information on test data performance during testing and gradients during training for
	  analysis