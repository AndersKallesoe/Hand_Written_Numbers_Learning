This project has been made using the numpy, json, cv2 and mlxtend libraries

The main.py file takes parameters: 
	"train [n] [b]" -> perform n training sessions on the network splitting the training data into b randomized batches 
			   for each training session
	"test" 		-> test how the network performs on the test data
	"draw"		-> render graphical user interface for drawing a number.
			   right-clicking resets the canvas and space-bar submits the picture to the network
			   and prints the network's guess on the input to the terminal.

TODO:
	- Add command-line functionality for choosing which weight and bias files to read from and write to
	- Add possibility of training with dynamic learning rate
	- Add command-line control over learning rate
	- Add functionality to log information on test data performance during testing and gradients during training for analysis