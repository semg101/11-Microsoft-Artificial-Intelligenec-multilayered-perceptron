'''
Problem: Optical Character Recognition (OCR) is a hot research area and there is a great demand for automation.
The MNIST data is comprised of hand-written digits with little background noise making it a nice dataset to create, 
experiment and learn deep learning models with reasonably small comptuing resources.
'''



# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import cntk as C
C.cntk_py.set_fixed_random_seed(1)

# Select the right target device when this notebook is being tested:------------------------------------------1
'''
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

# Test for CNTK version-------------------------------------------------------------------------------------2
if not C.__version__ == "2.0":
    raise Exception("this lab is designed to work with 2.0. Current Version: " + C.__version__) 
'''

#Initialization--------------------------------------------------------------------------------------------3
# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define the data dimensions
input_dim = 784
num_output_classes = 10

#Data reading----------------------------------------------------------------------------------------------4---------------------------------A
# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file-------------------A 4.1
def create_reader(path, is_training, input_dim, num_label_classes):
    
    labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    
    deserailizer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))
            
    return C.io.MinibatchSource(deserailizer,
       randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Ensure the training and test data is generated and available for this lab.-------------------------------A 4.2
# We search in two locations in the toolkit for the cached MNIST data set.
data_found = False

for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                 os.path.join("data", "MNIST")]:
    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found = True
        break
        
if not data_found:
    raise ValueError("Please generate the data by completing the MNIST data loader in Lab 1")
    
print("Data directory is {0}".format(data_dir))


#Model Creation------------------------------------------------------------------------------------------------------------------------B 5

#Our multi-layer perceptron will be relatively simple with 2 hidden layers (num_hidden_layers). The number of nodes in the hidden layer being 
#a parameter specified by hidden_layers_dim. 
num_hidden_layers = 2
hidden_layers_dim = 400

#Network input and output:
input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)


#Multi-layer Perceptron setup-----------------------------------------------------------------------------------------------------B 5.1

#The CNTK Layers module provides a Dense function that creates a fully connected layer -----------------------------------------------B 5.1.1
#which performs the above operations of weighted input summing and bias addition.
def create_model(features):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
            h = features
            for _ in range(num_hidden_layers):
                h = C.layers.Dense(hidden_layers_dim)(h)
            r = C.layers.Dense(num_output_classes, activation = None)(h)
            return r
        
z = create_model(input)



#z will be used to represent the output of a network.----------------------------------------------------------------------------------B 5.1.2
# Scale the input to 0-1 range by dividing each pixel by 255.
input_s = input/255
z = create_model(input_s)


#Training---------------------------------------------------------------------------------------------------------------------------C 6

#Below, we define the Loss function, which is used to guide weight changes during training.
#As explained in the lectures, we use the softmax function to map the accumulated evidences 
#or activations to a probability distribution over the classes (Details of the softmax function and other activation functions).

#We minimize the cross-entropy between the label and predicted probability by the network.
loss = C.cross_entropy_with_softmax(z, label)


#Evaluation-----------------------------------------------------------------------------------------------------------------------D 7

#Below, we define the Evaluation (or metric) function that is used to report a measurement of how well our model is performing.

#For this problem, we choose the classification_error() function as our metric, 
#which returns the average error over the associated samples (treating a match as "1", 
#where the model's prediction matches the "ground truth" label, and a non-match as "0").
label_error = C.classification_error(z, label)


#Configure training--------------------------------------------------------------------------------------------------------------E 8

#The trainer strives to reduce the loss function by different optimization approaches, Stochastic Gradient Descent (sgd) being one of the most popular.
#Typically, one would start with random initialization of the model parameters. The sgd optimizer would calculate the loss or error between the predicted 
#label against the corresponding ground-truth label and using gradient-decent generate a new set model parameters in a single iteration.

#The aforementioned model parameter update using a single observation at a time is attractive since it does not require the entire data set (all observation) 
#to be loaded in memory and also requires gradient computation over fewer datapoints, thus allowing for training on large data sets. However, the updates 
#generated using a single observation sample at a time can vary wildly between iterations. An intermediate ground is to load a small set of observations and 
#use an average of the loss or error from that set to update the model parameters. This subset is called a minibatch.

#With minibatches, we sample observations from the larger training dataset. We repeat the process of model parameters update using different combination of 
#training samples and over a period of time minimize the loss (and the error metric). When the incremental error rates are no longer changing significantly 
#or after a preset number of maximum minibatches to train, we claim that our model is trained.

#One of the key optimization parameters is called the learning_rate. For now, we can think of it as a scaling factor that modulates how much we change the 
#parameters in any iteration. With this information, we are ready to create our trainer.

# Instantiate the trainer object to drive the model training---------------------------------------------------------------------E 8.1
learning_rate = 0.2
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, label_error), [learner])

#First let us create some helper functions that will be needed to visualize different functions associated with training.--------E 8.2

# Define a utility function to compute the moving average sum.-------------------------------------------------------------------E 8.2.1
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress---------------------------------------------------------------------------E 8.2.2
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        
    return mb, training_loss, eval_error



#Run the trainer---------------------------------------------------------------------------------------------------------------F 9

#We are now ready to train our fully connected neural net. We want to decide what data we need to feed into the training engine.

#In this example, each iteration of the optimizer will work on minibatch_size sized samples. We would like to train on all 60000 observations. 
#Additionally we will make multiple passes through the data specified by the variable num_sweeps_to_train_with. With these parameters we can proceed
#with training our simple feed forward network.

# Initialize the parameters for the trainer---------------------------------------------------------------------------------F 9.1
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# Create the reader to training data set------------------------------------------------------------------------------------F 9.2
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the input and labels.----------------------------------------------------------------------------F 9.3
input_map = {
    label  : reader_train.streams.labels,
    input  : reader_train.streams.features
} 

# Run the trainer on and perform model training---------------------------------------------------------------------------F 9.4
training_progress_output_freq = 500

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):
    
    # Read a mini batch from the training data file
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    
    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)


#Let us plot the errors over the different training minibatches. Note that as we progress in our training, ---------------------H 10
#the loss decreases though we do see some intermediate bumps.

# Compute the moving average loss to smooth out the noise in SGD----------------------------------------------------------------H 10.1
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error---------------------------------------------------------------------------------H 10.2
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()


#Evaluation / Testing--------------------------------------------------------------------------------------------------------I 11

#Now that we have trained the network, let us evaluate the trained network on the test data. This is done using trainer.test_minibatch

# Read the test data
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):
    
    # We are loading test data in batches specified by test_minibatch_size
    # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
    # with one pixel per dimension that we will encode / decode with the 
    # trained model.
    data = reader_test.next_minibatch(test_minibatch_size,
                                      input_map = test_input_map)

    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


#We have so far been dealing with aggregate measures of error. Let us now get the probabilities associated with individual data points.---------------J 12
#For each observation, the eval function returns the probability distribution across all the classes. The classifier is trained to recognize digits, 
#hence has 10 classes. First let us route the network output through a softmax function. This maps the aggregated activations across the network to 
#probabilities across the 10 classes.

out = C.softmax(z)

#Let us test a small minibatch sample from the test data.--------------------------------------------------------------------------------------------J 12.1

# Read the data for evaluation
reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
eval_input_map = {input: reader_eval.streams.features} 

data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)

img_label = data[label].asarray()
img_data = data[input].asarray()
predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]

# Find the index with the maximum value for both predicted as well as the ground truth
pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]

print("Label    :", gtlabel[:25])
print("Predicted:", pred)



#As you can see above, our model is not yet perfect.------------------------------------------------------------------------------------------------J 12.2
#Let us visualize one of the test images and its associated label. Do they match?

# Plot a random image
sample_number = 5
plt.imshow(img_data[sample_number].reshape(28,28), cmap="gray_r")
plt.axis('off')
plt.show()

img_gt, img_pred = gtlabel[sample_number], pred[sample_number]
print("Image Label: ", img_pred)