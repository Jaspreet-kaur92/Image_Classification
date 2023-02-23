# Image_Classification

## importing the required libraries. These are:

1. tensorflow: Used to work on image classification
2. numpy: used for rescaling the images and to work with matrix
3. keras.preprocessing.image.ImageDataGenerator: This method is used to genrate the batches of tensor image data with real-time data augmentation. I applied this method on tarining dataset and testing dataset to rescale the images to bring value between 0 and 1, Because neural network can't take value more than 0 and 1. From ImageDataGenerator class I used flow_from_directory() method to take a path of a directory and to generate batches of augmented data.
4. matplotlib.pyplot: This library is used to show the difference between tarin and validation loss as well as accuracy with the help of graphs.

## Convolution Neural Network (CNN)
The Convolution Neural Network (CNN) is used to classify the images. I performed data pre-processing to rescale the train and test dataset. I split the datasets into 0.7 for train set and 0.2 for test set i.e 270 images for train and 30 images for test set . After splitting the dataset build the model. Basic building block of any neural network is neural netwok layer. I defined some layers. It will feed information into hidden layers and the it will start training the model. Convolution Conv2D() is used from keras.layers class to create the convolution layers, 3 convolution layers are created. These layers will create a convolution kernel that is convolved with the layer input to produce a tensor of outputs for this layer 'relu' activation is used because it prevents the Vanishing Gradient problem. I uesd flatten layer to  transform data into one-dimension. Dense method of layers class is used to create output layer. In Dense(3) it represent the number of class labels. In this datase 3 classes and the "softmax" activation function is used for output layer. Summary() function is used to show the summary of the model. 

# Model Compilation
The compile() method is uesd for model compilation with this method loss function, optimizor and matrix are defined.
- Loss function which tell us how much is the accuarcy of the model while training and testing.
- Optimization is used to overcome/overfitting of the model during training. Sometimes the model works very well on the training dataset but works very bad on the test dataset.

# Fit() model
fit model on x_train and y_train dataset. In this epochs means when the model run for the first time the weights will be intialized randomly and in the next step, weights will be updated via back propagation. I used epochs = 15, which means the model is getting trian 15 times on the whole data and weights are getting updated 15 times. Updation of the weights by using back-propagation.

# Model evalution
This model works well. The loss rate is decreasing and accuracy increased as shown in the graphs. 
* loss: 0.1545 
* accuracy: 0.9333

# Conclusion
I randomly selected images and apply this model. It predicts the images accuratly. In this, I have done pre-processing on data, train the CNN Deep learning model , build the model, predict the results and created flask app using python for Furniture Classification based on Deep Learning model. 

