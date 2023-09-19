# PyTorch_Handwriting_Recognition

This project was is one of my first experiments with PyTorch and ML! For this starter project, I utilize the Modified National Institute of Standards and Technology (MNIST) Database of handwritten digits for a simple computer vision classification model implementation.

**Model Structure**

The model implemented follows a straightfowrard structure as shown below:

1. nn.Linear(784, 128) --> Accepts all pixels of the 28*28 images flattened out to 784 pixels and processes them to 128 feature values
2. nn.ReLU layer --> Introduces non-linearity to allow for more complex mathematical equations and higher accuracy of model
3. nn.Linear(128, 64) --> Further processes the 128 feature values resulting from first layer and processes them to 64 feature values
4. nn.ReLU layer --> Introduces non-linearity in mathematical equations once more. The 2nd layer allows for processing of higher level structures including curves in digits like 9.
5. nn.Linear(64, 10) --> Processes the final 64 feature values into 10 labels corresponding to the 10 digits of the numerical system.

This was an incredibly straightforward, yet exciting project that allowed me to explore PyTorch and get a fresh start into Machine Learning! The finally trained model achieved a final accuracy of 98.7%. The high accuracy can be largely attributed to the large database provided by MNIST and a total training time spanning 50 epochs, along with the simplicity of this particular project! 
  
# How to Implement

1. Create a custom conda environment via the following command: `conda create -n PyTorch_MNIST python=3.11`
2. Obtain required dependencies from the following command: `conda install -c conda-forge torch torchvision`
3. Implement model training and save the model by executing the python file: `python3 main.py`

Thank you so much for your interest!
   
