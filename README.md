
3. 24/12/2024 (Christmas Eve):
    - Huge progress: Custom Dataset and CNN Model creation. Learned a lot today
    - Added example code using simple greyscale matrix for convolution and pooling
    - CNN model class:
        - Formed the structure of the CNN in the __init__ function with 3 rounds of convolution, activation functions, and max pooling (along with additional functions that I am still learning the functions of but know they are necessary)
        - Created the forward passing function in action to use that structure
    - Custom Dataset:
        - Used pandas library for CSV parsing. Initially formatted the images and opened/converted them into RGB values using PIL library. 
        - Used os library for joining of file paths to access images and establish a root directory.
        - Filtered access of the data in the csv dependant on their purpose (train/test/validate)
        - Instantiates column values in the CSV as variables to access
    - Combined the model and the dataset together to iterate through epochs of training data through the model to train it including backpropogation techniques (with cross entropy loss)
    
    *NEXT TASK* -> Debug code



References used:
    - https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/
    - https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/
    - https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification/code
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - https://deeplizard.com/learn/video/Zi-0rlM4RDs#:~:text=The%20training%20set%20is%20what%20it%20sounds%20like.
    

NOTE: ChatGPT was used in aiding in the understanding of concepts for CNNs and understanding the function of sections of code used in referenced resources. 
=======
    -PyTorch tutorial
