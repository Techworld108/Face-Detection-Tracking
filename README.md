# Face-Detection-Tracking
Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images. 
With face detection, you can get the information you need to perform tasks like embellishing selfies and portraits, or generating avatars from a user's photo. Because ML Kit can perform face detection in real time, you can use it in applications like video chat or games that respond to the player's expressions.

# Haar Cascade FrontalFace Algorithm
It is based on the Haar Wavelet technique to analyze pixels in the image into squares by function. 
This uses machine learning techniques to get a high degree of accuracy from what is called “training data”. 
This uses “integral image” concepts to compute the “features” detected. 
Haar Cascades use the Adaboost learning algorithm which selects a small number of important features from a large set to give an efficient result of classifiers.

# detectMultiScale
### faces = face_cascade.detectMultiScale(src, scalefactor,minNeighbors)

 faces = face_cascade.detectMultiScale(gray, 1.3, 4)

scaleFactor — Parameter specifying how much the image size is reduced at each image scale.

minNeighbors — Parameter specifying how many neighbors each candidate rectangle should have to retain it.

# Workflow of Face Detection
Loading HaarCascadeFace Algorithm

Initializing Camera

Reading Frame from Camera

Converting Color image into Grayscale Image

Obtaining Face coordinates by passing algorithm 

Drawing Rectangle on the Face Coordinates

Display the output Frame




