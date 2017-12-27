# K-Nearest-Neighbours
Determine orientation of an image in computer vision using KNN
To run the problem, simply type:

python3 orient.py **\<mode> \<test OR train-file.txt> \<model-file.txt> \<model>**

where,

* mode = train or test (_you have to train before testing for custom data sets_)
* test/train file - a .txt file that contains vectors(_192 dimensions_) for the images we're training on.
* model-file - Using pickle we store the trained model in this file (used when classifying testing data)
* model - enter 'knn' here

#### Note
This program stores the trained model in a temporary file (_using pickle_) called model-file.txt
