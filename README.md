# Animal_Classification
I've developed this Animal Classification Application using TensorFlow and PyQt. 

 ![](https://github.com/sumanta1997/Animal_Classification/blob/master/animal%20classifier.gif)

Here transfer learning technique has been used and only the output layer of MobileNetV2 Architecture has been trained to classify among 4 different classes of animals namely Buffalo,Elephant,Zebra and Rhino.

The dataset I've used in this project has been downloaded from Kaggle and split into Train,Test and Validation.

Dataset link -https://www.kaggle.com/datasets/ayushv322/animal-classification

To run this application, following libraries must be installed on the system along with Python 3:
1.PyQt5
pip install command: pip install PyQt5

2.TensorFlow

pip install command : pip install tensorflow

3.Numpy

pip install command: pip install numpy

4. Matplotlib(optional)

pip install command: pip install matplotlib

After installing libraries, users have to follow these steps:

1.Open command prompt or bash shell and browse to the directory where final.py is located using cd command

2.then run the final.py by using the following command "python final.py"

3.Click on browse button

4.Select and image

5.Click on classify button and result will pop up in bottom textedit window

6.If you want to train the model, click on train model
