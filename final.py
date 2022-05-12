

from PyQt5 import QtCore, QtGui, QtWidgets
import time
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing import image
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Dropout


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1393, 940)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(900, 190, 241, 71))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(90, 160, 751, 411))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 1121, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(900, 290, 241, 71))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 590, 271, 91))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(890, 410, 261, 71))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(90, 680, 631, 81))
        self.textEdit.setObjectName("textEdit")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(890, 590, 256, 231))
        self.listWidget.setObjectName("listWidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(900, 520, 231, 51))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1393, 43))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
         
        self.BrowseImage.clicked.connect(self.loadImage)

        self.Classify.clicked.connect(self.classifyFunction)

        self.Training.clicked.connect(self.trainingFunction)
        
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Sumanta\'s Animal Classifier"))
        self.BrowseImage.setText(_translate("MainWindow", "OPEN"))
        self.label_2.setText(_translate("MainWindow", "ANIMAL CLASSIFICATION USING CNN"))
        self.Classify.setText(_translate("MainWindow", "CLASSIFY"))
        self.label.setText(_translate("MainWindow", "RECOGNIZED CLASS"))
        self.Training.setText(_translate("MainWindow", "TRAINING"))
        self.label_3.setText(_translate("MainWindow", "RESULT LOGS"))
    
    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "",
                                                        "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)")  # Ask for file
        if fileName:  # If the user gives a file
            print(fileName)
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.imageLbl.setPixmap(pixmap)  # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
            self.textEdit.setText(" ")


    def classifyFunction(self):
        
        loaded_model=tf.keras.models.load_model('animal_model.h5')
       
        print("Loaded model from disk");
        label = ["Buffalo", "Elephant", "Rhino", "Zebra",]

        path2 = self.file
        print(path2)
        test_image = image.load_img(path2, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)

        fresult = np.max(result)
        label2 = label[result.argmax()]
        print(label2)
        self.textEdit.setText(label2)
        


    def trainingFunction(self):
        train_dir = "./Dataset/Training"
        test_dir = "./Dataset/Test"
        valid_dir = "./Dataset/Validation"
        train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                     image_size=(224, 224),
                                                                     label_mode="categorical",
                                                                     batch_size=32)
        test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                    image_size=(224, 224),
                                                                    label_mode="categorical",
                                                                    batch_size=32)
        valid_data = tf.keras.preprocessing.image_dataset_from_directory(valid_dir,
                                                                     image_size=(224, 224),
                                                                     label_mode="categorical",
                                                                     batch_size=32)

        # base model
        base_model = tf.keras.applications.MobileNetV2(include_top=False)
        #  create the input layer
        inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")
        # passing to base model
        x = base_model(inputs)
        # pass to global average pooling
        x = tf.keras.layers.GlobalAveragePooling2D(name='average_pooling_layer')(x)
        # output layer
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        # combine model
        model = tf.keras.Model(inputs, outputs)

        # compile the model
        model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

        # fit the model
        model.fit(train_data,
              steps_per_epoch=len(train_data),
              epochs=10,
              validation_data=valid_data,
              validation_steps=len(valid_data))
        model.save('animal_model.h5')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
