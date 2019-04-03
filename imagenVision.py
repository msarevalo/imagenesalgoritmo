from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

import numpy as np
import os
import glob

import io

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

raiz=Tk()

def desarrolla():
    messagebox.showinfo("Desarrolladores", "Manuel Santiago Arévalo Corredor\nMaria Alejandra Rojas Florez")

def salir():
    valor=messagebox.askokcancel("Salir", "¿Estas seguro que deseas salir de la aplicacion?")
    if valor:
        raiz.destroy()

def abrefichero(path, pattern):
    fichero=filedialog.askopenfilename(title="Abrir Archivo", filetypes=(("Imagenes", "*.jpg"), ("Imagenes", "*.png"), ("Imagenes", "*.jpeg")))
    class_names={}
    class_id=0
    x = []
    y = []
    for d in glob.glob(os.path.join(path, '*')):
        clname = os.path.basename(d)
        for f in glob.glob(os.path.join(d, pattern)): 
            if not clname in class_names:
                class_names[clname]=class_id 
                class_id += 1
            img = image.load_img(f, target_size=(224, 224))
            npi = image.img_to_array(img)     
            npi = preprocess_input(npi)
            for i in range(4):
                npi=np.rot90(npi, i)
                x.append(npi)
                y.append(class_names[clname])
    return np.array(x), np.array(y), class_names


x, y, class_names = ('C:\\Animales', '*.jpg')
num_classes = len(class_names)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


model = InceptionResNetV2(weights='imagenet', include_top=False)

x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x) 
model = Model(input=model.input, output=predictions)


LAYERS_TO_FREEZE=700
for layer in model.layers[:LAYERS_TO_FREEZE]:
    layer.trainable = False


model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)


raiz.title("Analisis de Imagenes")
#raiz.geometry("500x500")
raiz.config(bg="#858585")
raiz.resizable(0,0)

barraMenu=Menu(raiz)
raiz.config(menu=barraMenu)

archivoMenu=Menu(barraMenu, tearoff=0)
archivoMenu.add_command(label="Abrir", command=abrefichero)
archivoMenu.add_command(label="Salir", command=salir)

acercaMenu=Menu(barraMenu, tearoff=0)
acercaMenu.add_command(label="Desarrolladores", command=desarrolla)

barraMenu.add_cascade(label="Archivo", menu=archivoMenu)
barraMenu.add_cascade(label="Acerca", menu=acercaMenu)


miFrame=Frame()
miFrame.pack(fill="both", expand="true")
miFrame.config(bg="#858585")
miFrame.config(width="300", height="150")

miLabel=Label(miFrame, text="Por favor ingresa\nuna imagen", bg="#858585", font=("Times News Roman",14), fg="white").place(x="75", y="10")

Button(raiz, text="Abrir Archivo", command=abrefichero).place(x="110", y="75")
Button(raiz, text="Salir", command=salir).place(x="130", y="110")


raiz.mainloop()