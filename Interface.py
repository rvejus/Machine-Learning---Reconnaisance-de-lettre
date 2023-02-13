import itertools
import os
import random
from ctypes import *

import matplotlib.pyplot as plt
import numpy as np
import cv2

_dll = cdll.LoadLibrary(
    "C:/Users/33783/Documents/5A3DJV/Machine Learning/Machine-Learning---Reconnaisance-de-lettre/dll_machineLearning.dll")

_dll.initPMC.argtypes = [POINTER(c_int), c_int]
_dll.initPMC.restype = c_void_p

_dll.propagatePMC.argtypes = [c_void_p, POINTER(c_float), c_bool]
_dll.propagatePMC.restype = None

_dll.predictPMC.argtypes = [c_void_p, POINTER(c_float), c_bool]
_dll.predictPMC.restype = POINTER(c_float)

_dll.trainPMC.argtypes = [c_void_p, POINTER(POINTER(c_float)), c_int, POINTER(POINTER(c_float)), c_bool, c_float, c_int]
_dll.trainPMC.restype = None

_dll.EntrainementLineaireImage.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float),POINTER(c_float), c_int, c_int]
_dll.EntrainementLineaireImage.restype = POINTER(c_float)

_dll.predictImage.argtypes = [POINTER(c_float),POINTER(c_float), c_int]
_dll.predictImage.restype = c_int

_dll.free_float.argtypes = [POINTER(c_float)]
_dll.free_float.restype = None

#classe du PMC comportant toutes les fonction necessairee a le faire fonctioner
class PMC(object):
    def __init__(self, npl, nplSize):
        self._as_parameter_ = _dll.initPMC(npl, nplSize)

    def _propagate(self, inputs, is_classification):
        _dll.propagatePMC(self, inputs, is_classification)

    def predict(self, inputs, is_classification):
        return _dll.predictPMC(self, inputs, is_classification)

    def train(self, X_train, X_train_size, Y_train, is_classification, alpha=0.1, nb_iter=10000):
        return _dll.trainPMC(self, X_train, X_train_size, Y_train, is_classification, alpha, nb_iter)


def color_grid(W, width, height):
    print(W[0])
    print(W[1])
    print(W[2])
    test_points = []
    test_colors = []
    for row in range(0, 300):
        for col in range(0, 300):
            p = np.array([col / 100, row / 100])
            c = 'lightcyan' if np.matmul(np.transpose(W), np.array([1.0, *p])) >= 0 else 'pink'
            test_points.append(p)
            test_colors.append(c)
    test_points = np.array(test_points)
    test_colors = np.array(test_colors)
    return test_points, test_colors


# Fonction load des images du dataset d'entrainement
def loadDataSet(C,classe, N, classeN):
    data = []
    classes = []
    nbImage = 0
    for filename in os.listdir(C):
        img = cv2.imread(os.path.join(C, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dims = (28, 28)
        resized = cv2.resize(gray, dims, interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        vectorized = normalized.reshape(1, 28 * 28).astype(c_double)
        data.append(vectorized.tolist()[0])
        classes.append(classe)
        nbImage = nbImage + 1

        for filename in os.listdir(N):
            img = cv2.imread(os.path.join(N, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dims = (28, 28)
            resized = cv2.resize(gray, dims, interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0
            vectorized = normalized.reshape(1, 28 * 28).astype(c_double)
            data.append(vectorized.tolist()[0])
            classes.append(classeN)
            nbImage = nbImage + 1

    nbValue = dims[0] * dims[1]

    return nbImage, data, classes, nbValue

def loadImage(C):
    data = []

    img = cv2.imread(C)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dims = (28, 28)
    resized = cv2.resize(gray, dims, interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    vectorized = normalized.reshape(1, 28 * 28).astype(c_double)
    data.append(vectorized.tolist()[0])

    nbValue = dims[0] * dims[1]

    return data, nbValue

#Converti un tableau en Double*
def ConvertirPointerFloat(flat_data):
    my_array = (c_float * len(flat_data))
    c_array = my_array(*flat_data)
    # Convertir le tableau ctypes en un pointeur
    pointer_to_ctypes_array = pointer(c_array)
    double_pointer = cast(pointer_to_ctypes_array, POINTER(c_float))
    return double_pointer

######################################################
#              LINEAR SIMPLE                         #
######################################################
def ClassificationLinearSimple():


    points = [1.0,1.0,2.0,2.0,1.0,3.0]
    c_arrayPoint = ConvertirPointerFloat(points)

    classes = [1.0, -1.0, -1.0]
    c_arrayClasses = ConvertirPointerFloat(classes)

    # on Initialise les poids en random
    W = []
    for i in range(3):
        W.append(2 * random.random() - 1)

    c_arrayW = (c_float * len(W))(*W)
    Xk = [i for i in range(3)]
    Xk_ptr = ConvertirPointerFloat(Xk)
    # On lance l'entrainement linéaire
    W = _dll.EntrainementLineaireImage(c_arrayPoint,c_arrayClasses,c_arrayW,Xk_ptr,3,2,2)
    test_point , test_color = color_grid(c_arrayW,300,300)



    classes_string = ["blue" if i == 1 else "red" for i in classes]
    array2D = np.reshape(points, (3, 2))
    # on affiche le tout

    plt.scatter(test_point[:, 0], test_point[:, 1], c=test_color)
    plt.scatter(array2D[:,0], array2D[: ,1], c=classes_string)
    plt.show()


######################################################
#            LINEAR MULTIPLE                         #
######################################################
def ClassificationLinearMultiple():
    class Point(Structure):
        _fields_ = [("x", c_float), ("y", c_float)]

    _dll.EntrainementLineaire.argtypes = [POINTER(Point), POINTER(c_int), POINTER(c_double), c_int]
    _dll.EntrainementLineaire.restype = POINTER(c_double)

    X = []
    for i in range(50):
        X.append(Point(random.random() * 0.9 + 1, random.random() * 0.9 + 1))
    for i in range(50):
        X.append(Point(random.random() * 0.9 + 2, random.random() * 0.9 + 2))

    # Génération du vecteur de cibles
    Y = []
    for i in range(50):
        Y.append(1)
    for i in range(50):
        Y.append(-1)

    x_points = X[0:50]
    x_points1 = X[50:100]

    c_arrayPoint = (Point * len(X))(*X)
    pointSize = len(X)
    c_arrayClasses = (c_int * len(Y))(*Y)

    W = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    c_arrayW = (c_double * len(W))(*W)

    _dll.EntrainementLineaire(c_arrayPoint, c_arrayClasses, c_arrayW, pointSize)

    test_points = (Point * 90000)()
    test_colors = (c_int * 90000)()

    _dll.AffichageSeparation(c_arrayW, test_points, test_colors)

    xx = []
    xy = []

    yx = []
    yy = []

    for Point in x_points:
        xx.append(float(Point.x))
        xy.append(float(Point.y))

    for Point in x_points1:
        yx.append(float(Point.x))
        yy.append(float(Point.y))
    plt.scatter(xx, xy, color='blue')
    plt.scatter(yx, yy, color='red')

    x_test_points = []
    y_test_points = []

    for Point in test_points:
        x_test_points.append(float(Point.x))
        y_test_points.append(float(Point.y))

    x_points = []
    y_points = []

    for Point in X:
        x_points.append(float(Point.x))
        y_points.append(float(Point.y))

    test_color_string = ["lightcyan" if i == 0 else "pink" for i in test_colors]
    classes_string = ["blue" if i == 1 else "red" for i in Y]

    # on affiche le tout

    plt.scatter(x_test_points, y_test_points, c=test_color_string)
    plt.scatter(x_points, y_points, c=classes_string)
    plt.show()


######################################################
#                         XOR                        #
######################################################
def ClassificationXOR():
    class Point(Structure):
        _fields_ = [("x", c_float), ("y", c_float)]

    points = [Point(1, 0), Point(0, 1), Point(0, 0), Point(1, 1)]
    c_arrayPoint = (Point * len(points))(*points)

    pointSize = len(points)

    classes = [1, 1, 0, 0]
    c_arrayClasses = (c_int * len(classes))(*classes)

    W = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    c_arrayW = (c_double * len(W))(*W)

    _dll.EntrainementLineaire(c_arrayPoint, c_arrayClasses, c_arrayW, pointSize)

    test_points = (Point * 90000)()
    test_colors = (c_int * 90000)()

    _dll.AffichageSeparation(c_arrayW, test_points, test_colors)
    x_test_points = []
    y_test_points = []

    for Point in test_points:
        x_test_points.append(float(Point.x))
        y_test_points.append(float(Point.y))

    x_points = []
    y_points = []

    for Point in points:
        x_points.append(float(Point.x))
        y_points.append(float(Point.y))

    test_color_string = ["lightcyan" if i == 0 else "pink" for i in test_colors]
    classes_string = ["blue" if i == 0 else "red" for i in classes]

    # on affiche le tout

    plt.scatter(x_test_points, y_test_points, c=test_color_string)
    plt.scatter(x_points, y_points, c=classes_string)
    plt.show()

def RegressionLineaireSimple():
    class Point(Structure):
        _fields_ = [("x", c_float), ("y", c_float)]

    points = [Point(1, 2), Point(2, 3)]
    c_arrayPoint = (Point * len(points))(*points)

    pointSize = len(points)

    classes = [0, 0]
    c_arrayClasses = (c_int * len(classes))(*classes)

    # on Initialise les poids en random
    W = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

    c_arrayW = (c_float * len(W))(*W)
    res = [0, 0, 0]
    c_arrayRes = (c_float * len(res))(*res)

    # On lance l'entrainement linéaire
    W = _dll.EntrainementLineaire(c_arrayPoint, c_arrayClasses, c_arrayW, len(points), True)

    test_points = (Point * 90000)()
    test_colors = (c_int * 90000)()

    # On calcul les zones de couleur
    _dll.AffichageSeparation(c_arrayW, test_points, test_colors)

    x_test_points = []
    y_test_points = []

    for Point in test_points:
        x_test_points.append(float(Point.x))
        y_test_points.append(float(Point.y))

    x_points = []
    y_points = []

    for Point in points:
        x_points.append(float(Point.x))
        y_points.append(float(Point.y))

    test_color_string = ["lightcyan" if i == 0 else "pink" for i in test_colors]
    classes_string = ["blue" if i == 0 else "red" for i in classes]

    # on affiche le tout

    plt.scatter(x_test_points, y_test_points, c=test_color_string)
    plt.scatter(x_points, y_points, c=classes_string)
    plt.show()





def PMCXor():
    xor_points = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ], dtype=np.float32)

    rows, cols = xor_points.shape

    ptr = (POINTER(c_float) * rows)()
    for i in range(rows):
        ptr[i] = xor_points[i].ctypes.data_as(POINTER(c_float))

    color = [[0], [0], [1], [1]]

    array_1d = np.array(color).flatten()
    xor_colors = ['blue' if c == 1 else 'red' for c in array_1d]

    n_elements = len(color) * len(color[0])

    # Créer un tableau de flottants en C
    color_array = (c_float * n_elements)()

    color_array_ptr = (POINTER(c_float) * len(color))()
    for i, couleur in enumerate(color):
        # Créer un tableau de flottants pour chaque point
        color_array = (c_float * len(couleur))(*couleur)
        # Stocker ce tableau dans le tableau de pointeurs
        color_array_ptr[i] = color_array

    npl = [2, 2, 1]
    npl_ptr = (c_int * len(npl))(*npl)
    size = len(npl)

    pmc = PMC(npl_ptr, size)
    pmc.train(ptr, len(xor_points), color_array_ptr, True)

    test_points = []
    test_colors = []
    for row in range(0, 300):
        for col in range(0, 300):
            p = (col / 100 - 1, row / 100 - 1)
            p_list = list(p)
            p_ptr = (c_float * len(p_list))(*p_list)
            c = 'lightcyan' if pmc.predict(p_ptr, True)[0] >= 0 else 'pink'
            # print(pmc.predict(p_ptr,True)[0])
            test_points.append(p)
            test_colors.append(c)
    test_points = np.array(test_points)
    test_colors = np.array(test_colors)

    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
    plt.scatter(xor_points[:, 0], xor_points[:, 1], c=xor_colors)
    plt.show()


def StockW(W_ptr, nomStokage , nbValue):
    W_value = [W_ptr[i] for i in range(nbValue)]
    with open(nomStokage, "w") as f:
        f.writelines([str(x) + "\n" for x in W_value])

def LoadW(nomStokage):
    with open(nomStokage, "r") as f:
        lines = f.readlines()
        W_ptr = [float(line.strip()) for line in lines]
        return W_ptr

def EntrainementlineaireImage():
    #Entrainement d'un modele pour la lettre C
    nbImage, data, classes, nbValue = loadDataSet("Dataset/C/",1,"Dataset/N/",-1)
    W = [i for i in range(nbValue)]
    Xk = [i for i in range(nbValue)]

    flat_data = list(itertools.chain.from_iterable(data))
    data_ptr = ConvertirPointerFloat(flat_data)
    classe_ptr = ConvertirPointerFloat(classes)

    W_ptr = ConvertirPointerFloat(W)
    Xk_ptr = ConvertirPointerFloat(Xk)

    W_ptr = _dll.EntrainementLineaireImage(data_ptr, classe_ptr, W_ptr,Xk_ptr, nbImage, nbValue,28)
    StockW(W_ptr,"EntrainementLineaireC.txt",nbValue)

    # Entrainement d'un modele pour la lettre N

    #nbImageN, dataN, classesN, nbValueN = loadDataSet()
    #WN = [i for i in range(nbValueN)]
    #XkN = [i for i in range(nbValueN)]
    #flat_dataN = list(itertools.chain.from_iterable(dataN))
    #data_ptrN = ConvertirPointerFloat(flat_dataN)
    #classe_ptrN = ConvertirPointerFloat(classesN)
#
    #W_ptrN = ConvertirPointerFloat(WN)
    #Xk_ptrN = ConvertirPointerFloat(XkN)
#
    #W_ptrN = _dll.EntrainementLineaireImage(data_ptrN, classe_ptrN, W_ptrN,Xk_ptrN, nbImageN, nbValueN, 28)
    #StockW(W_ptrN, "EntrainementLineaireN.txt", nbValueN)

    # Entrainement d'un modele pour la lettre S

   #nbImageS, dataS, classesS, nbValueS = loadDataSet("Dataset/S/",3)
   #WS = [i for i in range(nbValueS)]
   #XkS = [i for i in range(nbValueS)]

   #flat_dataS = list(itertools.chain.from_iterable(dataS))
   #data_ptrS = ConvertirPointerFloat(flat_dataS)
   #classe_ptrS = ConvertirPointerFloat(classesS)

   #W_ptrS = ConvertirPointerFloat(WS)
   #Xk_ptrS = ConvertirPointerFloat(XkS)

   #W_ptrS = _dll.EntrainementLineaireImage(data_ptrS, classe_ptrS, W_ptrS, Xk_ptrS,nbImageS, nbValueS, 28)
   #StockW(W_ptrS, "EntrainementLineaireS.txt", nbValueS)

    print ("fin train")

def CheckImageLineaire():
    WC = LoadW("EntrainementLineaireC.txt")
    WC_ptr = ConvertirPointerFloat(WC)
    #WN = LoadW("EntrainementLineaireN.txt")
    #WN_ptr = ConvertirPointerFloat(WN)
    #WS = LoadW("EntrainementLineaireS.txt")
    #WS_ptr = ConvertirPointerFloat(WS)

    data , nbValue = loadImage("C (67).png")
    flat_data = list(itertools.chain.from_iterable(data))
    data_ptr = ConvertirPointerFloat(flat_data)

    resC = _dll.predictImage(data_ptr,WC_ptr,nbValue)
    print (resC)
    #resN = _dll.predictImage(data_ptr, WN_ptr, nbValue)
    #resS = _dll.predictImage(data_ptr, WS_ptr, nbValue)
    if(resC == 1):
        print("Lettre C")
    else :
        print("Lettre N")
    #if (resN ==1):
    #    print("Lettre N")
    #if (resS ==1):
    #    print("Lettre S")


EntrainementlineaireImage()
CheckImageLineaire()
# PMCXor()
#ClassificationLinearMultiple()
#ClassificationLinearSimple()
# RegressionLineaireSimple()
# ClassificationXOR()