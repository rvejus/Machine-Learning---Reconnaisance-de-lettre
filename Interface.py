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

_dll.EntrainementLineaireImage.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int]
_dll.EntrainementLineaireImage.restype = POINTER(c_double)


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
    test_points = []
    test_colors = []
    for row in range(height):
        for col in range(width):
            p = (col / 100, row / 100)
            tmp = W[0] + W[1] * p[0] + W[2] * p[1]
            c = 'lightcyan' if tmp >= 0 else 'pink'
            test_points.append(p)
            test_colors.append(c)
    return test_points, test_colors


# Fonction load des images du dataset d'entrainement
def loadDataSet():
    C = "Dataset/C/"
    N = "Dataset/N/"
    S = "Dataset/S/"

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
        classes.append(1)
        nbImage = nbImage + 1

    for filename in os.listdir(N):
        img = cv2.imread(os.path.join(N, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dims = (28, 28)
        resized = cv2.resize(gray, dims, interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        vectorized = normalized.reshape(1, 28 * 28).astype(c_double)
        data.append(vectorized.tolist()[0])
        classes.append(2)
        nbImage = nbImage + 1

    for filename in os.listdir(S):
        img = cv2.imread(os.path.join(S, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dims = (28, 28)
        resized = cv2.resize(gray, dims, interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        vectorized = normalized.reshape(1, 28 * 28).astype(c_double)
        data.append(vectorized.tolist()[0])
        classes.append(3)
        nbImage = nbImage + 1

    nbValue = dims[0] * dims[1]

    return nbImage, data, classes, nbValue


######################################################
#              LINEAR SIMPLE                         #
######################################################
def ClassificationLinearSimple():
    class Point(Structure):
        _fields_ = [("x", c_float), ("y", c_float)]

    _dll.EntrainementLineaire.argtypes = [POINTER(POINTER(c_int)), POINTER(c_int), POINTER(c_double), c_int]
    _dll.EntrainementLineaire.restype = POINTER(c_double)

    _dll.AffichageSeparation.argtypes = [POINTER(c_double), POINTER(POINTER(c_int)), POINTER(c_int)]
    _dll.AffichageSeparation.restype = None

    points = [Point(1, 1), Point(2, 3), Point(3, 3)]
    c_arrayPoint = (Point * len(points))(*points)

    pointSize = len(points)

    classes = [1, 0, 0]
    c_arrayClasses = (c_int * len(classes))(*classes)

    # on Initialise les poids en random
    W = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

    c_arrayW = (c_double * len(W))(*W)

    # On lance l'entrainement linéaire
    W = _dll.EntrainementLineaire(c_arrayPoint, c_arrayClasses, c_arrayW, pointSize)

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


######################################################
#            LINEAR MULTIPLE                         #
######################################################
def ClassificationLinearMultiple():
    class Point(Structure):
        _fields_ = [("x", c_float), ("y", c_float)]

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


###########################################
#                 CROSS                   #
###########################################
# points = []
# for i in range(500):
#   points.append(Point(random.random() * 2.0 - 1.0, random.random() * 2.0 - 1.0))
#
# c_arrayPoint = (Point * len(points))(*points)
#
# pointSize = len(points)
#
# classes = [1 if abs(p.x) <= 0.3 or abs(p.y) <= 0.3 else -1 for p in points]
# c_arrayClasses = (c_int * len(classes))(*classes)
#
# W = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
# c_arrayW = (c_float * len(W))(*W)
#
# _dll.EntrainementLineaire(c_arrayPoint,c_arrayClasses,c_arrayW,pointSize)
#
#
# X_pos = [(i, p) for i, p in enumerate(points) if classes[i] == 1]
# X_pos_x = [p[1][0] for p in X_pos]
# X_pos_y = [p[1][1] for p in X_pos]
# plt.scatter(X_pos_x, X_pos_y, color='blue')
#
# X_neg = [(i, p) for i, p in enumerate(points) if classes[i] == -1]
# X_neg_x = [p[1][0] for p in X_neg]
# X_neg_y = [p[1][1] for p in X_neg]
# plt.scatter(X_neg_x, X_neg_y, color='red')
# plt.show()
# plt.clf()


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


# ClassificationLinearMultiple()
# ClassificationLinearSimple()
# RegressionLineaireSimple()
# ClassificationXOR()


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

def ConvertirPointerDouble(flat_data):
    my_array = (c_double * len(flat_data))
    c_array = my_array(*flat_data)
    # Convertir le tableau ctypes en un pointeur
    pointer_to_ctypes_array = pointer(c_array)
    double_pointer = cast(pointer_to_ctypes_array, POINTER(c_double))
    return double_pointer

def EntrainementlineaireImage():
    nbImage, data, classes, nbValue = loadDataSet()
    W = [i for i in range(nbImage)]

    flat_data = list(itertools.chain.from_iterable(data))
    data_ptr = ConvertirPointerDouble(flat_data)
    classe_ptr = ConvertirPointerDouble(classes)

    W_ptr = ConvertirPointerDouble(W)

    _dll.EntrainementLineaireImage(data_ptr, classe_ptr, W_ptr, nbImage, nbValue)


EntrainementlineaireImage()
# PMCXor()
