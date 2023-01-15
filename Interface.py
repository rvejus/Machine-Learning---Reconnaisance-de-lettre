import random
from ctypes import *
import matplotlib.pyplot as plt
import numpy as np

# import dll
_dll = cdll.LoadLibrary(
    "C:/Users/33783/Documents/5A3DJV/Machine Learning/Machine-Learning---Reconnaisance-de-lettre/dll_machineLearning.dll")

class Point(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]

# récupération et definitions des fonctions utiles

_dll.EntrainementLineaire.argtypes = [POINTER(Point), POINTER(c_int), POINTER(c_float),c_int]
_dll.EntrainementLineaire.restype = POINTER(c_float)

_dll.AffichageSeparation.argtypes = [POINTER(c_float), POINTER(Point), POINTER(c_int)]
_dll.AffichageSeparation.restype = None


#def color_grid(W, width, height):
#    test_points = []
#    test_colors = []
#    for row in range(height):
#        for col in range(width):
#            p = (col / 100, row / 100)
#            tmp = W[0] + W[1] * p[0] + W[2] * p[1]
#            c = 'lightcyan' if tmp >= 0 else 'pink'
#            test_points.append(p)
#            test_colors.append(c)
#    return test_points, test_colors


points = [Point(1, 1), Point(2, 1), Point(2, 2)]
c_arrayPoint = (Point * len(points))(*points)

pointSize = len(points)

classes = [0,0,-1]
c_arrayClasses = (c_int * len(classes))(*classes)

# on Initialise les poids en random
W = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]

c_arrayW = (c_float * len(W))(*W)
res = [0,0,0]
c_arrayRes = (c_float * len(res))(*res)

# On lance l'entrainement linéaire
W = _dll.EntrainementLineaire(c_arrayPoint, c_arrayClasses, c_arrayW, len(points))

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

# X = []
# for i in range(50):
#    X.append(Point(random.random()*0.9+1, random.random()*0.9+1))
# for i in range(50):
#    X.append(Point(random.random()*0.9+2, random.random()*0.9+2))
#
## Génération du vecteur de cibles
# Y = []
# for i in range(50):
#    Y.append(1)
# for i in range(50):
#    Y.append(-1)
#
# x_points = X[0:50]
# x_points1 = X[50:100]
#
#
#
# c_arrayPoint = (Point * len(X))(*X)
# pointSize = len(X)
# c_arrayClasses = (c_int * len(Y))(*Y)
#
# W = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
# c_arrayW = (c_float * len(W))(*W)
#
# _dll.EntrainementLineaire(c_arrayPoint,c_arrayClasses,c_arrayW,pointSize)
#
# test_points = (Point * 90000)()
# test_colors = (c_int * 90000)()
#
# _dll.AffichageSeparation(c_arrayW,test_points,test_colors)
#
# xx = []
# xy = []
#
# yx = []
# yy = []
#
# for Point in x_points :
#    xx.append(float(Point.x))
#    xy.append(float(Point.y))
#
# for Point in x_points1 :
#    yx.append(float(Point.x))
#    yy.append(float(Point.y))
# plt.scatter(xx, xy, color='blue')
# plt.scatter(yx, yy, color='red')
#
#
#
# x_test_points = []
# y_test_points = []
#
# for Point in test_points :
#    x_test_points.append(float(Point.x))
#    y_test_points.append(float(Point.y))
#
# x_points = []
# y_points = []
#
# for Point in X :
#    x_points.append(float(Point.x))
#    y_points.append(float(Point.y))
#
# test_color_string = ["lightcyan" if i == 0 else "pink" for i in test_colors]
# classes_string = ["blue" if i == 1 else "red" for i in Y]
#
##on affiche le tout
#
# plt.scatter(x_test_points, y_test_points, c= test_color_string)
# plt.scatter(x_points, y_points, c= classes_string)
# plt.show()


#	//std::vector<float> testbis = std::vector<float>(2, 1);
#	//PMC model = PMC::PMC(test);
#	//std::vector<float> testris = model.predict(testbis, true);
#	//model.train(model.X, model.X, true, 0.01, 10000);

# npl = (c_int * 3)(2, 3, 1)
# nplSize = len(npl)
#
#
# _dll.PMC.argtypes = [POINTER(c_int), c_int]
# _dll.PMC.restype = c_void_p

# pmc_instance = pmc_constructor(npl,nplSize)
