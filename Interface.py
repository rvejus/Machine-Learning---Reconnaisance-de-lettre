import random
from ctypes import *
import matplotlib.pyplot as plt

#import dll
_dll = cdll.LoadLibrary("C:/Users/33783/Documents/5A3DJV/Machine Learning/Machine-Learning---Reconnaisance-de-lettre/dll_machineLearning.dll")


# Création des classes utiles
class Point(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]

# récupération et definitions des fonctions utiles

_dll.EntrainementLineaire.argtypes = [POINTER(Point), POINTER(c_int), POINTER(c_float),c_int]
_dll.EntrainementLineaire.restype = None

_dll.AffichageSeparation.argtypes = [POINTER(c_float), POINTER(Point), POINTER(c_int)]
_dll.AffichageSeparation.restype = None




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


_dll.EntrainementLineaire(c_arrayPoint,c_arrayClasses,c_arrayW,pointSize)

test_points = (Point * 90000)()
test_colors = (c_int * 90000)()

_dll.AffichageSeparation(c_arrayW,test_points,test_colors)


x_test_points = []
y_test_points = []

for Point in test_points :
    x_test_points.append(float(Point.x))
    y_test_points.append(float(Point.y))

x_points = []
y_points = []

for Point in points :
    x_points.append(float(Point.x))
    y_points.append(float(Point.y))

test_color_string = ["lightcyan" if i == 0 else "pink" for i in test_colors]
classes_string = ["green" if i == 0 else "red" for i in classes]


size = min(len(x_test_points), len(y_test_points), len(test_color_string))

plt.scatter(x_test_points, y_test_points, c= test_color_string)
plt.scatter(x_points, y_points, c= classes_string)
plt.show()