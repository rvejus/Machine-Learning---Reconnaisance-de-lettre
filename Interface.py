import random
from ctypes import *
import numpy as np

_dll = cdll.LoadLibrary("C:/Users/33783/Documents/5A3DJV/Machine Learning/Machine-Learning---Reconnaisance-de-lettre/dll_machineLearning.dll")


# Création des classes utiles
class Point(Structure):
    _fields_ = [("x", c_int), ("y", c_int)]



# récupération et definitions des fonctions utiles


_dll.EntrainementLineaire.argtypes = [POINTER(Point), POINTER(c_int), POINTER(c_float),c_int]
_dll.EntrainementLineaire.restype = None

#_dll.RandomFloat.argtypes = [c_float, c_float]
#_dll.RandomFloat.restype = c_float

#_dll.AffichageSeparation.argtypes = [VectorFloat]
#_dll.AffichageSeparation.restype = None




points = [Point(1, 1), Point(2, 1), Point(2, 2)]
c_arrayPoint = (Point * len(points))(*points)

pointSize = len(points)

classes = [1,1,-1]
c_arrayClasses = (c_int * len(classes))(*classes)

# on Initialise les poids en random
W = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]

c_arrayW = (c_float * len(W))(*W)

_dll.EntrainementLineaire(c_arrayPoint,c_arrayClasses,c_arrayW,pointSize)
#_dll.AffichageSeparation