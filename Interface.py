import random
from ctypes import *

_dll = cdll.LoadLibrary(
    "C:/Users/33783/Documents/5A3DJV/Machine Learning/Machine-Learning---Reconnaisance-de-lettre/Machine_Learning_semestre1.dll")


# Création des classes utiles
class Point(Structure):
    _fields_ = [("x", c_int), ("y", c_int)]


class VectorInt(Structure):
    _fields_ = [("data", POINTER(c_int)),
                ("size", c_size_t),
                ("capacity", c_size_t)]


class VectorFloat(Structure):
    _fields_ = [("data", POINTER(c_float)),
                ("size", c_size_t),
                ("capacity", c_size_t)]


class VectorPoint(Structure):
    _fields_ = [("data", POINTER(Point)),
                ("size", c_size_t),
                ("capacity", c_size_t)]


# récupération et definitions des fonctions utiles

_dll.EntrainementLineaire.argtypes = [VectorPoint, VectorInt, VectorFloat]
_dll.EntrainementLineaire.restype = None

_dll.RandomFloat.argtypes = [c_float, c_float]
_dll.RandomFloat.restype = c_float

_dll.AffichageSeparation.argtypes = [VectorFloat]
_dll.AffichageSeparation.restype = None

_dll.add_to_vector.argtypes = [POINTER(VectorFloat), c_float]
_dll.add_to_vector.restype = None



points = VectorPoint(Point(1, 1), Point(2, 1), Point(2, 2))
classes = VectorInt(1,1,-1)

# on Initialise les poids en random
W = VectorFloat(random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1))

_dll.EntrainementLineaire(points,classes,W)
_dll.AffichageSeparation