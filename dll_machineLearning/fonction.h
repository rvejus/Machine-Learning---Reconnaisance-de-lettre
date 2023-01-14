#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
using namespace std;


#ifdef FONCTION_EXPORTS
#define FONCTION_API __declspec(dllexport)
#else
#define FONCTION_API __declspec(dllimport)
#endif

extern "C"	FONCTION_API struct point
{
	float x;
	float y;
	point(float a, float b) {
		x = a;
		y = b;
	}
};


extern "C"	FONCTION_API void HelloWorld();
extern "C"	FONCTION_API void EntrainementLineaire(point *points, int *classes, float *W, int nbElem);
extern "C"	FONCTION_API float RandomFloat(float min, float max);
extern "C"	FONCTION_API void AffichageSeparation(float* W, point * test_points, int* test_classes);


class __declspec(dllexport) PMC
{
public:
	PMC(int *npl,int nplSize);
	~PMC();
	// Liste des poids
	float*** W;
	//le nombre de neurones appartenant à la couche l 
	int *D;

	int D_size;
	//identifiant de la derniere couche 
	int L;
	//Valeurs de sortie des Neurones
	float** X;
	float** delta;

	void _propagate(float* inputs, bool is_classification);
	float *predict(float* inputs, bool is_classification);
	void train(float** X_train, int X_train_size, float** Y_train, int Y_train_size, bool is_classification, float alpha, int nb_iter);
};