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
extern "C"	FONCTION_API double* EntrainementLineaire(point *points, int *classes, double *W, int nbElem);
extern "C"	FONCTION_API void AffichageSeparation(double* W, point * test_points, int* test_classes);

extern "C"	FONCTION_API double CalculDistance(point A, point B);

extern "C"	FONCTION_API point * initCentreRBFNaif(point * points, point * centres, int nbCluster);
extern "C"	FONCTION_API int* predictionRBFNaif(point* points, int nbPoints, point* centres, int nbCluster);

extern "C"	FONCTION_API void propagatePMC(float*** W, float** X, int* D, int D_size, float* inputs, bool is_classification);
extern "C"  FONCTION_API float* predictPMC(float*** W, float** X, int* D, int D_size, int L, float* inputs, bool is_classification);
extern "C"  FONCTION_API void trainPMC(float** delta, float*** W, float** X, int* D, int D_size, int L, float** X_train, int X_train_size, float** Y_train, bool is_classification, float alpha = 0.01, int nb_iter = 10000);
extern "C"  FONCTION_API void initPMC(float*** W, float* D, int D_size, int L, float** X, float** delta, int* npl, int nplSize);
	class __declspec(dllexport) PMC
	{
	public:
		PMC(int* npl, int nplSize);
		~PMC();
		// Liste des poids
		float*** W;
		//le nombre de neurones appartenant à la couche l 
		int* D;
		int D_size;
		//identifiant de la derniere couche 
		int L;
		//Valeurs de sortie des Neurones
		float** X;
		float** delta;

		void _propagate(float* inputs, bool is_classification);
		float* predict(float* inputs, bool is_classification);
		void train(float** X_train, int X_train_size, float** Y_train, bool is_classification, float alpha, int nb_iter);
	};

