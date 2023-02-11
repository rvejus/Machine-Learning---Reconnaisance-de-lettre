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
extern "C"	FONCTION_API double* EntrainementLineaire(float **points, int *classes, double *W, int nbElem);
extern "C"	FONCTION_API void AffichageSeparation(double* W, float** test_points, int* test_classes);

extern "C"	FONCTION_API double CalculDistance(point A, point B);

extern "C"	FONCTION_API point * initCentreRBF(point * points, point * centres, int nbCluster);
extern "C"	FONCTION_API int* predictionRBF(point* points, int nbPoints, point* centres, int nbCluster);




	class __declspec(dllexport) PMC
	{
	public:
		PMC(int* npl, int nplSize);
		~PMC();

		static PMC* createInstance(int* npl, int nplsize) {
			return new(std::nothrow) PMC(npl, nplsize);
		}
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

	extern "C"	FONCTION_API void propagatePMC(PMC * instance, float* inputs, bool is_classification);
	extern "C"  FONCTION_API float* predictPMC(PMC * instance, float* inputs, bool is_classification);
	extern "C"  FONCTION_API void trainPMC(PMC * instance, float** X_train, int X_train_size, float** Y_train, bool is_classification, float alpha = 0.01, int nb_iter = 10000);
	extern "C"  FONCTION_API PMC * initPMC(int* npl, int nplSize);
	