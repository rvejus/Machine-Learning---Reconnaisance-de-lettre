#pragma once
#include <iostream>
#include <list>
#include <vector>
using namespace std;


#ifdef __cplusplus
extern "C" {
#endif

	class __declspec(dllexport) PMC
	{
	public:
		PMC(vector<int> npl);
		~PMC();
		// Liste des poids
		vector<vector<vector<float>>> W;
		//le nombre de neurones appartenant à la couche l 
		vector<int> D;
		//identifiant de la derniere couche 
		int L;
		//Valeurs de sortie des Neurones
		vector<vector<float>> X{};
		vector<vector<float>> delta{};

		void _propagate(vector<float> inputs, bool is_classification);
		vector<float> predict(vector<float> inputs, bool is_classification);
		void train(vector<vector<float>> X_train, vector<vector<float>> Y_train, bool is_classification, float alpha, int nb_iter);
	};

	struct point
	{
		float x;
		float y;
		point(float a, float b) {
			x = a;
			y = b;
		}
	};


	extern "C"	__declspec(dllexport) void EntrainementLineaire(vector<point> points, vector<int> classes, vector<float> W);
	extern "C"	__declspec(dllexport) float RandomFloat(float min, float max);
	extern "C"	__declspec(dllexport) void AffichageSeparation(vector<float> W);
	extern "C"	__declspec(dllexport) void add_to_vector(vector<int>*vec, float value);
		


#ifdef __cplusplus
}
#endif



