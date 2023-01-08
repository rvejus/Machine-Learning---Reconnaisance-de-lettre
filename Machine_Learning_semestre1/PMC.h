#pragma once
#include <iostream>
#include <list>
#include <vector>
using namespace std;

class PMC
{
public :
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

