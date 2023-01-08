#pragma once
#include <iostream>
#include <list>
#include <vector>
using namespace std;

class PMC
{
public : 
	// Liste des poids
	vector<vector<float>> W;
	//le nombre de neurones appartenant à la couche l 
	vector<int> D; 

	//identifiant de la derniere couche 
	int L;
	//Valeurs de sortie des Neurones
	vector<vector<float>> X{};
	vector<vector<float>> delta{};

	PMC(vector<int> npl);
};

