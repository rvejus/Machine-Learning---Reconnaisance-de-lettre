#pragma once
#include <iostream>
#include <list>
using namespace std;

class PMC
{
public : 
	// Liste des poids
	list<list<float>> W;
	//le nombre de neurones appartenant à la couche l 
	list<int> D; 

	//identifiant de la derniere couche 
	int L;
	//Valeurs de sortie des Neurones
	list<float> X{};
	list<float> delta{};

	PMC(list<int> npl);
};

