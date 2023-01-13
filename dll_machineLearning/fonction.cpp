#include "pch.h"
#include "fonction.h"

void add_to_vectorFloat(std::vector<float>* vec, float value) {
	vec->push_back(value);
}

void add_to_vectorPoint(std::vector<point>* vec, point value) {
	vec->push_back(value);
}

void add_to_vectorInt(std::vector<int>* vec, int value) {
	vec->push_back(value);
}

void EntrainementLineaire(point *points, int *classes, float *W, int nbElem) {

	for (int i = 0; i < 10000; i++) {
		int k = rand() % nbElem;
		int yK = classes[k];
		int Xk[3] = { 1, points[k].x, points[k].y };
		

		int gXk = 0;
		float resgXk = 0;
		resgXk = (W[0] * Xk[0]) + (W[1] * Xk[1]) + (W[2] * Xk[2]);
		if (resgXk >= 0) {
			gXk = 1;
		}
		else {
			gXk = -1;
		}
		float temp = 0.01 * (yK - gXk);
		/*float res = temp * Xk[0];
		res += temp * Xk[1];
		res += temp * Xk[2];
		W[k] += res; */
		
		for (int i = 0; i < 3; i++) {
			W[i] += temp * Xk[i];
		}

	}
	cout << W[0] << " / " << W[1] << " / " << W[2] << endl;
	cout << "Entrainement termine" << endl;
}


void AffichageSeparation(float *W) {

	std::vector<point> test_points = {};
	std::vector<int> test_classes = {};

	for (float row = 0; row < 300; row++) {
		for (float col = 0; col < 300; col++) {
			std::vector<point> p = { point(col / 100,row / 100) };
			float couleur = (W[0] * 1) + (W[1] * p[0].x) + (W[2] * p[0].y);
			if (couleur >= 0) {
				couleur = 1;
			}
			else {
				couleur = 2;
			}
			test_points.push_back(p[0]);
			test_classes.push_back(couleur);
		}
	}
}




void HelloWorld() {
	printf("HelloWorld");
}