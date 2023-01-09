
#include "PMC.h"
#include <iostream>
#include <vector>





void main() {

	//Creation dataset de point 
	point p1(1, 1);
	point p2(2, 1);
	point p3(2, 2);

	std::vector<point> points = { p1,p2,p3 };

	std::vector<int> classes = { 1,1,-1 };

	vector<float> W;
	for (int i = 0; i < 3; i++) {
		float r = RandomFloat(-1, 1);
		W.push_back(r);
	}
	// On entraine notre modele 
	//EntrainementLineaire(points,classes, W);

	//AffichageSeparation(W);
	//std::vector<int> test = std::vector<int>(2, 1);
	//std::vector<float> testbis = std::vector<float>(2, 1);
	//PMC model = PMC::PMC(test);
	//std::vector<float> testris = model.predict(testbis, true);
	//model.train(model.X, model.X, true, 0.01, 10000);
}

float RandomFloat(float min, float max)
{

	float random = ((float)rand()) / (float)RAND_MAX;

	float range = max - min;
	return (random * range) + min;
}

void EntrainementLineaire(vector<point> points, vector<int> classes, vector<float> W) {

	for (int i = 0; i < 10000; i++) {
		int k = rand() % points.size();
		int yK = classes[k];
		vector<int> Xk;
		Xk.push_back(1);
		Xk.push_back(points[k].x);
		Xk.push_back(points[k].y);

		int gXk = 0;
		float resgXk = 0;
		resgXk = (W[0] * Xk[0]) + (W[1] * Xk[1]) + (W[2] * Xk[2]); 
		if (resgXk>=0) {
			gXk = 1;
		}
		else {
			gXk = -1;
		} 
		float temp = 0.01 * (yK - gXk);
		float res = temp * Xk[0];
		res += temp * Xk[1];
		res += temp * Xk[2];
		W[k] += res;
		//W[k] += 0.01 * (yK - gXk) * Xk;

	}
	cout << "Entrainement termine" << endl;
}

void AffichageSeparation(vector<float> W) {

	std::vector<point> test_points = {};
	std::vector<int> test_classes = {};

	for (float row = 0; row < 300; row++) {
		for (float col = 0; col < 300; col++) {
			std::vector<point> p = {point(col/100,row/100)};
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

void add_to_vector(std::vector<int>* vec, float value) {
	vec->push_back(value);
}