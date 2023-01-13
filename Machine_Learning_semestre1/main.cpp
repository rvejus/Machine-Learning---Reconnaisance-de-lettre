
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

