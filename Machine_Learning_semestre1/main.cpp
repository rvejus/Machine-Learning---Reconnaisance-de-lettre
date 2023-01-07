
#include "PMC.h"
#include <iostream>
#include <vector>


struct point
{
	float x;
	float y;
	point(float a, float b) {
		x = a;
		y = b;
	}
};

void EntrainementLineaire(vector<point> points, vector<int> classes, vector<float> W);

void main() {

	//Creation dataset de point 
	point p1(1, 1);
	point p2(2, 1);
	point p3(2, 2);

	std::vector<point> points = { p1,p2,p3 };

	std::vector<int> classes = { 1,1,-1 };

	vector<float> W;
	
	//EntrainementLineaire(points,classes, W);




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
		//W[k] += 0, 01 * (yK - gXk) * Xk;

	}
}