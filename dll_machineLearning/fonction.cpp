#include "pch.h"
#include "fonction.h"



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
			gXk = 0;
		}
		float tmp = 0.01 * (yK - gXk);
		cout << "temp : " << tmp << endl;
	/*float res = temp * Xk[0];
		res += temp * Xk[1];
		res += temp * Xk[2];
		W[k] += res; */
		
		for (int i = 0; i < 3; i++) {
			W[i] += tmp * Xk[i];
		} 

	}
	cout << W[0] << " / " << W[1] << " / " << W[2] << endl;
	cout << "Entrainement termine" << endl;
	
}


void AffichageSeparation(float *W , point *test_points , int* test_classes) {

	
	point p(0,0);
	int i = 0;
	for (float row = 0; row < 300; row++) {
		for (float col = 0; col < 300; col++) {
			p.x = col / 100;
			p.y = row / 100;
			float couleur = (W[0] * 1) + (W[1] * p.x) + (W[2] * p.y);
			if (couleur >= 0) {
				couleur = 0;
			}
			else {
				couleur = 1;
			}
			test_points[i] = p;
			test_classes[i] = couleur;
			i++;
		}
	}
}




void HelloWorld() {
	printf("HelloWorld");
}