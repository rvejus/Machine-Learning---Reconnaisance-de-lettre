#include "pch.h"
#include "fonction.h"



double* EntrainementLineaire(float** points, int* classes, double* W, int nbElem) {


	std::cout << points[0][0] << "-" << points[0][1] << " / " << points[1][0] << "-" << points[1][1] << " / " << points[2][0] << "-" << points[2][1] << endl;
	std::cout << classes[0] << " / " << classes[1] << " / " << classes[2] << endl;
	for (int i = 0; i < 100000; i++) {
		int k = rand() % nbElem;

		int yK = classes[k];

		float Xk[3] = { 1, points[k][0], points[k][1] };


		float gXk = 0;

		gXk += W[0] * Xk[0];
		gXk += W[1] * Xk[1];
		gXk += W[2] * Xk[2];


		if (gXk >= 0) {
			gXk = 1;
		}
		else {
			gXk = -1;
		}

		W[0] += 0.01 * (yK - gXk) * Xk[0];
		W[1] += 0.01 * (yK - gXk) * Xk[1];
		W[2] += 0.01 * (yK - gXk) * Xk[2];

	}
	std::cout << "W entrainer : " << W[0] << " / " << W[1] << " / " << W[2] << endl;
	std::cout << "Entrainement termine" << endl;
	return W;
}


void AffichageSeparation(double* W, float** test_points, int* test_classes) {

	float p[2];
	int i = 0;
	for (float row = 0; row < 300; row++) {
		for (float col = 0; col < 300; col++) {
			p[0] = col / 100;
			p[1] = row / 100;
			float couleur = (W[0] * 1) + (W[1] * p[0]) + (W[2] * p[1]);
			if (couleur >= 0) {
				couleur = 0;
			}
			else {
				couleur = 1;
			}
			test_points[i][0] = p[0];
			test_points[i][1] = p[1];
			test_classes[i] = couleur;
			i++;
		}
	}
}


void HelloWorld() {
	printf("HelloWorld");
}


void propagatePMC(PMC* instance ,float* inputs, bool is_classification) {
	instance->_propagate(inputs, is_classification);
}

float* predictPMC(PMC* instance, float* inputs, bool is_classification) {
	return instance->predict(inputs, is_classification);

}

void trainPMC(PMC* instance, float** X_train, int X_train_size, float** Y_train, bool is_classification, float alpha , int nb_iter) {
	for (int i = 0; i < X_train_size; i++) {
		cout << "Color : " << Y_train[i][0] << endl;
	}
	for (int i = 0; i < X_train_size; i++) {
		cout << "point : " <<"X : " << X_train[i][0] << " Y : " << X_train[i][1] << endl;
	}
	instance->train(X_train, X_train_size, Y_train, is_classification, alpha, nb_iter);
}

PMC* initPMC(int* npl, int nplSize) {
	for (int i = 0 ; i < nplSize; i++) {
		cout << "Npl : " << npl[i] << endl;
	}
	return new PMC(npl, nplSize);
}



 PMC::PMC(int* npl, int nplSize) {
	this->D = npl;
	this->D_size = nplSize;
	this->L = nplSize - 1;

	// Initialisation des W

	this->W = (float***)malloc(sizeof(float**) * nplSize);

	for (int l = 0; l < nplSize; l++) {
		this->W[l] = (float**)malloc(sizeof(float*) * (npl[l - 1] + 2));
		if (l == 0) {
			//std::cout << "l=0" << std::endl;
			continue;
		}
		for (int i = 0; i <= D[l - 1] + 1; i++) {

			//std::cout << "i" << std::endl;
			this->W[l][i] = (float*)malloc(sizeof(float) * (npl[l] + 2));
			for (int j = 1; j <= D[l] + 1; j++) {
				//std::cout << "j" << std::endl;

				if (j == 0) {
					this->W[l][i][j] = 0.0;
				}
				else {
					float random = (float)rand() / RAND_MAX * 2 - 1;
					this->W[l][i][j] = random;
				}
			}
		}
	}
	std::cout << "finished W" << std::endl;
	//Initialisation des X et des deltas
	//Init de la mémoire de la 1ere dimention pour les tableaux
	this->X = (float**)malloc(sizeof(float*) * nplSize);
	this->delta = (float**)malloc(sizeof(float*) * nplSize);
	for (int l = 0; l < nplSize; l++) {
		//std::cout << "l" << std::endl;
		X[l] = (float*)malloc(sizeof(float) * (npl[l] + 2));
		delta[l] = (float*)malloc(sizeof(float) * (npl[l] + 2));
		for (int j = 0; j <= D[l] + 1; j++) {
			//std::cout << "j" << std::endl;
			delta[l][j] = 0;
			if (j == 0) {
				X[l][j] = 1;
			}
			else {
				X[l][j] = 0;
			}
		}
	}

	std::cout << "finished X & delta" << std::endl;
	
}

PMC::~PMC() {
	// Desaloc de W
	free(this->W);

	free(this->X);

	free(this->delta);

	free(this->D);
}

void PMC::_propagate(float* inputs, bool is_classification) {
	for (int j = 1; j <= D[0] + 1; j++) {
		X[0][j] = inputs[j - 1];
	}
	//std::cout << "finished first for" << std::endl;


	for (int l = 1; l < D_size; l++) {
		//std::cout << "l" << std::endl;

		for (int j = 1; j < D[l] + 1; j++) {
			//std::cout << "j" << std::endl;
			int total = 0;
			for (int i = 0; i <= D[l - 1] + 1; i++) {
				//std::cout <<"l= " << l << std::endl;
				//std::cout << "i= " << i << std::endl;
				//std::cout << "j= " << j << std::endl;
				//std::cout << "W[l][0][0]= " << W[l][0][0] << std::endl;
				//std::cout << "W[l][i][0]= " << W[l][i][0] << std::endl;
				//std::cout << "D[l - 1] + 1= " << D[l - 1] + 1 << std::endl;
				//std::cout << "X[l - 1][i]= " << X[l - 1][i] << std::endl;
				total += this->W[l][i][j] * X[l - 1][i];
			}
			//std::cout << "finished i for" << std::endl;
			X[l][j] = total;
			if (is_classification || l < L) {
				//std::cout << "is_class" << std::endl;
				X[l][j] = std::tanh(total);
				//std::cout << "is_class passed" << std::endl;
			}
			//std::cout << "end j for" << std::endl;
		}
	}
	//std::cout << "finished propagate" << std::endl;
}

float* PMC::predict(float* inputs, bool is_classification) {
	this->_propagate(inputs, is_classification);
	//D[L]+1 pour la taille du dernier Layer et -1 pour retirer le biais 
	int output_size = D[L] + 1 - 1;
	float* output = (float*)malloc(output_size * sizeof(float));
	for (int i = 0; i < output_size; i++) {
		output[i] = X[L][i + 1];
	}
	return output;
}

void PMC::train(float** X_train,
	int X_train_size,
	float** Y_train,
	bool is_classification,
	float alpha = 0.01,
	int nb_iter = 10000)
{
	//std::cout << "train" << std::endl;
	for (int it = 0; it <= nb_iter; it++) {
		//std::cout << "it= " << it << std::endl;
		int k = rand() % (X_train_size);
		//std::cout << "k= " << k << std::endl;
		float* Xk = X_train[k];
		//std::cout << "Xk.size()= " << Xk.size() << std::endl;
		float* Yk = Y_train[k];
		//std::cout << "Yk.size()= " << Yk.size() << std::endl;

		this->_propagate(Xk, is_classification);

		//std::cout << "D len " << D.size() << std::endl;
		for (int j = 1; j < D[L] + 1; j++) {
			//std::cout << "j= " << j << std::endl;

			delta[L][j] = X[L][j] - Yk[j - 1];
			if (is_classification) {
				//std::cout << "is_class" << std::endl;
				delta[L][j] = delta[L][j] * (std::pow(1 - X[L][j], 2));
				//std::cout << "is_class finished" << std::endl;
			}
		}
		//std::cout << "first for finished" << std::endl;

		for (int l = D_size - 1; l >= 2; l--) {
			//std::cout << "l= "<<l << std::endl;
			for (int i = 1; i < D[l - 1] + 1; i++) {
				//std::cout << "i= " << i << std::endl;
				float total = 0;
				for (int j = 1; j < D[l - 1] + 1; j++) {
					//std::cout << "gonna calc total " << std::endl;
					//std::cout << "delta[l][0]= " << delta[l][0] << std::endl;
					//std::cout << "W[l][i][j]= " << this->W[l][i][j] <<std::endl;
					total += this->W[l][i][j] * delta[l][j];
					//std::cout << "total calculated " << i << std::endl;
				}
				delta[l - 1][i] = (std::pow(1 - X[l - 1][i], 2)) * total;
			}
		}
		//std::cout << "second for finished" << std::endl;
		for (int l = 1; l < D_size; l++) {
			//std::cout << "D.size()= " << D.size() << std::endl;
			//std::cout << "l= " << l << std::endl;
			for (int i = 0; i < D[l - 1] + 1; i++) {
				//std::cout << "i= " << i << std::endl;
				for (int j = 1; j < D[l] + 1; j++) {
					//std::cout << "j= " << j << std::endl;
					//std::cout << "delta[l][j]= " << delta[l][j] << std::endl;
					//std::cout << "W[l][i][j]= " << this->W[l][i][j] << std::endl;
					this->W[l][i][j] += -alpha * X[l - 1][i] * delta[l][j];
				}
			}
		}
	}
}

double CalculDistance(point A, point B) {
	return sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2));
}


point* initCentreRBF(point* points, point* centres, int nbCluster) {

	for (int i = 0; i < nbCluster; i++) {
		centres[i].x = rand() / RAND_MAX;
		centres[i].y = rand() / RAND_MAX;
	}
	return centres;
}

double gaussian(point A, point B, float sigma) {
	double dist = CalculDistance(A, B);
	return exp(-dist * dist / (2 * sigma * sigma));
}

int* predictionRBF(point* points, int nbPoints, point* centres, int nbCluster) {
	
	int* assignement = (int*)malloc(sizeof(int) * nbPoints);

	for (int r = 0; r < 1000; r++) {
		for (int i = 0; i < nbPoints; i++) {
			int indexCentreMin = 0;
			double distanceMin = CalculDistance(points[i], centres[0]);
			for (int j = 1; j < nbCluster; j++) {
				double distance = CalculDistance(points[i], centres[j]);
				if (distance < distanceMin) {
					indexCentreMin = j;
					distanceMin = distance;
				}
			}
			assignement[i] = indexCentreMin;
		}
		// on recalcul le centres des points 
		point* majCentre = (point*)malloc(sizeof(point) * nbCluster);
		int* nbPointDansCluster = (int*)malloc(sizeof(int) * nbCluster);

		for (int i = 0; i < nbPoints; i++) {
			int a = assignement[i];

			majCentre[a].x += points[a].x;
			majCentre[a].y += points[a].y;
			nbPointDansCluster[a]++;	
		}

		for (int i = 0; i < nbCluster; i++) {
			centres[i].x = majCentre[i].x / nbPointDansCluster[i];
			centres[i].y = majCentre[i].y / nbPointDansCluster[i];
		}
	}


	return assignement;

}