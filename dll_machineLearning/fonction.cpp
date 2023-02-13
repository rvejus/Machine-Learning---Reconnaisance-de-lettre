#include "pch.h"
#include "fonction.h"
#include <iostream>



double* EntrainementLineaire(point* points, int* classes, double* W, int nbElem) {
	srand(time(NULL));
	for (int i = 0; i < 100000; i++) {
		int k = rand() % nbElem;

		int yK = classes[k];

		float Xk[3] = { 1, points[k].x, points[k].y };


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


void AffichageSeparation(double* W, point* test_points, int* test_classes) {

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
			test_points[i].x = p.x;
			test_points[i].y = p.y;
			test_classes[i] = couleur;
			i++;
		}
	}
}


void propagatePMC(PMC* instance ,float* inputs, bool is_classification) {
	instance->_propagate(inputs, is_classification);
}

float* predictPMC(PMC* instance, float* inputs, bool is_classification) {
	return instance->predict(inputs, is_classification);

}

void trainPMC(PMC* instance, float** X_train, int X_train_size, float** Y_train, bool is_classification, float alpha , int nb_iter) {
	for (int i = 0; i < X_train_size; i++) {
		std::cout << "Color : " << Y_train[i][0] << endl;
	}
	for (int i = 0; i < X_train_size; i++) {
		std::cout << "point : " <<"X : " << X_train[i][0] << " Y : " << X_train[i][1] << endl;
	}
	instance->train(X_train, X_train_size, Y_train, is_classification, alpha, nb_iter);
}

PMC* initPMC(int* npl, int nplSize) {
	for (int i = 0 ; i < nplSize; i++) {
		std::cout << "Npl : " << npl[i] << endl;
	}
	return new PMC(npl, nplSize);
}



 PMC::PMC(int* npl, int nplSize) {
	this->D = npl;
	this->D_size = nplSize;
	this->L = nplSize - 1;
	srand(time(NULL));
	// Initialisation des W

	this->W = (float***)malloc(sizeof(float**) * nplSize);

	for (int l = 0; l < nplSize; l++) {
		this->W[l] = (float**)malloc(sizeof(float*) * (npl[l - 1] +1));
		if (l == 0) {
			//std::cout << "l=0" << std::endl;
			continue;
		}
		for (int i = 0; i < this->D[l - 1] + 1; i++) {

			//std::cout << "i" << std::endl;
			this->W[l][i] = (float*)malloc(sizeof(float) * (npl[l]+1));
			for (int j = 1; j < this->D[l] + 1; j++) {
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
		this->X[l] = (float*)malloc(sizeof(float) * (npl[l] + 2));
		this->delta[l] = (float*)malloc(sizeof(float) * (npl[l] + 2));
		for (int j = 0; j < this->D[l] + 1; j++) {
			//std::cout << "j" << std::endl;
			this->delta[l][j] = 0.0;
			if (j == 0) {
				this->X[l][j] = 1.0;
			}
			else {
				this->X[l][j] = 0.0;
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
	for (int j = 1; j < this->D[0] + 1; j++) {
		this->X[0][j] = inputs[j - 1];
	}
	//std::cout << "finished first for" << std::endl;


	for (int l = 1; l < this->D_size; l++) {
		//std::cout << "l" << std::endl;

		for (int j = 1; j < this->D[l] + 1; j++) {
			//std::cout << "j" << std::endl;
			int total = 0;
			for (int i = 0; i < this->D[l - 1] + 1; i++) {
				//std::cout <<"l= " << l << std::endl;
				//std::cout << "i= " << i << std::endl;
				//std::cout << "j= " << j << std::endl;
				//std::cout << "W[l][0][0]= " << W[l][0][0] << std::endl;
				//std::cout << "W[l][i][0]= " << W[l][i][0] << std::endl;
				//std::cout << "D[l - 1] + 1= " << D[l - 1] + 1 << std::endl;
				//std::cout << "X[l - 1][i]= " << X[l - 1][i] << std::endl;
				total += this->W[l][i][j] * this->X[l - 1][i];
			}
			//std::cout << "finished i for" << std::endl;
			//std::cout << "finished i for" << std::endl;
			this->X[l][j] = total;
			if (is_classification==true || l < this->L) {
				//std::cout << "is_class" << std::endl;
				this->X[l][j] = std::tanh(total);
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
	int output_size = sizeof(this->X[this->L]) / sizeof(this->X[this->L][0]);
	float* output = (float*)malloc(output_size * sizeof(float));
	for (int i = 0; i < output_size; i++) {
		output[i] = this->X[this->L][i + 1];
	}
	std::cout<<"from c++ " << output[0] << std::endl;
	return output;
}

void PMC::train(float** X_train,
	int X_train_size,
	float** Y_train,
	bool is_classification,
	float alpha,
	int nb_iter)
{
	srand(time(NULL));
	//std::cout << "train" << std::endl;
	for (int it = 0; it < nb_iter; it++) {
		//std::cout << "it= " << it << std::endl;
		int k = rand() % (X_train_size);
		//std::cout << "k= " << k << std::endl;
		float* Xk = X_train[k];
		//std::cout << "Xk.size()= " << Xk.size() << std::endl;
		float* Yk = Y_train[k];
		//std::cout << "Yk.size()= " << Yk.size() << std::endl;

		this->_propagate(Xk, is_classification);

		//std::cout << "D len " << D.size() << std::endl;
		for (int j = 1; j < this->D[this->L] + 1; j++) {
			//std::cout << "j= " << j << std::endl;

			this->delta[this->L][j] = this->X[this->L][j] - Yk[j - 1];
			if (is_classification==true) {
				//std::cout << "is_class" << std::endl;
				this->delta[this->L][j] = this->delta[this->L][j] * (1 - std::pow(this->X[this->L][j], 2));
				//std::cout << "is_class finished" << std::endl;
			}
		}
		//std::cout << "first for finished" << std::endl;

		for (int l = this->D_size - 1; l >= 2; l--) {
			//std::cout << "l= "<<l << std::endl;
			for (int i = 1; i < this->D[l - 1] + 1; i++) {
				//std::cout << "i= " << i << std::endl;
				float total = 0.0;
				for (int j = 1; j < this->D[l] + 1; j++) {
					//std::cout << "gonna calc total " << std::endl;
					//std::cout << "delta[l][0]= " << delta[l][0] << std::endl;
					//std::cout << "W[l][i][j]= " << this->W[l][i][j] <<std::endl;
					total += this->W[l][i][j] * this->delta[l][j];
					//std::cout << "total calculated " << i << std::endl;
				}
				this->delta[l - 1][i] = (1 - std::pow(this->X[l - 1][i], 2)) * total;
			}
		}
		//std::cout << "second for finished" << std::endl;
		for (int l = 1; l < this->D_size; l++) {
			//std::cout << "D.size()= " << D.size() << std::endl;
			//std::cout << "l= " << l << std::endl;
			for (int i = 0; i < this->D[l - 1] + 1; i++) {
				//std::cout << "i= " << i << std::endl;
				for (int j = 1; j < this->D[l] + 1; j++) {
					//std::cout << "j= " << j << std::endl;
					//std::cout << "delta[l][j]= " << delta[l][j] << std::endl;
					//std::cout << "W[l][i][j]= " << this->W[l][i][j] << std::endl;
					this->W[l][i][j] += -alpha * this->X[l - 1][i] * this->delta[l][j];
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

float* EntrainementLineaireImage(float* data, float* classe, float* W,float* Xk,int nbImages, int nbValue, int dimension) {
	srand(time(NULL));

	// On randomise nos poids initial
	for (int i = 0; i < nbValue + 1; i++) {
		W[i] = ((float)rand() / RAND_MAX*2)-1;
	} 
	
	/*for (int n = 0; n < 6; n++) {
		cout << data[n] << endl;
	} */
	for (int i = 0; i < 10000; i++) {
		int k = rand() % nbImages;
		
		float Yk = classe[k];
		
		Xk[0] = 1.0;
		for (int j = 0; j < nbValue; j++) {
			Xk[j + 1] = data[k * dimension + j];
			//std::cout << "XK :" << Xk[j+1] << endl;
		}
		float dotProduct = 0;
		float GXk ;
		for (int j = 0; j < nbValue; j++) {
			dotProduct = (W[j] * Xk[j])+ dotProduct;
		}
		if (dotProduct >= 0) {
			GXk = 1;
		}
		else { 
			GXk = -1;
		}
		for (int j = 0; j < nbValue +1 ; j++) {
			W[j] = 0.001 * (Yk - GXk) * Xk[j] + W[j];
		}
		for (int j = 0; j < nbValue + 1; j++) {
			Xk[j] = 0;
		}
	}
	
	/*for (int i = 0; i < nbValue; i++) {
		cout << "W[" << i << "] : " << W[i] << endl;
	} */
	return W;
}


int predictImage(float* data , float* W, int nbValue) {
	float dotProduct = 0;
	for (int i = 0; i < nbValue; i++) {
		dotProduct += (W[i] * data[i]);
	}
	if (dotProduct >= 0) {
		return 1;
	}
	else {
		return -1;
	}
}

void free_float(float* ptr) {
	free(ptr);
}