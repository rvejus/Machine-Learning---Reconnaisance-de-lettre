#include "PMC.h"
#include <cstdlib>

PMC::PMC(std::vector<int> npl) {
	this->D = npl;
	this->L = npl.size() - 1;

	// Initialisation des W

	//std::vector<std::vector<std::vector<float>>> W;

	for (int l = 0; l<D.size(); l++) {
		this->W.emplace_back(std::vector<std::vector<float>>());
		if (l == 0) {
			//std::cout << "l=0" << std::endl;
			continue;
		}
		for (int i = 0; i<=D[l - 1] + 1; i++) {

			//std::cout << "i" << std::endl;
			this->W[l].emplace_back(std::vector<float>());
			for (int j = 1; j<=D[l] + 1; j++) {
				//std::cout << "j" << std::endl;

				if (j == 0) {
					this->W[l][i].emplace_back(0);
				}
				else {
					float random = (float)rand() / RAND_MAX * 2 - 1;
					this->W[l][i].emplace_back(random);
				}
			}
		}
	}
	//std::cout << "finished W" << std::endl;
	//Initialisation des X et des deltas
	for (int l = 0; l<D.size(); l++) {
		//std::cout << "l" << std::endl;
		X.emplace_back(std::vector<float>());
		delta.emplace_back(std::vector<float>());
		for (int j = 0; j<=D[l] + 1; j++) {
			//std::cout << "j" << std::endl;
			delta[l].emplace_back(0);
			if (j == 0) {
				X[l].emplace_back(1);
			}
			else {
				X[l].emplace_back(0);
			}
		}
	}

	//std::cout << "finished X & delta" << std::endl;
}

PMC::~PMC() {
	// Desaloc de W

	/*for (int i = 0; i < this->W.size(); i++) {
		for (int j = 0; i < this->W[i].size(); i++) {
			this->W[i][j].clear();
		}
		this->W[i].clear();
	}*/
	this->W.clear();

	/*for (int i = 0; i < this->X.size(); i++) {
		this->X[i].clear();
	}*/
	this->X.clear();

	/*for (int i = 0; i < this->delta.size(); i++) {
		this->delta[i].clear();
	}*/
	this->delta.clear();

	this->D.clear();
}

void PMC::_propagate(std::vector<float> inputs, bool is_classification) {
	for (int j = 1; j<=D[0] + 1;j++) {
		X[0][j] = inputs[j - 1];
	}
	//std::cout << "finished first for" << std::endl;


	for (int l = 1; l<D.size(); l++) {
		//std::cout << "l" << std::endl;

		for (int j = 1; j<D[l] + 1; j++) {
			//std::cout << "j" << std::endl;
			int total = 0;
			for (int i = 0; i<=D[l - 1] + 1;i++) {
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
			if (is_classification || l<L) {
				//std::cout << "is_class" << std::endl;
				X[l][j] = std::tanh(total);
				//std::cout << "is_class passed" << std::endl;
			}
			//std::cout << "end j for" << std::endl;
		}
	}
	//std::cout << "finished propagate" << std::endl;
}

vector<float> PMC::predict(std::vector<float> inputs, bool is_classification) {
	this->_propagate(inputs, is_classification);
	return std::vector<float>(X[L].begin() + 1, X[L].end());
}

void PMC::train(vector<vector<float>> X_train,
	vector<vector<float>> Y_train,
	bool is_classification,
	float alpha = 0.01,
	int nb_iter=10000) 
{
	//std::cout << "train" << std::endl;
	for (int it = 0; it<=nb_iter; it++) {
		//std::cout << "it= " << it << std::endl;
		int k = rand() % (X_train.size());
		//std::cout << "k= " << k << std::endl;
		std::vector<float> Xk = X_train[k];
		//std::cout << "Xk.size()= " << Xk.size() << std::endl;
		std::vector<float> Yk = Y_train[k];
		//std::cout << "Yk.size()= " << Yk.size() << std::endl;

		this->_propagate(Xk, is_classification);

		//std::cout << "D len " << D.size() << std::endl;
		for (int j = 1; j<D[L] + 1; j++) {
			//std::cout << "j= " << j << std::endl;

			delta[L][j] = X[L][j] - Yk[j - 1];
			if (is_classification) {
				//std::cout << "is_class" << std::endl;
				delta[L][j] = delta[L][j] * (std::pow(1 - X[L][j],2));
				//std::cout << "is_class finished" << std::endl;
			}
		}
		//std::cout << "first for finished" << std::endl;

		for (int l = D.size()-1; l >= 2; l--) {
			//std::cout << "l= "<<l << std::endl;
			for (int i = 1; i<D[l - 1] + 1; i++) {
				//std::cout << "i= " << i << std::endl;
				float total = 0;
				for (int j = 1; j<D[l - 1] + 1; j++) {
					//std::cout << "gonna calc total " << std::endl;
					//std::cout << "delta[l][0]= " << delta[l][0] << std::endl;
					//std::cout << "W[l][i][j]= " << this->W[l][i][j] <<std::endl;
					total += this->W[l][i][j] * delta[l][j];
					//std::cout << "total calculated " << i << std::endl;
				}
				delta[l - 1][i] = (std::pow(1 - X[l - 1][i],2))*total;
			}
		}
		//std::cout << "second for finished" << std::endl;
		for (int l = 1; l<D.size(); l++) {
			//std::cout << "D.size()= " << D.size() << std::endl;
			//std::cout << "l= " << l << std::endl;
			for (int i = 0; i<D[l - 1] + 1; i++) {
				//std::cout << "i= " << i << std::endl;
				for (int j = 1; j<D[l] + 1; j++) {
					//std::cout << "j= " << j << std::endl;
					//std::cout << "delta[l][j]= " << delta[l][j] << std::endl;
					//std::cout << "W[l][i][j]= " << this->W[l][i][j] << std::endl;
					this->W[l][i][j] += -alpha * X[l - 1][i] * delta[l][j];
				}
			}
		}
	}
}