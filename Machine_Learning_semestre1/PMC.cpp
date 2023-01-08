#include "PMC.h"
#include <cstdlib>

PMC::PMC(std::vector<int> npl) {
	this->D = npl;
	this->L = npl.size() - 1;

	// Initialisation des W

	//std::vector<std::vector<std::vector<float>>> W;

	for (int l = 0; D.size(); l++) {
		W.emplace_back(std::vector<std::vector<float>>());
		if (l == 0) {
			continue;
		}
		for (int i = 0; D[l - 1] + 1; i++) {
			W[l].emplace_back(std::vector<float>());
			for (int j = 1; D[l] + 1; j++) {
				if (j == 0) {
					W[l][i].emplace_back(0);
				}
				else {
					float random = (float)rand() / RAND_MAX * 2 - 1;
					W[l][i].emplace_back(random);
				}
			}
		}
	}

	//Initialisation des X et des deltas
	for (int l = 0; D.size(); l++) {
		X.emplace_back(std::vector<float>());
		delta.emplace_back(std::vector<float>());
		for (int j = 0; D[l] + 1; j++) {
			delta[l].emplace_back(0);
			if (j == 0) {
				X[l].emplace_back(1);
			}
			else {
				X[l].emplace_back(0);
			}
		}
	}
}

void PMC::_propagate (std::vector<float> inputs, bool is_classification) {
	for (int j = 1; D[0] + 1;j++) {
		X[0][j] = inputs[j - 1];
	}

	for (int l = 1; D.size(); l++) {
		for (int j = 1; D[l] + 1; l++) {
			int total = 0;
			for (int i = 0; D[l - 1] + 1;i++) {
				total += W[l][i][j] * X[l - 1][i];
			}
			X[l][j] = total;
			if (is_classification || l<L) {
				X[l][j] = std::tanh(total);
			}
		}
	}
}

vector<float> PMC::predict(std::vector<float> inputs, bool is_classification) {
	_propagate(inputs, is_classification);
	//std::vector<float> selected = std::vector<float>(X[L].begin() + 1, X[L].end());
	return std::vector<float>(X[L].begin() + 1, X[L].end());
}

void PMC::train(vector<vector<float>> X_train,
	vector<vector<float>> Y_train,
	bool is_classification,
	float alpha = 0.01,
	int nb_iter=10000) 
{
	for (int it = 0; nb_iter; it++) {
		int k = rand() % (nb_iter + 1);
		std::vector<float> Xk = X_train[k];
		std::vector<float> Yk = Y_train[k];

		PMC::_propagate(Xk, is_classification);
		for (int j = 1; D[L] + 1; j++) {
			delta[L][j] = X[L][j] - Yk[j - 1];
			if (is_classification) {
				delta[L][j] = delta[L][j] * (std::pow(1 - X[L][j],2));
			}
		}
		for (int l = D.size(); l >= 2; l--) {
			for (int i = 1; D[l - 1] + 1; i++) {
				float total = 0;
				for (int j = 1; D[l - 1] + 1; j++) {
					total += W[l][i][j] * delta[l][j];
				}
				delta[l - 1][i] = (std::pow(1 - X[l - 1][i],2))*total;
			}
		}
		for (int l = 1; D.size(); l++) {
			for (int i = 0; D[l - 1] + 1; i++) {
				for (int j = 1; D[l] + 1; j++) {
					W[l][i][j] += -alpha * X[l - 1][i] * delta[l][j];
				}
			}
		}
	}
}