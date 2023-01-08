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