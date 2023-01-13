#pragma once
#include <vector>
#include <iostream>
using namespace std;


#ifdef FONCTION_EXPORTS
#define FONCTION_API __declspec(dllexport)
#else
#define FONCTION_API __declspec(dllimport)
#endif

extern "C"	FONCTION_API struct point
{
	float x;
	float y;
	point(float a, float b) {
		x = a;
		y = b;
	}
};


extern "C"	FONCTION_API void HelloWorld();
extern "C"	FONCTION_API void EntrainementLineaire(point *points, int *classes, float *W, int nbElem);
extern "C"	FONCTION_API float RandomFloat(float min, float max);
extern "C"	FONCTION_API void AffichageSeparation(vector<float> W);
extern "C"	FONCTION_API void add_to_vectorFloat(vector<float>*vec, float value);
extern "C"	FONCTION_API void add_to_vectorPoint(vector<point>*vec, point value);
extern "C"	FONCTION_API void add_to_vectorInt(vector<int>*vec, int value);