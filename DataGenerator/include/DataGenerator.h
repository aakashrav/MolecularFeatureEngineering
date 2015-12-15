/* 
 * Provides functions for generating data - namely vectors in R^d. 
 * These vectors can be combined to generate clusters of desired "density",
 * and the interface also enables generating clusters of dnesity in the desired
 * Euclidean subspaces
 */

#ifndef DATA_GENERATOR_H

#define DATA_GENERATOR_H

#include <random>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
class DataGenerator
{
public:
	static double 
	GetEuclideanDistance(const vector<double> & p1, const vector<double> & p2);

	static double 
	GetAverageDistanceFromCenter(const vector<vector<double> > cluster,
	const vector<double> center);

	static vector<double> 
	GenerateRandomVector(const int dimensions, const double max_deviation,
	const double multiplier, const vector<double> shifter, bool SUBSPACE_FLAG = false,
	vector<bool> * subspace_indices = NULL, double shift_range_min=0, double shift_range_max=0);

	static vector<vector<double> > 
	GenerateCluster(const int dimensions, const int amount,
	const double max_deviation, const double multiplier, const vector<double> shifter);

	static vector<vector<double> > 
	GenerateSubspaceCluster(const int dimensions, const int amount,
	const double max_deviation, const double multiplier, const vector<double> shifter,
	vector<bool> * subspace_indices, double shift_range_min, double shift_range_max);

};

#endif