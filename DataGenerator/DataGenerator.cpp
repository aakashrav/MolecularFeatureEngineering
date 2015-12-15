#include "DataGenerator.h"

/* Compute standard Euclidean distance*/
double
DataGenerator::GetEuclideanDistance(const vector<double> & p1, const vector<double> & p2)
{
	vector<double>::const_iterator itr;
	vector<double>::const_iterator itr2 = p2.begin();
	double distance = 0;

	for (itr = p1.begin(); itr!=p1.end(); ++itr)
	{
		distance += pow((*itr) - (*itr2),2);
		++itr2;
	}

	return sqrt(distance);
}

/* Compute the average distance of a set of vectors from the cluster center 
 * Useful for testing purposes.
 */
double 
DataGenerator::GetAverageDistanceFromCenter(const vector<vector<double> > cluster,
	const vector<double> center)
{
	double average =0;
	for (int i =0; i < cluster.size(); i++)
	{
		average += GetEuclideanDistance(cluster[i], center);
	}

	average /= cluster.size();
	return average;
}

/* 
 * Generates a random vector according to the prescribed max_deviation.
 * By making sure we choose values from a normal distribution with standard
 * deviation max_deviation/3, we ensure that the probability that we get a vector
 * beyond this distribution is quite low and therefore our cluster center won't
 * get skewed. We also specify a shifter and multiplier, which will scale and move 
 * all the vectors to somewhere far away from the origin. Ideally, for each cluster
 * we generate, we specify a random multiplier and shifter, giving us clusters positioned
 * in various places in the Euclidean space. 
 * TODO: Make the multiplier not just a scalar,
 * but a matrix so that we get linear transformations and not just scalar multiples
 */
vector<double> 
DataGenerator::GenerateRandomVector(const int dimensions, const double max_deviation,
	const double multiplier, const vector<double> shifter, bool SUBSPACE_FLAG,
	vector<bool> * subspace_indices, double shift_range_min, double shift_range_max)
{
	vector<double> random_vector;
	//Get a random seed from hardware
	random_device rd;
	//Seed a random number generator
	mt19937 gen(rd());

	//Define a normal distribution centered at 0 with deviation max_deviation/3
	normal_distribution<> d(0,max_deviation/3);

	for (int i =0; i < dimensions; i++)
	{
		// Choose a random value from the distribution for a specific attribute
		random_vector.push_back(d(gen));
	
		// If we are not doing subspace clustering, OR this is one of the subspaces
		// we want to be clustered, we perform the regular scaling and shifting.
		if (!SUBSPACE_FLAG || (*subspace_indices)[i] == true)
		{
			random_vector[i] = (multiplier * random_vector[i]) + shifter[i];
			continue;
		}
		
		// However, if we find a subspace that we do not wish to be clustered in,
		// we need to make this specific attribute noisy, therefore we shift by some
		// random noise generator, where the range of the shift is specified as a parameter
		if ((*subspace_indices)[i] == false)
		{
			// Define the range of the shift, specified in the parameter
			uniform_int_distribution<> distr(shift_range_min, shift_range_max);
			double random_shift = distr(gen);
			// Do some noisy attribute scaling, making sure this attribute 
			// has a low probability of being clustered
			random_vector[i] = (multiplier * random_vector[i]) + random_shift;
		}
	}

	return random_vector;
}

/* Generate a regular (no subspace) cluster*/
vector<vector<double> > 
DataGenerator::GenerateCluster(const int dimensions, const int amount,
	const double max_deviation, const double multiplier, const vector<double> shifter)
{
	vector<vector<double> > cluster;
	for (int i =0; i < amount; i++)
	{
		vector<double> member = GenerateRandomVector(dimensions, max_deviation, multiplier,
			shifter);
		cluster.push_back(member);
	}

	return cluster;
}

/* Generate a subspace cluster, with indices that we desire to be clustered
 * specified as  parameters "subspcae_indices"
 */
vector<vector<double> > 
DataGenerator::GenerateSubspaceCluster(const int dimensions, const int amount,
	const double max_deviation, const double multiplier, const vector<double> shifter,
	vector<bool> * subspace_indices, double shift_range_min, double shift_range_max)
{
	vector<vector<double> > cluster;
	for (int i =0; i < amount; i++)
	{
		vector<double> member = GenerateRandomVector(dimensions, max_deviation, multiplier,
			shifter, true, subspace_indices, shift_range_min, shift_range_max);
		cluster.push_back(member);
	}

	return cluster;
}
