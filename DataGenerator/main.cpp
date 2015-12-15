#include "DataGenerator.h"

// #define REGULAR_CLUSTERS
#define SUBSPACE_CLUSTERS

#include <vector>
#include <iostream>

using namespace std;

double DEVIATION_FROM_CENTER = 1;
double NUM_DIMENSIONS = 5;

int main (int argc, char * argv[])
{
	/* Define first the shift we would like away from the origin */
	std::vector<double> shifter;
	shifter.push_back(6);
	shifter.push_back(5);
	shifter.push_back(3);
	shifter.push_back(3);
	shifter.push_back(4);

	#ifdef REGULAR_CLUSTERS
	std::vector<vector<double> > test = DataGenerator::GenerateCluster(NUM_DIMENSIONS, 4, DEVIATION_FROM_CENTER,
		3, shifter);

	for (int i =0; i < 4; i++)
	{
		std::cout << "Vector " << i << " in cluster:"  << std::endl;

		for (int j =0; j < NUM_DIMENSIONS; j++)
		{
			std::cout << test[i][j] << " ";
		}
		
		std::cout << std::endl;
	}
	#endif

	#ifdef SUBSPACE_CLUSTERS
	std::vector<bool> subspace_indices(NUM_DIMENSIONS, 0);

	/* 
	 * Here we mark the indices that should have clusters in them.
	 * Other indices will be shifted with a noisy value
	*/
	subspace_indices[2]=true;
	subspace_indices[4]=true;

	std::vector<vector<double> > test = DataGenerator::GenerateSubspaceCluster(NUM_DIMENSIONS, 4, DEVIATION_FROM_CENTER,
		3, shifter,&subspace_indices,0,10);

	for (int i =0; i < 4; i++)
	{
		std::cout << "Vector " << i << "in cluster:"  << std::endl;

		for (int j =0; j < NUM_DIMENSIONS; j++)
		{
			std::cout << test[i][j] << " ";
		}

		std::cout << std::endl;
	}
	#endif

	std::cout << "Average distance from the cluster center= " << DataGenerator::GetAverageDistanceFromCenter(test, shifter) << std::endl;
}