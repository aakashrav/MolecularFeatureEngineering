#!/bin/sh
DATA_DIRECTORY=../TestFragmentDescriptorData
rm -rf $DATA_DIRECTORY || true
mkdir $DATA_DIRECTORY

TEST_DATA_COUNTER=0
ACTIVES_INACTIVES_RATIO=.1

# Grid testing of parameters
for DIMENSIONS in {2..10}
do
	for INTERCLUSTER_DISTANCE in {100..300}
	do
		for DENSITY in {5..20}
		do
			for ACTIVES_INACTIVES_RATIO_INT in {1..5}
			do
				CLUSTER_DIRECTORY=$DATA_DIRECTORY/$TEST_DATA_COUNTER
				mkdir $CLUSTER_DIRECTORY
				python generate_clusters.py $CLUSTER_DIRECTORY $DIMENSIONS $INTERCLUSTER_DISTANCE $DENSITY $ACTIVES_INACTIVES_RATIO

				(( TEST_DATA_COUNTER += 1))
				ACTIVES_INACTIVES_RATIO=`echo $ACTIVES_INACTIVES_RATIO + .1|bc`
				echo "Generated a cluster!"
			done
			(( DENSITY+=4 ))
		done
		(( INTERCLUSTER_DISTANCE+=49 ))
	done
	(( DIMENSIONS+=1 ))
done

echo "Test cluster generation finished!"