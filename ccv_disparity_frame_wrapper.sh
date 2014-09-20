#!/bin/bash

CODE_DIR=/home/aarslan/prj/code/dorsoventral/disparity
ANGLES_DIR=/gpfs/data/tserre/Users/aarslan/motion_morphing_dataset_angles
TARGET_DIR=/gpfs/data/tserre/Users/aarslan/motion_morphing_dataset_stereo/features_stereo

data_in=$ANGLES_DIR
data_out=$TARGET_DIR

type=frames_proto

find $data_out -size 0 -type f -delete 

for deg in {2..360..30}
do 
	for bod in {human,a,y}
	do
		for action in $ANGLES_DIR/$deg/$type/$bod/*
		do
			for seq in $action/*
			do
				sbatch ./ccv_disparity_frame_core.sh $data_in $(($deg+6)) $deg `basename $action` `basename $seq` $data_out $bod; 
				sleep 1
			done
		done
	done
	sleep 10
done

