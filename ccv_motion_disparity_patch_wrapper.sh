#!/bin/bash

data_in='/gpfs/data/tserre/Users/aarslan/foil_rotating/frames'
data_out='/gpfs/data/tserre/Users/aarslan/foil_rotating/features_motion'

#find $data_out -size 0 -type f -delete 

for deg in {2..360..30} #{2..360..30}

do 

	for action in $data_in/features_motion/$(($deg+6))-$deg/$bod/*
	do
		for seq in $action/*
		do
			sbatch ./ccv_motion_disparity_patch_core.sh $data_in $(($deg+6)) $deg `basename $action` `basename $seq` $data_out $bod; 
			#./ccv_motion_disparity_patch_core.sh $data_in $(($deg+6)) $deg `basename $action` `basename $seq` $data_out $bod; 				
			sleep 0.5
		done
	done

	sleep 10
done

