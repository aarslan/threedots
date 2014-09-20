#!/bin/bash

data_in='/gpfs/data/tserre/Users/aarslan/foil_rotating/frames'
data_out='/gpfs/data/tserre/Users/aarslan/foil_rotating/features_motion'

find $data_out -size 0 -type f -delete 

for deg in {2..360..30} #{2..360..30}

do 
	for seq in $data_in/$deg/*
	do
		echo sbatch ./ccv_motion_frame_core.sh $data_in $(($deg+6)) $deg `basename $seq` $data_out ; 
		sleep 1
	done
	sleep 90
done

