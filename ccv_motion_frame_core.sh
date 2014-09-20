#!/bin/bash

#SBATCH --time=00:59:00
#SBATCH --mem=64G
##SBATCH -n 8
##SBATCH -N 8-16 
#SBATCH --qos=pri-aarslan ##bibs-tserre-condo
#SBATCH --exclusive
#SBATCH --exclude=smp012,smp013,smp014,smp015

#SBATCH -J motion_xtract_trial
#SBATCH -o /users/aarslan/out/motion_xtract_%j.out


module unload python
module load enthought
module unload cuda

echo 'processing' $2 $3 $4 $5 
src_code_dir='/users/aarslan/code/threedots'
joblist='/users/aarslan/joblists/'$2_$3_$4_$5'.jlist'
echo before $PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/users/aarslan/tools/hmax/models/HNORM
echo after $PYTHONPATH

rm $joblist -f

for fr in {1..45}
do
	FILE=$6/$2-$3/$7/$4/$5/${fr}'_mt.mat'
	#echo checking $FILE
	if [ ! -f "$FILE" ]
	then
	echo python $src_code_dir/process_directory_motion.py --src_dir $1 --deg_r $2 --deg_l $3 --seq $4 --target_dir $5 --this_fr $fr >> $joblist	
	fi
done

parallel -j6 -a $joblist
