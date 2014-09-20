import sys
import computation as comp, models as mod, params
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import argparse
import time
#from hmax.models.dorsal import get_c1, prepare_cuda_kernels, pyramid_vid

def process_dir(src_dir, deg_l, deg_r, vid_type, target_dir, body_type, act, seq, this_fr):
	sys.path
	loop_proc = True;
	frame_cnt = 45
	filter_len = int(10)
	target_stereo_dir = os.path.join(src_dir, deg_r+'-'+deg_l, body_type )
	target_motion_dir = os.path.join(target_dir, deg_r+'-'+deg_l, body_type )
	imH = 432
	imW = 324
	#import ipdb; ipdb.set_trace()
	par = params.dorsal_pattern_simple()
	#par = params.dorsal_pattern()
	#import ipdb; ipdb.set_trace()
	n_size = par['filters']['gabors_sizes'].shape[0]
	n_freq = par['filters']['gabors_temporal_frequencies'].shape[0]
	n_ori = par['filters']['gabors_number_of_orientations']

	if loop_proc: #if the video loops seamlessly, you can pad the ends with frames from beginning and end
		vid = np.ones([frame_cnt, imH, imW])
##		for fr in range(1, frame_cnt): #load the regular features here first
##			stereo_dir_name = os.path.join(target_stereo_dir, act, seq)
##			stereo_mat_name = os.path.join(stereo_dir_name, str(fr))
##			target_dir_name = os.path.join(target_motion_dir, act, seq)
##			target_mat_name = os.path.join(target_dir_name, str(fr))
##			frame_dict = sp.io.loadmat(stereo_mat_name)
##			temp = frame_dict['fr']
##			temp = (temp != 0)*255
##			#import ipdb; ipdb.set_trace()
##			vid[fr,:,:] = np.array(temp)
##			#print str(fr+filt_fr)
##		vid2 = np.concatenate((vid[-1*(half_filter+1):-1,:,:], vid, vid[0:half_filter,:,:]))
##		
##		if this_fr != -99:
##				real_fr = this_fr+half_filter-1
##				start = real_fr-half_filter
##				end = real_fr+half_filter
##				im = vid2[start:end,:,:]
##				#import ipdb; ipdb.set_trace()
##				features= mod.dorsal_pattern_divisive(par, im)
##				target_dir_name = os.path.join(target_motion_dir, act, seq)
##				target_mat_V1_name = os.path.join(target_dir_name, str(this_fr)+'_v1')
##				target_mat_MT_name = os.path.join(target_dir_name, str(this_fr)+'_mt')
##				if not os.path.exists(target_dir_name):
##					os.makedirs(target_dir_name)
##				sp.io.savemat(target_mat_V1_name, {'fr': np.array(features['V1'], dtype='Float32')}, do_compression=True)
##				sp.io.savemat(target_mat_MT_name, {'fr': np.array(features['MT'], dtype='Float32')}, do_compression=True)
##				#import ipdb; ipdb.set_trace()
##		else:
##			for ii,f_fr in enumerate(range(half_filter, frame_cnt+half_filter)):
##				start = f_fr-half_filter
##				end = f_fr+half_filter
##				im = vid2[start:end,:,:]
##				#import ipdb; ipdb.set_trace()
##				features= mod.dorsal_pattern_divisive(par, im)
##				target_dir_name = os.path.join(target_motion_dir, act, seq)
##				target_mat_V1_name = os.path.join(target_dir_name, str(ii+1)+'_v1')
##				target_mat_MT_name = os.path.join(target_dir_name, str(ii+1)+'_mt')
##				if not os.path.exists(target_dir_name):
##					os.makedirs(target_dir_name)
##				sp.io.savemat(target_mat_V1_name, {'fr': np.array(features['V1'], dtype='Float32')}, do_compression=True)
##				sp.io.savemat(target_mat_MT_name, {'fr': np.array(features['MT'], dtype='Float32')}, do_compression=True)
##				#import ipdb; ipdb.set_trace()
				
		for fr in range(1, frame_cnt): #load the regular features here first
			stereo_dir_name = os.path.join(target_stereo_dir, act, seq)
			stereo_mat_name = os.path.join(stereo_dir_name, str(fr))
			target_dir_name = os.path.join(target_motion_dir, act, seq)
			target_mat_name = os.path.join(target_dir_name, str(fr))
			frame_dict = sp.io.loadmat(stereo_mat_name)
			temp = frame_dict['fr']
			temp = (temp != 0)*255
			#import ipdb; ipdb.set_trace()
			vid[fr,:,:] = np.array(temp)
			#print str(fr+filt_fr)
		vid2 = np.concatenate((vid[-1*(filter_len):-1,:,:], vid))
		
		if this_fr != -99:
				end = this_fr+filter_len-2
				start = end-filter_len+1
				im = vid2[start:end+1,:,:]
				#import ipdb; ipdb.set_trace()
				features= mod.dorsal_pattern_divisive(par, im)
				target_dir_name = os.path.join(target_motion_dir, act, seq)
				target_mat_V1_name = os.path.join(target_dir_name, str(this_fr)+'_v1')
				target_mat_MT_name = os.path.join(target_dir_name, str(this_fr)+'_mt')
				if not os.path.exists(target_dir_name):
					os.makedirs(target_dir_name)
				#sp.io.savemat(target_mat_V1_name, {'fr': np.array(features['V1'], dtype='Float32')}, do_compression=True)
				sp.io.savemat(target_mat_MT_name, {'fr': np.array(features['MT'], dtype='Float32')}, do_compression=True)
				#import ipdb; ipdb.set_trace()
		else:
			for ii,f_fr in enumerate(range(filter_len, frame_cnt+filter_len)):
				start = f_fr-filter_len
				end = f_fr
				im = vid2[start:end,:,:]
				#import ipdb; ipdb.set_trace()
				features= mod.dorsal_pattern_divisive(par, im)
				target_dir_name = os.path.join(target_motion_dir, act, seq)
				target_mat_V1_name = os.path.join(target_dir_name, str(ii+1)+'_v1')
				target_mat_MT_name = os.path.join(target_dir_name, str(ii+1)+'_mt')
				if not os.path.exists(target_dir_name):
					os.makedirs(target_dir_name)
				#sp.io.savemat(target_mat_V1_name, {'fr': np.array(features['V1'], dtype='Float32')}, do_compression=True)
				sp.io.savemat(target_mat_MT_name, {'fr': np.array(features['MT'], dtype='Float32')}, do_compression=True)
				#import ipdb; ipdb.set_trace()
			
	else:
		for ii,fr in enumerate(range(half_filter,frame_cnt-half_filter+1)):
			im = np.ones([filter_len, imH, imW])
			print 'base',fr
			for filt_fr in range(-half_filter+1, half_filter+1):
				stereo_dir_name = os.path.join(target_stereo_dir, act, seq)
				stereo_mat_name = os.path.join(stereo_dir_name, str(fr+filt_fr))
				target_dir_name = os.path.join(target_motion_dir, act, seq)
				target_mat_name = os.path.join(target_dir_name, str(fr))
				import ipdb; ipdb.set_trace()
				frame_dict = sp.io.loadmat(stereo_mat_name)
				temp = frame_dict['fr']
				temp = (temp != 0)*255
				#import ipdb; ipdb.set_trace()
				im[filt_fr,:,:] = np.array(temp)
				#print str(fr+filt_fr)
			features, dummy = mod.dorsal_pattern(par, im)
			if not os.path.exists(target_dir_name):
				os.makedirs(target_dir_name)
			sp.io.savemat(target_mat_name, {'fr': features})
	#import ipdb; ipdb.set_trace()

#plt.imshow(res)
#plt.show()


def main():
	parser = argparse.ArgumentParser(description=""" """)
	parser.add_argument('--src_dir', type=str, default='/home/aarslan/prj/data/motion_morphing_dataset_stereo/features_stereo')
	parser.add_argument('--deg_l', type=str, default='2')
	parser.add_argument('--deg_r', type=str, default='8')
	parser.add_argument('--vid_type', type=str, default='frames_proto')
	parser.add_argument('--body_type', type=str, default='human')
	parser.add_argument('--act', type=str, default='balletjump')
	parser.add_argument('--seq', type=str, default='05_16')
	parser.add_argument('--this_fr', type=int, default=-99)
	parser.add_argument('--target_dir', type=str, default='/home/aarslan/prj/data/motion_morphing_dataset_stereo/features_motion')
	

	args = parser.parse_args()
	src_dir = args.src_dir
	deg_l = args.deg_l
	deg_r = args.deg_r
	vid_type = args.vid_type
	body_type = args.body_type
	act = args.act
	seq =  args.seq
	this_fr = args.this_fr
	target_dir = args.target_dir
	process_dir(src_dir, deg_l, deg_r, vid_type, target_dir, body_type, act, seq, this_fr)

if __name__=="__main__":
	main()
    
