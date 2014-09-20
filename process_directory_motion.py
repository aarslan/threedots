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

def process_dir(src_dir, deg_l, deg_r, vid_type, target_dir, seq, this_fr):
	sys.path
	loop_proc = True;
	frame_cnt = 45
	middle = np.mean([int(deg_r), int(deg_l)], dtype='int')
	mid_angle_dir = os.path.join(src_dir, str(middle))
	target_motion_dir = os.path.join(target_dir, deg_r+'-'+deg_l)
	imH = 409
	imW = 311
	#import ipdb; ipdb.set_trace()
	par = params.dorsal_velocity()
	#par = params.dorsal_pattern()
	#import ipdb; ipdb.set_trace()
	n_size = par['filters']['gabors_sizes'].shape[0]
	n_freq = par['filters']['gabors_temporal_frequencies'].shape[0]
	n_ori = par['filters']['gabors_number_of_orientations']
	filter_len = int(par['filters']['gabors_number_of_frames'])
	stereo_dir_name = os.path.join(mid_angle_dir, seq)

	crop = get_crop(stereo_dir_name, frame_cnt, imH, imW, par['filters']['gabors_sizes'])
	if loop_proc: #if the video loops seamlessly, you can pad the ends with frames from beginning and end
		#vid = np.ones([frame_cnt, imH, imW])
		vid_fnames = []
		for fr in range(1, frame_cnt): #load the regular features here first
			file_name = os.path.join(stereo_dir_name, str(fr)+'.png')
			target_dir_name = os.path.join(target_motion_dir)
			target_mat_name = os.path.join(target_dir_name, str(fr))
			#frame = load_frame(file_name)
			#import ipdb; ipdb.set_trace()
			#vid[fr,:,:] = np.array(frame)
			vid_fnames.append(file_name)
			#print str(fr+filt_fr)
		#vid2 = np.concatenate((vid[-1*(filter_len):-1,:,:], vid))
		vid_fnames2 = np.concatenate((vid_fnames[-1*(filter_len):-1], vid_fnames)).tolist()
		
		if this_fr != -99:
				end = this_fr+filter_len-2
				start = end-filter_len+1
				#im = vid2[start:end+1,:,:]
				im = vid_fnames2[start:end+1]
				#import ipdb; ipdb.set_trace()
				features= mod.dorsal_velocity(par, im, crop = crop)
				target_dir_name = os.path.join(target_motion_dir)
				target_mat_V1_name = os.path.join(target_dir_name, str(this_fr)+'_v1')
				target_mat_MT_name = os.path.join(target_dir_name, str(this_fr)+'_mt')
				if not os.path.exists(target_dir_name):
					os.makedirs(target_dir_name)
				sp.io.savemat(target_mat_MT_name, {'fr': np.array(features['MT'], dtype='Float32')}, do_compression=True)
				#import ipdb; ipdb.set_trace()
		else:
			for ii,f_fr in enumerate(range(filter_len, frame_cnt+filter_len)):
				start = f_fr-filter_len
				end = f_fr
				#im = vid2[start:end,:,:]
				im = vid_fnames2[start:end+1]
				#import ipdb; ipdb.set_trace()
				features= mod.dorsal_velocity(par, im, crop = crop)
				target_dir_name = os.path.join(target_motion_dir)
				target_mat_V1_name = os.path.join(target_dir_name, str(ii+1)+'_v1')
				target_mat_MT_name = os.path.join(target_dir_name, str(ii+1)+'_mt')
				if not os.path.exists(target_dir_name):
					os.makedirs(target_dir_name)
				#import ipdb; ipdb.set_trace()
				np.save(target_mat_MT_name, np.array(features[1], dtype='Float32'))
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
				#import ipdb; ipdb.set_trace()
				frame = load_frame(file_name)
				#import ipdb; ipdb.set_trace()
				im[filt_fr,:,:] = np.array(frame)
				#print str(fr+filt_fr)
			features, dummy = mod.dorsal_velocity(par, im)
			if not os.path.exists(target_dir_name):
				os.makedirs(target_dir_name)
			sp.io.savemat(target_mat_name, {'fr': features})
	#import ipdb; ipdb.set_trace()

#plt.imshow(res)
#plt.show()

def load_frame(file_names, frame_cnt, imW, imH):
	vid = np.ones([frame_cnt, imH, imW])
	if file_name[-3:] is 'mat':
		temp = sp.io.loadmat(file_name)['fr']
		frame = (temp != 0)*255
		vid[fr,:,:] = np.array(frame)
	vid2 = np.concatenate((vid[-1*(filter_len):-1,:,:], vid))

def get_crop(stereo_dir_name, frame_cnt, imH, imW, gabor_sizes):
	vid = np.ones([frame_cnt, imH, imW])
	for i in range(1,frame_cnt):
		vid[i,:,:] = np.mean(sp.misc.imread(os.path.join(stereo_dir_name, str(i)+'.png')), axis=2)
	border = int(np.max(gabor_sizes)/2+1)
	av = np.mean(vid,axis=0)
	aaa = np.where(av<av[1,1])

	crop = ((aaa[0].min()-border, aaa[0].max()+border), ((aaa[1].min()-border, aaa[1].max()+border)))
	import ipdb; ipdb.set_trace()
	return crop



def main():
	parser = argparse.ArgumentParser(description=""" """)
	parser.add_argument('--src_dir', type=str, default='/home/aarslan/prj/data/foil_rotating/frames')
	parser.add_argument('--deg_l', type=str, default='2')
	parser.add_argument('--deg_r', type=str, default='8')
	parser.add_argument('--vid_type', type=str, default='frames')
	parser.add_argument('--seq', type=str, default='143_01')
	parser.add_argument('--this_fr', type=int, default=-99)
	parser.add_argument('--target_dir', type=str, default='/home/aarslan/prj/data/foil_rotating/features_motion')
	

	args = parser.parse_args()
	src_dir = args.src_dir
	deg_l = args.deg_l
	deg_r = args.deg_r
	vid_type = args.vid_type
	seq =  args.seq
	this_fr = args.this_fr
	target_dir = args.target_dir
	process_dir(src_dir, deg_l, deg_r, vid_type, target_dir, seq, this_fr)

if __name__=="__main__":
	main()
    
