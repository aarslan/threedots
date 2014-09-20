import computation as comp, models as mod, params_ali
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import argparse
import time
#from hmax.models.dorsal import get_c1, prepare_cuda_kernels, pyramid_vid

def process_dir(src_dir, deg_l, deg_r, vid_type, target_dir, body_type, act, seq, this_fr):
	frame_cnt = 45
	target_stereo_dir = os.path.join(target_dir, deg_r+'-'+deg_l, body_type, act, seq)
	imH = 432
	imW = 324

	par = params_ali.ventral_absolute_disparity_simple_new()
	#import ipdb; ipdb.set_trace()
	if this_fr == -99:
		my_range = range(1, frame_cnt+1)
	else:
		my_range = [this_fr]

	for fr in my_range:
		print fr
		start_time = time.time()
		imleft = os.path.join(src_dir, deg_l, vid_type, body_type, act, seq, str(fr)+'.png');
		imright = os.path.join(src_dir, deg_r, vid_type, body_type, act, seq, str(fr)+'.png');
		print 'loading '+imright
		print 'loading '+imleft
		im = []; im.append(imleft); im.append(imright);
		features = mod.absolute_disparity(par, im)
		
		av_features = np.mean(np.mean(features, axis = 0),axis=1)
		D = np.argmax(av_features, axis=0);
		mD = np.squeeze(np.max(av_features, axis=0, keepdims=True));
		res = sp.array(D*(mD>0.9)/1, dtype='uint8')
		#plt.matshow(res)
		#plt.show()
		#import ipdb; ipdb.set_trace()
		mat_name = os.path.join(target_stereo_dir, str(fr))

		if not os.path.exists(target_stereo_dir):
			os.makedirs(target_stereo_dir)
		#sp.io.savemat(mat_name, {'fr': res})
		sp.io.savemat(mat_name, {'fr': np.array(np.mean(features,axis=2),dtype='Float32')})

		elapsed_time = time.time()-start_time
		print elapsed_time

#plt.imshow(res)
#plt.show()


def main():
	parser = argparse.ArgumentParser(description=""" """)
	parser.add_argument('--src_dir', type=str, default='/home/aarslan/prj/data/motion_morphing_dataset_angles/')
	parser.add_argument('--deg_l', type=str, default='2')
	parser.add_argument('--deg_r', type=str, default='8')
	parser.add_argument('--vid_type', type=str, default='frames_proto')
	parser.add_argument('--body_type', type=str, default='human')
	parser.add_argument('--act', type=str, default='balletjump')
	parser.add_argument('--seq', type=str, default='05_16')
	parser.add_argument('--this_fr', type=int, default=-99)
	parser.add_argument('--target_dir', type=str, default='/home/aarslan/prj/data/motion_morphing_dataset_stereo/features_stereo')
	

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
    
