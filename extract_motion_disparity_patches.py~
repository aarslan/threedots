import computation as comp, models as mod, params
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import argparse
import random

#gpfs/data/tserre/Users/aarslan/motion_morphing_dataset_stereo/features_motion_fsize21/248-242/y/walk/35_01
#----#
def extract_patches(src_dir, deg_l, deg_r, vid_type, target_stem_dir, body_type, act, seq, this_fr, num_patches, value_method = 0, patch_ratio =1/2.):
	#(source_stem_dir, target_stem_dir, disp_string, act, seq, num_patches):
	source_disparity_dir = os.path.join(src_dir, 'features_stereo', deg_r+'-'+deg_l, body_type ) 
	source_motion_dir = os.path.join(src_dir, 'features_motion',deg_r+'-'+deg_l, body_type )
	#import ipdb; ipdb.set_trace()
	pool_shape_m = (1,1,54,54)
	pool_shape_d = (1,1,54,54)
	if this_fr != -99:
		my_range = [this_fr]
	else:
		my_range = range(1,45)

	for fr in my_range:
		motion_mat_name = os.path.join(source_motion_dir, act, seq, str(fr)+'_mt')
		f_motion = sp.io.loadmat(motion_mat_name)
		f_motion = f_motion['fr']

		disparity_mat_name = os.path.join(source_disparity_dir, act, seq, str(fr))
		f_disparity = sp.io.loadmat(disparity_mat_name)
		f_disparity = f_disparity['fr']
	
		pooled_motion = comp.flexpooling('max', pool_shape_m, f_motion,  downsample = True)
		pooled_disparity = comp.flexpooling('max', pool_shape_d, f_disparity, downsample = True) # 
		

		#av_features =pooled_disparity
		#D = np.argmax(av_features, axis=0);
		#mD = np.squeeze(np.max(av_features, axis=0, keepdims=True));
		#pooled_disparity = sp.array(D*(mD>0.9)/1, dtype='uint8') #here we got rid of the disparity bands and things are back to normal. Notice the thresholding is applied here (0.9)
		#import ipdb; ipdb.set_trace()
		#plt.matshow(res)
		#plt.show()
		
		#plt.matshow(np.mean(pooled_motion,axis=(0,1)))
		#plt.show()
		#import ipdb; ipdb.set_trace()
		arg0 = np.argmax(np.amax(pooled_motion, axis=0), axis=0)
		arg1 = np.argmax(np.amax(pooled_motion, axis=1), axis=0)
		eee = cartesian([range(arg0.max()+1), range(arg1.max()+1)])
		motion_size = pooled_motion.shape
		disparity_size = pooled_disparity.shape
		new_motion = np.zeros((motion_size[-2], motion_size[-1]))
		for (x,y), value in np.ndenumerate(arg0):
			new_motion[x,y] = int(np.argwhere(np.sum(eee == [arg0[x,y], arg1[x,y]], axis=1)==2))
		
		null1, p_w, null2, p_h = get_coor(pooled_disparity, patch_ratio)
		
		if num_patches is not 0:
			patch_count = num_patches
		else:
			patch_count = pow(1/patch_ratio, 2) #this is the case for parcellation
		#import ipdb; ipdb.set_trace()

		patch_m = np.ones((patch_count, motion_size[0], motion_size[1], p_h, p_w), dtype = 'Float32')
		patch_d = np.ones((patch_count, disparity_size[-3], p_h, p_w), dtype = 'Float32')
		
		if num_patches != 0:
			for pp in range(num_patches):
				h_rand, hori_bound, v_rand, vert_bound = get_coor(pooled_disparity, patch_ratio)
				patch_m[pp,:] = get_patch(pooled_motion, h_rand, hori_bound, v_rand, vert_bound)
				patch_d[pp,:] = get_patch(pooled_disparity, h_rand, hori_bound, v_rand, vert_bound)
		else:
			h_starts, h_bound, v_starts, v_bound = get_parcels(pooled_disparity, patch_ratio)
			for pp in range(len(h_starts)):
				patch_m[pp,:] = get_patch(pooled_motion, h_starts[pp], h_bound, v_starts[pp], v_bound)
				patch_d[pp,:] = get_patch(pooled_disparity, h_starts[pp], h_bound, v_starts[pp], v_bound)
		target_dir = os.path.join(target_stem_dir,'motion_disparity_patches',  deg_r+'-'+deg_l, body_type,act, seq)
		patch_mat_name = os.path.join(target_dir, str(fr))
		#import ipdb; ipdb.set_trace()
		if not os.path.exists(target_dir):
			os.makedirs(target_dir)
		sp.io.savemat(patch_mat_name, {'patch_m': patch_m, 'patch_d': patch_d})

#----#
def get_patch(feat, h_rand, hori_bound, v_rand, vert_bound):
	#import ipdb; ipdb.set_trace()
	if feat.ndim == 4:
		return feat[:,:,v_rand:v_rand+vert_bound, h_rand:h_rand+hori_bound]
	else:
		return feat[:, v_rand:v_rand+vert_bound, h_rand:h_rand+hori_bound]

#----#
def get_coor(feat, rat = 1/3.):
	hori_bound = int(feat.shape[-1]*rat)
	vert_bound = int(feat.shape[-2]*rat)
	
	h_rand = random.randint(0, hori_bound)
	v_rand = random.randint(0, vert_bound)
	return h_rand, hori_bound, v_rand, vert_bound

def get_parcels(feat, rat = 1/3.):
	
	feat_size = feat.shape;
	
	vert_size = range(0, feat_size[0], int(feat_size[0]*rat))
	hori_size = range(0, feat_size[1], int(feat_size[1]*rat))
	h_bound = int(feat.shape[-1]*rat)
	v_bound = int(feat.shape[-2]*rat)
	
	num_parcels = min(len(hori_size), len(vert_size))
	
	v_starts = []
	h_starts = []
	v_bounds = []
	h_bounds = []
	for ii in range(num_parcels):
		for jj in range(num_parcels):
			v_starts.append(vert_size[ii])
			h_starts.append(hori_size[jj])
			v_bounds.append(vert_size[ii]+v_bound)
			h_bounds.append(hori_size[jj]+h_bound)
	#import ipdb; ipdb.set_trace()
	return h_starts, h_bound, v_starts, v_bound

#----#

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

#----#
def main():
	parser = argparse.ArgumentParser(description=""" """)
	parser.add_argument('--src_dir', type=str, default='/home/aarslan/prj/data/motion_morphing_dataset_stereo/')
	parser.add_argument('--deg_l', type=str, default='2')
	parser.add_argument('--deg_r', type=str, default='8')
	parser.add_argument('--vid_type', type=str, default='frames_proto')
	parser.add_argument('--body_type', type=str, default='human')
	parser.add_argument('--act', type=str, default='balletjump')
	parser.add_argument('--seq', type=str, default='05_16')
	parser.add_argument('--this_fr', type=int, default=-99)
	parser.add_argument('--target_dir', type=str, default='/home/aarslan/prj/data/motion_morphing_dataset_stereo/')
	parser.add_argument('--num_patches', type=int, default=10)

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
	num_patches = args.num_patches
	extract_patches(src_dir, deg_l, deg_r, vid_type, target_dir, body_type, act, seq, this_fr, num_patches)

if __name__=="__main__":
	main()
    
