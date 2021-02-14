
# Created by: Gabriel Sarch
# Jan 2021
# Distance feature functions obtained with permission from: Mark Lescroart 
# Consistency model: https://github.com/EPFL-VILAB/XTConsistency

from modules.unet import UNet, UNetReshade
import os

import torch
import torchvision
from torchvision import transforms

import glob
import PIL
from PIL import Image

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm,colors
from matplotlib import transforms as mtransforms
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import color as skcol
import ipdb
st = ipdb.set_trace

import scipy.io

coco_dir = '/lab_data/tarrlab/common/datasets/NSD_images'  
save_dir = '/lab_data/tarrlab/common/datasets/GRNN/NSD_feats_allimages/mark' 

id_file = '/lab_data/tarrlab/common/datasets/GRNN/NSD_shared1000_cocoIDs.npy'
XTC_loc = '/home/gsarch/repo/grnn/XTConsistency'

do_Mark_features = True
do_alexnet = False

process_all_NSD_images = True # do I want to run through all 73000 images or just 1000

# coco_dir = '/lab_data/tarrlab/common/datasets/NSD_images'  
# save_dir = '/Users/gabrielsarch/Documents/repo/data/coco_normals' 

# id_file = '/Users/gabrielsarch/Documents/repo/data/NSD_shared1000_cocoIDs.npy'
# XTC_loc = '/Users/gabrielsarch/Documents/repo/XTConsistency'

NORM_BIN_CENTERS = np.array([[-1, 0, 0],  # Cardinal directions: up, down, left, right
                             [1, 0, 0],
                             [0, 0, -1],
                             [0, 0, 1],
                             [-1, 1, -1],  # Oblique directions
                             [1, 1, -1],
                             [1, 1, 1],
                             [-1, 1, 1],
                             [0, 1, 0]])  # Straight ahead
# NOTE: Reasonable disance divisions will depend on the distance input. If absolute
# distance at approximately human scales is used, this scaling makes sense. If
# some measure of relative distance in the scene is desired, this makes less
# sense (unless that relative distance is scaled 0-100 or some such)
# N_BINS_DIST = 10
N_BINS_DIST = 10
# MAX_DIST = 100
MAX_DIST = 100
DIST_BIN_EDGES = np.logspace(np.log10(1), np.log10(MAX_DIST), N_BINS_DIST)
DIST_BIN_EDGES = np.hstack([0, DIST_BIN_EDGES[:-1], 999])

# Colormap(s)
from matplotlib.colors import LinearSegmentedColormap
RET = LinearSegmentedColormap.from_list('RET', 
        [(1, 0, 0), (1., 1., 0), (0, 0, 1), (0, 1., 1), (1., 0, 0)])
blue = (0, 0, 1.0)
cyan = (0, 0.5, 1.0)
white = (1.0, 0.85, 1.0)
orange = (1.0, 0.5, 0)
red = (1.0, 0, 0)
color_cycle = [blue, cyan, white, orange, red]
alpha_cycle_0 = (1.0, 0.625, 0.25, 0.625, 1.0)
alpha_cycle_1 = (1.0, 0.5, 0.0, 0.5, 1.0)
# blue, cyan, white, orange, red
# (this is effectively a higher-contrast RdBu_r)
bcwor = LinearSegmentedColormap.from_list('bcwor', color_cycle)
bcwora = LinearSegmentedColormap.from_list('bcwora', [col + tuple([a]) for col, a in zip(color_cycle, alpha_cycle_0)])
bcworaa = LinearSegmentedColormap.from_list('bcworaa', [col + tuple([a]) for col, a in zip(color_cycle, alpha_cycle_1)])


class BrainModel():
    def __init__(self):
        print("INIT MODEL")


        # # Initialize vgg
        # vgg16 = torchvision.models.vgg16(pretrained=True).double().cuda()
        # vgg16.eval()
        # print(torch.nn.Sequential(*list(vgg16.features.children())))
        # self.vgg_feat_extractor = torch.nn.Sequential(*list(vgg16.features.children())[:2])
        # print(self.vgg_feat_extractor)
        # self.vgg_mean = torch.from_numpy(np.array([0.485,0.456,0.406]).reshape(1,3,1,1))
        # self.vgg_std = torch.from_numpy(np.array([0.229,0.224,0.225]).reshape(1,3,1,1))
        
        if do_alexnet:
            print("GETTING ALEXNET")

            # Initialize alexnet
            alexnet = torchvision.models.alexnet(pretrained=True).double().cuda()
            alexnet.eval()
            # print(torch.nn.Sequential(*list(alexnet.features.children())))

            layers = []
            layers.extend(list(alexnet.features.children()))
            
            self.avgpool = alexnet.avgpool

            classifier = []
            classifier.extend(list(alexnet.classifier.children()))

            self.alexnet_feat_extractor = []
            self.alexnet_class_extractor = []
            self.alexnet_names = []
            for i in range(len(layers)):
                self.alexnet_feat_extractor.append(torch.nn.Sequential(*list(layers[:i+1])))
                name = 'alexnet_' + str(layers[i]).split('(')[0] + '_' + str(i)
                self.alexnet_names.append(name)
            
            for i in range(len(classifier)):
                self.alexnet_class_extractor.append(torch.nn.Sequential(*list(classifier[:i+1])))
                name = 'alexnet_' + str(classifier[i]).split('(')[0] + '_' + str(len(layers) + i)
                self.alexnet_names.append(name)

            print(self.alexnet_feat_extractor)
        
        if do_Mark_features:
            print("GETTING MARK MODEL")

            target_tasks = ['normal','depth','reshading'] #options for XTC model

            #initialize XTC model
            ### DEPTH MODEL %%%%%%%%%%
            task_index = target_tasks.index('depth')
            models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
            self.XTCmodel_depth = models[task_index]

            pretrained_model = 'consistency_wimagenet'
            path = os.path.join(XTC_loc, 'models', 'rgb2'+'depth'+'_'+pretrained_model+'.pth')
            map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
            model_state_dict = torch.load(path, map_location=map_location)
            self.XTCmodel_depth.load_state_dict(model_state_dict)

            ### NORMALS MODEL %%%%%%%%%%
            task_index = target_tasks.index('normal')
            models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
            self.XTCmodel_normals = models[task_index]

            pretrained_model = 'consistency'
            path = os.path.join(XTC_loc, 'models', 'rgb2'+'normal'+'_'+pretrained_model+'.pth')
            map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
            model_state_dict = torch.load(path, map_location=map_location)
            self.XTCmodel_normals.load_state_dict(model_state_dict)

        # file paths
        print("GETTING FILES")
        self.files = glob.glob("{0}/*.jpg".format(coco_dir))

        if process_all_NSD_images:
            self.files_iter = self.files
        else:
            self.file_ids = np.load(id_file)
            
            # keep only files shared between NSD and BOLD5000
            self.files_iter = []
            for file in self.files:
                file_id = os.path.split(file)
                file_id = file_id[-1]
                file_id = os.path.splitext(file_id)[0]
                if int(file_id) in self.file_ids:
                    self.files_iter.append(file)

        print("Found ", len(self.files_iter), "files.")

        self.W = 256
        self.H = 256

    def run_model(self):

        files_s = range(len(self.files_iter)) #range(len(self.files))
        print("Iterating over ", len(self.files_iter), " files.")

        # files_s = files_s[0:5]
        
        # get depth maps
        depths = []
        normals = []
        file_order = []
        # images = []

        # initialize layer variables
        feats = {}

        s_num = 0 

        # currently batch size is 1 but would be good to have this as a parameter 
        for s in files_s:

            print("Processing file ", s)

            im_path = self.files_iter[s]
            print("Loading ", im_path)

            im = Image.open(im_path)

            file_id = os.path.split(im_path)
            file_id = file_id[-1]
            file_id = os.path.splitext(file_id)[0]
            print("FILE PATH: ", file_id)
            file_order.append(int(file_id))

            if do_Mark_features:
                #file_id = self.file_ids[s]
                trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                                    transforms.CenterCrop(256),
                                                    transforms.ToTensor()])
                
                im_t = trans_totensor(im)[:3].unsqueeze(0)

                depth_cam = self.XTCmodel_depth(im_t)#.clamp(min=0, max=1)

                normals_cam = self.XTCmodel_normals(im_t)

                depth_cam = depth_cam.squeeze().detach().cpu().numpy() 
                normals_cam = normals_cam.squeeze().detach().cpu().numpy()

                # image = im_t.squeeze().detach().cpu().numpy() 
                # images.append(image)

                # get depths in 0-100 range approx.
                depth_cam = (depth_cam / 0.7) * 100

                depths.append(depth_cam)
                normals.append(normals_cam)

                if False:
                    print("SAVING: ", file_id)
                    np.save(f'{save_dir}/mark/{file_id}.npy', normals_cam)
                    np.save(f'{save_dir}/mark/{file_id}_img.npy', im_t.squeeze().detach().cpu().numpy())
                    np.save(f'{save_dir}/mark/{file_id}_depth.npy', depth_cam)
            



            if do_alexnet:

                normalize = transforms.Compose([
                    transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                
                
                # normalize before running through features
                norm_img = normalize(im)[:3].unsqueeze(0).double().cuda()

                if s_num == 0:
                    for i in range(len(self.alexnet_class_extractor) + len(self.alexnet_feat_extractor) + 1):
                        feats[i] = []
                        print(i)

                # get output of each layer
                with torch.no_grad():
                    for i in range(len(self.alexnet_feat_extractor)):
                        feat_extractor = self.alexnet_feat_extractor[i]
                        # print(i)
                        img_feat = feat_extractor(norm_img)
                        feats[i].append(img_feat.squeeze().detach().cpu().float().numpy())
                    
                    feats_end = self.alexnet_feat_extractor[-1]
                    img_feat = feats_end(norm_img)
                    img_feat = self.avgpool(img_feat)
                    feats[len(self.alexnet_feat_extractor)].append(img_feat.squeeze().detach().cpu().float().numpy())
                    # print(len(self.alexnet_feat_extractor))
                    idx = 0
                    for i in range(len(self.alexnet_feat_extractor) + 1, len(self.alexnet_class_extractor) + len(self.alexnet_feat_extractor) + 1):
                        feat_extractor = self.alexnet_class_extractor[idx]
                        # print(i)
                        img_feat = feats_end(norm_img)
                        img_feat = self.avgpool(img_feat)
                        img_feat = torch.flatten(img_feat, 1)
                        img_feat = feat_extractor(img_feat)
                        feats[i].append(img_feat.squeeze().detach().cpu().float().numpy())
                        idx += 1
                
                if s_num == 0:
                    s_num += 1
            

                # with torch.no_grad():
                #     img_feat = self.alexnet_Conv2d_6(norm_img)
                # Conv2d_6.append(img_feat.squeeze().detach().cpu().float().numpy())

                # with torch.no_grad():
                #     img_feat = self.alexnet_ReLU_7(norm_img)
                # ReLU_7.append(img_feat.squeeze().detach().cpu().float().numpy())

        file_order = np.array(file_order)

        if do_Mark_features:
            # np.save(f'{save_dir}/XTConsistency/depths.npy', np.array(depths))
            # np.save(f'{save_dir}/XTConsistency/normals.npy', np.array(normals))
            # np.save(f'{save_dir}/XTConsistency/images.npy', np.array(images))
            # np.save(f'{save_dir}/XTConsistency/file_order.npy', file_order)
            # scipy.io.savemat(f'{save_dir}/XTConsistency/depths.mat', dict(depths=np.array(depths)))
            # scipy.io.savemat(f'{save_dir}/XTConsistency/normals.mat', dict(normals=np.array(normals)))
            # scipy.io.savemat(f'{save_dir}/XTConsistency/images.mat', dict(images=np.array(images)))
            # scipy.io.savemat(f'{save_dir}/XTConsistency/file_order.mat', dict(file_order=file_order))

            depths = np.transpose(np.array(depths), (0, 1, 2))
            normals = np.transpose(np.array(normals), (0, 2, 3, 1))

            output, params = self.compute_distance_orientation_bins(normals, depths)
            
            np.save(f'{save_dir}/mark/feats.npy', np.array(output))
            np.save(f'{save_dir}/mark/file_order.npy', file_order)

            

        if do_alexnet:

            # Sequential(
            # (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            # (1): ReLU(inplace=True)
            # (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            # (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
            # (4): ReLU(inplace=True)
            # (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            # (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (7): ReLU(inplace=True)
            # (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (9): ReLU(inplace=True)
            # (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # (11): ReLU(inplace=True)
            # (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            # )]

            # ReLU_7 = np.array(ReLU_7)
            # Conv2d_6 = np.array(Conv2d_6)
            # np.save(f'{save_dir}/alexnet/alexnet_Conv2d_0.npy', np.array(feats[0]))
            # np.save(f'{save_dir}/alexnet/alexnet_ReLU_1.npy', np.array(feats[1]))
            # np.save(f'{save_dir}/alexnet/alexnet_MaxPool2d_2.npy', np.array(feats[2]))
            # np.save(f'{save_dir}/alexnet/alexnet_Conv2d_3.npy', np.array(feats[3]))
            # np.save(f'{save_dir}/alexnet/alexnet_ReLU_4.npy', np.array(feats[4]))
            # np.save(f'{save_dir}/alexnet/alexnet_MaxPool2d_5.npy', np.array(feats[5]))
            # np.save(f'{save_dir}/alexnet/alexnet_Conv2d_6.npy', np.array(feats[6]))
            # np.save(f'{save_dir}/alexnet/alexnet_ReLU_7.npy', np.array(feats[7]))
            # np.save(f'{save_dir}/alexnet/alexnet_Conv2d_8.npy', np.array(feats[8]))
            # np.save(f'{save_dir}/alexnet/alexnet_ReLU_9.npy', np.array(feats[9]))
            # np.save(f'{save_dir}/alexnet/alexnet_Conv2d_10.npy', np.array(feats[10]))
            # np.save(f'{save_dir}/alexnet/alexnet_ReLU_11.npy', np.array(feats[11]))
            # np.save(f'{save_dir}/alexnet/alexnet_MaxPool2d_12.npy', np.array(feats[12]))

            for name_idx in range(len(self.alexnet_names)):
                name = self.alexnet_names[name_idx]
                np.save(f'{save_dir}/alexnet2/{name}.npy', np.array(feats[name_idx]))

            # np.save(f'{save_dir}/alexnet_Conv2d_6.npy', Conv2d_6)

            np.save(f'{save_dir}/alexnet2/alexnet_file_order.npy', file_order)



        return True
    
# BELOW OBTAINED FROM: Mark Lescroart 
    def compute_distance_orientation_bins(self,normals,
                                        distance,
                                        camera_vector=None,
                                        norm_bin_centers=NORM_BIN_CENTERS,
                                        dist_bin_edges=DIST_BIN_EDGES,
                                        sky_channel=True,
                                        remove_camera_rotation=False,
                                        assure_normals_equal_1=True,
                                        pixel_norm=False,
                                        dist_normalize=False,
                                        n_bins_x=1,
                                        n_bins_y=1,
                                        ori_norm=2,
                                        ):
        """Compute % of pixels in specified distance & orientation bins
        Preprocessing for normal & depth map images to compute scene features
        for Lescroart & Gallant, 2018
        Parameters
        ----------
        normals: array
            4D array of surface normal images (x, y, xyz_normal, frames)
        distance: array
            3D array of distance images (x, y, frames)
        norm_bin_centers: array
            array of basis vectors that specify the CENTERS (not edges) of bins for
            surface orientation
        dist_bin_edges: array
            array of EDGES (not centers) of distance bins
        sky_channel: bool
        If true, include a separate channel for sky (all depth values above max
        value in dist_bin_edges)
        dist_normalize
        remove_camera_rotation: bool or array of bools
            whether to remove X, Y, or Z rotation of camera. Defaults to False
            (do nothing to any rotation)
        ori_norm: scalar
            How to normalize norms (??): 1 = L1 (max), 2 = L2 (Euclidean)
        """
        # Cleanup of some distance files
        distance[np.isnan(distance)] = 1000

        bins_x = np.linspace(0, 1, n_bins_x+1)
        bins_x[-1] = np.inf
        bins_y = np.linspace(0, 1, n_bins_y+1)
        bins_y[-1] = np.inf

        # Computed parameters
        n_norm_bins = norm_bin_centers.shape[0]
        n_dist_bins = dist_bin_edges.shape[0] - 1
        # Normalize bin vectors
        L2norm = np.linalg.norm(norm_bin_centers, axis=1, ord=2)
        norm_bin_centers = norm_bin_centers / L2norm[:, np.newaxis]

        # Note, that the bin width for these bins will not be well-defined (or,
        # will not be uniform). For now, take the average min angle between bins
        if n_norm_bins == 1:
            # Normals can't deviate by more than 90 deg (unless they're un-
            # rotated) BUT: We don't actually want to soft bin if there is only one
            # normal, we want to assign ALL pixels EQUALLY to the ONE BIN.
            # Thus d = np.inf
            d = np.inf
        else:
            # Soft-bin normals
            d = np.arccos(norm_bin_centers.dot(norm_bin_centers.T))
            d[np.abs(d) < 0.00001] = np.nan
        norm_bin_width = np.mean(np.nanmin(d))
        # Add an extra buffer to this? We don't want "stray" pixels with normals
        # that don't fall into any bin (but we also don't want to double-count
        # pixels)

        # Optionaly remove any camera rotations
        if remove_camera_rotation is False:
            remove_camera_rotation = np.array([False, False, False])
        if np.any(remove_camera_rotation):
            normals = remove_rotation(normals,
                                    -camera_angles,
                                    is_normalize_normals=is_normalize_normals,
                                    angle_to_remove=remove_camera_rotation)
        # Get number of images
        n_ims, x, y  = distance.shape
        n_tiles = n_bins_y * n_bins_x
        n_dims = n_tiles * n_dist_bins * n_norm_bins
        if sky_channel:
            n_dims = n_dims + n_tiles
        else:
            n_dims = n_tiles * n_dist_bins * n_norm_bins

        output = np.zeros((n_ims, n_dims)) * np.nan
        for iS in range(n_ims):
            if n_ims>200:
                if iS % 200 == 0:
                    print("Done to image %d / %d"%(iS, n_ims)) #progressdot(iS,200,2000,n_ims)
            elif (n_ims < 200) and (n_ims > 1):
                print('computing Scene Depth Normals...')
            # Pull single image for preprocessing
            z = distance[iS]
            n = normals[iS]
            height, width, nd = n.shape
            xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            idx = np.arange(n_norm_bins)
            for d_st, d_fin in zip(dist_bin_edges[:-1], dist_bin_edges[1:]):
                dIdx = (z >= d_st) & (z < d_fin)
                for ix in range(n_bins_x):
                    hIdx = (xx >= bins_x[ix]) & (xx < bins_x[ix+1])
                    for iy in range(n_bins_y):
                        vIdx = (yy >= bins_y[iy]) & (yy < bins_y[iy + 1])
                        this_section = (dIdx & hIdx) & vIdx
                        if this_section.sum()==0:
                            output[iS, idx] = 0
                            idx += n_norm_bins
                            continue
                        if n_norm_bins > 1:
                            nn = n[this_section, :]
                            # Compute orientation of pixelwise surface normals relative
                            # to all normal bins
                            o = nn.dot(norm_bin_centers.T)
                            #print(o.shape)
                            #L2nn = np.linalg.norm(nn, axis=1, ord=2)
                            #o = bsxfun(@rdivide,o,Lb) # Norm of norm_bin_centers should be 1
                            o /= np.linalg.norm(nn, axis=1, ord=2)[:, np.newaxis]
                            if np.max(o-1) > 0.0001:
                                raise Exception('The magnitude of one of your normal bin vectors crossed with a stimulus normal is > 1 - Check on your stimulus / normal vectors!')
                            # Get rid of values barely > 1 to prevent imaginary output
                            o = np.minimum(o, 1)  
                            angles = np.arccos(o)
                            # The following is a "soft" histogramming of normals.
                            # i.e., if a given normal falls partway between two
                            # normal bins, it is partially assigned to each of the
                            # nearest bins (not exclusively to one).
                            tmp_out = np.maximum(0, norm_bin_width - angles) / norm_bin_width
                            # Sum over all pixels w/ depth in this range
                            tmp_out = np.sum(tmp_out, axis=0)
                            # Normalize across different normal orientation bins
                            tmp_out = tmp_out / np.linalg.norm(tmp_out, ord=ori_norm)
                            if pixel_norm:
                                tmp_out = tmp_out * len(nn)/len(dIdx.flatten())
                        else:
                            # Special case: one single normal bin
                            # compute the fraction of screen pixels in this screen
                            # tile at this depth
                            tmp_out = np.mean(this_section)
                        # Illegal for more than two bins of normals within the same
                        # depth / horiz/vert tile to be == 1
                        if sum(tmp_out == 1) > 1:
                            error('Found two separate normal bins equal to 1 - that should be impossible!')
                        if dist_normalize and not (n_norm_bins == 1):
                            # normalize normals by n pixels at this depth/screen tile
                            tmp_out = tmp_out * pct_pix_this_depth
                        output[iS, idx] = tmp_out
                        idx += n_norm_bins
            # Do sky channel(s) after last depth channel, add (n tiles) sky channels
            if sky_channel and (np.max(dist_bin_edges) < np.inf):
                dSky = z >= dist_bin_edges[-1]
                skyidx = np.arange((n_dims - n_tiles), n_dims)
                tmp = np.zeros((n_bins_y, n_bins_x))
                for x_st, x_fin in zip(bins_x[:-1], bins_x[1:]):
                    hIdx = (xx >= x_st) & (xx < x_fin)
                    for y_st, y_fin in zip(bins_y[:-1], bins_y[1:]):
                        vIdx = (yy >= y_st) & (yy < y_fin)
                        tmp[iy, ix] = np.mean(dSky & hIdx & vIdx)
                output[iS, skyidx] = tmp.flatten()

        # Cleanup
        output[np.isnan(output)] = 0
        params = dict(
            norm_bin_centers=norm_bin_centers,
            dist_bin_edges=dist_bin_edges,
            sky_channel=sky_channel,
            remove_camera_rotation=remove_camera_rotation,
            assure_normals_equal_1=assure_normals_equal_1,
            pixel_norm=pixel_norm,
            dist_normalize=dist_normalize,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            ori_norm=ori_norm,
            )
        return output, params


    def remove_rotation(self,normals, camera_vector, angle_to_remove=(True, False, False), do_normalize=True):
        """Remove rotation about one or more axes from normals
        Parameters
        ----------
        normals : array
            array of normals
        camera_vector : array
            vector to 
        angle_to_remove: tuple or list
            list of boolean values indicating whether to remove [x, y, z] rotations
        do_normalize: bool
            whether to re-normalize angles after rotation is removed.
        """
        # Load normals for test scene, w/ camera moving around square block:
        n_frames, vdim, hdim, n_channels = normals.shape
        output = np.zeros_like(normals)

        if do_normalize:
            camera_vector /= np.linalg.norm(camera_vector, axis=1)

        for frame in range(n_frames):
            if frame % 100 == 0:
                print("Done to frame %d/%d"%(frame, n_frames))
            c_vec = camera_vector[frame]
            camera_matrix = vector_to_camera_matrix(c_vec, ~angle_to_remove)
            n = normals[frame].reshape(-1, 3) 
            nT = camera_matrix.dot(n.T)
            nn = nT.T.reshape((vdim, hdim, n_channels))
            output[frame] = nn
        return output


    def vector_to_camera_matrix(self,c_vec, ignore_rot_xyz=(False, True, False)):
        """Gets the camera (perspective) transformation matrix, given a vector
        Vector should be from camera->fixation.  Optionally, sets one (or more)
        axes of rotation for the camera to zero (this provides a matrix that
        "un-rotates" the camera perspective, but leaves whatever axes are set to
        "true" untouched (i.e., in their original image space).
        Deals with ONE VECTOR AT A TIME
        IgnoreRot = [false,true,false] by default (there should be no y rotation
        [roll] of cameras anyway!)
        """
        xr, yr, zr = (~np.array(ignore_rot_xyz)).astype(np.bool)
        # Vector to Euler angles:
        if xr:
            xr = np.arctan2(c_vec[2], (c_vec[0]**2 + c_vec[1]**2)**0.5)
        if yr:
            raise Exception("SORRY I don't compute y rotations! Pls consult wikipedia!")
        else:
            yr = 0
        if zr:
            zr = -np.arctan2(c_vec[0], c_vec[1])
        # Rotation matrices, given Euler angles:
        # X rotation
        xRot = np.array([[1., 0., 0.],
                        [0., np.cos(xr), np.sin(xr)],
                        [0., -np.sin(xr), np.cos(xr)]])
        # Y rotation
        yRot = np.array([[np.cos(yr), 0., -np.sin(yr)],
                        [0., 1., 0.],
                        [np.sin(yr), 0., np.cos(yr)]])
        # Z rotation
        zRot = np.array([[np.cos(zr), np.sin(zr), 0.],
                        [-np.sin(zr), np.cos(zr), 0.],
                        [0., 0., 1.]])
        # Multiply rotations to get final matrix
        camera_matrix = xRot.dot(yRot).dot(zRot)
        return camera_matrix


    def circ_dist(self,a, b):
        """Angle between two angles, all in radians
        """
        phi = np.e**(1j*a) / np.e**(1j*b)
        ang_dist = np.arctan2(phi.imag, phi.real)
        return ang_dist


    def tilt_slant(self,img, make_1d=False):
        """Convert a pixelwise surface normal image into tilt, slant values
        Parameters
        ----------
        nimg: array
            Pixelwise normal image, [x,y,3] - 3rd dimension should represent 
            the surface normal (x,y,z vector, summing to 1) at each pixel
        """
        sky = np.all(img==0, axis=2)
        # Tilt
        tau = np.arctan2(img[:,:,2], img[:,:,0])
        # Slant
        sig = np.arccos(img[:,:,1])
        tau[sky] = np.nan
        sig[sky] = np.nan
        tau = circ_dist(tau, -np.pi / 2) + np.pi
        if make_1d:
            tilt = tau[~np.isnan(tau)].flatten()
            slant = sig[~np.isnan(sig)].flatten()
            return tilt, slant
        else:
            return tau, sig


    def norm_color_image(self,nimg, cmap=RET, vmin_t=0, vmax_t=2 * np.pi,
                        vmin_s=0, vmax_s=np.pi/2):
        """Convert normal image to colormapped normal image"""
        from matplotlib.colors import Normalize
        tilt, slant = tilt_slant(nimg, make_1d=False)
        # Normalize tilt (-pi to pi) -> (0, 1)
        norm_t = Normalize(vmin=vmin_t, vmax=vmax_t, clip=True)
        # Normalize slant (0 to pi/2) -> (0, 1)
        norm_s = Normalize(vmin=vmin_s, vmax=vmax_s, clip=True)
        # Convert normalized tilt to RGB color
        tilt_rgb_orig = cmap(norm_t(tilt))
        # Convert to HSV, replace saturation w/ normalized slant value
        tilt_hsv = skcol.rgb2hsv(tilt_rgb_orig[...,:3])
        tilt_hsv[:,:,1] = norm_s(slant)
        # Convert back to RGB
        tilt_rgb = skcol.hsv2rgb(tilt_hsv)
        tilt_rgb = np.dstack([tilt_rgb, 1-np.isnan(slant).astype(np.float)])
        # Compute better alpha
        a_im = np.dstack([tilt_rgb_orig[...,:3], norm_s(slant)])
        aa_im = tilt_rgb_orig[...,:3] * norm_s(slant)[..., np.newaxis] + np.ones_like(tilt_rgb_orig[...,:3]) * 0.5 * (1-norm_s(slant)[...,np.newaxis])
        aa_im = np.dstack([aa_im, 1-np.isnan(tilt).astype(np.float)])
        
        return aa_im


    def tilt_slant_hist(self,tilt, slant, n_slant_bins = 30, n_tilt_bins = 90, do_log=True, 
                        vmin=None, vmax=None, H=None, ax=None, **kwargs):
        """Plot a polar histogram of tilt and slant values
        
        if H is None, computes & plots histogram of tilt & slant
        if H is True, computes histogram of tilt & slant & returns histogram count
        if H is a value, plots histogram of H"""
        if (H is None) or (H is True) or (H is False):
            return_h = H is True
            tbins = np.linspace(0, 2*np.pi, n_tilt_bins)      # 0 to 360 in steps of 360/normals.
            sbins = np.linspace(0, np.pi/2, n_slant_bins) 
            H, xedges, yedges = np.histogram2d(tilt, slant, bins=(tbins,sbins), normed=True) #, weights=pwr)
            #H /= H.sum()
            if do_log:
                #print(H.shape)
                H = np.log(H)
                #H[np.isinf(H)] = np.nan
            if return_h:
                return H

        if do_log:
            if vmin is None:
                vmin=-8
            if vmax is None:
                vmax = 4

        e1 = n_tilt_bins * 1j
        e2 = n_slant_bins * 1j

        # Grid to plot your data on using pcolormesh
        theta, r = np.mgrid[0:2*np.pi:e1, 0:np.pi/2:e2]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        pc = ax.pcolormesh(theta, r, H, vmin=vmin, vmax=vmax, **kwargs)
        # Remove yticklabels, set limits
        ax.set_ylim([0, np.pi/2])
        ax.set_theta_offset(-np.pi/2)
        if ax is None:
            plt.colorbar(pc)


    def show_sdn(self,wts, params, mn_mx=None, lw=1, cmap=bcwora, ax=None, show_axis=False, 
                azim=-80, elev=10, dst_spacing=3, pane_scale=1, cbar=False):
        """Show scene distance/normal model channels
        """
        # forget tiled models for now - they don't work anyway.
        #if params['sky_channel']:
        #    sky = wts[-1]
        #    wts = wts[:-1]
        if mn_mx is None:
            # Default to min/max of wts, respecting zero
            mx = np.max(np.abs(wts)) * 0.8
            mn_mx = (-mx,mx)
        bin_centers = params['norm_bin_centers']
        nD = len(params['dist_bin_edges'])-1
        DstAdd = np.array([0,1,0]);
        
        # Base patch, facing -y direction
        base_patch = np.array([[-1,0,-1],[-1,0,1],[1,0,1],[1,0,-1],[-1,0,-1]])* pane_scale
        #wts[np.isnan(wts)] = 0;
        faces = []
        # Loop over different distances ...
        for iD in range(nD): #= 1:nD
            # ...and vectors in normal bin centers
            for iP,bc in enumerate(bin_centers):
                #ct = iP+iD*len(bin_centers)
                xyz = -dst_spacing*bc - DstAdd*iD;
                # re-set direction of normal vector to make coordinate conventions consistent
                xyz = xyz*np.array([1,-1,1])
                # rotate patch by camera transformation
                cam_mat = vector_to_camera_matrix(bc)
                patch_rot = cam_mat.dot(base_patch.T).T
                patch_rot_shift = xyz[None,:]+patch_rot
                # add patch to list of faces
                faces.append(patch_rot_shift)
        if params['sky_channel']:
            bc = np.array([0, 1, 0])
            xyz = -dst_spacing * bc - DstAdd*nD
            xyz = xyz * np.array([1, -1, 1])
            cam_mat = vector_to_camera_matrix(bc)
            patch_rot = cam_mat.dot(base_patch.T * 3).T
            patch_rot_shift = xyz[None,:] + patch_rot
            faces.append(patch_rot_shift)
        ## -- Set face colors -- ##
        norm = colors.Normalize(*mn_mx)
        cmapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cols = cmapper.to_rgba(wts)
        ## -- Plot poly collection -- ##
        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            fig = ax.get_figure()
        #print("faces are %d long"%len(faces))
        #print("colors are %d long"%len(cols))
        nfaces = len(faces)
        for ii,ff in enumerate(faces):
            polys = Poly3DCollection([ff], linewidths=lw)
            polys.set_facecolors([cols[ii]])
            polys.set_edgecolors([0.5, 0.5, 0.5])
            ax.add_collection3d(polys)
        xl,yl,zl = zip(np.min(np.vstack([f for f in faces]),axis=0),
                    np.max(np.vstack([f for f in faces]),axis=0))
        plt.setp(ax,xlim=xl,ylim=yl,zlim=zl)
        ax.view_init(azim=azim,elev=elev)
        if show_axis:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            ax.set_axis_off()
        if cbar:
            fig.colorbar(polys, ax=ax)


if __name__ == '__main__':
    model = BrainModel()
    OK = model.run_model()