import os

import torch
import torch.nn as nn

import kaolin as kal
import numpy as np

from collections import defaultdict

from utils.config import SHINEConfig

class FeatureOctreeBase(nn.Module):

    def __init__(self, config: SHINEConfig):
        
        super().__init__()

        # [0 1 2 3 ... max_level-1 max_level], 0 level is the root, which have 8 corners.
        self.max_level = config.tree_level_world 
        # the number of levels with feature (begin from bottom)
        self.leaf_vox_size = config.leaf_vox_size 
        self.featured_level_num = config.tree_level_feat 
        self.free_level_num = self.max_level - self.featured_level_num + 1
        self.feature_dim = config.feature_dim
        self.item_dim = self.feature_dim
        self.feature_std = config.feature_std
        self.polynomial_interpolation = config.poly_int_on
        self.device = config.device

        # Initialize the look up tables 
        self.corners_lookup_tables = [] # from corner morton to corner index (top-down)
        self.nodes_lookup_tables = []   # from nodes morton to corner index (top-down)
        # Initialize the look up table for each level, each is a dictionary
        for l in range(self.max_level+1):
            self.corners_lookup_tables.append({})
            self.nodes_lookup_tables.append({}) # actually the same speed as below
            # default_indices = [-1 for i in range(8)]
            # nodes_dict = defaultdict(lambda:default_indices)
            # self.nodes_lookup_tables.append(nodes_dict)
            
        # Initialize the hierarchical grid feature list 
        if self.featured_level_num < 1:
            raise ValueError('No level with grid features!')
        
        # top-down: leaf node level is stored in the last row (dim feature_level_num-1)
        # but only for the featured levels
        self.hierarchical_items = nn.ParameterList([])

        # the temporal stuffs that can be cleared
        # hierachical feature grid indices for the input batch point
        self.hierarchical_indices = [] 
        # bottom-up: stored from the leaf node level (dim 0)

        self.to(config.device)

    # the last element of the each level of the hier_features is the trashbin element
    # after the optimization, we need to set it back to zero vector
    def set_zero(self):
        with torch.no_grad():
            for n in range(len(self.hierarchical_items)):
                self.hierarchical_items[n][-1] = torch.zeros(1,self.item_dim) 

    def forward(self, x):
        feature = self.query_feature(x)
        return feature

    def get_morton(self, sample_points, level):
        points = kal.ops.spc.quantize_points(sample_points, level)  # quantize to interger coords
        points_morton = kal.ops.spc.points_to_morton(points) # to 1d morton code
        sample_points_with_morton = torch.hstack((sample_points, points_morton.view(-1, 1)))
        morton_set = set(points_morton.cpu().numpy())
        return sample_points_with_morton, morton_set

    def get_octree_nodes(self, level): # top-down
        nodes_morton = list(self.nodes_lookup_tables[level].keys())
        nodes_morton = torch.tensor(nodes_morton).to(self.device, torch.int64)
        nodes_spc = kal.ops.spc.morton_to_points(nodes_morton)
        nodes_spc_np = nodes_spc.cpu().numpy()
        node_size = 2**(1-level) # in the -1 to 1 kaolin space
        nodes_coord_scaled = (nodes_spc_np * node_size) - 1. + 0.5 * node_size  # in the -1 to 1 kaolin space
        return nodes_coord_scaled

    def is_empty(self):
        return len(self.hierarchical_items) == 0

    # clear the temp data (used for one batch) that is not needed
    def clear_temp(self):
        self.hierarchical_indices = []

    def update(self, surface_points, incremental_on = False):
        # [0 1 2 3 ... max_level-1 max_level]
        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(surface_points, self.max_level) 
        pyramid = spc.pyramids[0].cpu()
        for i in range(self.max_level+1): # for each level (top-down)            
            if i < self.free_level_num: # free levels (skip), only need to consider the featured levels
                continue
            nodes = spc.point_hierarchies[pyramid[1, i]:pyramid[1, i+1]] # nodes at certain level
            new_item_size = self.update_lookup_table(i, nodes) # update the corners & nodes lookup table
            if new_item_size > 0:
                self.update_items(i, new_item_size, incremental_on) # update the items
    
    def update_lookup_table(self, i, nodes, sort_corners = False):
        pre_size = len(self.corners_lookup_tables[i])
        nodes_morton = kal.ops.spc.points_to_morton(nodes).cpu().numpy().tolist() # nodes at certain level
        new_nodes_index = []
        for idx in range(len(nodes_morton)):
            if nodes_morton[idx] not in self.nodes_lookup_tables[i]:
                new_nodes_index.append(idx) # nodes to corner dictionary: key is the morton code
        new_nodes = nodes[new_nodes_index] # get the newly added nodes
        if new_nodes.shape[0] > 0:
            corners = kal.ops.spc.points_to_corners(new_nodes).reshape(-1,3)
            unique_corners = torch.unique(corners, dim=0)
            unique_corners_morton = kal.ops.spc.points_to_morton(unique_corners)
            if sort_corners:
                unique_corners_morton = torch.sort(unique_corners_morton)[0].cpu().numpy().tolist() # sort the morton code to restore the octree structure
            else:
                unique_corners_morton = unique_corners_morton.cpu().numpy().tolist()

            # update the corners lookup table
            if len(self.corners_lookup_tables[i]) == 0: # for the first frame
                corners_dict = dict(zip(unique_corners_morton, range(len(unique_corners_morton))))
                self.corners_lookup_tables[i] = corners_dict          
            else: # update for new frames
                for m in unique_corners_morton:
                    if m not in self.corners_lookup_tables[i]: # add new keys
                        self.corners_lookup_tables[i][m] = len(self.corners_lookup_tables[i])

            # update the nodes lookup table (from node morton code to 8-corner indices)                        
            corners_morton = kal.ops.spc.points_to_morton(corners).cpu().numpy().tolist()
            indexes = torch.tensor([self.corners_lookup_tables[i][x] for x in corners_morton]).reshape(-1,8).numpy().tolist()
            new_nodes_morton = kal.ops.spc.points_to_morton(new_nodes).cpu().numpy().tolist()
            for k in range(len(new_nodes_morton)):
                self.nodes_lookup_tables[i][new_nodes_morton[k]] = indexes[k]

        return len(self.corners_lookup_tables[i]) - pre_size # return the number of new corners

    def update_items(self, i, new_item_size, incremental_on = False):
        new_items = self.feature_std*torch.randn(new_item_size + 1, self.item_dim, device=self.device)
        new_items[-1] = torch.zeros(1,self.item_dim)
        if len(self.corners_lookup_tables[i]) == new_item_size:
            self.hierarchical_items.append(nn.Parameter(new_items))
        else:
            items = torch.cat((self.hierarchical_items[i-self.free_level_num][:-1],new_items),0)
            self.hierarchical_items[i-self.free_level_num] = nn.Parameter(items)
        
    # tri-linear (or polynomial) interplation of feature at certain octree level at certain spatial point x 
    def interpolate(self, x, level):
        coords = ((2**level)*(x*0.5+0.5)) 
        d_coords = torch.frac(coords)
        if self.polynomial_interpolation:
            tx = 3*(d_coords[:,0]**2) - 2*(d_coords[:,0]**3)
            ty = 3*(d_coords[:,1]**2) - 2*(d_coords[:,1]**3)
            tz = 3*(d_coords[:,2]**2) - 2*(d_coords[:,2]**3)
        else: # linear 
            tx = d_coords[:,0]
            ty = d_coords[:,1]
            tz = d_coords[:,2]
        _1_tx = 1-tx
        _1_ty = 1-ty
        _1_tz = 1-tz
        p0 = _1_tx*_1_ty*_1_tz
        p1 = _1_tx*_1_ty*tz
        p2 = _1_tx*ty*_1_tz
        p3 = _1_tx*ty*tz
        p4 = tx*_1_ty*_1_tz
        p5 = tx*_1_ty*tz
        p6 = tx*ty*_1_tz
        p7 = tx*ty*tz

        p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.unsqueeze(2)
        return p

    # get the unique indices of the feature node at spatial points x for each level
    # TODO: speed up !!!
    def get_indices(self, coord):
        self.hierarchical_indices = [] # initialize the hierarchical indices list for the batch points x
        for i in range(self.featured_level_num): # bottom-up, for each level
            current_level = self.max_level - i
            points = kal.ops.spc.quantize_points(coord,current_level) # quantize to interger coords
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist() # convert to 1d morton code for the voxel center
            features_last_row = [-1 for t in range(8)] # if not in the look up table, then assign all -1
            # look up the 8 corner nodes' unique indices for each 1d morton code in the look up table [nx8], the most time-consuming part 
            # [actually a kind of hashing realized by python dictionary]
            
            indices_list = [self.nodes_lookup_tables[current_level].get(p,features_last_row) for p in points_morton] 
            # indices_list = [self.nodes_lookup_tables[current_level][p] for p in points_morton]  # actually the same speed as above when using the defaultdict
            
            # if p is not found in the key lists of cur_lookup_table, use features_last_row, 
            # which is the all-zero trashbin vector of the level's feature
            indices_torch = torch.tensor(indices_list, device=self.device) 
            self.hierarchical_indices.append(indices_torch) # l level {nx8}  # bottom-up
        
        return self.hierarchical_indices

    # get the hierachical-sumed interpolated feature at spatial points x
    def query_feature_with_indices(self, coord, hierarchical_indices):
        # TODO: filter the -1 for faster back-propgation
        sum_features = torch.zeros(coord.shape[0], self.feature_dim, device=self.device)
        for l in range(self.featured_level_num): # for each level
            current_level = self.max_level - l
            item_level = self.featured_level_num-l-1
            # Interpolating
            # get the interpolation coefficients for the 8 neighboring corners, corresponding to the order of the hierarchical_indices
            coeffs = self.interpolate(coord,current_level,self.polynomial_interpolation) 
            sum_features += (self.hierarchical_items[item_level][hierarchical_indices[l]]*coeffs).sum(1) 
            # corner index -1 means the queried voxel is not in the leaf node. If so, we will get the trashbin row of the feature grid, 
            # and get the value 0, the feature for this level will then be 0
        return sum_features

    # all-in-one function to get the octree features for a batch of points
    def query_feature(self, coord, faster = False):
        self.set_zero() # set the trashbin feature vector back to 0 after the feature update
        if faster:
            indices = self.get_indices_fast(coord) # it would only be faster when the input coords have a lot share the same morton code
        else:
            indices = self.get_indices(coord)
        features = self.query_feature_with_indices(coord, indices)
        return features

    def list_duplicates(self, seq):
        dd = defaultdict(list)
        for i,item in enumerate(seq):
            dd[item].append(i)
        return [(key,locs) for key,locs in dd.items() if len(locs)>=1] 
                                
    # speed up for the batch sdf inferencing during meshing
    # points in the same voxel would be grouped and getting indices together
    # more efficient only when there are lots of samples from the same voxel in the batch (the case when conducting meshing)
    # This function contains some problem which would make the mesh worse, check it later (solved)
    def get_indices_fast(self, coord):
        self.hierarchical_indices = []
        for i in range(self.featured_level_num): # bottom-up
            current_level = self.max_level - i
            points = kal.ops.spc.quantize_points(coord,current_level) # quantize to interger coords
            points_morton = kal.ops.spc.points_to_morton(points).cpu().numpy().tolist() # convert to 1d morton code for the voxel center
            features_last_row = [-1 for t in range(8)] # if not in the look up table, then assign -1

            dups_in_mortons = dict(self.list_duplicates(points_morton)) # list the x with the same morton code (samples inside the same voxel)
            dups_indices = np.zeros((len(points_morton), 8))
            # print(len(dups_in_mortons.keys()), len(points_morton))
            for p in dups_in_mortons.keys():
                idx = dups_in_mortons[p] # indices, p is the point morton
                # get indices only once for these samples sharing the same voxel 
                corner_indices = self.nodes_lookup_tables[current_level].get(p,features_last_row)
                dups_indices[idx,:] = corner_indices
            indices = torch.tensor(dups_indices, device=self.device).long()
            self.hierarchical_indices.append(indices) # l level {nx8} 
        
        return self.hierarchical_indices
    
    def save_model(self, experiment_path):
        self.save_structure(experiment_path)
        self.sort_hierarchical_items()
        self.save_items(experiment_path)
    
    def load_model(self, experiment_path):
        self.load_structure(experiment_path)
        self.load_items(experiment_path)
        self.to(self.device)
    
    def save_structure(self, experiment_path):
        hierarchical_nodes = []
        for l in range(self.max_level+1):
            hierarchical_nodes.append(list(self.nodes_lookup_tables[l].keys()))
        
        torch.save(
            hierarchical_nodes,
            os.path.join(experiment_path, "neural_map", "structure.pth")
        )

    def load_structure(self, experiment_path):
        hierarchical_nodes_morton = torch.load(os.path.join(experiment_path, "neural_map", "structure.pth"))
        for i in range(self.max_level+1): # for each level (top-down)            
            if i < self.free_level_num: # free levels (skip), only need to consider the featured levels
                continue
            # level storing features (i>=free_level_num)
            nodes = kal.ops.spc.morton_to_points(torch.tensor(hierarchical_nodes_morton[i]).to(self.device))
            self.update_lookup_table(i, nodes, sort_corners = True)
    
    def save_items(self, experiment_path):
        torch.save(
            self.hierarchical_items,
            os.path.join(experiment_path, "neural_map", "item.pth")
        )
    
    def load_items(self, experiment_path):
        self.hierarchical_items = torch.load(os.path.join(experiment_path, "neural_map", "item.pth"))

    def sort_hierarchical_items(self):
        """
        Sorts the items in the hierarchical structure based on Morton codes.
        """
        for i in range(self.max_level+1): # for each level (top-down)            
            if i < self.free_level_num: # free levels (skip), only need to consider the featured levels
                continue
            corners_morton = list(self.corners_lookup_tables[i].keys())
            sorted_corners_morton, _ = torch.sort(torch.tensor(corners_morton))
            sorted_corners_morton = sorted_corners_morton.tolist()
            items_indices = []
            for m in sorted_corners_morton:
                items_indices.append(self.corners_lookup_tables[i][m])
            items_indices.append(-1)
            self.sort_items(i, items_indices)

    def sort_items(self, i, items_indices):
        self.hierarchical_items[i-self.free_level_num] = nn.Parameter(self.hierarchical_items[i-self.free_level_num][items_indices])

    def print_detail(self):
        print("Current Octomap:")
        total_vox_count = 0
        for level in range(self.featured_level_num):
            level_vox_size = self.leaf_vox_size*(2**(self.featured_level_num-1-level))
            level_vox_count = self.hierarchical_items[level].shape[0]
            print("%.2f m: %d voxel corners" %(level_vox_size, level_vox_count))
            total_vox_count += level_vox_count
        total_map_memory = total_vox_count * self.feature_dim * 4 / 1024 / 1024 # unit: MB
        print("memory: %d x %d x 4 = %.3f MB" %(total_vox_count, self.feature_dim, total_map_memory)) 
        print("--------------------------------")  