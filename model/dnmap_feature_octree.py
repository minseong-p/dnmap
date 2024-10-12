import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import numpy as np

from model.feature_octree_base import FeatureOctreeBase

from utils.config import DNMapConfig

class DecompositionBasedEmbeddingSpace(nn.Module):
    def __init__(self, bitwidth, embedding_dims, baked=False):
        super().__init__()
        self.bitwidth = bitwidth
        self.embedding_dims = embedding_dims

        # Component vector set (embedding offsets and bias)
        self.embedding_offsets = nn.Parameter(0.01 * torch.randn(self.embedding_dims, 2 * self.bitwidth))
        self.embedding_bias = nn.Parameter(0.01 * torch.randn(self.embedding_dims))

        self.baked = baked

    def reinit_embedding_offsets(self):
        self.embedding_offsets = nn.Parameter(0.01 * torch.randn(self.embedding_dims, self.bitwidth))

    def bake(self):
        if not self.baked:
            assert self.embedding_offsets.shape[1] == 2 * self.bitwidth
            self.eval()
            embedding_offsets_0 = self.embedding_offsets[:,:self.bitwidth]
            embedding_offsets_1 = self.embedding_offsets[:,self.bitwidth:] 
            embedding_bias = self.embedding_bias + torch.sum(embedding_offsets_0, dim=1, keepdim=False)
            
            self.embedding_offsets = nn.Parameter(embedding_offsets_1 - embedding_offsets_0)
            self.embedding_bias = nn.Parameter(embedding_bias)

            self.baked = True

    def forward(self, cmps: torch.Tensor):
        return F.linear(cmps, self.embedding_offsets, self.embedding_bias)

class DNMapFeatureOctree(FeatureOctreeBase):

    def __init__(self, config: DNMapConfig):

        super().__init__(config)

        self.bitwidth = config.bitwidth
        self.item_dim = self.bitwidth
        self.use_continuous = config.use_continuous
        self.efficient_implementation = config.efficient_implementation

        self.continuous_embedding_level = config.continuous_embedding_level
        self.continuous_embeddings = None

        embedding_space = []
        for l in range(self.featured_level_num):
            embedding_space.append(DecompositionBasedEmbeddingSpace(self.bitwidth, self.feature_dim))
        self.embedding_space = nn.ModuleList(embedding_space)

        self.baked = False
        self.to(config.device)

    def query_feature_with_indices(self, coord, hierarchical_indices):
        out_features = torch.zeros(coord.shape[0], self.feature_dim, device=self.device)
        for l in range(self.featured_level_num): # for each level
            cur_level = self.max_level - l
            feature_level = self.featured_level_num-l-1
            # Interpolating
            # get the interpolation coefficients for the 8 neighboring corners, corresponding to the order of the hierarchical_indices
            coeffs = self.interpolate(coord,cur_level)

            # Continuous embedding
            if self.use_continuous:
                if feature_level == self.continuous_embedding_level:
                    out_features += (self.continuous_embeddings[hierarchical_indices[l]]*coeffs).sum(1)

            # Composition
            # cmps: composition indicators
            if self.baked:
                inp_cmps = self.hierarchical_items[feature_level]
            else:
                soft_cmps = torch.sigmoid(self.hierarchical_items[feature_level])
                hard_cmps = torch.zeros_like(soft_cmps)
                hard_cmps[soft_cmps > 0.5] = 1.0
                cmps = soft_cmps + (hard_cmps - soft_cmps).detach()
                inp_cmps = torch.cat((1-cmps, cmps), dim=1)
            
            if self.efficient_implementation:
                inp_cmps = (inp_cmps[hierarchical_indices[l]]*coeffs).sum(1)
                out_features += self.embedding_space[l](inp_cmps)
            else:
                cur_features = self.embedding_space[l](inp_cmps)
                out_features += (cur_features[hierarchical_indices[l]]*coeffs).sum(1)

        return out_features

    def bake(self):
        self.eval()
        # Binarize the composition indicators      
        for l in range(self.featured_level_num):
            cur_items = torch.sigmoid(self.hierarchical_items[l])
            cmps = torch.zeros_like(cur_items)
            cmps[cur_items > 0.5] = 1.0
            self.hierarchical_items[l] = nn.Parameter(cmps)

        # Bake the embedding space
        for l in range(self.featured_level_num):
            self.embedding_space[l].bake()

        self.baked = True

    def save_items(self, experiment_path):
        if not self.baked:
            self.bake()

        for l in range(self.featured_level_num):
            cmps = self.hierarchical_items[l].to(torch.uint8)
            cmps = cmps.detach().cpu().numpy()
            cmps = np.packbits(cmps.flatten())
            with open(os.path.join(experiment_path, "neural_map", "cmps_%d.bin" % l), "wb") as f:
                f.write(cmps.tobytes())
        
        if self.use_continuous:
            torch.save(
                self.continuous_embeddings,
                os.path.join(experiment_path, "neural_map", "continuous_embedding.pth")
            )
        
        torch.save(
            self.embedding_space.state_dict(),
            os.path.join(experiment_path, "neural_map", "embedding_space.pth")
        )
    
    def load_items(self, experiment_path):
        for l in range(self.featured_level_num):
            self.embedding_space[l].reinit_embedding_offsets()
            self.embedding_space[l].baked = True
        self.baked = True

        # Load composition indicators from saved binary files
        for i in range(self.max_level+1):            
            if i < self.free_level_num:
                continue
            item_size = len(self.corners_lookup_tables[i]) + 1
            with open(os.path.join(experiment_path, "neural_map", "cmps_%d.bin" % (i-self.free_level_num)), "rb") as f:
                cmps = f.read()
                cmps = np.unpackbits(np.frombuffer(cmps, dtype=np.uint8))
                cmps = torch.tensor(cmps[:item_size*self.bitwidth], dtype=torch.uint8).reshape(-1, self.bitwidth)
            self.hierarchical_items.append(nn.Parameter(cmps.to(torch.float32)))

        if self.use_continuous:
            self.continuous_embeddings = nn.Parameter(torch.load(os.path.join(experiment_path, "neural_map", "continuous_embedding.pth")))
        
        self.embedding_space.load_state_dict(
            torch.load(os.path.join(experiment_path, "neural_map", "embedding_space.pth"))
        )
        self.to(self.device)
    
    def update_items(self, i, new_item_size, incremental_on):
        super().update_items(i, new_item_size, incremental_on)
        # Update continuous embedding
        if self.use_continuous:
            if (i-self.free_level_num) == self.continuous_embedding_level:
                new_items = self.feature_std*torch.randn(new_item_size + 1, self.feature_dim, device=self.device)
                new_items[-1] = torch.zeros(1,self.feature_dim)
                if len(self.corners_lookup_tables[i]) == new_item_size:
                    self.continuous_embeddings = nn.Parameter(new_items)
                else:
                    continuous_embeddings = torch.cat((self.continuous_embeddings[:-1],new_items),0)
                    self.continuous_embeddings = nn.Parameter(continuous_embeddings)
    
    def sort_items(self, i, items_indices):
        super().sort_items(i, items_indices)
        if self.use_continuous:
            if (i-self.free_level_num) == self.continuous_embedding_level:
                self.continuous_embeddings = nn.Parameter(self.continuous_embeddings[items_indices])

    def set_zero(self):
        if not self.baked:
            super().set_zero()

    def print_detail(self):
        print("Current Octomap:")
        total_vox_count = 0
        for level in range(self.featured_level_num):
            level_vox_size = self.leaf_vox_size*(2**(self.featured_level_num-1-level))
            level_vox_count = self.hierarchical_items[level].shape[0]
            print("%.2f m: %d voxel corners" %(level_vox_size, level_vox_count))
            total_vox_count += level_vox_count

        discrete_map_memory = total_vox_count * self.bitwidth / 8 / 1024 # unit: kB
        print("discrete representation memory: %.3f kB" %(discrete_map_memory))

        embedding_space_memory = self.featured_level_num * (self.bitwidth + 1) * self.feature_dim * 4 / 1024 # unit: MB
        print("embedding space memory: %.3f kB" %(embedding_space_memory))
        
        lrce_map_memory = 0.0
        if self.use_continuous:
            lrce_vox_count = self.continuous_embeddings.shape[0]
            lrce_map_memory = lrce_vox_count * self.feature_dim * 4 / 1024 # unit: MB
            print("continuous representation memory: %.3f kB" %(lrce_map_memory))

        print("total: %.3f kB" %(discrete_map_memory + embedding_space_memory + lrce_map_memory))
        print("--------------------------------")