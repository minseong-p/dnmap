import sys
import torch

from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer, random_color_table

from model.dnmap_feature_octree import DNMapFeatureOctree as FeatureOctree
from model.decoder import Decoder

def map_visualization():

    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        sys.exit(
            "Please provide the path to the experiment folder."
        )

    config: SHINEConfig = torch.load(
        os.path.join(experiment_path, "config.pth"),
    )

    # initialize the feature octree
    octree = FeatureOctree(config)
    octree.load_model(experiment_path)

    # initialize the mlp decoder
    geo_mlp = Decoder(config, is_geo_encoder=True)
    sem_mlp = Decoder(config, is_geo_encoder=False)
    loaded_decoder = torch.load(
        os.path.join(experiment_path, "neural_map", "decoders.pth"),
    )
    geo_mlp.load_state_dict(loaded_decoder["geo_decoder"])

    mesher = Mesher(config, octree, geo_mlp, sem_mlp)

    vis = MapVisualizer()

    mesh_path = experiment_path + '/mesh/mesh_vis.ply'
    map_path = experiment_path + '/map/sdf_map_vis.ply'

    cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
    vis.update_mesh(cur_mesh)
    vis.stop()

    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    map_visualization()