from model.feature_octree_base import FeatureOctreeBase

from utils.config import SHINEConfig

class SHINEFeatureOctree(FeatureOctreeBase):

    def __init__(self, config: SHINEConfig):

        super().__init__(config)