from eval_utils import eval_mesh
import os

gt_pcd_path = "/mnt/hdd/mai_city/mai_city/gt_map_pc_mai.ply"

pred_mesh_path = "xxx.ply"

down_sample_vox = 0.02
dist_thre = 0.1
truncation_dist_acc = 0.2 
truncation_dist_com = 2.0

# down_sample_vox = 0.02
# dist_thre = 0.2
# truncation_dist_acc = 0.4
# truncation_dist_com = 2.0

# evaluation
eval_metric = eval_mesh(pred_mesh_path, gt_pcd_path, down_sample_res=down_sample_vox, threshold=dist_thre, 
                        truncation_acc = truncation_dist_acc, truncation_com = truncation_dist_com, gt_bbx_mask_on = True) 

print(eval_metric)

evals = [eval_metric]

csv_columns = ['MAE_accuracy (m)', 'MAE_completeness (m)', 'Chamfer_L1 (m)', 'Chamfer_L2 (m)', \
        'Precision [Accuracy] (%)', 'Recall [Completeness] (%)', 'F-score (%)', 'Spacing (m)', \
        'Inlier_threshold (m)', 'Outlier_truncation_acc (m)', 'Outlier_truncation_com (m)']
