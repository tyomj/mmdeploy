_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='MultiViewDetection', model_type='end2end')
onnx_config = dict(
    input_names=[
        'imgs', 'ranks_bev', 'ranks_depth', 'ranks_feat', 'interval_starts',
        'interval_lengths', 'mlp_inputs'
    ],
    # need to change output_names for head with multi-level features
    output_names=['cls_score0', 'bbox_pred0', 'dir_cls_pred0'])
