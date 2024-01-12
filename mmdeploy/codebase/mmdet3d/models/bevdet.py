# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_ir_config


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d_custom.models.detectors.bevdet.BEVDet.forward')
def bevdet__forward(self, imgs, ranks_bev, ranks_depth, ranks_feat,
                    interval_starts, interval_lengths, mlp_inputs, **kwargs):
    """Rewrite this func to utilize TRTBEVPoolv2.

    Args:
        imgs (torch.Tensor): Input images.
        ranks_bev (torch.Tensor): Ranks of BEV features.
        ranks_depth (torch.Tensor): Ranks of depth features.
        ranks_feat (torch.Tensor): Ranks of intermediate features.
        interval_starts (torch.Tensor): Start indices of each interval.
        interval_lengths (torch.Tensor): Lengths of each interval.
        mlp_inputs (torch.Tensor): MLP inputs.

    Returns:
        tuple: A tuple of classification scores, bbox and direction
            classification prediction.

            - cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, the channels number
                is num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, the channels number
                is num_base_priors * C.
            - dir_cls_preds (list[Tensor|None]): Direction classification
                predictions for all scale levels, each is a 4D-tensor,
                the channels number is num_base_priors * 2.
    """
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg

    # Inference
    img_feat = self.image_encoder(imgs)
    B, N, C, H, W = img_feat.shape
    img_feat = img_feat.view(B * N, C, H, W)
    img_feat = self.vtransform.depth_net(img_feat, mlp_inputs)
    depth = img_feat[:, :self.vtransform.D].softmax(dim=1)
    tran_feat = img_feat[:, self.vtransform.D:(self.vtransform.D +
                                               self.vtransform.out_channels)]
    tran_feat = tran_feat.view(B, N, self.vtransform.out_channels, H, W)
    tran_feat = tran_feat.permute(0, 1, 3, 4, 2)
    depth = depth.view(B, N, self.vtransform.D, H, W)
    tran_feat = tran_feat.squeeze(0)
    depth = depth.squeeze(0)
    x = self.trt_bev_pool_apply(depth, tran_feat, ranks_depth, ranks_feat,
                                ranks_bev, interval_starts, interval_lengths)
    x = x.permute(0, 3, 1, 2).contiguous()
    bev_feat = self.bev_encoder(x)
    outs = self.pts_bbox_head(bev_feat)

    # Slight post-processing
    if type(outs[0][0]) is dict:
        bbox_preds, scores, dir_scores = [], [], []
        for task_res in outs:
            bbox_preds.append(task_res[0]['reg'])
            bbox_preds.append(task_res[0]['height'])
            bbox_preds.append(task_res[0]['dim'])
            if 'vel' in task_res[0].keys():
                bbox_preds.append(task_res[0]['vel'])
            scores.append(task_res[0]['heatmap'])
            dir_scores.append(task_res[0]['rot'])
        bbox_preds = torch.cat(bbox_preds, dim=1)
        scores = torch.cat(scores, dim=1)
        dir_scores = torch.cat(dir_scores, dim=1)
        return scores, bbox_preds, dir_scores
    else:
        preds = []
        expect_names = []
        for i in range(len(outs[0])):
            preds += [outs[0][i], outs[1][i], outs[2][i]]
            expect_names += [
                f'cls_score{i}', f'bbox_pred{i}', f'dir_cls_pred{i}'
            ]
        # check if output_names is set correctly.
        onnx_cfg = get_ir_config(deploy_cfg)
        output_names = onnx_cfg['output_names']
        if output_names != list(expect_names):
            raise RuntimeError(f'`output_names` should be {expect_names} '
                               f'but given {output_names}\n'
                               f'Deploy config:\n{deploy_cfg.pretty_text}')
        return tuple(preds)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d_custom.models.necks.vtransformer.LSSDepthTransform.forward')
def lss_depth_transform__forward(self, args, **kwargs):
    """Rewrite this func to be a placeholder."""
    pass
