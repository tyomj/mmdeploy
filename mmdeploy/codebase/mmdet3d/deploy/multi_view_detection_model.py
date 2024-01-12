# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine import Config
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, InstanceData

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

__BACKEND_MODEL = Registry('backend_multi_view_detectors')


@__BACKEND_MODEL.register_module('end2end')
class MultiViewDetectionModel(BaseBackendModel):
    """End to end model for inference of multi-view 3D object detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        model_cfg (str | Config): The model config.
        deploy_cfg (str|Config): Deployment config file or loaded
            Config object.
        data_preprocessor (dict|torch.nn.Module): The input preprocessor
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: Union[str, Config],
                 deploy_cfg: Union[str, Config],
                 data_preprocessor: Optional[Union[dict,
                                                   torch.nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize backend wrapper.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        imgs, ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths, mlp_inputs = transform_batch_inputs(  # noqa: E501
            self.model_cfg, data)
        inputs = {
            'imgs': imgs,
            'ranks_bev': ranks_bev,
            'ranks_depth': ranks_depth,
            'ranks_feat': ranks_feat,
            'interval_starts': interval_starts,
            'interval_lengths': interval_lengths,
            'mlp_inputs': mlp_inputs,
        }
        input_dict = {
            'inputs': inputs,
            'data_samples': data['data_samples'],
        }
        return self._run_forward(input_dict, mode='predict')  # type: ignore

    def forward(self,
                inputs: dict,
                data_samples: Optional[List[BaseDataElement]] = None,
                **kwargs) -> Any:
        """Run forward inference.

        Args:
            inputs (dict): A dict of inputs
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).

        Returns:
            list: A list contains predictions.
        """
        input_dict = {
            'imgs': inputs['imgs'].to(self.device),
            'ranks_bev': inputs['ranks_bev'].to(self.device),
            'ranks_depth': inputs['ranks_depth'].to(self.device),
            'ranks_feat': inputs['ranks_feat'].to(self.device),
            'interval_starts': inputs['interval_starts'].to(self.device),
            'interval_lengths': inputs['interval_lengths'].to(self.device),
            'mlp_inputs': inputs['mlp_inputs'].to(self.device),
        }

        outputs = self.wrapper(input_dict)
        num_level = len(outputs) // 3
        new_outputs = dict(
            cls_score=[outputs[f'cls_score{i}'] for i in range(num_level)],
            bbox_pred=[outputs[f'bbox_pred{i}'] for i in range(num_level)],
            dir_cls_pred=[
                outputs[f'dir_cls_pred{i}'] for i in range(num_level)
            ])
        outputs = new_outputs
        if data_samples is None:
            return outputs

        prediction = MultiViewDetectionModel.postprocess(
            model_cfg=self.model_cfg,
            deploy_cfg=self.deploy_cfg,
            outs=outputs,
            metas=data_samples)

        return prediction

    @staticmethod
    def convert_to_datasample(
        data_samples: SampleList,
        data_instances_3d: Optional[List[InstanceData]] = None,
        data_instances_2d: Optional[List[InstanceData]] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

    @staticmethod
    def postprocess(model_cfg: Union[str, Config],
                    deploy_cfg: Union[str, Config], outs: Dict, metas: Dict):
        """postprocess outputs to datasamples.

        Args:
            model_cfg (Union[str, Config]): The model config from
                trainning repo
            deploy_cfg (Union[str, Config]): The deploy config to specify
                backend and input shape
            outs (Dict): output bbox, cls and score
            metas (Dict): DataSample3D for bbox3d render

        Raises:
            NotImplementedError: Only support mmdet3d model with `bbox_head`

        Returns:
            DataSample3D: datatype for render
        """
        if 'cls_score' not in outs or 'bbox_pred' not in outs or 'dir_cls_pred' not in outs:  # noqa: E501
            raise RuntimeError('output tensor not found')

        if 'test_cfg' not in model_cfg.model:
            raise RuntimeError('test_cfg not found')

        from mmengine.registry import MODELS
        cls_score = outs['cls_score']
        bbox_pred = outs['bbox_pred']
        dir_cls_pred = outs['dir_cls_pred']
        batch_input_metas = [data_samples.metainfo for data_samples in metas]

        head = None
        cfg = None
        if 'bbox_head' in model_cfg.model:
            # pointpillars postprocess
            model_cfg.model['bbox_head'].train_cfg = model_cfg.model.train_cfg
            model_cfg.model['bbox_head'].test_cfg = model_cfg.model.test_cfg
            head = MODELS.build(model_cfg.model['bbox_head'])
            cfg = model_cfg.model.test_cfg
        elif 'pts_bbox_head' in model_cfg.model:
            # centerpoint postprocess
            model_cfg.model[
                'pts_bbox_head'].train_cfg = model_cfg.model.train_cfg.pts
            model_cfg.model[
                'pts_bbox_head'].test_cfg = model_cfg.model.test_cfg.pts
            head = MODELS.build(model_cfg.model['pts_bbox_head'])
            cfg = model_cfg.model.test_cfg.pts
        else:
            raise NotImplementedError('mmdet3d model bbox_head not found')

        if not hasattr(head, 'task_heads'):
            data_instances_3d = head.predict_by_feat(
                cls_scores=cls_score,
                bbox_preds=bbox_pred,
                dir_cls_preds=dir_cls_pred,
                batch_input_metas=batch_input_metas,
                cfg=cfg)

            data_samples = MultiViewDetectionModel.convert_to_datasample(
                data_samples=metas, data_instances_3d=data_instances_3d)

        else:
            cls_score = cls_score[0]
            bbox_pred = bbox_pred[0]
            dir_cls_pred = dir_cls_pred[0]

            pts = model_cfg.model.test_cfg.pts

            rets = []
            scores_range = [0]
            bbox_range = [0]
            dir_range = [0]
            for i, _ in enumerate(head.task_heads):
                scores_range.append(scores_range[i] + head.num_classes[i])
                bbox_range.append(bbox_range[i] + 8)
                dir_range.append(dir_range[i] + 2)

            for task_id in range(len(head.num_classes)):
                num_class_with_bg = head.num_classes[task_id]

                batch_heatmap = cls_score[:,
                                          scores_range[task_id]:scores_range[
                                              task_id + 1], ...].sigmoid()

                batch_reg = bbox_pred[:,
                                      bbox_range[task_id]:bbox_range[task_id] +
                                      2, ...]
                batch_hei = bbox_pred[:, bbox_range[task_id] +
                                      2:bbox_range[task_id] + 3, ...]

                if head.norm_bbox:
                    batch_dim = torch.exp(bbox_pred[:, bbox_range[task_id] +
                                                    3:bbox_range[task_id] + 6,
                                                    ...])
                else:
                    batch_dim = bbox_pred[:, bbox_range[task_id] +
                                          3:bbox_range[task_id] + 6, ...]

                batch_vel = bbox_pred[:, bbox_range[task_id] +
                                      6:bbox_range[task_id + 1], ...]

                batch_rots = dir_cls_pred[:,
                                          dir_range[task_id]:dir_range[task_id
                                                                       + 1],
                                          ...][:, 0].unsqueeze(1)
                batch_rotc = dir_cls_pred[:,
                                          dir_range[task_id]:dir_range[task_id
                                                                       + 1],
                                          ...][:, 1].unsqueeze(1)

                temp = head.bbox_coder.decode(
                    batch_heatmap,
                    batch_rots,
                    batch_rotc,
                    batch_hei,
                    batch_dim,
                    batch_vel,
                    reg=batch_reg,
                    task_id=task_id)

                assert pts['nms_type'] in ['circle', 'rotate']
                batch_reg_preds = [box['bboxes'] for box in temp]
                batch_cls_preds = [box['scores'] for box in temp]
                batch_cls_labels = [box['labels'] for box in temp]
                if pts['nms_type'] == 'circle':
                    boxes3d = temp[0]['bboxes']
                    scores = temp[0]['scores']
                    labels = temp[0]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    from mmdet3d.models.layers import circle_nms
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            pts['min_radius'][task_id],
                            post_max_size=pts['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task = [ret]
                    rets.append(ret_task)
                else:
                    rets.append(
                        head.get_task_detections(
                            num_class_with_bg,
                            batch_cls_preds,
                            batch_reg_preds,
                            batch_cls_labels,
                            batch_input_metas,
                            task_id=task_id))

            # Merge branches results
            num_samples = len(rets[0])

            ret_list = []
            for i in range(num_samples):
                temp_instances = InstanceData()
                for k in rets[0][i].keys():
                    if k == 'bboxes':
                        bboxes = torch.cat([ret[i][k] for ret in rets])
                        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                        bboxes = batch_input_metas[i]['box_type_3d'](
                            bboxes, head.bbox_coder.code_size)
                    elif k == 'scores':
                        scores = torch.cat([ret[i][k] for ret in rets])
                    # Original CenterHead version
                    elif k == 'labels' and type(
                            head).__name__ != 'OutriderCenterHead':
                        flag = 0
                        for j, num_class in enumerate(head.num_classes):
                            rets[j][i][k] += flag
                            flag += num_class
                        labels = torch.cat([ret[i][k].int() for ret in rets])
                    # OutriderCenterHead version
                    elif k == 'labels' and type(
                            head).__name__ == 'OutriderCenterHead':
                        for task_idx, num_class in enumerate(head.num_classes):
                            mapping = {
                                key: head.prediction_classes.index(cls_name)
                                for key, cls_name in enumerate(
                                    head.class_names[task_idx])
                            }
                            assert num_class == len(mapping)
                            new_labels = [
                                mapping[cls_id.item()]
                                for cls_id in rets[task_idx][i][k]
                            ]
                            rets[task_idx][i][k] = torch.tensor(
                                new_labels,
                                dtype=torch.long,
                                device=rets[task_idx][i][k].device)
                        labels = torch.cat([ret[i][k].int() for ret in rets])
                temp_instances.bboxes_3d = bboxes
                temp_instances.scores_3d = scores
                temp_instances.labels_3d = labels
                ret_list.append(temp_instances)

            data_samples = MultiViewDetectionModel.convert_to_datasample(
                metas, data_instances_3d=ret_list)

        return data_samples


def build_multi_view_detection_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build multi-view 3d object detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.

    Returns:
        VoxelDetectionModel: Detector for a configured backend.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_detector = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_detector


def transform_batch_inputs(model_cfg, data):
    """Transform batch inputs to model inputs.

    Args:
        model_cfg (Config): Model config.
        data (dict): Batch input data.
        device (str): Device to input model.

    Returns:
        imgs: (torch.Tensor): Images.
        ranks_bev: (torch.Tensor): BEV voxel indices.
        ranks_depth: (torch.Tensor): Depth voxel indices.
        ranks_feat: (torch.Tensor): Feature voxel indices.
        interval_starts: (torch.Tensor): Voxel pooling interval starts.
        interval_lengths: (torch.Tensor): Voxel pooling interval lengths.
        mlp_inputs: (torch.Tensor): MLP inputs.
    """
    # Import detection model i.e. BEVDepth
    from mmdet3d.registry import MODELS
    model = MODELS.get(model_cfg.model.type)

    # Build Lift-splat-shoot depth transform
    vtransform_config = deepcopy(model_cfg.model.vtransform)
    vtransform_config['depth_net'] = None  # don't need nn parameters
    vtransform = MODELS.build(vtransform_config)

    batch_input_dict = data['inputs']
    batch_input_metas = [item.metainfo for item in data['data_samples']]

    # to tensors
    imgs, points, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix = model.prepare_inputs(  # noqa: E501
        batch_input_dict, batch_input_metas)

    # stack MLP inputs
    mlp_inputs = vtransform.get_mlp_inputs(
        camera2lidar=camera2lidar,
        intrins=camera_intrinsics[..., :3, :3],
        post_rots=img_aug_matrix[..., :3, :3],
        post_trans=img_aug_matrix[..., :3, 3],
        lidar_aug_matrix=lidar_aug_matrix[:, :, :3, :3])

    # Get coords
    coor = vtransform.get_lidar_coor(
        rots=camera2lidar[..., :3, :3],
        trans=camera2lidar[..., :3, 3],
        cam2imgs=camera_intrinsics[..., :3, :3],
        post_rots=img_aug_matrix[..., :3, :3],
        post_trans=img_aug_matrix[..., :3, 3])
    # Get bev pooling inputs
    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = vtransform.voxel_pooling_prepare_v2(  # noqa: E501
        coor)
    return imgs, ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths, mlp_inputs  # noqa: E501
