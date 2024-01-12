# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmdet3d.structures import get_box_type  # noqa
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate  # noqa
from mmengine.model import BaseDataPreprocessor

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task
from .mmdet3d import MMDET3D_TASK


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """

    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        if 'metainfo' in dataset_cfg:
            return dataset_cfg.metainfo
    return None


@MMDET3D_TASK.register_module(Task.MV_DETECTION.value)
class MultiViewDetection(BaseTask):

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Sequence[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .multi_view_detection_model import \
            build_multi_view_detection_model

        data_preprocessor = deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        data_preprocessor.setdefault('type', 'mmdet3D.Det3DDataPreprocessor')

        model = build_multi_view_detection_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model

    def create_input(
        self,
        pcd: Union[str, Sequence[str]],
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for detector.

        Args:
            pcd (str, Sequence[str]): Input pcd file path.
            input_shape (Sequence[int], optional): model input shape.
                Defaults to None.
            data_preprocessor (Optional[BaseDataPreprocessor], optional):
                model input preprocess. Defaults to None.

        Returns:
            tuple: (data, input), meta information for the input pcd
                and model input.
        """
        # TODO rewrite this with a single sample
        from mmengine.runner import Runner

        from .multi_view_detection_model import transform_batch_inputs

        test_dataloader_cfg = deepcopy(self.model_cfg.test_dataloader)
        test_dataloader_cfg.batch_size = 1
        test_dataloader_cfg.num_workers = 0
        test_dataloader_cfg.persistent_workers = False
        test_dataloader = Runner.build_dataloader(test_dataloader_cfg, seed=42)
        for data in test_dataloader:
            break

        if data_preprocessor is None:
            from mmengine.registry import MODELS
            data_preprocessor = MODELS.build(
                self.model_cfg.model.data_preprocessor)

        data = data_preprocessor(data)
        inputs = transform_batch_inputs(self.model_cfg, data)
        return data, inputs

    def visualize(self,
                  image: Union[str, np.ndarray],
                  model: torch.nn.Module,
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  draw_gt: bool = False,
                  **kwargs):
        """visualize backend output.

        Args:
            image (Union[str, np.ndarray]): pcd file path
            model (torch.nn.Module): input pytorch model
            result (list): output bbox, score and type
            output_file (str): the directory to save output
            window_name (str, optional): display window name
            show_result (bool, optional): show result or not.
                Defaults to False.
            draw_gt (bool, optional): show gt or not. Defaults to False.
        """
        # TODO
        pass

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        raise NotImplementedError

    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmdet.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        raise NotImplementedError

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        raise NotImplementedError
