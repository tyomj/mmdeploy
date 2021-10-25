import mmcv
import numpy as np
import pytest
import torch
from mmocr.models.textdet.necks import FPNC

from mmdeploy.utils.test import (WrapModel, get_model_outputs,
                                 get_rewrite_outputs)


class FPNCNeckModel(FPNC):

    def __init__(self, in_channels, init_cfg=None):
        super().__init__(in_channels, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.neck = FPNC(in_channels, init_cfg=init_cfg)

    def forward(self, inputs):
        neck_inputs = [
            torch.ones(1, channel, inputs.shape[-2], inputs.shape[-1])
            for channel in self.in_channels
        ]
        output = self.neck.forward(neck_inputs)
        return output


def get_bidirectionallstm_model():
    from mmocr.models.textrecog.layers.lstm_layer import BidirectionalLSTM
    model = BidirectionalLSTM(32, 16, 16)

    model.requires_grad_(False)
    return model


def get_single_stage_text_detector_model():
    from mmocr.models.textdet import SingleStageTextDetector
    backbone = dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe')
    neck = dict(
        type='FPNC',
        in_channels=[64, 128, 256, 512],
        lateral_channels=4,
        out_channels=4)
    bbox_head = dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=16,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True))
    model = SingleStageTextDetector(backbone, neck, bbox_head)

    model.requires_grad_(False)
    return model


def get_encode_decode_recognizer_model():
    from mmocr.models.textrecog import EncodeDecodeRecognizer

    cfg = dict(
        preprocessor=None,
        backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
        encoder=dict(type='TFEncoder'),
        decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
        loss=dict(type='CTCLoss'),
        label_convertor=dict(
            type='CTCConvertor',
            dict_type='DICT36',
            with_unknown=False,
            lower=True),
        pretrained=None)

    model = EncodeDecodeRecognizer(
        backbone=cfg['backbone'],
        encoder=cfg['encoder'],
        decoder=cfg['decoder'],
        loss=cfg['loss'],
        label_convertor=cfg['label_convertor'])
    model.requires_grad_(False)
    return model


def get_crnn_decoder_model(rnn_flag):
    from mmocr.models.textrecog.decoders import CRNNDecoder
    model = CRNNDecoder(32, 4, rnn_flag=rnn_flag)

    model.requires_grad_(False)
    return model


def get_fpnc_neck_model():
    model = FPNCNeckModel([2, 4, 8, 16])

    model.requires_grad_(False)
    return model


def get_base_recognizer_model():
    from mmocr.models.textrecog import CRNNNet

    cfg = dict(
        preprocessor=None,
        backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
        encoder=None,
        decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
        loss=dict(type='CTCLoss'),
        label_convertor=dict(
            type='CTCConvertor',
            dict_type='DICT36',
            with_unknown=False,
            lower=True),
        pretrained=None)

    model = CRNNNet(
        backbone=cfg['backbone'],
        decoder=cfg['decoder'],
        loss=cfg['loss'],
        label_convertor=cfg['label_convertor'])
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', ['ncnn'])
def test_bidirectionallstm(backend_type):
    """Test forward rewrite of bidirectionallstm."""
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
    bilstm = get_bidirectionallstm_model()
    bilstm.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 1, 32)

    # to get outputs of pytorch model
    model_inputs = {
        'input': input,
    }
    model_outputs = get_model_outputs(bilstm, 'forward', model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(bilstm, 'forward')
    rewrite_inputs = {'input': input}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


def test_simple_test_of_single_stage_text_detector():
    """Test simple_test single_stage_text_detector."""
    single_stage_text_detector = get_single_stage_text_detector_model()
    single_stage_text_detector.eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='default'),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextDetection',
            )))

    input = torch.rand(1, 3, 64, 64)
    img_metas = [{
        'ori_shape': [64, 64, 3],
        'img_shape': [64, 64, 3],
        'pad_shape': [64, 64, 3],
        'scale_factor': [1., 1., 1., 1],
    }]

    x = single_stage_text_detector.extract_feat(input)
    model_outputs = single_stage_text_detector.bbox_head(x)

    wrapped_model = WrapModel(single_stage_text_detector, 'simple_test')
    rewrite_inputs = {'img': input, 'img_metas': img_metas[0]}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend_type', ['ncnn'])
@pytest.mark.parametrize('rnn_flag', [True, False])
def test_crnndecoder(backend_type, rnn_flag):
    """Test forward rewrite of crnndecoder."""
    pytest.importorskip(backend_type, reason=f'requires {backend_type}')
    crnn_decoder = get_crnn_decoder_model(rnn_flag)
    crnn_decoder.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 32, 1, 64)
    out_enc = None
    targets_dict = None
    img_metas = None

    # to get outputs of pytorch model
    model_inputs = {
        'feat': input,
        'out_enc': out_enc,
        'targets_dict': targets_dict,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(crnn_decoder, 'forward_train',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        crnn_decoder,
        'forward_train',
        out_enc=out_enc,
        targets_dict=targets_dict,
        img_metas=img_metas)
    rewrite_inputs = {'feat': input}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize(
    'img_metas', [[None], [{
        'resize_shape': [32, 32],
        'valid_ratio': 1.0
    }]])
@pytest.mark.parametrize('is_dynamic', [True, False])
def test_forward_of_base_recognizer(img_metas, is_dynamic):
    """Test forward base_recognizer."""
    base_recognizer = get_base_recognizer_model()
    base_recognizer.eval()

    if not is_dynamic:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type='ncnn'),
                onnx_config=dict(input_shape=None),
                codebase_config=dict(
                    type='mmocr',
                    task='TextRecognition',
                )))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type='ncnn'),
                onnx_config=dict(
                    input_shape=None,
                    dynamic_axes={
                        'input': {
                            0: 'batch',
                            2: 'height',
                            3: 'width'
                        },
                        'output': {
                            0: 'batch',
                            2: 'height',
                            3: 'width'
                        }
                    }),
                codebase_config=dict(
                    type='mmocr',
                    task='TextRecognition',
                )))

    input = torch.rand(1, 1, 32, 32)

    feat = base_recognizer.extract_feat(input)
    out_enc = None
    if base_recognizer.encoder is not None:
        out_enc = base_recognizer.encoder(feat, img_metas)
    model_outputs = base_recognizer.decoder(
        feat, out_enc, None, img_metas, train_mode=False)
    wrapped_model = WrapModel(
        base_recognizer, 'forward', img_metas=img_metas[0])
    rewrite_inputs = {
        'img': input,
    }
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


def test_simple_test_of_encode_decode_recognizer():
    """Test simple_test encode_decode_recognizer."""
    encode_decode_recognizer = get_encode_decode_recognizer_model()
    encode_decode_recognizer.eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='default'),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 1, 32, 32)
    img_metas = [{'resize_shape': [32, 32], 'valid_ratio': 1.0}]

    feat = encode_decode_recognizer.extract_feat(input)
    out_enc = None
    if encode_decode_recognizer.encoder is not None:
        out_enc = encode_decode_recognizer.encoder(feat, img_metas)
    model_outputs = encode_decode_recognizer.decoder(
        feat, out_enc, None, img_metas, train_mode=False)

    wrapped_model = WrapModel(
        encode_decode_recognizer, 'simple_test', img_metas=img_metas)
    rewrite_inputs = {'img': input}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.parametrize('backend_type', ['tensorrt'])
def test_forward_of_fpnc(backend_type):
    """Test forward rewrite of fpnc."""
    fpnc = get_fpnc_neck_model()
    fpnc.eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(
                type=backend_type,
                common_config=dict(max_workspace_size=1 << 30),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            input=dict(
                                min_shape=[1, 3, 64, 64],
                                opt_shape=[1, 3, 64, 64],
                                max_shape=[1, 3, 64, 64])))
                ]),
            onnx_config=dict(input_shape=[64, 64], output_names=['output']),
            codebase_config=dict(type='mmocr', task='TextDetection')))

    input = torch.rand(1, 3, 64, 64).cuda()
    model_inputs = {
        'inputs': input,
    }
    model_outputs = get_model_outputs(fpnc, 'forward', model_inputs)
    wrapped_model = WrapModel(fpnc, 'forward')
    rewrite_inputs = {
        'inputs': input,
    }
    rewrite_outputs, is_need_name = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_need_name:
        model_output = model_outputs[0].squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs['output'].squeeze().cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        for model_output, rewrite_output in zip(model_outputs,
                                                rewrite_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze().cpu().numpy()
            assert np.allclose(
                model_output, rewrite_output, rtol=1e-03, atol=1e-05)
