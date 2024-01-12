_base_ = [
    './multi_view-detection_static.py',
    '../../_base_/backends/tensorrt-fp16.py'
]
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                imgs=dict(
                    min_shape=[1, 6, 3, 448, 704],
                    opt_shape=[1, 6, 3, 448, 704],
                    max_shape=[1, 6, 3, 448, 704]),
                ranks_bev=dict(
                    min_shape=[622435], opt_shape=[622435], max_shape=[622435
                                                                       ]),
                ranks_depth=dict(
                    min_shape=[622435], opt_shape=[622435], max_shape=[622435
                                                                       ]),
                ranks_feat=dict(
                    min_shape=[622435], opt_shape=[622435], max_shape=[622435
                                                                       ]),
                interval_starts=dict(
                    min_shape=[14019], opt_shape=[14019], max_shape=[14019]),
                interval_lengths=dict(
                    min_shape=[14019], opt_shape=[14019], max_shape=[14019]),
                mlp_inputs=dict(
                    min_shape=[1, 6, 27],
                    opt_shape=[1, 6, 27],
                    max_shape=[1, 6, 27]),
            ))
    ])
