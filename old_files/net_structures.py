from new_layers.Transformer import Transformer, Mask

# Neural network structure
dc_net_transformer_with_agent = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],

    # Transformer Block
    # First embeddings
    [
        {
            'type': 'input',
            'names': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'output',
            'name': 'items_emb'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['agent']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'output',
            'name': 'agent_emb'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['target']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'output',
            'name': 'target_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type' : Mask,
            'names' : ['items', 'agent', 'target'],
            'value': 99.0
        },
        {
            'type': 'output',
            'name': 'mask'
        }

    ],

    # Self attention
    [
        {
            'type' : 'input',
            'names' : ['items_emb', 'agent_emb', 'target_emb'],
            'aggregation_type': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 256,
            "pooling": 'avg',
            'num_entities': 22
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'trans_out'
        }
    ],

    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'trans_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'output',
            'name': 'first_FC'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'action_out'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['first_FC', 'action_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "internal_lstm",
            "size": 256,
        }
    ]
]
dc_net_transformer = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 118,
            "size": 128
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],

    #Transformer
    [
        {
            'type' : 'input',
            'names' : ['items']
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 256,
            "pooling": 'avg'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'items_out'
        }
    ],

    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out', 'items_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ],
]
dc_net_transformer_fc = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 118,
            "size": 128
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],

    #Transformer
    [
        {
            'type' : 'input',
            'names' : ['items']
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 512,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'items_out'
        }
    ],

    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out', 'items_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ],
]


# Baseline net structure
dc_baseline_transformer_with_agent = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],

    # Transformer Block
    # First embeddings
    [
        {
            'type': 'input',
            'names': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'output',
            'name': 'items_emb'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['agent']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'output',
            'name': 'agent_emb'
        }
    ],
    [
        {
            'type': 'input',
            'names': ['target']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 256,
            'activation': 'relu'
        },
        {
            'type': 'output',
            'name': 'target_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type' : Mask,
            'names' : ['items', 'agent', 'target'],
            'value': 99.0
        },
        {
            'type': 'output',
            'name': 'mask'
        }

    ],

    # Self attention
    [
        {
            'type' : 'input',
            'names' : ['items_emb', 'agent_emb', 'target_emb'],
            'aggregation_type': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 256,
            "pooling": 'avg',
            'num_entities': 22
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'trans_out'
        }
    ],

    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'trans_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ],
]
dc_baseline_transformer = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 118,
            "size": 128
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],

    #Transformer
    [
        {
            'type': 'input',
            'names': ['items']
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 256,
            "pooling": 'avg'
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'output',
            'name': 'items_out'
        }
    ],

    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out', 'items_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]
dc_baseline_transformer_fc = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 4,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 118,
            "size": 128
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],

    #Transformer
    [
        {
            'type': 'input',
            'names': ['items']
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 512,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'output',
            'name': 'items_out'
        }
    ],

    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out', 'items_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]

dc_net = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 118,
            "size": 64
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]
irl_net = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 110,
            "size": 64
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ]
]
irl_net_no_stats = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ]
]
dc_baseline = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 118,
            "size": 64
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]
irl_baseline = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['stats']
        },
        {
            "type": "embedding",
            "indices": 110,
            "size": 64
        },
        {
            'type': 'flatten'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'stats_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two', 'stats_out'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ]
]
irl_baseline_no_stats = [
    [
        {
            'type' : 'input',
            'names' : ['global_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'global_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out'
        }
    ],
    [
        {
            'type' : 'input',
            'names' : ['local_in_two']
        },
        {
            "type": "embedding",
            "indices": 12,
            "size": 32
        },
        {
            "type": "conv2d",
            "size": 32,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            "type": "conv2d",
            "size": 64,
            "window": (3, 3),
            "stride": 1,
            "activation": 'relu'
        },
        {
            'type': 'flatten'
        },
        {
            'type' : 'output',
            'name' : 'local_out_two'
        }
    ],
    [
        {
            'type' : 'input',
            'names': ['global_out', 'local_out', 'local_out_two'],
            'aggregation_type': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type' : 'output',
            'name' : 'first_FC'
        }
    ]
]