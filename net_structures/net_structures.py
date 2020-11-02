from new_layers.Transformer import Transformer, Mask, OutputPositionItem, ScatterEmbedding
from new_layers.Transformer import Transformer, Mask, OutputPositionItem, ScatterEmbedding

net_transformer_map_scattered = [

    # First Entities Embedding
    [
        {
            'type': 'retrieve',
            'tensors': 'melee_weapons'
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'tanh'
        },
        {
            'type': 'register',
            'tensor': 'melee_embeddings'
        }
    ],
    # Second Entities Embedding
    [
        {
            'type': 'retrieve',
            'tensors': 'range_weapons'
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'tanh'
        },
        {
            'type': 'register',
            'tensor': 'range_embeddings'
        }
    ],
    # Third Entities Embedding
    [
        {
            'type': 'retrieve',
            'tensors': 'potions'
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'tanh'
        },
        {
            'type': 'register',
            'tensor': 'potions_embeddings'
        }
    ],
    # Transformer Block
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['melee_weapons', 'range_weapons', 'potions'],
            'value': 99.0,
            'num_entities': 60
        },
        {
            'type': 'register',
            'tensor': 'mask'
        }
    ],
    # Global Transformer Layer and scattered
    [
        {
            'type': 'retrieve',
            'tensors': ['melee_embeddings', 'range_embeddings', 'potions_embeddings'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 60,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'global_trans_embeddings'
        }
    ],
    # Global Scattered Map
    [
        {
          'type': 'retrieve',
          'tensors': ['global_trans_embeddings']
        },
        {
            'type': ScatterEmbedding,
            'indices_name': 'global_indices',
            'size': 10,
            'hidden_size': 64
        },
        {
            'type': 'register',
            'tensor': 'global_scattered'
        }
    ],

    # Local Transformer Layer and scattered
    [
        {
            'type': 'retrieve',
            'tensors': ['melee_embeddings', 'range_embeddings', 'potions_embeddings'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 60,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'local_trans_embeddings'
        }
    ],
    # Local Scattered Map
    [
        {
          'type': 'retrieve',
          'tensors': ['local_trans_embeddings']
        },
        {
            'type': ScatterEmbedding,
            'indices_name': 'local_indices',
            'size': 5,
            'hidden_size': 64
        },
        {
            'type': 'register',
            'tensor': 'local_scattered'
        }
    ],

    # Local Transformer Layer and scattered
    [
        {
            'type': 'retrieve',
            'tensors': ['melee_embeddings', 'range_embeddings', 'potions_embeddings'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 60,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'local_two_trans_embeddings'
        }
    ],
    # Local two Scattered Map
    [
        {
          'type': 'retrieve',
          'tensors': ['local_two_trans_embeddings']
        },
        {
            'type': ScatterEmbedding,
            'indices_name': 'local_two_indices',
            'size': 3,
            'hidden_size': 64
        },
        {
            'type': 'register',
            'tensor': 'local_two_scattered'
        }
    ],

    # Global Cell type Embeddings and concatenation with Scattered
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'global_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['global_emb', 'global_scattered'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'global_out'
        }
    ],

    # Local Cell type Embeddings and concatenation with Scattered
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_emb', 'local_scattered'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'local_out'
        }
    ],

    # Local two Cell type Embeddings and concatenation with Scattered
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_two_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_two_emb', 'local_two_scattered'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'local_out_two'
        }
    ],

    # This remain the same as before
    [
        {
            'type': 'retrieve',
            'tensors': ['agent_stats']
        },
        {
            "type": "embedding",
            "size": 256,
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
            'type': 'register',
            'tensor': 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "embedding",
            "size": 256
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
            'type': 'register',
            'tensor': 'target_stats_out'
        }
    ],

    [
        {
            'type': 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'first_FC'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['prev_action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'register',
            'tensor': 'action_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['first_FC', 'action_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]



baseline_transformer_map_scattered = [

    # First Entities Embedding
    [
        {
            'type': 'retrieve',
            'tensors': 'melee_weapons'
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'tanh'
        },
        {
            'type': 'register',
            'tensor': 'melee_embeddings'
        }
    ],
    # Second Entities Embedding
    [
        {
            'type': 'retrieve',
            'tensors': 'range_weapons'
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'tanh'
        },
        {
            'type': 'register',
            'tensor': 'range_embeddings'
        }
    ],
    # Third Entities Embedding
    [
        {
            'type': 'retrieve',
            'tensors': 'potions'
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'tanh'
        },
        {
            'type': 'register',
            'tensor': 'potions_embeddings'
        }
    ],
    # Transformer Block
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['melee_weapons', 'range_weapons', 'potions'],
            'value': 99.0,
            'num_entities': 60
        },
        {
            'type': 'register',
            'tensor': 'mask'
        }
    ],
    # Global Transformer Layer and scattered
    [
        {
            'type': 'retrieve',
            'tensors': ['melee_embeddings', 'range_embeddings', 'potions_embeddings'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 60,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'global_trans_embeddings'
        }
    ],
    # Global Scattered Map
    [
        {
          'type': 'retrieve',
          'tensors': ['global_trans_embeddings']
        },
        {
            'type': ScatterEmbedding,
            'indices_name': 'global_indices',
            'size': 10,
            'hidden_size': 64
        },
        {
            'type': 'register',
            'tensor': 'global_scattered'
        }
    ],

    # Local Transformer Layer and scattered
    [
        {
            'type': 'retrieve',
            'tensors': ['melee_embeddings', 'range_embeddings', 'potions_embeddings'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 60,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'local_trans_embeddings'
        }
    ],
    # Local Scattered Map
    [
        {
          'type': 'retrieve',
          'tensors': ['local_trans_embeddings']
        },
        {
            'type': ScatterEmbedding,
            'indices_name': 'local_indices',
            'size': 5,
            'hidden_size': 64
        },
        {
            'type': 'register',
            'tensor': 'local_scattered'
        }
    ],

    # Local Transformer Layer and scattered
    [
        {
            'type': 'retrieve',
            'tensors': ['melee_embeddings', 'range_embeddings', 'potions_embeddings'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 60,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'local_two_trans_embeddings'
        }
    ],
    # Local two Scattered Map
    [
        {
          'type': 'retrieve',
          'tensors': ['local_two_trans_embeddings']
        },
        {
            'type': ScatterEmbedding,
            'indices_name': 'local_two_indices',
            'size': 3,
            'hidden_size': 64
        },
        {
            'type': 'register',
            'tensor': 'local_two_scattered'
        }
    ],

    # Global Cell type Embeddings and concatenation with Scattered
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'global_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['global_emb', 'global_scattered'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'global_out'
        }
    ],

    # Local Cell type Embeddings and concatenation with Scattered
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_emb', 'local_scattered'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'local_out'
        }
    ],

    # Local two Cell type Embeddings and concatenation with Scattered
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_two_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_two_emb', 'local_two_scattered'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'local_out_two'
        }
    ],

    # This remain the same as before
    [
        {
            'type': 'retrieve',
            'tensors': ['agent_stats']
        },
        {
            "type": "embedding",
            "size": 256,
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
            'type': 'register',
            'tensor': 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "embedding",
            "size": 256
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
            'type': 'register',
            'tensor': 'target_stats_out'
        }
    ],

    [
        {
            'type': 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        },
        {
            'type': 'register',
            'tensor': 'first_FC'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['prev_action']
        },
        {
            'type': 'flatten'
        },
        {
            'type': 'register',
            'tensor': 'action_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['first_FC', 'action_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 256,
            "activation": 'relu'
        }
    ]
]
net_transformer_map = [
    # Transformer Block
    # First embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items'],
            'value': 99.0,
            'num_entities': 100
        },
        {
            'type': 'register',
            'tensor': 'mask'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['items_emb'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 100,
            'mask_name': 'mask'
        },
        {
            'type': 'register',
            'tensor': 'trans_global'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'global_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['global_emb', 'trans_global'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'global_out'
        }
    ],
    # Transformer Block
    # Second embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb_local'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local'],
            'value': 99.0,
            'num_entities': 25
        },
        {
            'type': 'register',
            'tensor': 'mask_local'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['items_emb_local'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 25,
            'mask_name': 'mask_local',
        },
        {
            'type': 'register',
            'tensor': 'trans_local'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_emb', 'trans_local'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'local_out'
        }
    ],
    # Transformer Block
    # third embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local_two']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'items_emb_local_two'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local_two'],
            'value': 99.0,
            'num_entities': 9
        },
        {
            'type': 'register',
            'tensor': 'mask_local_two'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['items_emb_local_two'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 9,
            'mask_name': 'mask_local_two',
        },
        {
            'type': 'register',
            'tensor': 'trans_local_two'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'local_two_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['local_two_emb', 'trans_local_two'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'local_out_two'
        }
    ],

    [
        {
            'type' : 'retrieve',
            'tensors' : ['agent_stats']
        },
        {
            "type": "embedding",
            "size": 256,
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
            'type' : 'register',
            'tensor' : 'agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "embedding",
            "size": 256
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
            'type': 'register',
            'tensor': 'target_stats_out'
        }
    ],

    [
        {
            'type' : 'retrieve',
            'tensors': ['global_out', 'local_out', 'local_out_two', 'agent_stats_out', 'target_stats_out'],
            'aggregation': 'concat',
        },
        {
            "type": "dense",
            "size": 128,
            "activation": 'relu'
        },
        {
            "type": "dense",
            "size": 128,
            "activation": 'relu'
        }
    ]
]

baseline_transformer_map = [
    # Transformer Block
    # First embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items'],
            'value': 99.0,
            'num_entities': 100
        },
        {
            'type': 'register',
            'tensor': 'base_mask'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['base_items_emb'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 100,
            'mask_name': 'base_mask',
        },
        {
            'type': 'register',
            'tensor': 'base_trans_global'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['global_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'base_global_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['base_global_emb', 'base_trans_global'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'base_global_out'
        }
    ],
    # Transformer Block
    # Second embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb_local'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local'],
            'value': 99.0,
            'num_entities': 25
        },
        {
            'type': 'register',
            'tensor': 'base_mask_local'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['base_items_emb_local'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 25,
            'mask_name': 'base_mask_local',
        },
        {
            'type': 'register',
            'tensor': 'base_trans_local'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'base_local_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['base_local_emb', 'base_trans_local'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'base_local_out'
        }
    ],
    # Transformer Block
    # third embeddings
    [
        {
            'type': 'retrieve',
            'tensors': ['items_local_two']
        },
        {
            'type': 'conv1d',
            'window': 1,
            'stride': 1,
            'size': 64,
            'activation': 'relu'
        },
        {
            'type': 'register',
            'tensor': 'base_items_emb_local_two'
        }
    ],
    # Creating Mask
    [
        {
            'type': Mask,
            'tensors': ['items_local_two'],
            'value': 99.0,
            'num_entities': 9
        },
        {
            'type': 'register',
            'tensor': 'base_mask_local_two'
        }

    ],
    # Self attention
    [
        {
            'type': 'retrieve',
            'tensors': ['base_items_emb_local_two'],
            'aggregation': 'concat',
        },
        {
            "type": Transformer,
            "n_head": 2,
            "hidden_size": 64,
            "pooling": 'none',
            'num_entities': 9,
            'mask_name': 'base_mask_local_two',
        },
        {
            'type': 'register',
            'tensor': 'base_trans_local_two'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['local_in_two']
        },
        {
            "type": "embedding",
            "size": 32
        },
        {
            'type': 'register',
            'tensor': 'base_local_two_emb'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['base_local_two_emb', 'base_trans_local_two'],
            'aggregation': 'concat',
            'axis': 2
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
            'type' : 'register',
            'tensor' : 'base_local_out_two'
        }
    ],
    [
        {
            'type' : 'retrieve',
            'tensors' : ['agent_stats']
        },
        {
            "type": "embedding",
            "size": 256,
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
            'type' : 'register',
            'tensor' : 'base_agent_stats_out'
        }
    ],
    [
        {
            'type': 'retrieve',
            'tensors': ['target_stats']
        },
        {
            "type": "embedding",
            "size": 256
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
            'type': 'register',
            'tensor': 'base_target_stats_out'
        }
    ],

    [
        {
            'type' : 'retrieve',
            'tensors': ['base_global_out', 'base_local_out', 'base_local_out_two', 'base_agent_stats_out', 'base_target_stats_out'],
            'aggregation': 'concat',
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

