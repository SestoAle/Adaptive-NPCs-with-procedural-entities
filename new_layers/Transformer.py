from tensorforce.core.layers.layer import Layer
import tensorflow as tf
import numpy as np
from tensorforce.core import Module, parameter_modules
from tensorforce import TensorforceError, util


class Transformer(Layer):
    def __init__(self, name, n_head, hidden_size, num_entities, mlp_layer=1, mask_name='', pooling='average', residual=True,
                 masking=True, with_embeddings=False, with_ffn=True, post_norm=True, input_spec=None, pre_norm = True,
                 num_block=1,
                 summary_labels=()):
        """
        Transformer Layer
        """
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        self.mlp_layer = mlp_layer
        self.pooling = pooling
        while self.pooling not in ['avg', 'max', 'none']:
            self.pooling = 'none'

        self.residual = residual
        self.masking = masking
        self.with_embeddings = with_embeddings
        self.with_ffn = with_ffn
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.name = name
        self.num_block=num_block
        self.mask_name = mask_name

        super(Transformer, self).__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def get_output_spec(self, input_spec):
        if self.pooling is not 'none':
            return dict(type='float', shape=(self.hidden_size))
        else:
            size = int(np.sqrt(self.num_entities))
            return dict(type='float', shape=(self.num_entities, self.hidden_size))

    def linear(self, a, b, bias):
        return tf.nn.bias_add(tf.matmul(a,b), bias)

    def tf_initialize(self):
        super().tf_initialize()
        # qkv embeddings weights
        self.qk_weights = self.add_variable(
            name='qk_weights', dtype='float', shape=(self.input_spec['shape'][1], self.hidden_size*2),
            is_trainable=True, initializer='orthogonal'
        )
        self.qk_bias = self.add_variable(
            name='qk_bias', dtype='float', shape=(self.hidden_size*2,),
            is_trainable=True, initializer='zeros'
        )
        self.v_weights = self.add_variable(
            name='v_weights', dtype='float', shape=(self.input_spec['shape'][1], self.hidden_size),
            is_trainable=True, initializer='orthogonal'
        )
        self.v_bias = self.add_variable(
            name='v_bias', dtype='float', shape=(self.hidden_size,),
            is_trainable=True, initializer='zeros'
        )

        # FFN
        self.mlp_layers_weights = []
        self.mlp_layers_bias = []
        for i in range(self.mlp_layer):
            self.mlp_layers_weights.append(self.add_variable(
                name='mlp' + str(i) + '_weights', dtype='float', shape=(self.input_spec['shape'][1], self.hidden_size),
                is_trainable=True, initializer='orthogonal'
            ))
            self.mlp_layers_bias.append(self.add_variable(
                name='mlp' + str(i) + '_bias', dtype='float', shape=(self.hidden_size,),
                is_trainable=True, initializer='zeros'
            ))

        # If with initial embedding
        if self.with_embeddings:
            self.init_emb_weights = self.add_variable(
                name='init_emb_weights', dtype='float', shape=(self.input_spec['shape'][1], self.hidden_size),
                is_trainable=True, initializer='orthogonal'
            )
            self.init_emb_bias = self.add_variable(
                name='init_emb_bias', dtype='float', shape=(self.hidden_size,),
                is_trainable=True, initializer='zeros'
            )

        if self.post_norm:
            self.post_norm_layer = tf.keras.layers.LayerNormalization(axis=3)
            self.post_norm_layer.build(input_shape=((None,1) + self.input_spec['shape']))
            for variable in self.post_norm_layer.trainable_weights:
                name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
                self.variables[name] = variable
                self.trainable_variables[name] = variable

        if self.pre_norm:
            self.pre_norm_layer = tf.keras.layers.LayerNormalization(axis=3)
            self.pre_norm_layer.build(input_shape=((None,1) + self.input_spec['shape']))
            for variable in self.pre_norm_layer.trainable_weights:
                name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
                self.variables[name] = variable
                self.trainable_variables[name] = variable

    def tf_apply(self, x):

        x = x[:, tf.newaxis, :, :]
        bs, t, NE, feature = self.shape_list(x)
        mask = None
        if self.masking:
            mask = Module.retrieve_tensor(name=self.mask_name)
        size = np.sqrt(NE)

        x, mask = self.apply_attention(x, mask)

        if self.pooling is not 'none':
            if self.pooling == 'avg':
                x = self.entity_avg_pooling_masked(x, mask)
            elif self.pooling == 'max':
                x = self.entity_max_pooling_masked(x, mask)
            x = tf.reshape(x, (bs, feature))
        else:

            # x = tf.reshape(x, (bs, size, size, feature))
            #
            # mask = tf.reshape(mask, (bs, size, size))
            mask = tf.expand_dims(mask, -1)

            x = x * mask
            x = tf.reshape(x, [bs, self.num_entities, self.hidden_size])

        return super().tf_apply(x=x)

    def apply_attention(self, x, mask):

        # Create a first embedding for each object
        if self.with_embeddings:
            x = self.linear(x, self.init_emb_weights, self.init_emb_bias)

        a = self.self_attention(x, mask, self.n_head, self.hidden_size)

        if self.with_ffn:
            for i in range(self.mlp_layer):
                a = self.linear(a, self.mlp_layers_weights[i], self.mlp_layers_bias[i])

        if self.residual:
            x = x + a
        else:
            x = a

        if self.post_norm:
            x = self.post_norm_layer(x)

        return x, mask

    def self_attention(self, inp, mask, heads, n_embd):

        bs, T, NE, features = self.shape_list(inp)
        # Put mask in format correct for logit matrix
        entity_mask = None
        if mask is not None:
            assert np.all(np.array(mask.get_shape().as_list()) == np.array(inp.get_shape().as_list()[:3])), \
                f"Mask and input should have the same first 3 dimensions. {self.shape_list(mask)} -- {self.shape_list(inp)}"
            entity_mask = mask
            mask = tf.expand_dims(mask, -2)  # (BS, T, 1, NE)

        query, key, value = self.qkv_embed(inp, heads, n_embd)
        logits = tf.matmul(query, key, name="matmul_qk_parallel")  # (bs, T, heads, NE, NE)
        logits /= np.sqrt(n_embd / heads)
        softmax = self.stable_masked_softmax(logits, mask)

        att_sum = tf.matmul(softmax, value, name="matmul_softmax_value")  # (bs, T, heads, NE, features)

        out = tf.transpose(att_sum, (0, 1, 3, 2, 4))  # (bs, T, n_output_entities, heads, features)
        n_output_entities = self.shape_list(out)[2]
        out = tf.reshape(out, (bs, T, n_output_entities, n_embd))  # (bs, T, n_output_entities, n_embd)

        return out

    def stable_masked_softmax(self, logits, mask):

        #  Subtract a big number from the masked logits so they don't interfere with computing the max value
        if mask is not None:
            mask = tf.expand_dims(mask, 2)
            logits -= (1.0 - mask) * 1e10

        #  Subtract the max logit from everything so we don't overflow
        logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
        unnormalized_p = tf.exp(logits)

        #  Mask the unnormalized probibilities and then normalize and remask
        if mask is not None:
            unnormalized_p *= mask
        normalized_p = unnormalized_p / (tf.reduce_sum(unnormalized_p, axis=-1, keepdims=True) + 1e-10)
        if mask is not None:
            normalized_p *= mask
        return normalized_p

    def qkv_embed(self, inp, heads, n_embd):

        bs, T, NE, features = self.shape_list(inp)
        if self.pre_norm:
            inp = self.pre_norm_layer(inp)


        qk = self.linear(inp, self.qk_weights, self.qk_bias)
        qk = tf.reshape(qk, (bs, T, NE, heads, n_embd // heads, 2))

        # (bs, T, NE, heads, features)
        query, key = [tf.squeeze(x, -1) for x in tf.split(qk, 2, -1)]

        value = self.linear(inp, self.v_weights, self.v_bias)
        value = tf.reshape(value, (bs, T, NE, heads, n_embd // heads))

        query = tf.transpose(query, (0, 1, 3, 2, 4),
                             name="transpose_query")  # (bs, T, heads, NE, n_embd / heads)
        key = tf.transpose(key, (0, 1, 3, 4, 2),
                           name="transpose_key")  # (bs, T, heads, n_embd / heads, NE)
        value = tf.transpose(value, (0, 1, 3, 2, 4),
                             name="transpose_value")  # (bs, T, heads, NE, n_embd / heads)

        return query, key, value

    def shape_list(self, x):
        '''
            deal with dynamic shape in tensorflow cleanly
        '''
        ps = x.get_shape().as_list()
        ts = tf.shape(x)
        return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

    def create_mask(self, x):
        '''
            Create mask from the input. If the first element is 99, then mask it.
            The mask must be 1 for the input and 0 for the
        '''

        # x = bs, NE, feature
        mask = 1 - tf.cast(tf.equal(x[:,:,:,0], 99999999.0), tf.float32)
        return mask

    def entity_avg_pooling_masked(self, x, mask):
        '''
            Masks and pools x along the second to last dimension. Arguments have dimensions:
                x:    batch x time x n_entities x n_features
                mask: batch x time x n_entities
        '''
        mask = tf.expand_dims(mask, -1)
        masked = x * mask
        summed = tf.reduce_sum(masked, -2)
        denom = tf.reduce_sum(mask, -2) + 1e-5
        return summed / denom

    def entity_max_pooling_masked(self, x, mask):
        '''
            Masks and pools x along the second to last dimension. Arguments have dimensions:
                x:    batch x time x n_entities x n_features
                mask: batch x time x n_entities
        '''
        mask = tf.expand_dims(mask, -1)
        has_unmasked_entities = tf.sign(tf.reduce_sum(mask, axis=-2, keepdims=True))
        offset = (mask - 1) * 1e9
        masked = (x + offset) * has_unmasked_entities
        return tf.reduce_max(masked, -2)

class Mask(Layer):
    def __init__(self, name, num_entities, tensors, value=99.0, input_spec=None, summary_labels=()):
        """
        Transformer Layer
        """
        self.value = value
        self.num_entities = num_entities
        self.tensors = (tensors,) if isinstance(tensors, str) else tuple(tensors)

        super(Mask, self).__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

    def tf_apply(self, x):

        tensors = list()
        for tensor in self.tensors:
            if tensor == '*':
                tensors.append(x)
            else:
                last_scope = Module.global_scope.pop()
                tensors.append(Module.retrieve_tensor(name=tensor))
                Module.global_scope.append(last_scope)

        shape = self.output_spec['shape']
        for n, tensor in enumerate(tensors):
            for axis in range(util.rank(x=tensor), len(shape)):
                tensor = tf.expand_dims(input=tensor, axis=axis)
            tensors[n] = tensor

        masks = []
        for tensor in tensors:
            tensor = tensor[:, tf.newaxis, :, :]
            tensor = tf.cast(tensor, tf.float32)
            mask = 1 - tf.cast(tf.equal(tensor[:, :, :, 0], self.value), tf.float32)
            masks.append(mask)

        mask = tf.concat(values=masks, axis=2)

        return mask

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        # mask: batch x time x n_entities
        return dict(type='float', shape=(1, self.num_entities))

class OutputPositionItem(Layer):
    def __init__(self, name, t, input_spec=None, summary_labels=()):
        self.t = t
        super(OutputPositionItem, self).__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

    def tf_apply(self, x):

        x = x[:,:,0:2]
        return x

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        # mask: batch x time x n_entities
        if self.t == 'items':
            return dict(type='float', shape=(20, 2))
        else:
            return dict(type='float', shape=(1, 2))

class ScatterEmbedding(Layer):
    def __init__(self, name, indices_name = 'global', size = 10, hidden_size = 64,
                 base = False, input_spec = None, summary_labels=()):
        """
        This layer will create the scattered map. It takes as input the items embedding and global/local indices.
        It returns a map (batch_size, w, h, features).
        """
        self.indices_name = indices_name
        self.size = size
        self.size = size
        self.hidden_size = hidden_size
        self.base = base
        self.indices_name = indices_name
        super(ScatterEmbedding, self).__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

    def tf_apply(self, x):

        BS, entities, features = self.shape_list(x)
        self.features = features
        size = self.size

        indices = Module.retrieve_tensor(name=self.indices_name)
        indices = tf.reshape(indices, (BS, entities))
        indices = tf.cast(indices, tf.int32)
        if self.indices_name is not 'global_indices':
            indices = tf.where(tf.greater_equal(indices, 0), indices, -(size*size*BS - 1))
            indices = tf.where(tf.less_equal(indices, size*size - 1), indices, -(size*size*BS - 1))  

        # @tf.function
        # def create_scatter(a):
        #     ind = a[0]
        #     ind_int = tf.cast(ind, tf.int32)
        #     items = a[1]
        #     scatter_b = tf.scatter_nd(ind_int, items, [size*size, features])
        #     return [scatter_b, ind]
        #
        # def dummy_fn(a):
        #     return a
        #
        # scattered_map = tf.map_fn(create_scatter, [indices, x])[0]
        # scattered_map = tf.reshape(scattered_map, (BS, size, size, features))

        x = tf.reshape(x, (BS*entities, features))
        a_rows = tf.expand_dims(tf.range(BS, dtype=tf.int32), 1)
        a_rows *= (size*size)
        indices = indices + a_rows
        indices = tf.reshape(indices, (BS*entities, 1))

        scattered_map = tf.scatter_nd(indices, x, [BS*size*size, features])
        scattered_map = tf.reshape(scattered_map, (BS, size, size, features))

        return scattered_map

    def shape_list(self, x):

        ps = x.get_shape().as_list()
        ts = tf.shape(x)
        return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        return dict(type='float', shape=(self.size, self.size, self.hidden_size))
