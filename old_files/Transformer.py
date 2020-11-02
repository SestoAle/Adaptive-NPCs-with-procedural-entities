from tensorforce.core.networks.layer import Layer
import tensorflow as tf
import numpy as np


class Transformer(Layer):
    def __init__(self, n_head, hidden_size, num_entities, pooling='avg', named_tensors=None, scope='transformer', summary_labels=()):
        """
        Transformer Layer
        """

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.num_entities = num_entities

        self.with_initial_embedding = False

        super(Transformer, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update, layer_norm=True, post_sa_layer_norm=True,
                      n_mlp=1, qk_w=0.125, v_w=0.125, post_w=0.125,
                      mlp_w1=0.125, mlp_w2=0.125,
                      scope="residual_sa_block", reuse=False):

        x = x[:, tf.newaxis, :, :]

        mask = self.named_tensors['mask']

        # Create a first embedding for each object
        if self.with_initial_embedding:
            embs_scale = np.sqrt(post_w / self.hidden_size)
            x = tf.layers.dense(x,
                                self.hidden_size,
                                kernel_initializer=tf.random_normal_initializer(stddev=embs_scale),
                                name="embs1")

        a = self.self_attention(x, mask, self.n_head, self.hidden_size, layer_norm=layer_norm, qk_w=qk_w, v_w=v_w,
                           scope='self_attention', reuse=False)

        post_scale = np.sqrt(post_w / self.hidden_size)
        post_a_mlp = tf.layers.dense(a,
                                     self.hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=post_scale),
                                     name="mlp1")

        x = x + post_a_mlp
        if post_sa_layer_norm:
            with tf.variable_scope('post_a_layernorm'):
                x = tf.contrib.layers.layer_norm(x, begin_norm_axis=3)

        if n_mlp > 1:
            mlp = x
            mlp2_scale = np.sqrt(mlp_w1 / self.hidden_size)
            mlp = tf.layers.dense(mlp,
                                  self.hidden_size,
                                  kernel_initializer=tf.random_normal_initializer(stddev=mlp2_scale),
                                  name="mlp2")
        if n_mlp > 2:
            mlp3_scale = np.sqrt(mlp_w2 / self.hidden_size)
            mlp = tf.layers.dense(mlp,
                                  self.hidden_size,
                                  kernel_initializer=tf.random_normal_initializer(stddev=mlp3_scale),
                                  name="mlp3")
        if n_mlp > 1:
            x = x + mlp

        # x = (bs, feature)
        if self.pooling == 'avg':
            x = self.entity_avg_pooling_masked(x, mask)
        else:
            x = self.entity_max_pooling_masked(x, mask)

        bs, t, feature = self.shape_list(x)
        x = tf.reshape(x, (bs, feature))

        return x

    def self_attention(self, inp, mask, heads, n_embd, layer_norm=False, qk_w=1.0, v_w=0.01,
                       scope='', reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            bs, T, NE, features = self.shape_list(inp)
            # Put mask in format correct for logit matrix
            entity_mask = None
            if mask is not None:
                with tf.variable_scope('expand_mask'):
                    assert np.all(np.array(mask.get_shape().as_list()) == np.array(inp.get_shape().as_list()[:3])), \
                        f"Mask and input should have the same first 3 dimensions. {self.shape_list(mask)} -- {self.shape_list(inp)}"
                    entity_mask = mask
                    mask = tf.expand_dims(mask, -2)  # (BS, T, 1, NE)

            query, key, value = self.qkv_embed(inp, heads, n_embd, layer_norm=layer_norm, qk_w=qk_w, v_w=v_w, reuse=reuse)
            logits = tf.matmul(query, key, name="matmul_qk_parallel")  # (bs, T, heads, NE, NE)
            logits /= np.sqrt(n_embd / heads)
            softmax = self.stable_masked_softmax(logits, mask)

            att_sum = tf.matmul(softmax, value, name="matmul_softmax_value")  # (bs, T, heads, NE, features)
            with tf.variable_scope('flatten_heads'):
                out = tf.transpose(att_sum, (0, 1, 3, 2, 4))  # (bs, T, n_output_entities, heads, features)
                n_output_entities = self.shape_list(out)[2]
                out = tf.reshape(out, (bs, T, n_output_entities, n_embd))  # (bs, T, n_output_entities, n_embd)

            return out

    def stable_masked_softmax(self, logits, mask):

        with tf.variable_scope('stable_softmax'):
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

    def qkv_embed(self, inp, heads, n_embd, layer_norm=False, qk_w=1.0, v_w=0.01, reuse=False):

        with tf.variable_scope('qkv_embed'):
            bs, T, NE, features = self.shape_list(inp)
            if layer_norm:
                with tf.variable_scope('pre_sa_layer_norm'):
                    inp = tf.contrib.layers.layer_norm(inp, begin_norm_axis=3)

            # qk shape (bs x T x NE x h x n_embd/h)
            qk_scale = np.sqrt(qk_w / features)
            qk = tf.layers.dense(inp,
                                 n_embd * 2,
                                 kernel_initializer=tf.random_normal_initializer(stddev=qk_scale),
                                 reuse=reuse,
                                 name="qk_embed")  # bs x T x n_embd*2
            qk = tf.reshape(qk, (bs, T, NE, heads, n_embd // heads, 2))

            # (bs, T, NE, heads, features)
            query, key = [tf.squeeze(x, -1) for x in tf.split(qk, 2, -1)]

            v_scale = np.sqrt(v_w / features)
            value = tf.layers.dense(inp,
                                    n_embd,
                                    kernel_initializer=tf.random_normal_initializer(stddev=v_scale),
                                    reuse=reuse,
                                    name="v_embed")  # bs x T x n_embd
            value = tf.reshape(value, (bs, T, NE, heads, n_embd // heads))

            query = tf.transpose(query, (0, 1, 3, 2, 4),
                                 name="transpose_query")  # (bs, T, heads, NE, n_embd / heads)
            key = tf.transpose(key, (0, 1, 3, 4, 2),
                               name="transpose_key")  # (bs, T, heads, n_embd / heads, NE)
            value = tf.transpose(value, (0, 1, 3, 2, 4),
                                 name="transpose_value")  # (bs, T, heads, NE, n_embd / heads)

        return query, key, value

    def shape_list(self, x):

        ps = x.get_shape().as_list()
        ts = tf.shape(x)
        return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

    def create_mask(self, x):

        # x = bs, T, NE, feature
        mask = 1 - tf.cast(tf.equal(x[:,:,:,0], 99.0), tf.float32)

        return mask

    def entity_avg_pooling_masked(self, x, mask):

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
    def __init__(self, names, value=99.0, named_tensors=None, scope='masking', summary_labels=()):
        """
        Transformer Layer
        """
        self.value = value
        self.names = names

        super(Mask, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):

        inputs = list()
        for name in self.names:
            if name == '*' or name == 'previous':
                # like normal list network_spec
                inputs.append(x)
            elif name in self.named_tensors:
                inputs.append(self.named_tensors[name])

        masks = []
        for tensor in inputs:
            tensor = tensor[:, tf.newaxis, :, :]
            mask = 1 - tf.cast(tf.equal(tensor[:, :, :, 0], self.value), tf.float32)
            masks.append(mask)

        mask = tf.concat(values=masks, axis=2)
        return mask

if __name__ == '__main__':

    with tf.Session() as sess:

        x = tf.placeholder(shape=(None, 10, 64), dtype=tf.float32)
        transformer = Transformer(4, 64)
        out = transformer.tf_apply(x, False)

        init = tf.global_variables_initializer()
        sess.run(init)

        inp = np.random.randn(32, 10, 64)

        feed_dict = {x: inp}

        for i in range(2):
            res = sess.run([out], feed_dict)
            print(res)

