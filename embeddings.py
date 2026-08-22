import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Embedding(object):

    def __init__(self, emb_type, embedding_dim):
        self.emb_type = emb_type
        self.embedding_dim = embedding_dim

    def __call__(self, input_pnt):
        raise NotImplementedError


class LinearEmbedding(Embedding):

    def __init__(self, embedding_dim, _scope=''):
        super(LinearEmbedding, self).__init__('linear', embedding_dim)

        self.project_emb = tf.keras.layers.Conv1D(
            filters=embedding_dim,
            kernel_size=1,
            name=_scope + 'embedding_conv1d'
        )

    def __call__(self, input_pnt):

        # input: [batch , nodes , input_dim]
        emb_inp_pnt = self.project_emb(input_pnt)

        # output: [batch , nodes , embedding_dim]
        return emb_inp_pnt


if __name__ == "__main__":

    sess = tf.InteractiveSession()

    input_pnt = tf.random.uniform([2, 10, 2])

    embedding_layer = LinearEmbedding(128)

    emb_inp_pnt = embedding_layer(input_pnt)

    sess.run(tf.global_variables_initializer())

    print(sess.run([emb_inp_pnt, tf.shape(emb_inp_pnt)]))
