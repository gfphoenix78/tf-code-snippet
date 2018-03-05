```
def load_frozen_graph(file_path):
    with tf.Graph() as g:
        g_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as f:
            g_def.ParseFromString(f.read())
        tf.import_graph_def(g_def)
    return g
```
