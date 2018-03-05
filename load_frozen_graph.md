```
def load_frozen_graph(file_path):
    g = tf.Graph()
    with g.as_default():
        gdef = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as f:
            gdef.ParseFromString(f.read())
        tf.import_graph_def(gdef)
    return g
```
