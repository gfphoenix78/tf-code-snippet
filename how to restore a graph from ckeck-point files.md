```
def load_model(path, choose=None):
    if choose is None:
        file = tf.train.latest_checkpoint(path)
    else:
        file = '{}-{}'.format(path, choose)

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(file + '.meta')
    return g, saver, file
```
this function is used to restore a training graph from ckt files.
you can specify the meta file in two ways:
* `path` is a training directory, and keep `choose` to `None` (use the newest ckt/meta)
* `path` is the file prefix, and `choose` is the candidate number (form `path-<choose>.meta`)
