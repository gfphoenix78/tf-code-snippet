```
def determine_meta(model_dir, choose=None):
    if not tf.gfile.IsDirectory(model_dir):
        raise ValueError('invalid model_dir')
    base_name = tf.train.latest_checkpoint(model_dir)
    if choose is not None:
        i = base_name.rfind('-')
        base_name = base_name[:i+1] + str(choose)
    return base_name

def freeze_graph(model_dir, outputs, choose=None, pb_name='frozen_model.pb'):
    if not isinstance(outputs, (list, tuple)):
        raise ValueError('outputs must be list/tuple of names')
    if len(outputs)==0:
        raise ValueError('output names must not be empty')
    for name in outputs:
        if not isinstance(name, str) or name=='':
            raise ValueError('each value in outputs must be a name')

    base_name = determine_meta(model_dir, choose)
    output_graph = os.path.join(model_dir, pb_name)
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(base_name + '.meta', clear_devices=True)
        saver.restore(sess, base_name)
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=outputs)
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(graph_def.SerializeToString())
        print('{} ops in the final graph'.format(len(graph_def.node)))
    return graph_def
```
some variant may use other functions in tf.graph_util:
* tf.graph_util.convert_variables_to_constants
* tf.graph_util.extract_sub_graph
* tf.graph_util.must_run_on_cpu
* tf.graph_util.remove_training_nodes
* tf.graph_util.tensor_shape_from_node_def_name
