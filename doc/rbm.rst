Restricted Boltzmann Machine (RBM)
==================================

Overview
--------

Examples
--------

Simple RBM Training
^^^^^^^^^^^^^^^^^^^

A simplpe example on how to use xRBM to train an RBM on image data: ::

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    from xrbm.models.rbm import RBM

    data_sets = input_data.read_data_sets('MNIST_data', False)

    r1 = RBM(num_vis=52, num_hid=300, vis_type='binary', 
             name='rbm_mnist_simple', 
             activation=tf.nn.sigmoid)

    with tf.Session() as sess: 
        init = tf.global_variables_initializer()
        sess.run(init)

        W, vb, hb, = r1.train(sess, 
                input_data=data_sets.train.images,
                training_epochs=200,
                learning_rate=0.01,
                batch_size=100,
                cd_k=10,
                wdecay=0.0002,
                momentum=0.0)


Step-by-step RBM Training
^^^^^^^^^^^^^^^^^^^^^^^^^


References
----------