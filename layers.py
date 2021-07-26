import tensorflow as tf


def lstm_network(input, lstm_hidden_size_layer=64,
                 lstm_latent_dim=16, lstm_num_layers=2, forget_bias=1.0, scope='lstm_network'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # tf.nn.rnn_cell
        def make_cell():
            cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer, forget_bias=forget_bias)
            return cell

        lstm_cells = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(lstm_num_layers)])

        # initial_state = lstm_cells.zero_state(batch_size,  tf.float32)

        outputs, states = tf.nn.dynamic_rnn(lstm_cells, input, dtype=tf.float32, initial_state=None)
        # print(z_sequence_output.get_shape())

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        print('LSTM output shape: ' + str(outputs.get_shape().as_list()))

        #outputs = tf.slice(outputs, [0, outputs.get_shape().as_list()[1]-1, 0], [-1, 1, -1])
        #outputs = tf.squeeze(outputs)
        #print('LSTM output shape: ' + str(outputs.get_shape().as_list()))

        z_sequence_output = outputs

        #states_concat = tf.concat([states[0].h, states[1].h], 1)
        #z_sequence_output = fc(states_concat, lstm_latent_dim, scope='linear_transform')
        # print('LSTM state shape: ' + str(states))

        #z_sequence_output = states[1].h

    return z_sequence_output


def bi_lstm_network(input, forget_bias=1.0, lstm_hidden_size_layer=64, lstm_latent_dim=16, lstm_num_layers=2, scope='bi_lstm_network'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Forward and backword cells
        def make_cell():
            cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer, forget_bias=forget_bias)
            return cell

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(lstm_num_layers)])
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(lstm_num_layers)])

        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, dtype=tf.float32)

        fw_output = tf.transpose(outputs[0], [1, 0, 2])
        bw_output = tf.transpose(outputs[1], [1, 0, 2])
        outputs = tf.concat([fw_output[-1], bw_output[-1]], -1)
        print('LSTM output shape: ' + str(outputs.get_shape().as_list()))
        z_sequence_output = fc(outputs, lstm_latent_dim, use_bias=True, scope='linear_transform')

        #states_fw, states_bw = states
        #state_concat = tf.concat([states_fw[1].h, states_bw[1].h], 1)

        # Linear Transform
        #z_sequence_output = fc(state_concat, lstm_latent_dim, use_bias=True, scope='linear_transform')
        #z_sequence_output = states_fw[1].h

    return z_sequence_output


def fc(input_data, out_dim, non_linear_fn=None, initial_value=None, use_bias=True, scope='fc'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_dims = input_data.get_shape().as_list()

        if len(input_dims) == 4:
            _, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_data, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input_data

        if initial_value is None:
            fc_weight = tf.get_variable("weights", shape=[in_dim, out_dim], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            fc_bias = tf.get_variable("bias", shape=[out_dim], initializer=tf.constant_initializer(0.0))
        else:
            fc_weight = tf.get_variable("weights", initializer=initial_value[0])
            fc_bias = tf.get_variable("bias", shape=[out_dim], initializer=initial_value[1])

        if use_bias:
            output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)
        else:
            output = tf.matmul(flat_input, fc_weight)

        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation


def batch_norm(x, b_train, scope, decay=0.999, epsilon=1e-3):
    with tf.variable_scope(scope):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

        return normed


def coord_conv(input, scope, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu, dilation=[1, 1, 1, 1], bias=False, sn=False):
    input_dims = input.get_shape().as_list()
    batch_size, height, width, channels = input_dims

    xx_ones = tf.ones([batch_size, width], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(height), 0), [batch_size, 1])
    xx_range = tf.expand_dims(xx_range, 1)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    print('Coordinate X: ' + str(xx_channel.get_shape().as_list()))

    yy_ones = tf.ones([batch_size, height], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(width), 0), [batch_size, 1])
    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)
    print('Coordinate Y: ' + str(yy_channel.get_shape().as_list()))

    xx_channel = tf.cast(xx_channel, tf.float32) / (width - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = tf.cast(yy_channel, tf.float32) / (height - 1)
    yy_channel = yy_channel * 2 - 1

    rr = tf.sqrt(tf.square(xx_channel) + tf.square(yy_channel))

    coord_tensor = tf.concat([input, xx_channel, yy_channel, rr], axis=-1)

    return conv(coord_tensor, scope, filter_dims, stride_dims, padding, non_linear_fn, dilation, bias, sn)


def conv(input, scope, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu, dilation=[1, 1, 1, 1], bias=False, sn=False):
    input_dims = input.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):

        conv_weight = tf.get_variable('conv_weight',
                                      shape=[filter_h, filter_w, num_channels_in, num_channels_out],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        if bias is True:
            conv_bias = tf.get_variable('conv_bias', shape=[num_channels_out],
                                        initializer=tf.zeros_initializer)

        #conv_weight = tf.Variable(
        #    tf.truncated_normal([filter_h, filter_w, num_channels_in, num_channels_out], stddev=0.1, dtype=tf.float32))

        #if bias is True:
        #    conv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))

        conv_filter = conv_weight

        if sn == True:
            conv_filter = spectral_norm(conv_weight, scope='sn')

        map = tf.nn.conv2d(input, filter=conv_filter, strides=[1, stride_h, stride_w, 1], padding=padding, dilations=dilation)

        if bias is True:
            map = tf.nn.bias_add(map, conv_bias)

        if non_linear_fn is not None:
            activation = non_linear_fn(map)
        else:
            activation = map

        # print(activation.get_shape().as_list())
        return activation


def batch_norm_conv(x, b_train, scope, decay=0.999, epsilon=1e-3):
    with tf.variable_scope(scope):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

        return normed


def add_dense_layer(layer, filter_dims, act_func=tf.nn.relu, scope='dense_layer', norm='layer',
                    b_train=False, use_bias=False, dilation=[1, 1, 1, 1], sn=False):
    with tf.variable_scope(scope):
        l = layer
        l = conv_normalize(l, norm=norm, b_train=b_train, scope='norm')
        l = act_func(l)
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], dilation=dilation,
                 non_linear_fn=None, bias=use_bias, sn=sn)
        l = tf.concat([l, layer], 3)

    return l


def add_residual_layer(layer, filter_dims, act_func=tf.nn.relu, scope='residual_layer',
                       norm='layer', b_train=False, use_bias=False, dilation=[1, 1, 1, 1], sn=False):
    with tf.variable_scope(scope):
        l = layer
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1],
                 dilation=dilation, non_linear_fn=None, bias=use_bias, sn=sn)
        l = conv_normalize(l, norm=norm, b_train=b_train, scope='norm')

        if act_func is not None:
            l = act_func(l)

    return l


def add_dense_transition_layer(layer, filter_dims, stride_dims=[1, 1], act_func=tf.nn.relu, scope='transition',
                               norm='layer', b_train=False, use_pool=True, use_bias=False, sn=False):
    with tf.variable_scope(scope):
        l = layer
        l = conv_normalize(l, norm=norm, b_train=b_train, scope='norm')
        l = act_func(l)
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=stride_dims,
                 non_linear_fn=None, bias=use_bias, sn=sn)

        if use_pool:
            l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return l


def global_avg_pool(input_data, output_length=1, padding='VALID', use_bias=False, scope='gloval_avg_pool'):
    input_dims = input_data.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in

    num_channels_in = input_dims[-1]
    height = input_dims[1]
    width = input_dims[2]

    with tf.variable_scope(scope):
        if num_channels_in != output_length:
            conv_weight = tf.get_variable('gap_weight', shape=[1, 1, num_channels_in, output_length],
                                          initializer=tf.truncated_normal_initializer(stddev=1.0))

            conv = tf.nn.conv2d(input_data, conv_weight, strides=[1, 1, 1, 1], padding='SAME')

            if use_bias == True:
                conv_bias = tf.get_variable('gap_bias', shape=[output_length], initializer=tf.zeros_initializer)
                conv = tf.nn.bias_add(conv, conv_bias)

            pool = tf.nn.avg_pool(conv, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)
        else:
            pool = tf.nn.avg_pool(input_data, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)

        if output_length != 1:
            pool = tf.squeeze(pool, axis=[1, 2])

        return pool


def avg_pool(input, filter_dims, stride_dims, padding='SAME', scope='avgpool'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        pool = tf.nn.avg_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding)

        return pool


def get_deconv2d_output_dims(input_dims, filter_dims, stride_dims, padding):
    batch_size, input_h, input_w, num_channels_in = input_dims
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    if padding == 'SAME':
        out_h = input_h * stride_h
    elif padding == 'VALID':
        out_h = (input_h - 1) * stride_h + filter_h

    if padding == 'SAME':
        out_w = input_w * stride_w
    elif padding == 'VALID':
        out_w = (input_w - 1) * stride_w + filter_w

    return [batch_size, out_h, out_w, num_channels_out]


def deconv(input_data, b_size, scope, filter_dims, stride_dims, padding='SAME', non_linear_fn=tf.nn.relu, sn=False):
    input_dims = input_data.get_shape().as_list()
    # print(scope, 'in', input_dims)
    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    input_dims = [b_size, input_dims[1], input_dims[2], input_dims[3]]
    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        deconv_weight = tf.get_variable('deconv_weight', shape=[filter_h, filter_w, num_channels_out, num_channels_in],
                                       initializer=tf.random_normal_initializer(stddev=0.1))

        deconv_bias = tf.get_variable('deconv_bias', shape=[num_channels_out], initializer=tf.zeros_initializer)

        conv_filter = deconv_weight

        if sn == True:
            conv_filter = spectral_norm(deconv_weight, scope='deconv_sn')

        map = tf.nn.conv2d_transpose(input_data, conv_filter, output_dims, strides=[1, stride_h, stride_w, 1],
                                     padding=padding)

        map = tf.nn.bias_add(map, deconv_bias)

        if non_linear_fn is not None:
            map = non_linear_fn(map)

        # print(scope, 'out', activation.get_shape().as_list())
        return map


def self_attention(x, channels=0, act_func=tf.nn.relu, scope='attention'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        if channels == 0:
            channels = num_channels

        f = conv(x, scope='f_conv', filter_dims=[1, 1, channels // 8], stride_dims=[1, 1], non_linear_fn=act_func)
        f = tf.layers.max_pooling2d(f, pool_size=4, strides=4, padding='SAME')
        print('attention f dims: ' + str(f.get_shape().as_list()))

        g = conv(x, scope='g_conv', filter_dims=[1, 1, channels // 8], stride_dims=[1, 1], non_linear_fn=act_func)
        print('attention g dims: ' + str(g.get_shape().as_list()))

        h = conv(x, scope='h_conv', filter_dims=[1, 1, channels // 8], stride_dims=[1, 1], non_linear_fn=act_func)
        h = tf.layers.max_pooling2d(h, pool_size=4, strides=4, padding='SAME')
        print('attention h dims: ' + str(h.get_shape().as_list()))

        # N = h * w
        g = tf.reshape(g, shape=[-1, g.shape[1] * g.shape[2], g.get_shape().as_list()[-1]])
        print('attention g flat dims: ' + str(g.get_shape().as_list()))

        f = tf.reshape(f, shape=[-1, f.shape[1] * f.shape[2], f.shape[-1]])
        print('attention f flat dims: ' + str(f.get_shape().as_list()))

        s = tf.matmul(g, f, transpose_b=True)  # # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        print('attention beta dims: ' + str(s.get_shape().as_list()))

        h = tf.reshape(h, shape=[-1, h.shape[1] * h.shape[2], h.shape[-1]])
        print('attention h flat dims: ' + str(h.get_shape().as_list()))

        o = tf.matmul(beta, h)  # [bs, N, C]
        print('attention o dims: ' + str(o.get_shape().as_list()))

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=[-1, height, width, num_channels // 8])  # [bs, h, w, C]
        o = conv(o, scope='attn_conv', filter_dims=[1, 1, channels], stride_dims=[1, 1], non_linear_fn=act_func)
        x = gamma * o + x

    return x


def spectral_norm(w, iteration=1, scope='sn'):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None

        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def moments_for_layer_norm(x, axes=1, name=None):
    # output for mean and variance should be [batch_size]
    # from https://github.com/LeavesBreathe/tensorflow_with_latest_papers
    epsilon = 1e-3  # found this works best.

    if not isinstance(axes, list):
        axes = [axes]

    mean = tf.reduce_mean(x, axes, keepdims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keepdims=True) + epsilon)

    return mean, variance


def layer_norm(x, scope="layer_norm", alpha_start=1.0, bias_start=0.0):
    # derived from:
    # https://github.com/LeavesBreathe/tensorflow_with_latest_papers, but simplified.
    with tf.variable_scope(scope):
        input_dims = x.get_shape().as_list()

        if len(input_dims) != 2:
            x = tf.reshape(x, [input_dims[0], -1])

        num_units = x.get_shape().as_list()[1]

        alpha = tf.get_variable('alpha', [num_units],
                                initializer=tf.constant_initializer(alpha_start), dtype=tf.float32)
        bias = tf.get_variable('bias', [num_units],
                               initializer=tf.constant_initializer(bias_start), dtype=tf.float32)

        mean, variance = moments_for_layer_norm(x)
        y = (alpha * (x - mean)) / (variance) + bias

        if len(input_dims) != 2:
           y = tf.reshape(y, input_dims)

    return y


def instance_norm(x, scope="instance_norm", alpha_start=1.0, bias_start=0.0, num_grp=8):
    with tf.variable_scope(scope):
        input_dims = x.get_shape().as_list()
        h = input_dims[1]
        w = input_dims[2]
        c = input_dims[3]
        eps = 1e-5
        g_c = c // num_grp
        x = tf.reshape(x, shape=[-1, num_grp, g_c, h, w])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        alpha = tf.get_variable('alpha', shape=[1, 1, 1, num_grp*g_c], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(alpha_start, 0.02, dtype=tf.float32))
        bias = tf.get_variable('bias', [1, 1, 1, num_grp*g_c],
                               initializer=tf.constant_initializer(bias_start), dtype=tf.float32)
        x = (x - mean) * tf.rsqrt(var + eps)
        x = tf.reshape(x, [-1, h, w, c])
        y = alpha * x + bias

    return y


def add_residual_dense_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, norm='layer', b_train=False,
                             scope='residual_dense_block', use_dilation=False, stochastic_depth=False,
                             stochastic_survive=0.9):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        bn_depth = num_channel_in // (num_layers * 2)
        #bn_depth = bottleneck_depth

        l = conv(l, scope='bt_conv', filter_dims=[1, 1, bn_depth], stride_dims=[1, 1], dilation=[1, 1, 1, 1],
                    non_linear_fn=None, bias=False, sn=False)

        for i in range(num_layers):
            l = add_dense_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=act_func, norm=norm, b_train=b_train,
                                       scope='layer' + str(i), dilation=dilation)

        l = add_dense_transition_layer(l, filter_dims=[1, 1, num_channel_in], act_func=act_func,
                                              scope='dense_transition_1', norm=norm, b_train=b_train, use_pool=False)
        l = conv_normalize(l, norm=norm, b_train=b_train, scope='norm2')
        pl = tf.constant(stochastic_survive)

        def train_mode():
            survive = tf.less(pl, tf.random_uniform(shape=[], minval=0.0, maxval=1.0))
            return tf.cond(survive, lambda: tf.add(l, in_layer), lambda: in_layer)

        def test_mode():
            return tf.add(tf.multiply(pl, l), in_layer)

        if stochastic_depth == True:
            return tf.cond(b_train, train_mode, test_mode)

        l = tf.add(l, in_layer)
        l = act_func(l)

        return l


def add_se_residual_block(in_layer, filter_dims, num_layers=2, act_func=tf.nn.relu, norm='layer',
                       b_train=False, use_residual=True, scope='residual_block', use_dilation=False,
                       sn=False, use_bottleneck=False):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        # Bottle Neck Layer
        if use_bottleneck is True:
            bn_depth = num_channel_out // (num_layers * 2)
            l = conv(l, scope='bt_conv1', filter_dims=[1, 1, bn_depth], stride_dims=[1, 1],
                     dilation=dilation,
                     non_linear_fn=None, bias=False, sn=False)
            l = conv_normalize(l, norm=norm, b_train=b_train, scope='bt_norm1')
            l = act_func(l)
        else:
            bn_depth = num_channel_out

        for i in range(num_layers - 1):
            l = add_residual_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=act_func, norm=norm, b_train=b_train,
                                          scope='layer' + str(i), dilation=dilation, sn=sn)

        if use_bottleneck is True:
            l = act_func(l)
            l = conv(l, scope='bt_conv2', filter_dims=[1, 1, num_channel_out], stride_dims=[1, 1],
                     dilation=dilation,
                     non_linear_fn=None, bias=False, sn=False)
            l = conv_normalize(l, norm=norm, b_train=b_train, scope='bt_norm2')

        # SE Path
        # Squeeze
        sl = global_avg_pool(l, output_length=num_channel_out, scope='squeeze')
        sl = fc(sl, out_dim=num_channel_out // 16, non_linear_fn=tf.nn.relu, scope='reduction')
        sl = fc(sl, out_dim=num_channel_out,  non_linear_fn=tf.nn.sigmoid, scope='transform')
        # Excitation
        sl = tf.expand_dims(sl, axis=1)
        sl = tf.expand_dims(sl, axis=2)
        l = tf.multiply(l, sl)

        if use_residual is True:
            l = tf.add(l, in_layer)
            l = act_func(l)

    return l


def add_residual_block(in_layer, filter_dims, num_layers=2, act_func=tf.nn.relu, norm='layer',
                       b_train=False, use_residual=True, scope='residual_block', use_dilation=False,
                       sn=False, use_bottleneck=False):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        # Bottle Neck Layer
        if use_bottleneck is True:
            bn_depth = num_channel_out // (num_layers * 2)
            l = conv(l, scope='bt_conv1', filter_dims=[1, 1, bn_depth], stride_dims=[1, 1],
                     dilation=[1, 1, 1, 1],
                     non_linear_fn=None, bias=False, sn=False)
            l = conv_normalize(l, norm=norm, b_train=b_train, scope='bt_norm1')
            l = act_func(l)
        else:
            bn_depth = num_channel_out

        for i in range(num_layers - 1):
            l = add_residual_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=act_func, norm=norm, b_train=b_train,
                                          scope='layer' + str(i), dilation=[1, 1, 1, 1], sn=sn)

        if use_dilation is True:
            dl = add_residual_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=None, norm=norm,
                                    b_train=b_train,
                                    scope='dilated_layer', dilation=dilation, sn=sn)
            #l = add_residual_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=None, norm=norm,
            #                       b_train=b_train,
            #                       scope='layer_last', dilation=dilation, sn=sn)
            l = tf.add(l, dl)
        else:
            l = add_residual_layer(l, filter_dims=[filter_dims[0], filter_dims[1], bn_depth], act_func=None, norm=norm,
                                   b_train=b_train,
                                   scope='layer_last', dilation=dilation, sn=sn)

        if use_bottleneck is True:
            l = act_func(l)
            l = conv(l, scope='bt_conv2', filter_dims=[1, 1, num_channel_out], stride_dims=[1, 1],
                     dilation=[1, 1, 1, 1],
                     non_linear_fn=None, bias=False, sn=False)
            l = conv_normalize(l, norm=norm, b_train=b_train, scope='bt_norm2')

        if use_residual is True:
            l = tf.add(l, in_layer)
            l = act_func(l)

    return l


def conv_normalize(input, norm='layer', b_train=True, scope='conv_norm'):
    with tf.variable_scope(scope):
        l = input

        if norm == 'layer':
            l = layer_norm(l, scope=scope)
        elif norm == 'batch':
            l = batch_norm_conv(l, b_train=b_train, scope=scope)
        elif norm == 'instance':
            l = instance_norm(l, scope=scope)

    return l
