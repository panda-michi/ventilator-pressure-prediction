import tensorflow as tf

def mask_mae(y_true, y_pred, u_out_col = 1):
    y = y_true[:, :, -1]

    error = tf.abs(y - y_pred[:, :, 0])
    u_out = 1 - y_true[:, :, u_out_col]
    # w = 1 - u_out
    w = u_out
    error = w * error
    return tf.reduce_sum(error, axis=-1) / tf.reduce_sum(w, axis=-1)