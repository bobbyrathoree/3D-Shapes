import tensorflow as tf


def log_this(callback, name, value, batch_number):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_number)
    callback.writer.flush()
