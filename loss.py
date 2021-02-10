import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy, cosine_similarity
import tensorflow as tf

def loss(num_classes, emb_size):
    def custom_loss(y_true, y_pred):

        y_true_label_1 = y_true[:,:num_classes]
        y_true_label_2 = y_true[:,num_classes:num_classes*2]
        y_pred_label_1 = y_pred[:, :num_classes]
        y_pred_label_2 = y_pred[:, num_classes:num_classes*2]

        y_pred_embedding_1 = y_pred[:,num_classes*2:num_classes*2 + emb_size]
        y_pred_embedding_2 = y_pred[:,num_classes*2 + emb_size:]

        class_loss_1 = categorical_crossentropy(y_true_label_1, y_pred_label_1)
        class_loss_2 = categorical_crossentropy(y_true_label_2, y_pred_label_2)
        embedding_loss = cosine_similarity(y_pred_embedding_1, y_pred_embedding_2)
        
        are_labels_equal = K.all(K.equal(y_true_label_1, y_true_label_2), axis=1)

        a = tf.where(are_labels_equal,
                     tf.fill([tf.shape(are_labels_equal)[0]], 1.0),
                     tf.fill([tf.shape(are_labels_equal)[0]], -1.0))

        result = class_loss_1 + class_loss_2 + tf.math.multiply(a, embedding_loss)

        return tf.math.reduce_mean(result)
    return custom_loss
