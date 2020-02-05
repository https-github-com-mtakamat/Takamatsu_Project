import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model
import numpy as np
import tensorflow as tf

image_shape=(256,256,3)

#dice loss
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

#l1 loss
def l1_loss(y_true,y_pred):
    return K.mean(K.abs(y_pred-y_true))

#l2 loss
def l1_loss(y_true,y_pred):
    return K.mean(K.square(y_pred-y_true))

#perceptual loss 100
def perceptual_loss_100(y_true,y_pred):
    return 100*perceptual_loss(y_true,y_pred)

#perceptual loss
def perceptual_loss(y_true,y_pred):
    return perceptual_loss_fixed

def perceptual_loss_fixed(y_true,y_pred):
    vgg=vgg19(include=False,weigt='imagenet',input_shape=image_shape)

    loss_model_1=Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
    loss_model_2=Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
    loss_model_3=Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv4').output)
    loss_model_4=Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv4').output)
    loss_model_5=Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)

    loss_model_1.trainable=False
    loss_model_2.trainable=False
    loss_model_3.trainable=False
    loss_model_4.trainable=False
    loss_model_5.trainable=False

#L1 lossで，context lossを計算
    p1=K.mean(K.abs(loss_model_1(y_true) - loss_model_1(y_pred)))
    p2=K.mean(K.abs(loss_model_2(y_true) - loss_model_1(y_pred)))
    p3=K.mean(K.abs(loss_model_3(y_true) - loss_model_1(y_pred)))
    p4=K.mean(K.abs(loss_model_4(y_true) - loss_model_1(y_pred)))
    p5=K.mean(K.abs(loss_model_5(y_true) - loss_model_1(y_pred)))

    return p1+p2+p3+p4+p5

#binary focal loss
# Focal loss function
def focal_loss(y_true, y_pred):
	return focal_loss_fixed(y_true, y_pred)

alpha=0.25
gamma=2.0

def focal_loss_fixed(y_true, y_pred):

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

#categorical focal loss
def categorical_focal_loss(y_true, y_pred):
    return categorical_focal_loss_fixed

alpha=0.25
gamma=2.0


    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Sum the losses in mini_batch
    return K.sum(loss, axis=1)
