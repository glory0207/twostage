"""
IterNet模型定义 - 兼容新版本TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras import losses


def get_unet(minimum_kernel=32, do=0, activation=layers.ReLU, iteration=1):
    """
    构建IterNet模型
    
    Args:
        minimum_kernel: 最小卷积核数量
        do: dropout率
        activation: 激活函数
        iteration: 迭代次数
    
    Returns:
        model: Keras模型
    """
    inputs = layers.Input((None, None, 3))
    conv1 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel, (3, 3), padding='same')(inputs)))
    conv1 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel, (3, 3), padding='same')(conv1)))
    a = conv1
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')(pool1)))
    conv2 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv2)))
    b = conv2
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 4, (3, 3), padding='same')(pool2)))
    conv3 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv3)))
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 8, (3, 3), padding='same')(pool3)))
    conv4 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv4)))
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 16, (3, 3), padding='same')(pool4)))
    conv5 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 16, (3, 3), padding='same')(conv5)))

    up6 = layers.concatenate([layers.Conv2DTranspose(minimum_kernel * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4],
                      axis=3)
    conv6 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 8, (3, 3), padding='same')(up6)))
    conv6 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 8, (3, 3), padding='same')(conv6)))

    up7 = layers.concatenate([layers.Conv2DTranspose(minimum_kernel * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3],
                      axis=3)
    conv7 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 4, (3, 3), padding='same')(up7)))
    conv7 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 4, (3, 3), padding='same')(conv7)))

    up8 = layers.concatenate([layers.Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                      axis=3)
    conv8 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')(up8)))
    conv8 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')(conv8)))

    up9 = layers.concatenate([layers.Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel, (3, 3), padding='same')(up9)))
    conv9 = layers.Dropout(do)(activation()(layers.Conv2D(minimum_kernel, (3, 3), padding='same')(conv9)))

    # 定义可重用的层
    pt_conv1a = layers.Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation1a = activation()
    pt_dropout1a = layers.Dropout(do)
    pt_conv1b = layers.Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation1b = activation()
    pt_dropout1b = layers.Dropout(do)
    pt_pooling1 = layers.MaxPooling2D(pool_size=(2, 2))

    pt_conv2a = layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation2a = activation()
    pt_dropout2a = layers.Dropout(do)
    pt_conv2b = layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation2b = activation()
    pt_dropout2b = layers.Dropout(do)
    pt_pooling2 = layers.MaxPooling2D(pool_size=(2, 2))

    pt_conv3a = layers.Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_activation3a = activation()
    pt_dropout3a = layers.Dropout(do)
    pt_conv3b = layers.Conv2D(minimum_kernel * 4, (3, 3), padding='same')
    pt_activation3b = activation()
    pt_dropout3b = layers.Dropout(do)

    pt_tranconv8 = layers.Conv2DTranspose(minimum_kernel * 2, (2, 2), strides=(2, 2), padding='same')
    pt_conv8a = layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation8a = activation()
    pt_dropout8a = layers.Dropout(do)
    pt_conv8b = layers.Conv2D(minimum_kernel * 2, (3, 3), padding='same')
    pt_activation8b = activation()
    pt_dropout8b = layers.Dropout(do)

    pt_tranconv9 = layers.Conv2DTranspose(minimum_kernel, (2, 2), strides=(2, 2), padding='same')
    pt_conv9a = layers.Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation9a = activation()
    pt_dropout9a = layers.Dropout(do)
    pt_conv9b = layers.Conv2D(minimum_kernel, (3, 3), padding='same')
    pt_activation9b = activation()
    pt_dropout9b = layers.Dropout(do)

    conv9s = [conv9]
    outs = []
    a_layers = [a]
    
    for iteration_id in range(iteration):
        out = layers.Conv2D(1, (1, 1), activation='sigmoid', name=f'out1{iteration_id + 1}')(conv9s[-1])
        outs.append(out)

        # 构建迭代部分
        conv1_iter = pt_dropout1a(pt_activation1a(pt_conv1a(conv9s[-1])))
        conv1_iter = pt_dropout1b(pt_activation1b(pt_conv1b(conv1_iter)))
        a_layers.append(conv1_iter)
        conv1_iter = layers.concatenate(a_layers, axis=3)
        conv1_iter = layers.Conv2D(minimum_kernel, (1, 1), padding='same')(conv1_iter)
        pool1_iter = pt_pooling1(conv1_iter)

        conv2_iter = pt_dropout2a(pt_activation2a(pt_conv2a(pool1_iter)))
        conv2_iter = pt_dropout2b(pt_activation2b(pt_conv2b(conv2_iter)))
        pool2_iter = pt_pooling2(conv2_iter)

        conv3_iter = pt_dropout3a(pt_activation3a(pt_conv3a(pool2_iter)))
        conv3_iter = pt_dropout3b(pt_activation3b(pt_conv3b(conv3_iter)))

        up8_iter = layers.concatenate([pt_tranconv8(conv3_iter), conv2_iter], axis=3)
        conv8_iter = pt_dropout8a(pt_activation8a(pt_conv8a(up8_iter)))
        conv8_iter = pt_dropout8b(pt_activation8b(pt_conv8b(conv8_iter)))

        up9_iter = layers.concatenate([pt_tranconv9(conv8_iter), conv1_iter], axis=3)
        conv9_iter = pt_dropout9a(pt_activation9a(pt_conv9a(up9_iter)))
        conv9_iter = pt_dropout9b(pt_activation9b(pt_conv9b(conv9_iter)))

        conv9s.append(conv9_iter)

    # 最终输出
    out2 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='final_out')(conv9s[-1])
    outs.append(out2)

    model = Model(inputs=[inputs], outputs=outs)

    # 定义损失函数
    loss_funcs = {}
    for iteration_id in range(iteration):
        loss_funcs[f'out1{iteration_id + 1}'] = 'binary_crossentropy'
    loss_funcs['final_out'] = 'binary_crossentropy'

    # 定义指标
    metrics = {
        "final_out": ['accuracy']
    }

    # 编译模型 (使用新版本的API)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss_funcs, metrics=metrics)

    return model
