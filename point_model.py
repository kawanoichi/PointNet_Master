"""機械学習のモデル

以下の論文で使用されたモデル(python2から3に変更)
論文URL : https://arxiv.org/pdf/1612.00603.pdf
@author kawanoichi
"""
import tensorflow as tf

#モデル
def PointModel(OUTPUTPOINTS, BATCH_SIZE):
    """モデル.

    Args:
        OUTPUTPOINTS (int): 出力する点の数
        BATCH_SIZE (int): バッチサイズ
    return:
        model: モデル
    """

#----------------------------------------------------------------encoder----------------------------------------------------------------
    #192 256
    inputs = tf.keras.layers.Input(shape=(192, 256, 4), batch_size=BATCH_SIZE)
    x = tf.keras.layers.Conv2D(16, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(inputs)
    x = tf.keras.layers.Conv2D(16, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    # x0=x
    x = tf.keras.layers.Conv2D(32, (3,3),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #96 128
    x = tf.keras.layers.Conv2D(32, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(32, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x1=x
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #48 64
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x2=x
    x = tf.keras.layers.Conv2D(128, (3,3),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #24 32
    x = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x3=x
    x = tf.keras.layers.Conv2D(256, (5,5),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #12 16
    x = tf.keras.layers.Conv2D(256, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(256, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x4=x
    x = tf.keras.layers.Conv2D(512, (5,5),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #6 8
    x = tf.keras.layers.Conv2D(512, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(512, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x5=x
    x = tf.keras.layers.Conv2D(512, (5,5),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #----------------------------------------------------------------decoder----------------------------------------------------------------
    #3 4
    x_additional = tf.keras.layers.Flatten()(x)
    x_additional = tf.keras.layers.Dense(2048, activation='relu', activity_regularizer='L2')(x_additional)
    x = tf.keras.layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', activation='linear',  activity_regularizer='L2')(x)
    #6 8
    x5 = tf.keras.layers.Conv2D(256, (5,5),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x5)
    x=tf.nn.relu(tf.add(x,x5))
    x = tf.keras.layers.Conv2D(256, (5,5),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x5=x
    x = tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', activation='linear',  activity_regularizer='L2')(x)
    #12 16
    x4 = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x4)
    x=tf.nn.relu(tf.add(x,x4))
    x = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x4=x 
    x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', activation='linear',  activity_regularizer='L2')(x)
    #24 32
    x3 = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x3)
    x=tf.nn.relu(tf.add(x,x3))
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x3=x
    x = tf.keras.layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', activation='linear',  activity_regularizer='L2')(x)
    #48 64
    x2 = tf.keras.layers.Conv2D(32, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x2)
    x=tf.nn.relu(tf.add(x,x2))
    x = tf.keras.layers.Conv2D(32, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x2=x
    x = tf.keras.layers.Conv2DTranspose(16, (5,5), strides=(2,2), padding='same', activation='linear',  activity_regularizer='L2')(x)		# print("outputs x",x.shape)
    #----------------------------------------------------------------decoder----------------------------------------------------------------
    #96 128
    x1 = tf.keras.layers.Conv2D(16, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x1)
    x=tf.nn.relu(tf.add(x,x1))
    x = tf.keras.layers.Conv2D(16, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(32, (3,3),  strides=(2,2), padding = "same", activation='linear', activity_regularizer='L2')(x)
    #48 64
    x2 = tf.keras.layers.Conv2D(32, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x2)
    x=tf.nn.relu(tf.add(x,x2))
    x = tf.keras.layers.Conv2D(32, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    # x2=x
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #24 32
    x3 = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x3)
    x=tf.nn.relu(tf.add(x,x3))
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x3=x
    x = tf.keras.layers.Conv2D(128, (5,5),  strides=(2,2), padding = "same", activation='linear', activity_regularizer='L2')(x)
    #12 16
    x4 = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x4)
    x=tf.nn.relu(tf.add(x,x4))
    x = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x=x
    x = tf.keras.layers.Conv2D(256, (5,5),  strides=(2,2), padding = "same", activation='linear', activity_regularizer='L2')(x)
    #6 8
    x5 = tf.keras.layers.Conv2D(256, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x5)
    x=tf.nn.relu(tf.add(x,x5))
    x = tf.keras.layers.Conv2D(256, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x5=x
    x = tf.keras.layers.Conv2D(512, (5,5),  strides=(2,2), padding = "same", activation='relu', activity_regularizer='L2')(x)
    #---------------------------------------------------------------predictor---------------------------------------------------------------
    #3 4
    x_additional = tf.keras.layers.Dense(2048, activation='relu', activity_regularizer='L2')(x_additional)
    xx = tf.keras.layers.Flatten()(x)
    xx = tf.keras.layers.Dense(2048, activation='relu', activity_regularizer='L2')(xx)
    x_additional = tf.nn.relu(tf.add(x_additional, xx))
    x = tf.keras.layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', activation='linear',  activity_regularizer='L2')(x)
    #6 8
    x5 = tf.keras.layers.Conv2D(256, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x5)
    x=tf.nn.relu(tf.add(x,x5))
    x = tf.keras.layers.Conv2D(256, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    # x5=x
    x = tf.keras.layers.Conv2DTranspose(128, (5,5),  strides=(2,2), padding = "same", activation='linear', activity_regularizer='L2')(x)
    #12 16
    x4 = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x4)
    x=tf.nn.relu(tf.add(x,x4))
    x = tf.keras.layers.Conv2D(128, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    # x4=x
    x = tf.keras.layers.Conv2DTranspose(64, (5,5),  strides=(2,2), padding = "same", activation='linear', activity_regularizer='L2')(x)
    #24 32
    x3 = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x3)
    x=tf.nn.relu(tf.add(x,x3))
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)
    x = tf.keras.layers.Conv2D(64, (3,3),  strides=(1,1), padding = "same", activation='relu', activity_regularizer='L2')(x)    
    x_additional = tf.keras.layers.Dense(OUTPUTPOINTS, activation='relu', activity_regularizer='L2')(x_additional)
    x_additional = tf.keras.layers.Dense(256*3, activation='linear', activity_regularizer='L2')(x_additional)
    x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))

#---------------------------------------------------------------------------------------------------------------------------------------
    #  (32, 24, 32, 64)
    x = tf.keras.layers.Conv2D(3, (3,3),  strides=(1,1), padding = "same", activation='linear', activity_regularizer='L2')(x)
    x = tf.reshape(x,(BATCH_SIZE,32*24,3))
    x = tf.concat([x_additional,x],1)
    outputs = tf.reshape(x,(BATCH_SIZE,OUTPUTPOINTS,3))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="point_net")
    # model.summary()
    return model
