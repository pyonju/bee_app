import cv2
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#%matplotlib inline

# ディレクトリ中のファイル名をリストとして格納
path_wasp = os.listdir('/content/drive/MyDrive/images/wasp/')
path_paper_wasp = os.listdir('/content/drive/MyDrive/images/paper_wasp/')

# 変換した画像を格納する空リストを定義
img_wasp = []
img_paper_wasp = []

# 画像のサイズ
img_size = 200

# スズメバチ
for i in range(len(path_wasp)):
    # ディレクトリ内にある".DS_Store"というファイルを除くためif文を定義
    if path_wasp[i].split('.')[1] == 'jpg':
        img = cv2.imread('/content/drive/MyDrive/images/wasp/' + path_wasp[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_wasp.append(img)

# アシナガバチ
for i in range(len(path_paper_wasp)):
    if path_paper_wasp[i].split('.')[1] == 'jpg':
        img = cv2.imread('/content/drive/MyDrive/images/paper_wasp/' + path_paper_wasp[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_paper_wasp.append(img)
        
X = np.array(img_wasp + img_paper_wasp)
y = np.array([0]*len(img_wasp) + [1]*len(img_paper_wasp))

# 画像データをシャッフル
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]

# トレーニングデータとテストデータに分ける(8:2)
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]

# ラベルデータをone-hotベクトルに変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 画像を水増しする関数
def img_augment(x, y):
    X_augment = []
    y_augment = []
    i = 0

    # 水増しを20回繰り返す
    while i < 20:
        datagen = ImageDataGenerator(rotation_range=30,
                                    width_shift_range=0.3,
                                    height_shift_range=0.3,
                                    horizontal_flip = True,
                                    zoom_range = [0.9, 0.9])
        datagen = datagen.flow(X_train, y_train, shuffle = False, batch_size = len(X_train))
        X_augment.append(datagen.next()[0])
        y_augment.append(datagen.next()[1])
        i += 1

    # numpy配列に変換
    X_extend = np.array(X_augment).reshape(-1, img_size, img_size, 3)
    y_extend = np.array(y_augment).reshape(-1, 2)

    return X_extend, y_extend

# trainデータの水増し
img_add = img_augment(X_train, y_train)

# 元の画像データと水増しデータを統合
X_train = np.concatenate([X_train, img_add[0]])
y_train = np.concatenate([y_train, img_add[1]])

# VGG16
input_tensor = Input(shape=(img_size, img_size, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# vggのoutputを受け取り、2クラス分類する層を定義
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.1))
top_model.add(BatchNormalization())
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.1))
top_model.add(BatchNormalization())
top_model.add(Dense(2, activation='softmax'))

# vggと、top_modelを連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vggの層の重みを固定
for layer in model.layers[:15]:
    layer.trainable = False

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# 学習
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# モデルを保存
model.save('bee_app.h5')

# 可視化
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
