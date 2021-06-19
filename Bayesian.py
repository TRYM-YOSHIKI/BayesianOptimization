import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import copy
import warnings
warnings.simplefilter('ignore')
from bayes_opt import BayesianOptimization

#全データを読み取る, csvファイルからPandas DataFrameへ読み込み
data = pd.read_csv('train_3.csv', header=None, delimiter=',', low_memory=False)

#dataのtargetをカテゴリーに変換
data[6] = data[6].astype('category')

# ラベルエンコーディング（LabelEncoder）
le = LabelEncoder()
encoded = le.fit_transform(data[6].values)
decoded = le.inverse_transform(encoded)
data[6] = encoded


#メイン--------------------------------------------------------------
def main():
    optimizer = bayesOpt()
    print(optimizer.res)


#ベイズ最適化---------------------------------------------------------
def bayesOpt():
    # 最適化するパラメータの下限・上限
    pbounds = {
        'l1': (8, 50),
        'l2': (8, 50),
        'l1_drop': (0.0, 0.5),
        'l2_drop': (0.0, 0.5),
        'epochs': (5, 500),
        'batch_size': (8, 64)
    }
    # 関数と最適化するパラメータを渡す
    optimizer = BayesianOptimization(f=validate, pbounds=pbounds)
    # 最適化
    optimizer.maximize(init_points=5, n_iter=10, acq='ucb')
    return optimizer
    


#評価------------------------------------------------------------------
def validate(l1, l2, l1_drop, l2_drop, epochs, batch_size):

    #モデルを構築&コンパイル----------------------
    def set_model(l1, l2, l1_drop, l2_drop, epochs, batch_size):
        #モデルを構築
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(6,)),
            keras.layers.Dense(int(l1), activation='relu'),
            keras.layers.Dropout(l1_drop),
            keras.layers.Dense(int(l2), activation='relu'),
            keras.layers.Dropout(l2_drop),
            keras.layers.Dense(5, activation='softmax')
        ])

        #モデルをコンパイル
        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model


    #訓練データとテストデータに分割------------------
    def split_data(i):
        train_scope = [(251, 1000), (501, 250), (751, 500), (1, 750)]  #データを分割するための範囲
        test_scope = [(1, 250), (251, 500), (501, 750), (751, 1000)]  #データを分割するための範囲
        if i != 3 or i != 0:
            train = data[(data[7] >= train_scope[i][0]) | (data[7] <= train_scope[i][1])]
            test = data[(data[7] >= test_scope[i][0]) & (data[7] <= test_scope[i][1])]
        else:
            train = data[(data[7] >= train_scope[i][0]) & (data[7] <= train_scope[i][1])]
            test = data[(data[7] >= test_scope[i][0]) & (data[7] <= test_scope[i][1])]
        return train, test


    #交叉検証------------------------------------
    def Closs_validate(l1, l2, l1_drop, l2_drop, epochs, batch_size):
        eval_sum = 0.0  #評価を格納
        for i in range(4):
            #訓練データとテストデータに分割
            train, test = split_data(i)
            
            #データとラベルを分割する
            x_train, y_train = train.drop([6], axis=1).drop([7], axis=1), train[6]
            x_test, y_test = test.drop([6], axis=1).drop([7], axis=1), test[6]

            #モデルをセット
            model = set_model(l1, l2, l1_drop, l2_drop, epochs, batch_size)

            #モデルを学習
            model.fit(x_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

            #テストデータを適用
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

            #評価を格納
            eval_sum += test_acc
        return eval_sum/4
        
    return Closs_validate(l1, l2, l1_drop, l2_drop, epochs, batch_size)


if __name__ == '__main__':
    main()