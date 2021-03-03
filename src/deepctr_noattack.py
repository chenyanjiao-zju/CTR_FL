import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch
import copy

if __name__ == "__main__":
    data_dir = '../data/new_data.csv'
    data_full = pd.read_csv(data_dir)
    data = copy.deepcopy(data_full.iloc[0:len(data_full) // 2,:])

    sparse_features = ['uid','task_id','adv_id','creat_type_cd','adv_prim_id','dev_id',
    'inter_type_cd','slot_id','spread_app_id','tags','app_first_class','app_second_class',
    'city','city_rank','device_name','career','gender','net_type','residence','emui_dev',
    'up_membership_grade','consume_purchase','indu_name','pt_d']

    dense_features = ['age','device_size','his_app_size','his_on_shelf_time','app_score','list_time',
    'device_price','up_life_duration','up_membership_grade','membership_life_duration',
    'consume_purchase','communication_avgonline_30d']

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    # print(data[target].value_counts())
    # exit(0)  # 0:1 = 28.1


    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
                            for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    # # 3.1 使label=1 均匀分布
    # label_1_index = []   # 存放label=1的index的列表
    # for i in range(1,len(train)):
    #     if train.iloc[i, 1] == 1:
    #         label_1_index.append(i)
    # # print(len(train) / len(label_1_index)) # 数据集中正负样本比例是1：28.992735763393437
    # # exit(0)
    #
    # i = 0
    # while i * 29 < len(train) and i < len(label_1_index):
    #     print('i',i)
    #     index = int(i * 29)      # i*29，即每29个label=0，就会跟一个label=1
    #     a, b = train.iloc[index].copy(), train.iloc[label_1_index[i]].copy()
    #     train.iloc[index], train.iloc[label_1_index[i]] = b, a
    #     # 交换 index索引数据 和 label_1_index[i]索引数据
    #     i += 1


    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    pos_idx = np.where(test['label'] == 1)[0]
    neg_idx = np.delete(np.array([i for i in range(len(test))]), pos_idx)

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                    metrics=['binary_crossentropy'], )


    history = model.fit(train_model_input, train[target].values,
                        batch_size=1024, epochs=10, verbose=2, validation_split=0.1, )

    torch.save(model, "cleanmodel.pth")
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))



    #
    # np.save("pred.npy", np.array(pred_ans))
    # np.save("neg.npy", np.array(neg_idx))
    # np.save("pos.npy", np.array(pos_idx))