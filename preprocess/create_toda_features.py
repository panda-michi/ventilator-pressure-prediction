# refferenced from https://www.kaggle.com/takamichitoda/ventilator-train-classification

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class config:
    USE_LAG = 4
    CONT_FEATURES = ['u_in', 'u_out', 'time_step'] + ['u_in_cumsum', 'u_in_cummean', 'area', 'cross', 'cross2'] + ['R_cate', 'C_cate']
    LAG_FEATURES = ['breath_time']
    LAG_FEATURES += [f'u_in_lag_{i}' for i in range(1, USE_LAG+1)]
    LAG_FEATURES += [f'u_in_time{i}' for i in range(1, USE_LAG+1)]
    LAG_FEATURES += [f'u_out_lag_{i}' for i in range(1, USE_LAG+1)]
    ALL_FEATURES = CONT_FEATURES + LAG_FEATURES

def add_feature(df):
    df['time_delta'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['delta'] = df['time_delta'] * df['u_in']
    df['area'] = df.groupby('breath_id')['delta'].cumsum()

    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] / df['count']
    
    df = df.drop(['count','one'], axis=1)
    return df

def add_lag_feature(df):
    # https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173
    for lag in range(1, config.USE_LAG+1):
        df[f'breath_id_lag{lag}']=df['breath_id'].shift(lag).fillna(0)
        df[f'breath_id_lag{lag}same']=np.select([df[f'breath_id_lag{lag}']==df['breath_id']], [1], 0)

        # u_in 
        df[f'u_in_lag_{lag}'] = df['u_in'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        #df[f'u_in_lag_{lag}_back'] = df['u_in'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_in_time{lag}'] = df['u_in'] - df[f'u_in_lag_{lag}']
        #df[f'u_in_time{lag}_back'] = df['u_in'] - df[f'u_in_lag_{lag}_back']
        df[f'u_out_lag_{lag}'] = df['u_out'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        #df[f'u_out_lag_{lag}_back'] = df['u_out'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']

    # breath_time
    df['time_step_lag'] = df['time_step'].shift(1).fillna(0) * df[f'breath_id_lag{lag}same']
    df['breath_time'] = df['time_step'] - df['time_step_lag']

    drop_columns = ['time_step_lag']
    drop_columns += [f'breath_id_lag{i}' for i in range(1, config.USE_LAG+1)]
    drop_columns += [f'breath_id_lag{i}same' for i in range(1, config.USE_LAG+1)]
    df = df.drop(drop_columns, axis=1)

    # fill na by zero
    df = df.fillna(0)
    return df

c_dic = {10: 0, 20: 1, 50:2}
r_dic = {5: 0, 20: 1, 50:2}
rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}    

def add_category_features(df):
    df['C_cate'] = df['C'].map(c_dic)
    df['R_cate'] = df['R'].map(r_dic)
    df['RC_sum'] = (df['R'] + df['C']).map(rc_sum_dic)
    df['RC_dot'] = (df['R'] * df['C']).map(rc_dot_dic)
    return df

norm_features = config.CONT_FEATURES + config.LAG_FEATURES
def norm_scale(train_df, test_df):
    scaler = RobustScaler()
    all_u_in = np.vstack([train_df[norm_features].values, test_df[norm_features].values])
    scaler.fit(all_u_in)
    train_df[norm_features] = scaler.transform(train_df[norm_features].values)
    test_df[norm_features] = scaler.transform(test_df[norm_features].values)
    return train_df, test_df

def main():
	data_dir = '/content/drive/MyDrive/kaggle/dataset/ventilator-pressure-prediction/'
	train_df = pd.read_csv(data_dir + 'train.csv')
	test_df = pd.read_csv(data_dir + 'test.csv')
	
	train_df = add_feature(train_df)
	test_df = add_feature(test_df)
	train_df = add_lag_feature(train_df)
	test_df = add_lag_feature(test_df)
	
	train_df = add_category_features(train_df)
	test_df = add_category_features(test_df)
	
	train_df, test_df = norm_scale(train_df, test_df)
	
	train_df.to_csv(data_dir + 'train_toda_features.csv', index=False)
	test_df.to_csv(data_dir + 'test_toda_features.csv', index=False)

if __name__ == "__main__":
	main()