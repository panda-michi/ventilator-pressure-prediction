import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# credit https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173
def add_features_v1(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    
    df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
    df['breath_id_lag5']=df['breath_id'].shift(5).fillna(0)
    df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
    df['breath_id_lag5same']=np.select([df['breath_id_lag5']==df['breath_id']],[1],0)

    df['u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['u_in_lag'] = df['u_in_lag']*df['breath_id_lagsame']
    df['u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['u_in_lag2'] = df['u_in_lag2']*df['breath_id_lag2same']
    df['u_out_lag2'] = df['u_out'].shift(2).fillna(0)
    df['u_out_lag2'] = df['u_out_lag2']*df['breath_id_lag2same']

    df['u_in_runmean5'] = df['u_in'].rolling(5).mean().fillna(0)
    df['u_in_runmean5'] = df['u_in_runmean5']*df['breath_id_lag5same']
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['RC'] = df['R']+df['C']
    df = pd.get_dummies(df)

    return df

def add_features_v2(df, n_round=-1):
    if n_round > 0:
        u_in_value = df['u_in'].values
        u_in_value = np.round(u_in_value, n_round)
        df['u_in'] = u_in_value

    df['time_step_diff'] = (df['time_step']).groupby(df['breath_id']).diff(-1).fillna(0).abs()
    df['area'] = df['time_step_diff'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df.drop(['time_step_diff'], axis=1, inplace=True)

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['RC'] = df["R"].astype(str) + '_' + df["C"].astype(str)

    df = pd.get_dummies(df)

    return df

def add_features_v3(df):
    df['time_step_diff'] = (df['time_step']).groupby(df['breath_id']).diff(-1).fillna(0).abs()
    df['area_modified'] = df['time_step_diff'] * df['u_in']
    df['area_modified'] = df.groupby('breath_id')['area_modified'].cumsum()
    df.drop(['time_step_diff'], axis=1, inplace=True)

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    df['u_in_lag2'] = (df['u_in']).groupby(df['breath_id']).shift(2).fillna(0)
    df['u_in_lag4'] = (df['u_in']).groupby(df['breath_id']).shift(4).fillna(0)
	
    df['ewm_u_in_mean'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_std'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,drop=True)
    
    df['rolling_10_mean'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_10_max'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_10_std'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,drop=True) 

    df['expand_mean'] = df.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0,drop=True)
    df['expand_max'] = df.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0,drop=True)
    df['expand_std'] = df.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0,drop=True)

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df = pd.get_dummies(df)

    df.fillna(0, inplace=True)

    return df

def add_features_v4(df):
    df['time_step_diff'] = (df['time_step']).groupby(df['breath_id']).diff(-1).fillna(0).abs()
    df['area_modified'] = df['time_step_diff'] * df['u_in']
    df['area_modified'] = df.groupby('breath_id')['area_modified'].cumsum()
    df.drop(['time_step_diff'], axis=1, inplace=True)

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    df['u_in_lag1'] = (df['u_in']).groupby(df['breath_id']).shift(1).fillna(0)
    df['u_in_lag2'] = (df['u_in']).groupby(df['breath_id']).shift(2).fillna(0)
    df['u_in_lag3'] = (df['u_in']).groupby(df['breath_id']).shift(3).fillna(0)
    df['u_in_lag4'] = (df['u_in']).groupby(df['breath_id']).shift(4).fillna(0)
    df['u_in_lag_back1'] = (df['u_in']).groupby(df['breath_id']).shift(-1).fillna(0)
    df['u_in_lag_back2'] = (df['u_in']).groupby(df['breath_id']).shift(-2).fillna(0)
    df['u_in_lag_back3'] = (df['u_in']).groupby(df['breath_id']).shift(-3).fillna(0)
    df['u_in_lag_back4'] = (df['u_in']).groupby(df['breath_id']).shift(-4).fillna(0)

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag2']
	
    df['ewm_u_in_mean'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_std'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,drop=True)
    
    df['rolling_10_mean'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_10_max'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_10_std'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,drop=True) 

    df['expand_mean'] = df.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0,drop=True)
    df['expand_max'] = df.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0,drop=True)
    df['expand_std'] = df.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0,drop=True)

    df['u_in_diffmax'] = df.groupby('breath_id')['u_in'].transform('max') - df['u_in']
    df['u_in_diffmean'] = df.groupby('breath_id')['u_in'].transform('mean') - df['u_in']

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['RC'] = df['R']+df['C']
    df = pd.get_dummies(df)

    df.fillna(0, inplace=True)

    return df

def add_features_v5(df):
    df['time_step_diff'] = (df['time_step']).groupby(df['breath_id']).diff(-1).fillna(0).abs()
    df['area_modified'] = df['time_step_diff'] * df['u_in']
    df['area_modified'] = df.groupby('breath_id')['area_modified'].cumsum()
    df.drop(['time_step_diff'], axis=1, inplace=True)

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()

    df['sim_in_lag1'] = (df['similar_u_in']).groupby(df['breath_id']).shift(1).fillna(0)
    df['sim_in_lag2'] = (df['similar_u_in']).groupby(df['breath_id']).shift(2).fillna(0)
    df['sim_in_lag3'] = (df['similar_u_in']).groupby(df['breath_id']).shift(3).fillna(0)
    df['sim_in_lag4'] = (df['similar_u_in']).groupby(df['breath_id']).shift(4).fillna(0)
    df['sim_in_lag_back1'] = (df['similar_u_in']).groupby(df['breath_id']).shift(-1).fillna(0)
    df['sim_in_lag_back2'] = (df['similar_u_in']).groupby(df['breath_id']).shift(-2).fillna(0)
    df['sim_in_lag_back3'] = (df['similar_u_in']).groupby(df['breath_id']).shift(-3).fillna(0)
    df['sim_in_lag_back4'] = (df['similar_u_in']).groupby(df['breath_id']).shift(-4).fillna(0)
    df.fillna(0, inplace=True)

    df['sim_in_diff1'] = df['similar_u_in'] - df['sim_in_lag1']
    df['sim_in_diff2'] = df['similar_u_in'] - df['sim_in_lag2']
    df['sim_in_diff3'] = df['similar_u_in'] - df['sim_in_lag3']
    df['sim_in_diff4'] = df['similar_u_in'] - df['sim_in_lag4']
	
    df['ewm_sim_in_mean'] = df.groupby('breath_id')['similar_u_in'].ewm(halflife=10).mean().reset_index(level=0,drop=True)
    df['ewm_sim_in_std'] = df.groupby('breath_id')['similar_u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True)
    df['ewm_sim_in_corr'] = df.groupby('breath_id')['similar_u_in'].ewm(halflife=10).corr().reset_index(level=0,drop=True)

    df['sim_rolling_10_mean'] = df.groupby('breath_id')['similar_u_in'].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)
    df['sim_rolling_10_max'] = df.groupby('breath_id')['similar_u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,drop=True)
    df['sim_rolling_10_std'] = df.groupby('breath_id')['similar_u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,drop=True) 

    df['sim_expand_mean'] = df.groupby('breath_id')['similar_u_in'].expanding(2).mean().reset_index(level=0,drop=True)
    df['sim_expand_max'] = df.groupby('breath_id')['similar_u_in'].expanding(2).max().reset_index(level=0,drop=True)
    df['sim_expand_std'] = df.groupby('breath_id')['similar_u_in'].expanding(2).std().reset_index(level=0,drop=True)

    df['sim_in_diffmax'] = df.groupby('breath_id')['similar_u_in'].transform('max') - df['similar_u_in']
    df['sim_in_diffmean'] = df.groupby('breath_id')['similar_u_in'].transform('mean') - df['similar_u_in']

    df['diff_sim_u_in'] = df['u_in'] - df['similar_u_in'] 

    df['u_in_lag1'] = (df['u_in']).groupby(df['breath_id']).shift(1).fillna(0)
    df['u_in_lag2'] = (df['u_in']).groupby(df['breath_id']).shift(2).fillna(0)
    df['u_in_lag3'] = (df['u_in']).groupby(df['breath_id']).shift(3).fillna(0)
    df['u_in_lag4'] = (df['u_in']).groupby(df['breath_id']).shift(4).fillna(0)
    df['u_in_lag_back1'] = (df['u_in']).groupby(df['breath_id']).shift(-1).fillna(0)
    df['u_in_lag_back2'] = (df['u_in']).groupby(df['breath_id']).shift(-2).fillna(0)
    df['u_in_lag_back3'] = (df['u_in']).groupby(df['breath_id']).shift(-3).fillna(0)
    df['u_in_lag_back4'] = (df['u_in']).groupby(df['breath_id']).shift(-4).fillna(0)
    df.fillna(0, inplace=True)

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']

    df['ewm_u_in_mean'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_std'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,drop=True)

    df['rolling_10_mean'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_10_max'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_10_std'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,drop=True) 

    df['expand_mean'] = df.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0,drop=True)
    df['expand_max'] = df.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0,drop=True)
    df['expand_std'] = df.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0,drop=True)

    df['u_in_diffmax'] = df.groupby('breath_id')['u_in'].transform('max') - df['u_in']
    df['u_in_diffmean'] = df.groupby('breath_id')['u_in'].transform('mean') - df['u_in']

    df['u_out_lag1'] = (df['u_out']).groupby(df['breath_id']).shift(1).fillna(0)
    df['u_out_lag2'] = (df['u_out']).groupby(df['breath_id']).shift(2).fillna(0)
    df['u_out_lag3'] = (df['u_out']).groupby(df['breath_id']).shift(3).fillna(0)
    df['u_out_lag4'] = (df['u_out']).groupby(df['breath_id']).shift(4).fillna(0)
    df.fillna(0, inplace=True)
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']

    df['cross'] = df['u_in']*df['u_out']
    df['cross2'] = df['time_step']*df['u_out']
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)

    df = pd.get_dummies(df)

    df.fillna(0, inplace=True)

    return df

def add_features_v6(df, n_round=-1):
    if n_round >= 0:
        u_in_value = df['u_in'].values
        u_in_value = np.round(u_in_value, n_round)
        df['u_in'] = u_in_value
    
    print(len(df['u_in'].unique()))

    df['time_step_diff'] = (df['time_step']).groupby(df['breath_id']).diff(-1).fillna(0).abs()
    df['area'] = df['time_step_diff'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df.drop(['time_step_diff'], axis=1, inplace=True)

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']

    df['ewm_u_in_mean'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,drop=True)
    df['ewm_u_in_std'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True)
    df['ewm_u_in_corr'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,drop=True)

    df['rolling_10_mean'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)
    df['rolling_10_max'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,drop=True)
    df['rolling_10_std'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,drop=True) 

    df['expand_mean'] = df.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0,drop=True)
    df['expand_max'] = df.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0,drop=True)
    df['expand_std'] = df.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0,drop=True)
    df = df.fillna(0)
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['RC'] = df["R"].astype(str) + '_' + df["C"].astype(str)

    df = pd.get_dummies(df)

    return df

# reffered from https://www.kaggle.com/dlaststark/gb-vpp-pulp-fiction
def add_features_v7(df):
    df['cross']= df['u_in'] * df['u_out']
    df['cross2']= df['time_step'] * df['u_out']
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    print("Step-1...Completed")
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    print("Step-2...Completed")
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    print("Step-3...Completed")
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    print("Step-4...Completed")
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    
    df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
    df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
    df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']
    print("Step-5...Completed")
    
    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['ewm_u_in_mean'] = (df\
                           .groupby('breath_id')['u_in']\
                           .ewm(halflife=9)\
                           .mean()\
                           .reset_index(level=0,drop=True))
    df[["15_in_sum","15_in_min","15_in_max","15_in_mean"]] = (df\
                                                              .groupby('breath_id')['u_in']\
                                                              .rolling(window=15,min_periods=1)\
                                                              .agg({"15_in_sum":"sum",
                                                                    "15_in_min":"min",
                                                                    "15_in_max":"max",
                                                                    "15_in_mean":"mean"})\
                                                               .reset_index(level=0,drop=True))
    print("Step-6...Completed")
    
    df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']
    df['u_out_lagback_diff1'] = df['u_out'] - df['u_out_lag_back1']
    df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag_back2']
    df['u_out_lagback_diff2'] = df['u_out'] - df['u_out_lag_back2']
    print("Step-7...Completed")
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")
    
    return df

def apply_features(data_root, out, ver='v3'):
    extractor = {
        'v2': add_features_v2,
        'v3': add_features_v3,
        'v4': add_features_v4,
        'v5': add_features_v5,
        'v6': add_features_v6,
        }

    df_train = pd.read_csv(data_root + 'train.csv')
    df_test = pd.read_csv(data_root + 'test.csv')

    train_pressure = df_train['pressure'].values
    df_train.drop(['pressure'], axis=1, inplace=True)
    df_test = extractor[ver](df_test)

    RS = RobustScaler()
    train_values = RS.fit_transform(df_train)
    test_values = RS.transform(df_test)

    train_cols = df_train.columns.tolist()
    test_cols = df_test.columns.tolist()
    
    ignore_cols = ['id', 'breath_id']

    print(train_values.shape, len(train_cols))
    for i in range(len(train_cols)):
        if train_cols[i] in ignore_cols:
            continue
        df_train[train_cols[i]] = train_values[:, i]

    print(test_values.shape, len(test_cols))
    for i in range(len(test_cols)):
        if test_cols[i] in ignore_cols:
            continue
        df_test[test_cols[i]] = test_values[:, i]

    df_train['pressure'] = train_pressure
    df_train.to_csv(f'{out}train_{ver}-scaled.csv', index=False)
    df_test.to_csv(f'{out}test_{ver}-scaled.csv', index=False)

if __name__ == "__main__":
    data_root = '/content/drive/MyDrive/kaggle/dataset/ventilator-pressure-prediction/'
    apply_features(data_root, data_root, 'v7')
    apply_features(data_root, data_root,  'v7')