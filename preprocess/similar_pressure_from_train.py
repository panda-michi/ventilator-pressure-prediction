import pandas as pd
import numpy as np
import random 
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATADIR = '/content/drive/MyDrive/kaggle/dataset/ventilator-pressure-prediction/'

def split_train_referece(csv_dir, out):
    df = pd.read_csv(csv_dir)
    df['RC'] = df['R'].astype(str) + '_' + df['C'].astype(str)

    bids = df['breath_id'].unique().tolist()
    num_ref = len(bids) // 4
    refered_id = random.sample(bids, num_ref)

    ref_df = df[df['breath_id'].isin(refered_id)].reset_index().copy()
    train_df = df[~df['breath_id'].isin(refered_id)].reset_index().copy()

    ref_df.to_csv(out + 'train_using_ref_4.csv', index=False)
    train_df.to_csv(out + 'train_using_train_4.csv', index=False)

def get_similar_id(target_id, _table, _refer_table):
    table, refer_table = _table.copy(), _refer_table.copy()

    if target_id in table['breath_id'].tolist():
        target_rc = table[table['breath_id']==target_id]['RC'].values[0]
        # u_in
        target_vec = table[table['breath_id']==target_id].to_numpy()[:,2:]
        # same rc breath
        refer_table = refer_table[refer_table['RC']==target_rc].reset_index(drop=True)
        # return 
        refer_vec = refer_table.to_numpy()[:,2:]

        breaths = refer_table['breath_id'].unique().tolist()
        breath_map = {i:b for i,b in enumerate(breaths)}

        cs = cosine_similarity(target_vec, refer_vec)[0]

        similar_id = breath_map[np.argmax(cs)]
    else :
        similar_id = None

    return similar_id

def get_similar_info(target_id, table, refer_table, df):
    rc = table[table['breath_id']==target_id]['RC']
    similar_id = get_similar_id(target_id, table, refer_table)

    u_in = df[df['breath_id']==similar_id]['u_in'].values
    pressure = df[df['breath_id']==similar_id]['pressure'].values

    return u_in.tolist(), pressure.tolist()

def main():
    train_dir = DATADIR + 'train.csv' 

    split_train_referece(train_dir, DATADIR)

    df_ref = pd.read_csv(DATADIR + 'train_using_ref_4.csv')
    df_train = pd.read_csv(DATADIR + 'train_using_train_4.csv')

    df_test = pd.read_csv(DATADIR + 'test.csv')
    df_test['RC'] = df_test['R'].astype(str) + '_' + df_test['C'].astype(str)

    df_ref['num'] = df_ref.groupby(['breath_id', 'RC']).cumcount()
    df_train['num'] = df_train.groupby(['breath_id', 'RC']).cumcount()
    df_test['num'] = df_test.groupby(['breath_id', 'RC']).cumcount()

    u_in_ref = pd.pivot_table(df_ref, index=['breath_id', 'RC'], columns='num', values='u_in')
    u_in_train = pd.pivot_table(df_train, index=['breath_id', 'RC'], columns='num', values='u_in')
    u_in_test = pd.pivot_table(df_test, index=['breath_id', 'RC'], columns='num', values='u_in')

    u_in_ref = u_in_ref.reset_index()
    u_in_train = u_in_train.reset_index()
    u_in_test = u_in_test.reset_index()

    print('### apply train data ###')
    df_train['similar_pressure'] = 0.0
    df_train['similar_u_in'] = 0.0  

    for bid in tqdm(u_in_train['breath_id'].tolist()):
        sim_u_in, sim_pressure = get_similar_info(bid, u_in_train, u_in_ref, df_ref)

        df_train.loc[df_train['breath_id'] == bid, 'similar_u_in'] = sim_u_in
        df_train.loc[df_train['breath_id'] == bid, 'similar_pressure'] = sim_pressure

    df_train.to_csv(DATADIR + 'train_with_siminfo4.csv')

    print('### apply test data ###')
    df_test['similar_pressure'] = 0.0
    df_test['similar_u_in'] = 0.0  

    for bid in tqdm(u_in_test['breath_id'].tolist()):
        sim_u_in, sim_pressure = get_similar_info(bid, u_in_test, u_in_ref, df_ref)

        df_test.loc[df_test['breath_id'] == bid, 'similar_u_in'] = sim_u_in
        df_test.loc[df_test['breath_id'] == bid, 'similar_pressure'] = sim_pressure

    df_test.to_csv(DATADIR + 'test_with_siminfo4.csv')

if __name__ == "__main__":
    main()