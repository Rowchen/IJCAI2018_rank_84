import time
import pandas as pd
import numpy as np
import gc

def get_count(df, cols, cname, value):
    df_count = pd.DataFrame(df.groupby(cols)[value].count()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    del df_count
    gc.collect()
    return df

def get_sum(df, cols, cname, value):
    df_count = pd.DataFrame(df.groupby(cols)[value].sum()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    del df_count
    gc.collect()
    return df

def get_mean(df, cols, cname, value):
    df_mean = pd.DataFrame(df.groupby(cols)[value].mean()).reset_index()
    df_mean.columns = cols + [cname]
    df = df.merge(df_mean, on=cols, how='left')
    del df_mean
    gc.collect()
    return df

def get_std(df, cols, cname, value):
    df_std = pd.DataFrame(df.groupby(cols)[value].std()).reset_index()
    df_std.columns = cols + [cname]
    df = df.merge(df_std, on=cols, how='left')
    del df_std
    gc.collect()
    return df

def get_nunique(df, cols, cname, value):
    df_nunique = pd.DataFrame(df.groupby(cols)[value].nunique()).reset_index()
    df_nunique.columns = cols + [cname]
    df = df.merge(df_nunique, on=cols, how='left')
    del df_nunique
    gc.collect()
    return df
    
def get_cumcount(df, cols, cname):
    df[cname] = df.groupby(cols).cumcount() + 1
    return df

def get_hour(datetime):
    return datetime.hour
def get_day(datetime):
    return datetime.day

def get_rank(order_by,group_by):
    _ord = np.lexsort((order_by, group_by))
    _cs1 = np.zeros(group_by.size,dtype=np.int)
    _prev_grp = group_by[_ord[0]]
    for i in xrange(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            if order_by[i0]==order_by[_ord[i-1]]:
                _cs1[i] = _cs1[i - 1]
            else:
                _cs1[i] = _cs1[i - 1] + 1
        else:
            _cs1[i] = 0
            _prev_grp = group_by[i0]
    org_idx = np.zeros(group_by.size, dtype=np.int)
    org_idx[_ord] = np.asarray(xrange(group_by.size))
    return _cs1[org_idx]

def get_silde_sum(df,end_day,cols,cname,day_long=4):
    df=df.loc[(df.day>=end_day-day_long)&(df.day<end_day),cols+['is_trade']]
    tb=df.groupby(cols,as_index=False)['is_trade'].agg({cname:np.sum})
    return tb

def get_silde_cnt(df,end_day,cols,cname,day_long=4):
    df=df.loc[(df.day>=end_day-day_long)&(df.day<end_day),cols+['is_trade']]
    tb=df.groupby(cols,as_index=False)['is_trade'].agg({cname:'count'})
    return tb

def get_silde_sum2(df,end_day,cols,cname,day_long=4):
    df=df.loc[(df.day>=end_day-day_long)&(df.day<end_day),cols+['is_trade']]
    tb=df.groupby(cols,as_index=False)['is_trade'].agg({cname:np.sum})
    return tb

def get_silde_cnt2(df,end_day,cols,cname,day_long=4):
    df=df.loc[(df.day>=end_day-day_long)&(df.day<end_day),cols+['is_trade']]
    tb=df.groupby(cols,as_index=False)['is_trade'].agg({cname:'count'})
    return tb
def get_comb_feat2(data,data_fix,input_col1,input_col2,sumorcnt):
    feat_list=[]
    feat_list2=[]
    for col1 in input_col1:
        cnt=data.groupby([col1], as_index=False)['is_trade'].agg({str(col1)+'_cnt':sumorcnt})
        data_fix = pd.merge(data_fix, cnt, on=[col1], how='left')
    for col1 in input_col1:
        for col2 in input_col2:
            name1=col1 +'_'+ col2+'_prob'
            name2=col1 +'_'+ col2+'_cnt'
            if name2 in data_fix.columns:
                del data_fix[name2]
            feat_list+=[name1]
            feat_list2+=[name2]
            combine_cnt = data.groupby([col1,col2], as_index=False)['is_trade'].agg({str(col1) +'_'+ str(col2)+'_cnt':sumorcnt})
            data_fix = pd.merge(data_fix, combine_cnt, on=[col1, col2], how='left')
            data_fix[name2]=data_fix[name2].fillna(0)
            data_fix[name1] = data_fix[name2] / (data_fix[str(col1)+'_cnt'].astype(float)+1)
            #del data_fix[name2]
        del data_fix[str(col1)+'_cnt']
    return data_fix,feat_list,feat_list2