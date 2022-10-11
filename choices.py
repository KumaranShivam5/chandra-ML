# from unicodedata import name
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
model_dict = {
    'RF' : RandomForestClassifier() , 
    'RF_mod' : RandomForestClassifier(
        n_estimators = 500 , 
    ) , 
    'GB' : GradientBoostingClassifier() , 
}

import numpy as np
flag = {
    'conf_flag' : 0 , 
    'streak_src_flag' : 0 , 
    'extent_flag' : 0 , 
    'pileup_flag' : 0 , 
    }
classes =['AGN' ,'YSO' ,'STAR' ,'HMXB' ,'LMXB' ,'ULX' ,'CV' ,'PULSAR']
def get_train_data(flags=flag , classes=classes , offset = -1 , sig = 0, deets=0 , file = None ,ret_id_cols = ['class']):
    #print(flags)
    # data_id = pd.read_csv('../compiled_data_v3/id_frame.csv' , index_col='name')
    data_id = pd.read_csv('data/training_data/id_frame.csv' , index_col='name')
    id_col = data_id.columns.to_list()
    # default_data = '../compiled_data_v3/imputed_data_v2/x_phot_minmax_modeimp.csv'
    default_data = 'data/training_data/imputed/x_phot_minmax_modeimp.csv'
    if(file):
        x_data = pd.read_csv(file , index_col = 'name').iloc[:,  1:]
    else:
        x_data = pd.read_csv( default_data, index_col = 'name').iloc[:,  1:]
    x_col = x_data.columns.to_list()
    #print(x_col)
    data = pd.concat([data_id, x_data] , axis=1).sort_values(by='offset')
    data = pd.concat([data_id, x_data] , axis=1).sort_values(by='offset')
    data = pd.merge(data_id , x_data , left_index=True , right_index= True ,  how ='left')
    #data = data.drop_duplicates('name')
    data= data.drop_duplicates(['ra','dec'])
    #data = data.set_index('name')
    if(deets):
        deets(data , 1)
    for flag , val in zip(flags.keys() , flags.values()):
        data = data[data[flag]==val]
    eps = 0.01 # offset epsilon
    if(offset>0):
        data = data[data['offset']<=offset+eps]
    max = data['offset'].max()
    #print(f"offset:  \t{data['offset'].min() :.3f}|{data['offset'].max():.3f}")
    data = data[data['significance']> sig]
    #print(f"singinficance:  {data['significance'].min():.3f}|{data['significance'].max():.3f}")
    data = data[data['class'].isin(classes)]
    src_class = data['class'].to_list()
    x = data[x_col]
    #print(id_col)
    for l in ret_id_cols:
        #print((l in(id_col)))
        if(not l in(id_col)):
            raise KeyError(f'Entered variable "{l}" is not in the database. ')
        else:
            x.insert(0 , l , data[l].to_list())
    return x

#if __name__ == 'main':
data_dict = {
    'no_filter' : 
        get_train_data(
            flags = {
            'conf_flag' : 0 , 
            'streak_src_flag' : 0 , 
            'extent_flag' : 0 , 
            'pileup_flag' : 0 , 
            } , 
            classes = ['AGN' ,'STAR' , 'YSO' , 'PULSAR'  , 'CV' , 'LMXB' , 'HMXB' ,'ULX'] , 
            offset = -1 , 
            sig = 0
        ) ,

    'off_2_sig_3' : 
        get_train_data(
            flags = {
            'conf_flag' : 0 , 
            'streak_src_flag' : 0 , 
            'extent_flag' : 0 , 
            'pileup_flag' : 0 , 
            } , 
            classes = ['AGN' ,'STAR' , 'YSO' , 'PULSAR'  , 'CV' , 'LMXB' , 'HMXB' ,'ULX'] , 
            offset = 2 , 
            sig = 3
        ) 

}

param_dict ={
    'model_fit' : 
        ['powlaw_gamma',
        'powlaw_nh',
        'powlaw_ampl',
        'powlaw_stat',
        'bb_kt',
        'bb_nh',
        'bb_ampl',
        'bb_stat',
        'brems_kt',
        'brems_nh',
        'brems_norm',
        'brems_stat',
        ] , 
    'hard_var_col' : ['var_inter_hard_prob_hs', 'ks_intra_prob_b', 'var_inter_hard_sigma_hm', 'var_inter_hard_prob_ms', 'var_inter_hard_prob_hm',]
    ,
    'hardness' : 
        [
        'hard_hs',
        'hard_hm',
        'hard_ms',
            ] , 
    'variability' : [
         'var_inter_prob_b',
        'var_inter_sigma_b',
        'var_intra_prob_b',
        'var_inter_index_b',
        'kp_intra_prob_b',
        'var_intra_index_b',
    ], 
    'WISE' : ['W1','W2','W3','W4',] , 
    'IRAC' : [
        '4.5 microns (IRAC)',
        '8.0 microns (IRAC)',
        '3.6 microns (IRAC)',
        '5.8 microns (IRAC)',] , 
    '2MASS' : ['J','H','K',], 
    'GAIA' : ['G','Bp','Rp',] ,
    'color' : ['B-R','G-J','G-W2','Bp-H','Bp-W3','Rp-K','J-H','J-W1','W1-W2'] , 
    'SDSS' : [
        'u-sdss',
        'g-sdss',
        'r-sdss',
        'i-sdss',
        'z-sdss',
        ] , 
    'GALEX' :['FUV','NUV',] , 

    'MW' : [
        'J','H','K','u-sdss',
        'g-sdss',
        'r-sdss',
        'i-sdss',
        'z-sdss','FUV','NUV','G','Bp','Rp',
        'W1','W2','W3','W4', '4.5 microns (IRAC)',
        '8.0 microns (IRAC)',
        '3.6 microns (IRAC)',
        '5.8 microns (IRAC)',
        'B-R','G-J','G-W2','Bp-H','Bp-W3','Rp-K','J-H','J-W1','W1-W2'
    ] , 
    'IR' : [
        '4.5 microns (IRAC)',
        '8.0 microns (IRAC)',
        '3.6 microns (IRAC)',
        '5.8 microns (IRAC)',
        'W1','W2','W3','W4',
    ],
    'optical_uv' : [
        'u-sdss',
        'g-sdss',
        'r-sdss',
        'i-sdss',
        'z-sdss',
        'FUV','NUV',
        'G','Bp','Rp',
         'J','H','K',
    ],
    'hard_var_col' : ['var_inter_hard_prob_hs', 'ks_intra_prob_b', 'var_inter_hard_sigma_hm', 'var_inter_hard_prob_ms', 'var_inter_hard_prob_hm',] , 
'model_fit_col' :        ['powlaw_gamma',
        'powlaw_nh',
        'powlaw_ampl',
        'powlaw_stat',
        'bb_kt',
        'bb_nh',
        'bb_ampl',
        'bb_stat',
        'brems_kt',
        'brems_nh',
        'brems_norm',
        'brems_stat',
        ], 
'sparse_col' : [
    '0p5_2csc' , '2-10 keV (XMM)' , '1_2_csc' , '0p5_8_csc'
] , 

}


palette = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

]



def get_raw_data(df_index = [] , offset = 1 , significance = 0 ):
    df = pd.read_csv('data/mw_cat/chandra_filtered_sources.csv' , index_col = 'name')
    # df_id = pd.read_csv('compiled_data_v3/id_frame.csv' , index_col='name')[['offset' , 'class']]
    # off = offset + 0.01
    # df = pd.merge(df_id[df_id['offset']<off] , df , left_index=True , right_index =True , how='right')
    # sig = significance
    # #df = df[df['significance']>sig]
    # df = df.drop(columns = ['significance'  , 'offset' , 'ra' , 'dec', 'var_inter_hard_flag' , 'likelihood'])
    df = df.rename(columns = {
        'flux_aper_b' : 'b-csc' , 
        'flux_aper_h' : 'h-csc' ,
        'flux_aper_m': 'm-csc' ,
        'flux_aper_s': 's-csc' ,
        'flux_aper_u': 'u-csc' ,
    })
    df = pd.merge(
        df , pd.read_csv('data/mw_cat/sdss.csv' , index_col='name') ,
        left_index=True , 
        right_index = True , 
        how = 'left'
    )
    df = pd.merge(
        df , pd.read_csv('data/mw_cat/MIPS.csv' , index_col='name') ,
        left_index=True , 
        right_index = True , 
        how = 'left'
    )
    df = pd.merge(
        df , pd.read_csv('data/mw_cat/2mass_v2.csv' , index_col='name') ,
        left_index=True , 
        right_index = True , 
        how = 'left'
    )
    df = pd.merge(
        df , pd.read_csv('data/mw_cat/wise_combined.csv' , index_col='name') ,
        left_index=True , 
        right_index = True , 
        how = 'left'
    )
    df = pd.merge(
        df , pd.read_csv('data/mw_cat/galex_combined.csv' , index_col='name') ,
        left_index=True , 
        right_index = True , 
        how = 'left'
    )
    df = pd.merge(
        df , pd.read_csv('data/mw_cat/gaia.csv' , index_col='name') ,
        left_index=True , 
        right_index = True , 
        how = 'left'
    )
    df['Bp-R'] = df['bp_mag']-df['rp_mag']
    df['G-J'] = df[ 'g_mag'] - df['Jmag']
    df['G-W2'] = df['g_mag'] - df['W2mag']
    df['Bp-H'] = df['bp_mag'] - df[ 'Hmag']
    df['Bp-W3'] = df['bp_mag'] - df['W3mag']
    df['Rp-K'] = df['rp_mag'] - df['Kmag']
    df['J-H'] = df['Jmag'] - df['Hmag']
    df['J-W1'] = df['Jmag'] - df['W1mag']
    df['W1-W2'] = df['W1mag'] - df['W2mag']
    df['u-g'] = df['umag'] - df['gmag']
    df['g-r'] = df['gmag'] - df['rmag']
    df['r-z'] = df['rmag'] - df['zmag']
    df['i-z'] = df['imag'] - df['zmag']
    df['u-z'] = df['umag'] - df['zmag']
    df = df.rename(columns = {
        'gal_l' : 'gal_l2',
        'gal_b' : 'gal_b2' , 
        'umag' : 'u-sdss',
        'gmag' : 'g-sdss' ,
        'rmag' : 'r-sdss',
        'imag': 'i-sdss',
        'zmag' : 'z-sdss',
        'W1mag': 'W1',
        'W2mag': 'W2',
        'W3mag': 'W3',
        'W4mag': 'W4',
        'Jmag':  'J',
        'Hmag':  'H',
        'Kmag':  'K',
        'fuv_mag' : 'FUV',
        'nuv_mag': 'NUV',
        'g_mag' : 'G',
        'bp_mag' : 'Bp',
        'rp_mag' : 'Rp',

    })
    df = df.reset_index()
    if(len(df_index)>0):
        df_ret = df[df['name'].isin(df_index.index.to_list())].set_index('name')
        df_ret = pd.merge(df_index , df_ret , left_index=True , right_index=True)
        return df_ret
    else:
        return df




def get_source_info(df_index):
    df = pd.read_csv('data/source_info/all_csc_source_info.csv' , index_col = 'name')
    df = df.reset_index()
    if(len(df_index)>0):
        df_ret = df[df['name'].isin(df_index.index.to_list())].set_index('name')
        df_ret = pd.merge(df_index , df_ret , left_index=True , right_index=True)
        return df_ret
    else:
        return df


def get_source_list(flags):
    data = pd.read_csv('data/source_info/all_csc_source_info.csv', index_col='name')
    for flag , val in zip(flags.keys() , flags.values()):
        data = data[data[flag]==val]
    return data[[]]

from astroquery.ipac.ned import Ned
from tqdm import tqdm 
def get_NED_data(df):
    for n in tqdm(df.index.to_list()[:]):
    #print(n)
        try:
            result_table = Ned.get_table(n)
            res_df = result_table.to_pandas()
            #display(res_df)
            #res_df.to_csv('multi_wave_data/'+n+'.csv')
            res_df.to_csv('data/NED_data/'+n+'.csv')
        except Exception as e:
            print(e)