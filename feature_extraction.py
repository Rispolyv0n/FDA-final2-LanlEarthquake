import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import logging
from scipy import stats
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):   
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta

def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)

def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)   
    zc = np.fft.fft(xc) #快速傅立葉變换
    realFFT = np.real(zc)# 獲取實數部分
    imagFFT = np.imag(zc)#獲取虛數部分
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()#標準差
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'sum'] = xc.sum()
    X.loc[seg_id, 'mad'] = xc.mad()#根據平均值計算平均絕對離差
    X.loc[seg_id, 'kurt'] = xc.kurtosis()#峰度
    X.loc[seg_id, 'skew'] = xc.skew()#偏度，是統計數據分布偏斜方向和程度的度量，是統計數據分布非對稱程度的數字特徵。

    X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())
    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])

    X.loc[seg_id, 'med'] = xc.median()#算術中位數(50%分位數) 
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()#xc的絕對值，之後再取平均值
    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()
    X.loc[seg_id, 'abs_min'] = np.abs(xc).min()    
    X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))
    X.loc[seg_id, 'mean_change_rate'] = calc_change_rate(xc)
    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)

    #分位數之統計
    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)#95%分位數
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)#99%分位數
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)#5%分位數
    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)#1%分位數
    X.loc[seg_id, 'q999'] = np.quantile(xc,0.999)#99.9%分位數
    X.loc[seg_id, 'q001'] = np.quantile(xc,0.001)#0.1%分位數
    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
    
    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)
    
    #傅立葉變換之統計
    X.loc[seg_id, 'Rmean'] = realFFT.mean()
    X.loc[seg_id, 'Rstd'] = realFFT.std()
    X.loc[seg_id, 'Rmax'] = realFFT.max()
    X.loc[seg_id, 'Rmin'] = realFFT.min()
    X.loc[seg_id, 'Imean'] = imagFFT.mean()
    X.loc[seg_id, 'Istd'] = imagFFT.std()
    X.loc[seg_id, 'Imax'] = imagFFT.max()
    X.loc[seg_id, 'Imin'] = imagFFT.min()

    # trend
    X.loc[seg_id, 'trend'] = add_trend_feature(xc)
    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)

    #分段
    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
    X.loc[seg_id, 'std_first_25000'] = xc[:25000].std()
    X.loc[seg_id, 'std_last_25000'] = xc[-25000:].std()
    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()
    
    X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()
    X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()
    X.loc[seg_id, 'avg_first_25000'] = xc[:25000].mean()
    X.loc[seg_id, 'avg_last_25000'] = xc[-25000:].mean()
    X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()
    X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()
    
    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()
    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()
    X.loc[seg_id, 'min_first_25000'] = xc[:25000].min()
    X.loc[seg_id, 'min_last_25000'] = xc[-25000:].min()
    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()
    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()
    
    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()
    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()
    X.loc[seg_id, 'max_first_25000'] = xc[:25000].max()
    X.loc[seg_id, 'max_last_25000'] = xc[-25000:].max()
    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()
    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()
    
    X.loc[seg_id, 'mean_change_rate_first_50000'] = calc_change_rate(xc[:50000])
    X.loc[seg_id, 'mean_change_rate_last_50000'] = calc_change_rate(xc[-50000:])
    X.loc[seg_id, 'mean_change_rate_first_25000'] = calc_change_rate(xc[:25000])
    X.loc[seg_id, 'mean_change_rate_last_25000'] = calc_change_rate(xc[-25000:])
    X.loc[seg_id, 'mean_change_rate_first_10000'] = calc_change_rate(xc[:10000])
    X.loc[seg_id, 'mean_change_rate_last_10000'] = calc_change_rate(xc[-10000:])
    
    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()#希爾伯特轉換，是一個對函數 u(t) 產生定義域相同的函數 H(u)(t) 的線性算子。 
    
    X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
# 71
    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()
    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()
    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()
    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()
    X.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(xc, 50, 1000).mean()
    X.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta(xc, 100, 5000).mean()
    X.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta(xc, 333, 666).mean()
    X.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta(xc, 4000, 10000).mean()
# 79
    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
    
    ewma = pd.Series.ewm
    X.loc[seg_id, 'exp_Moving_average_300_mean'] = ewma(xc, span=300).mean().mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
    X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=30000).mean().mean(skipna=True)
# 83
    no_of_std = 3
    X.loc[seg_id,'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
    X.loc[seg_id,'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X.loc[seg_id,'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
    X.loc[seg_id,'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X.loc[seg_id,'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()
    X.drop('Moving_average_700_mean', axis=1, inplace=True)
# 89
    #rolling features
    for w in [10, 50, 100, 1000]:
        x_roll_abs_mean = xc.abs().rolling(w).mean().dropna().values
        x_roll_mean = xc.rolling(w).mean().dropna().values
        x_roll_std = xc.rolling(w).std().dropna().values
        x_roll_min = xc.rolling(w).min().dropna().values
        x_roll_max = xc.rolling(w).max().dropna().values
        
        X.loc[seg_id, 'ave_roll_std_' + str(w)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(w)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(w)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(w)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q10_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.10)
        X.loc[seg_id, 'q95_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.99)
        
        X.loc[seg_id, 'ave_roll_mean_' + str(w)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(w)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(w)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(w)] = x_roll_mean.min()
        X.loc[seg_id, 'q05_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.95)
        
        X.loc[seg_id, 'ave_roll_abs_mean_' + str(w)] = x_roll_abs_mean.mean()
        X.loc[seg_id, 'std_roll_abs_mean_' + str(w)] = x_roll_abs_mean.std()
        X.loc[seg_id, 'q05_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.05)
        X.loc[seg_id, 'q95_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.95)
        
        X.loc[seg_id, 'std_roll_min_' + str(w)] = x_roll_min.std()
        X.loc[seg_id, 'max_roll_min_' + str(w)] = x_roll_min.max()
        X.loc[seg_id, 'q05_roll_min_' + str(w)] = np.quantile(x_roll_min, 0.05)
        X.loc[seg_id, 'q95_roll_min_' + str(w)] = np.quantile(x_roll_min, 0.95)

        X.loc[seg_id, 'std_roll_max_' + str(w)] = x_roll_max.std()
        X.loc[seg_id, 'min_roll_max_' + str(w)] = x_roll_max.min()
        X.loc[seg_id, 'q05_roll_max_' + str(w)] = np.quantile(x_roll_max, 0.05)
        X.loc[seg_id, 'q95_roll_max_' + str(w)] = np.quantile(x_roll_max, 0.95)



