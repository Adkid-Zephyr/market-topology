"""Endpoint-only retrospective diagnostics for underspecified spectral preprocessing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import periodogram, windows
from scipy.stats import kendalltau

ROOT=Path(__file__).resolve().parent


def smooth(x,kernel):
    if kernel=="gaussian25":
        sigma=25*0.25/0.6744897501960817
        offsets=np.arange(-int(np.ceil(4*sigma)),int(np.ceil(4*sigma))+1)
        weights=np.exp(-.5*(offsets/sigma)**2)
    elif kernel=="box25":
        weights=np.ones(25)
    else:
        return np.zeros_like(x)
    return convolve1d(x,weights,mode="constant",cval=0)/convolve1d(np.ones_like(x),weights,mode="constant",cval=0)


def main():
    rows=[]
    for w in (50,100):
        df=pd.read_csv(ROOT/"results"/f"daily_w{w}.csv",index_col=0,parse_dates=True)
        for endpoint in ["2000-03-09","2008-09-12",str(df.index[-1].date())]:
            part=df.loc[:endpoint].tail(1000)
            for kernel in ["none","gaussian25","box25"]:
                x=part.l1.to_numpy()
                residual=x-smooth(x,kernel)
                variance=[];psd=[];r_style=[]
                for i in range(499,1000):
                    z=residual[i-499:i+1]
                    variance.append(np.var(z,ddof=1))
                    _,p=periodogram(z,detrend="constant")
                    psd.append(p[1:32].mean())
                    # R spectrum defaults detrend=True, taper=0.1; closest Python analogue.
                    _,p=periodogram(z,detrend="linear",window=windows.tukey(500,alpha=.2,sym=True))
                    r_style.append(p[2:63].mean())
                rows.append({"window":w,"endpoint":endpoint,"kernel":kernel,
                             "variance_tau250":kendalltau(np.arange(250),variance[-250:]).statistic,
                             "psd_tau250":kendalltau(np.arange(250),psd[-250:]).statistic,
                             "r_analogue_psd_tau250":kendalltau(np.arange(250),r_style[-250:]).statistic})
    pd.DataFrame(rows).to_csv(ROOT/"results"/"spectral_sensitivity.csv",index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__=="__main__":main()
