# Chandra-ML

This Repository contains all the necessary data tables and routines for the classification of sources in the **Chandra Source Catalog-2.0**

The section **Data** describes all the data tables in and various routines to obtain the required data in proper format from this table in general.

The section **Model Training and validation** Describes our application of LightGBM classification model on this data and the routines developed for it.

The last section **Application** Shows the applicatin of the model on the unclassified sources.

#### Requirements : 
```
astropy
astroquery
pandas
numpy
scickit-learn
lightgbm
```

#### Important Imports


```python
import numpy as np 
import pandas as pd 
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


# Data

### Directory Details

All the data for this work is in the folder *_data_* folder. 

<small>
Due to size constrains of this github, the data is not included in this repository. It is uploaded to the google drive *drive link* in a zipped file. Download and extract the folder in this directory and do not change the file names. the **data** folder should be in the same directory as *choices.py* and *utilities_v2.py* files
</small>



#### _data_ folder structure


```
├── data
│   ├── classified
│   │   ├── AGN.csv
│   │   ├── CV.csv
│   │   ├── HMXB.csv
│   │   ├── LMXB.csv
│   │   ├── PULSAR.csv
│   │   ├── STAR.csv
│   │   ├── TRAIN_SRC.csv
│   │   ├── ULX.csv
│   │   └── YSO.csv
│   ├── mw_cat
│   │   ├── 2mass_v2.csv
│   │   ├── chandra_filtered_sources.csv
│   │   ├── gaia.csv
│   │   ├── galex_combined.csv
│   │   ├── MIPS.csv
│   │   ├── sdss.csv
│   │   └── wise_combined.csv
│   ├── new_src_data
│   │   └── new_sources.csv
│   ├── source_info
│   │   └── all_csc_source_info.csv
│   └── training_data
│       ├── id_frame.csv
│       ├── imputed
│       │   ├── x_phot_minmax_10iter_rfimpimp.csv
│       │   ├── x_phot_minmax_constimp.csv
│       │   ├── x_phot_minmax_forestimp.csv
│       │   ├── x_phot_minmax_knnimp.csv
│       │   ├── x_phot_minmax_meanimp.csv
│       │   └── x_phot_minmax_modeimp.csv
│       ├── train_data_minmax.csv
│       └── x_phot_minmax.csv

```



* *classified* : Contains the data table for all the sources idnetified using the LightGBM in this work. The table consists of the class memberhip probabilities alongwith the MW data for all the sources.

* *mw_cat* : Multi-wavelength catalogs for all the sources. use csc names as the identifiers.

* *new_src_data* : normalized data corresponding to all the unclassified sources. This data tale is used in this work to preoduse the classification table and the CMPs available in the _classified_ folder

* *source_info* : The data-table in this folder contains all the necessary information (quality flags, position, observation info) for all the sources in the CSC-2.0

* *training_data* : contains the data-tabl of the sources cross-match and identified in various classes. The data in this folder are normalised and was used for the training of model in this work. All the imputed data are inside the _imputed_ folder.

#### General Data Retrival

We will start with the source list of all the sources in the CSC-2.0.
The data table **all_source_info.csv** in the folder _data/source_info_ contains the information of all the sources. With a minimum Pandas skill, one can select the object of choice from this csv file. But with the routine **get_source_info** in the _choices_ module, the list can be derived eaisly using flags parameter.


```python
flags = {
    'conf_flag' : 0 , 
    'streak_src_flag' : 0 , 
    'extent_flag' : 0 , 
    'pileup_flag' : 0 , 
    }
from choices import get_source_list
sources = get_source_list(flags)
sources
```



<!-- 
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
    <tr>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2CXO J003935.9-732725</th>
    </tr>
    <tr>
      <th>2CXO J003936.7-731249</th>
    </tr>
    <tr>
      <th>2CXO J004028.7-731106</th>
    </tr>
    <tr>
      <th>2CXO J004506.3-730056</th>
    </tr>
    <tr>
      <th>2CXO J004659.0-731918</th>
    </tr>
    <tr>
      <th>...</th>
    </tr>
    <tr>
      <th>2CXO J220613.7-495727</th>
    </tr>
    <tr>
      <th>2CXO J220614.6-500951</th>
    </tr>
    <tr>
      <th>2CXO J220618.4-500554</th>
    </tr>
    <tr>
      <th>2CXO J220626.0-500126</th>
    </tr>
    <tr>
      <th>2CXO J220642.7-495916</th>
    </tr>
  </tbody>
</table>
<p>277717 rows × 0 columns</p>
</div>



Now let's extract the information for 100 of these sources using the function **get_source_info**.
<small> Note: this function is also based on the file _all_source_info.csv_. The point of having an additional function for this is that at any stage of working with any data-table of N number of sources, we can alway pull out the information about the source.


```python
from choices import get_source_info
source_info = get_source_info(sources.sample(100))
source_info
```




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
 </style> 
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ra</th>
      <th>dec</th>
      <th>gal_l</th>
      <th>gal_b</th>
      <th>err_ellipse_r0</th>
      <th>err_ellipse_r1</th>
      <th>err_ellipse_ang</th>
      <th>conf_flag</th>
      <th>extent_flag</th>
      <th>sat_src_flag</th>
      <th>var_flag</th>
      <th>pileup_flag</th>
      <th>streak_src_flag</th>
      <th>significance</th>
      <th>acis_time</th>
      <th>hrc_time</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2CXO J002046.2-705659</th>
      <td>5.192620</td>
      <td>-70.949804</td>
      <td>306.523591</td>
      <td>-45.963984</td>
      <td>0.956380</td>
      <td>0.804594</td>
      <td>44.584318</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.588235</td>
      <td>22768.771028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J101020.3-124106</th>
      <td>152.584628</td>
      <td>-12.685272</td>
      <td>253.231259</td>
      <td>34.215853</td>
      <td>0.718688</td>
      <td>0.717721</td>
      <td>79.403259</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.390751</td>
      <td>51195.761895</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J165409.8-020219</th>
      <td>253.540954</td>
      <td>-2.038749</td>
      <td>16.632152</td>
      <td>24.790265</td>
      <td>2.002898</td>
      <td>1.185102</td>
      <td>104.896836</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>4.722222</td>
      <td>20772.938789</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J100301.2+021934</th>
      <td>150.755395</td>
      <td>2.326167</td>
      <td>237.190828</td>
      <td>42.708642</td>
      <td>1.161515</td>
      <td>1.079687</td>
      <td>75.542480</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.666667</td>
      <td>158905.462446</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J054244.6-404855</th>
      <td>85.686088</td>
      <td>-40.815288</td>
      <td>246.499422</td>
      <td>-29.796437</td>
      <td>3.891118</td>
      <td>2.875106</td>
      <td>178.855720</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.875000</td>
      <td>50405.925987</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2CXO J180325.0-295432</th>
      <td>270.854168</td>
      <td>-29.909153</td>
      <td>1.117724</td>
      <td>-3.835987</td>
      <td>0.805650</td>
      <td>0.779075</td>
      <td>6.289155</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.875000</td>
      <td>104822.403707</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J181647.4-162129</th>
      <td>274.197777</td>
      <td>-16.358283</td>
      <td>14.477496</td>
      <td>0.010426</td>
      <td>0.853008</td>
      <td>0.796700</td>
      <td>174.514643</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.052632</td>
      <td>18463.958710</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J173555.2+570356</th>
      <td>263.980184</td>
      <td>57.065711</td>
      <td>85.279886</td>
      <td>32.634090</td>
      <td>4.897011</td>
      <td>3.619107</td>
      <td>102.905240</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.111111</td>
      <td>10337.817452</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J134902.7+035728</th>
      <td>207.261495</td>
      <td>3.957986</td>
      <td>336.132887</td>
      <td>63.054502</td>
      <td>1.876542</td>
      <td>1.016800</td>
      <td>133.998688</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.082235</td>
      <td>11005.007980</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J100001.3-301359</th>
      <td>150.005799</td>
      <td>-30.233315</td>
      <td>264.363857</td>
      <td>19.517936</td>
      <td>0.775855</td>
      <td>0.752233</td>
      <td>103.729420</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>5.793844</td>
      <td>30677.516682</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 16 columns</p>
</div>



Now we will get the raw data for these sources using the function **get_raw_data**. 


```python
from choices import get_raw_data
src_data = get_raw_data(source_info)
src_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ra_x</th>
      <th>dec_x</th>
      <th>gal_l</th>
      <th>gal_b</th>
      <th>err_ellipse_r0</th>
      <th>err_ellipse_r1</th>
      <th>err_ellipse_ang</th>
      <th>conf_flag</th>
      <th>extent_flag</th>
      <th>sat_src_flag</th>
      <th>var_flag_x</th>
      <th>pileup_flag</th>
      <th>streak_src_flag</th>
      <th>significance_x</th>
      <th>acis_time</th>
      <th>hrc_time</th>
      <th>ra_y</th>
      <th>dec_y</th>
      <th>significance_y</th>
      <th>gal_l2</th>
      <th>gal_b2</th>
      <th>likelihood</th>
      <th>var_flag_y</th>
      <th>var_inter_hard_flag</th>
      <th>b-csc</th>
      <th>h-csc</th>
      <th>m-csc</th>
      <th>s-csc</th>
      <th>u-csc</th>
      <th>hard_hm</th>
      <th>hard_hs</th>
      <th>hard_ms</th>
      <th>var_intra_index_b</th>
      <th>var_intra_prob_b</th>
      <th>ks_intra_prob_b</th>
      <th>kp_intra_prob_b</th>
      <th>var_inter_index_b</th>
      <th>var_inter_prob_b</th>
      <th>var_inter_sigma_b</th>
      <th>u-sdss</th>
      <th>g-sdss</th>
      <th>r-sdss</th>
      <th>i-sdss</th>
      <th>z-sdss</th>
      <th>24_microns_(MIPS)</th>
      <th>J</th>
      <th>H</th>
      <th>K</th>
      <th>W1</th>
      <th>W2</th>
      <th>W3</th>
      <th>W4</th>
      <th>FUV</th>
      <th>NUV</th>
      <th>G</th>
      <th>Bp</th>
      <th>Rp</th>
      <th>Bp-R</th>
      <th>G-J</th>
      <th>G-W2</th>
      <th>Bp-H</th>
      <th>Bp-W3</th>
      <th>Rp-K</th>
      <th>J-H</th>
      <th>J-W1</th>
      <th>W1-W2</th>
      <th>u-g</th>
      <th>g-r</th>
      <th>r-z</th>
      <th>i-z</th>
      <th>u-z</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2CXO J002046.2-705659</th>
      <td>5.192620</td>
      <td>-70.949804</td>
      <td>306.523591</td>
      <td>-45.963984</td>
      <td>0.956380</td>
      <td>0.804594</td>
      <td>44.584318</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.588235</td>
      <td>22768.771028</td>
      <td>NaN</td>
      <td>5.192620</td>
      <td>-70.949804</td>
      <td>3.588235</td>
      <td>306.523591</td>
      <td>-45.963984</td>
      <td>127.049495</td>
      <td>0</td>
      <td>0</td>
      <td>5.067160e-15</td>
      <td>3.284658e-15</td>
      <td>1.163564e-15</td>
      <td>9.246993e-16</td>
      <td>NaN</td>
      <td>0.206121</td>
      <td>0.154903</td>
      <td>-0.059963</td>
      <td>2.0</td>
      <td>0.898197</td>
      <td>0.731318</td>
      <td>0.860139</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J101020.3-124106</th>
      <td>152.584628</td>
      <td>-12.685272</td>
      <td>253.231259</td>
      <td>34.215853</td>
      <td>0.718688</td>
      <td>0.717721</td>
      <td>79.403259</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.390751</td>
      <td>51195.761895</td>
      <td>NaN</td>
      <td>152.584628</td>
      <td>-12.685272</td>
      <td>9.390751</td>
      <td>253.231259</td>
      <td>34.215853</td>
      <td>625.852569</td>
      <td>0</td>
      <td>1</td>
      <td>2.397524e-14</td>
      <td>1.604152e-14</td>
      <td>4.974342e-15</td>
      <td>2.974315e-15</td>
      <td>NaN</td>
      <td>0.113679</td>
      <td>0.123673</td>
      <td>0.008745</td>
      <td>0.0</td>
      <td>0.394664</td>
      <td>0.806962</td>
      <td>0.768943</td>
      <td>3.0</td>
      <td>0.513402</td>
      <td>1.688237e-06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J165409.8-020219</th>
      <td>253.540954</td>
      <td>-2.038749</td>
      <td>16.632152</td>
      <td>24.790265</td>
      <td>2.002898</td>
      <td>1.185102</td>
      <td>104.896836</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>4.722222</td>
      <td>20772.938789</td>
      <td>NaN</td>
      <td>253.540954</td>
      <td>-2.038749</td>
      <td>4.722222</td>
      <td>16.632152</td>
      <td>24.790265</td>
      <td>72.716409</td>
      <td>0</td>
      <td>0</td>
      <td>2.547270e-14</td>
      <td>1.946782e-14</td>
      <td>3.889532e-15</td>
      <td>3.761376e-15</td>
      <td>NaN</td>
      <td>0.176140</td>
      <td>0.016240</td>
      <td>-0.167395</td>
      <td>0.0</td>
      <td>0.174089</td>
      <td>0.516371</td>
      <td>0.157145</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J100301.2+021934</th>
      <td>150.755395</td>
      <td>2.326167</td>
      <td>237.190828</td>
      <td>42.708642</td>
      <td>1.161515</td>
      <td>1.079687</td>
      <td>75.542480</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.666667</td>
      <td>158905.462446</td>
      <td>NaN</td>
      <td>150.755395</td>
      <td>2.326167</td>
      <td>2.666667</td>
      <td>237.190828</td>
      <td>42.708642</td>
      <td>81.028272</td>
      <td>0</td>
      <td>1</td>
      <td>2.186402e-15</td>
      <td>1.675384e-15</td>
      <td>5.853234e-16</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.149906</td>
      <td>0.433479</td>
      <td>0.151156</td>
      <td>0.0</td>
      <td>0.444484</td>
      <td>0.612317</td>
      <td>0.474779</td>
      <td>0.0</td>
      <td>0.456854</td>
      <td>7.386110e-08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J054244.6-404855</th>
      <td>85.686088</td>
      <td>-40.815288</td>
      <td>246.499422</td>
      <td>-29.796437</td>
      <td>3.891118</td>
      <td>2.875106</td>
      <td>178.855720</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.875000</td>
      <td>50405.925987</td>
      <td>NaN</td>
      <td>85.686088</td>
      <td>-40.815288</td>
      <td>3.875000</td>
      <td>246.499422</td>
      <td>-29.796437</td>
      <td>26.563678</td>
      <td>0</td>
      <td>0</td>
      <td>5.791593e-15</td>
      <td>3.559558e-15</td>
      <td>1.471395e-15</td>
      <td>9.266752e-16</td>
      <td>NaN</td>
      <td>-0.108682</td>
      <td>-0.039975</td>
      <td>0.074953</td>
      <td>0.0</td>
      <td>0.358657</td>
      <td>0.302306</td>
      <td>0.460818</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2CXO J180325.0-295432</th>
      <td>270.854168</td>
      <td>-29.909153</td>
      <td>1.117724</td>
      <td>-3.835987</td>
      <td>0.805650</td>
      <td>0.779075</td>
      <td>6.289155</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.875000</td>
      <td>104822.403707</td>
      <td>NaN</td>
      <td>270.854168</td>
      <td>-29.909153</td>
      <td>3.875000</td>
      <td>1.117724</td>
      <td>-3.835987</td>
      <td>44.912474</td>
      <td>1</td>
      <td>0</td>
      <td>1.286669e-15</td>
      <td>0.000000e+00</td>
      <td>5.750358e-16</td>
      <td>6.346875e-16</td>
      <td>NaN</td>
      <td>-0.463460</td>
      <td>-0.730793</td>
      <td>-0.371018</td>
      <td>6.0</td>
      <td>0.958639</td>
      <td>0.998059</td>
      <td>0.993966</td>
      <td>5.0</td>
      <td>0.716285</td>
      <td>3.706027e-08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J181647.4-162129</th>
      <td>274.197777</td>
      <td>-16.358283</td>
      <td>14.477496</td>
      <td>0.010426</td>
      <td>0.853008</td>
      <td>0.796700</td>
      <td>174.514643</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.052632</td>
      <td>18463.958710</td>
      <td>NaN</td>
      <td>274.197777</td>
      <td>-16.358283</td>
      <td>2.052632</td>
      <td>14.477496</td>
      <td>0.010426</td>
      <td>26.362438</td>
      <td>0</td>
      <td>0</td>
      <td>1.862584e-14</td>
      <td>1.672257e-14</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.999375</td>
      <td>0.999375</td>
      <td>-0.999375</td>
      <td>0.0</td>
      <td>0.474449</td>
      <td>0.587416</td>
      <td>0.283705</td>
      <td>0.0</td>
      <td>0.448714</td>
      <td>1.900174e-07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2CXO J173555.2+570356</th>
      <td>263.980184</td>
      <td>57.065711</td>
      <td>85.279886</td>
      <td>32.634090</td>
      <td>4.897011</td>
      <td>3.619107</td>
      <td>102.905240</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.111111</td>
      <td>10337.817452</td>
      <td>NaN</td>
      <td>263.980184</td>
      <td>57.065711</td>
      <td>2.111111</td>
      <td>85.279886</td>
      <td>32.634090</td>
      <td>16.895662</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>4.433949e-15</td>
      <td>NaN</td>
      <td>-0.999375</td>
      <td>-0.999375</td>
      <td>-0.999375</td>
      <td>0.0</td>
      <td>0.416042</td>
      <td>0.496184</td>
      <td>0.243796</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.743</td>
      <td>17.738</td>
      <td>16.831</td>
      <td>16.426</td>
      <td>16.143</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.005</td>
      <td>0.907</td>
      <td>0.688</td>
      <td>0.283</td>
      <td>3.600</td>
    </tr>
    <tr>
      <th>2CXO J134902.7+035728</th>
      <td>207.261495</td>
      <td>3.957986</td>
      <td>336.132887</td>
      <td>63.054502</td>
      <td>1.876542</td>
      <td>1.016800</td>
      <td>133.998688</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.082235</td>
      <td>11005.007980</td>
      <td>NaN</td>
      <td>207.261495</td>
      <td>3.957986</td>
      <td>9.082235</td>
      <td>336.132887</td>
      <td>63.054502</td>
      <td>357.519121</td>
      <td>0</td>
      <td>0</td>
      <td>1.027781e-13</td>
      <td>5.846297e-14</td>
      <td>1.823370e-14</td>
      <td>2.798401e-14</td>
      <td>0.0</td>
      <td>0.126171</td>
      <td>-0.294816</td>
      <td>-0.403498</td>
      <td>0.0</td>
      <td>0.147051</td>
      <td>0.218394</td>
      <td>0.527038</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.452</td>
      <td>21.149</td>
      <td>20.913</td>
      <td>20.646</td>
      <td>20.894</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.303</td>
      <td>0.236</td>
      <td>0.019</td>
      <td>-0.248</td>
      <td>0.558</td>
    </tr>
    <tr>
      <th>2CXO J100001.3-301359</th>
      <td>150.005799</td>
      <td>-30.233315</td>
      <td>264.363857</td>
      <td>19.517936</td>
      <td>0.775855</td>
      <td>0.752233</td>
      <td>103.729420</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>5.793844</td>
      <td>30677.516682</td>
      <td>NaN</td>
      <td>150.005799</td>
      <td>-30.233315</td>
      <td>5.793844</td>
      <td>264.363857</td>
      <td>19.517936</td>
      <td>171.645220</td>
      <td>0</td>
      <td>0</td>
      <td>2.537887e-14</td>
      <td>1.747985e-14</td>
      <td>3.152844e-15</td>
      <td>5.943509e-15</td>
      <td>NaN</td>
      <td>0.213616</td>
      <td>-0.131168</td>
      <td>-0.338538</td>
      <td>2.0</td>
      <td>0.747763</td>
      <td>0.974962</td>
      <td>0.900371</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 71 columns</p>
</div>



Note the above function retrives the data from the various MW catalog we have collected from various MW catalogs and the selected features obtained from the NED. 
<small> Note: The NED contains the information about the source from various other catalogs and other properties not included in this work.</small> 
For the sources we can obtain the data directly from the NED using the function **get_NED_data**. This will create seperate CSV files in the folder _data/NED_data_ .

Example is given for 10 sources from the above table.


```python
from choices import get_NED_data
# No need to provide the entire dataframe, an empty dataframe with name as the index will work.
get_NED_data(src_data.sample(10)[[]])
```

    100%|██████████| 10/10 [00:00<00:00, 48.77it/s]


> All the data used in this work with the selected flag filters and the features are stored in the following files, and will be used in the further section
* Training data : *data/training_data/train_data_minmax.csv*
* Unclassified sources : *data/new_src_data/new_sources.csv*

# Model Training and Validation

### Training Data

In the folder **training_data** all the data table consist of 13882 sources and 57 features. These sources are all the sources matched within a cross match radius of 10 arcses.

The information about the sources (including the source catalog, its name in the parent ctalog and the cross-match offset of its parent catalog wit CSC) is given in the data table *id_frame.csv*. 

Using this *id_frame.csv*, the training data with required constrains can be retrived.

We used the training data with the follwing constrains :

* 'conf_flag' : 0 , 
* 'streak_src_flag' : 0 , 
* 'extent_flag' : 0 , 
* 'pileup_flag' : 0 ,
* 'offset' $\leq 1$

Which gives 7703 sources and we have used 41 out of 57 features.
The training data used in this work is given in *train_data_minmax.csv*


```python
data_train = pd.read_csv('data/training_data/train_data_minmax.csv' , index_col='name')
data_train
```

### Import _make_model_ class
The class _make_model_ is takes in the training data, a classification model(scickit-learn compatible model). This class is can be used to validate the model using CCV method and to train and save the classifier for implementation on the test data.


```python
from utilities_v2 import make_model
```

### Build the Model: _make_model_ class

_make_model_ takes in the following components
*   name : user defined name of the model (can be any string)
*   train_data : as pandas dataframe
*   label : class label for the training data (list or pandas series)
*   classifier : classifier model
*   oversamples : Oversampling function like Scickit-Learn's _SMOTE_ object.

#### Data
the class _make_model_ takes in training data and the training label as pandas dataframe


```python
# Example Implementation ####################
# x = data.drop(columns=['class'])
# y = data['class']

x = data_train.drop(columns=['class'])
y = data_train['class']
```

#### Classifier

Next we will use a classifier from scickit-learn _RandomForestClassifier_ 

The user can supply their own classifier for the _make_model_ object with only condition that the classifier must implement the _fit_ function. (Need not worry, as most of the models in Scickit-Learn always implement the _fit_ function)

<small>Note: the parameters we are giving for the model that we are giving here is optained after hyper-parameter tuning of the model.</small>

##### Random Forest classifier


```python
# Create a new make_model object
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=400 , max_depth=30 , random_state=np.random.randint(0,999999))
```

##### LightGBM classifier


```python
import lightgbm as lgb 
def calc_weight(gamma , y):
    l = len(y)
    cl_weight = {}
    cl_dict = y.value_counts().to_dict()
    for cl , val in zip(cl_dict.keys() , cl_dict.values()):
        w = np.exp((l / val)*gamma)
        cl_weight[cl] = w
    #print(cl_weight)
    return cl_weight
```


```python
gamma = 0.07
cl_weight = calc_weight(gamma , y)
clf = lgb.LGBMClassifier(n_estimators = 100 ,class_weight = cl_weight , objective= 'multiclass', sparse=True , is_unbalance=True , metric=['auc_mu'] ,verbosity = 0 , random_state=42 , num_class=len(np.unique(y)) ,force_col_wise=True)
```

#### Oversampler


```python
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(k_neighbors=4)
```

#### Put everything together 


```python
# rf_model = make_model(model_name = 'test_model', classifier=clf, oversampler = oversampler, train_data = x, label=y)
lgb_model = make_model(model_name = 'lgb_model', classifier=clf, oversampler = None, train_data = x, label=y)
```

### Validate the Model

the object _make_model_ implements *validate* function ehich performs the Cumultive K fold cross validation for the supplied model and for the given data


```python
lgb_model.train()
```


```python
lgb_model.validate(save_predictions=True, multiprocessing=False, k_fold=20)
```

Let us see the validation result

The validation results are stored in the attribute _validation_model_ of the _make_model_ object


```python
# Print validation result
print("Confusion Matrix: ")
print(lgb_model.validation_score['class_labels'])
print(lgb_model.validation_score['confusion_matrix'])
print("Overall Scores: ")
print(lgb_model.validation_score['overall_scores'])
print("Class-Wise scores: ")
display(lgb_model.validation_score['class_wise_scores']*100)
```

### Train the model

Now the above validation function can be used by varying the classifier parameters and then checking the validation result as per the user requirement, and once the results are satisfactoory, the user call the _train_ function of the _make_model_ object which will train and store the supplied classifier. for training, unlike the cross validation where a fraction of th data is used, here the classifier is trained on the entire dataset.


```python
lgb_model.train()
```

### Save the Model

Next we will use the _save_ function of the object _make_model_ to save the classifier alongwith the validation scores and predictions on the training data


```python
lgb_model.save('models/lightGBM-example.joblib')
```

# Application

### Load Data: Unidentified sources


```python
all_new = pd.read_csv('data/new_src_data/new_sources.csv' , index_col='name')
```

### Load Saved Model


```python
import joblib
from utilities_v2 import make_model
lgb_model = joblib.load('models/lightGBM-example.joblib')
lgb_model
```

### Predict

The classifier is inside the *make_model* object and can be accessed by the clf attribute of the *make_model* object


```python
clf = lgb_model.clf
```

The following routine gets the predictions and the class membership probabilities in pretty dataframe format form the classifier applied on the given data of new sources.


```python
# from utilities import softmax , norm_prob
def get_pred_table(u):
    pred_prob = (clf.predict_proba(u))
    pred_prob_df = pd.DataFrame(pred_prob , columns=[f'prob_{el}' for el in clf.classes_] , index = u.index.to_list())
    pred_prob_df
    u_df = pd.DataFrame({
        'name' : u.index.to_list() ,
        'class' : clf.predict(u) , 
        'prob' : [np.amax(el) for el in pred_prob] ,
        'prob_margin' : [el[-1]-el[-2] for el in np.sort(pred_prob , axis=1 ,)]
    }).set_index('name')
    u_df = pd.merge(u_df , pred_prob_df , left_index=True , right_index=True)
    u_df.index.name = 'name'
    u_df 
    return u_df
```


```python
# u_df_var = get_pred_table(variable_src)
u_df = get_pred_table(all_new)
u_df
```

**The predictions from our classifier is stored in the folder *data/classified/* class-wise.** 


