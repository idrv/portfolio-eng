# Saint-Petersburg's realty market study

We have access to archive data from the [Yandex.Realty](https://realty.ya.ru/) service on apartments in St. Petersburg and neighboring settlements. The key objective is to learn how to determine market values of real estate. Study reslults might help to build an automated system to track anomalies and realty frauds. Some of these data were entered by service users, others were obtained automatically with Yandex GIS: distances to a settlement center, airport, nearest park or pond.

Our project is aimed to identify key parameters determining real estate market values.

Project plan:

1. Perform preliminary data analysis. 
2. Make exploratory data analysis (EDA) and preprocess the data:
    - identify missing values;
    - suggest possible explanations for missing data;
    - fill in the missing values, if necessary;
    - identify columns with data types to be changed;
    - observe outliers;
    - calculate several synthetic features:
        - square meter price;
        - day of week, month and year when an ad was created;
        - floor categories;
        - proportions of apartment parts square footage.
3. Perform correlation analysis, graphic trend analysis.
4. Summarize findings.

Input data:

- `'airports_nearest'` — distance to a nearest airport (meters)
- `'balcony'` — how many balconies an apartment has 
- `'ceiling_height'` — ceiling height высота потолков (meters)
- `'cityCenters_nearest'` — distance to a settlement center (meters)
- `'days_exposition'` — how long an ad stayed from publication to taking out (days)
- `'first_day_exposition'` — date of ad publication
- `'floor'` — floor
- `'floors_total'` — how many floors are in a building
- `'is_apartment'` — is an apartment considered to serve strictly commercial purposes (bool)
- `'kitchen_area'` — kitchen square footage (m²)
- `'last_price'` — price when an ad was taken out
- `'living_area'` — livable area square footage (m²)
- `'locality_name'` — settlement name
- `'open_plan'` — if an appartment has an open layout (bool)
- `'parks_around3000'` — how many parks are within 3 kilometers distance
- `'parks_nearest'` — distance to a nearest park (meters)
- `'ponds_around3000'` — how many ponds or lakes are within 3 kilometers distance
- `'ponds_nearest'` — distance to a nearest pond or lake (meters)
- `'rooms'` — how many rooms an apartment has 
- `'studio'` — if an apartment is a studio (bool)
- `'total_area'` — total apartment square footage (m²)
- `'total_images'` — how many aparment photos an ad contains

## First peek at the data

### Loading the data

Let's install and import all necessary libraries, functions and classes:


```python
!pip install -q phik
!pip install -q skimpy
!pip install -q pymystem3
```


```python
import pandas as pd
import matplotlib.pyplot as plt

import phik
from phik.report import plot_correlation_matrix
from phik import report

from seaborn import heatmap

from skimpy import clean_columns

from statsmodels.tsa.seasonal import seasonal_decompose

from pymystem3 import Mystem
```

It's time to load the dataset into a dataframe. From a sneak peek we know, that tabulation is used to separate values. That's why we use the following `sep` value.

We define all possible paths to load data from:


```python
path1 = '/datasets/'
path2 = '/Users/idrv/Yandex.Disk.localized/2022-Ya.Practicum/datasets/'
path3 = '/content/drive/MyDrive/Colab Notebooks/datasets/'
dataset_name = 'real_estate_data.csv'
```

`Try... except` construction is used to create the dataframe:


```python
try:
    realty = pd.read_csv(path1 + dataset_name, sep='\t')
except FileNotFoundError:
    try:
        realty = pd.read_csv(path2 + dataset_name, sep='\t')
    except FileNotFoundError:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            realty = pd.read_csv(path3 + dataset_name, sep='\t')
        except FileNotFoundError:
            print('File not found. Please, check the path!')
```


```python
del path1, path2, path3, dataset_name
```

### Data structure

`info()` method is one of the easiest first steps to explore the data. We use it along with viewing first 5 dataframe rows:


```python
realty.info()
realty.head(5)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23699 entries, 0 to 23698
    Data columns (total 22 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   total_images          23699 non-null  int64  
     1   last_price            23699 non-null  float64
     2   total_area            23699 non-null  float64
     3   first_day_exposition  23699 non-null  object 
     4   rooms                 23699 non-null  int64  
     5   ceiling_height        14504 non-null  float64
     6   floors_total          23613 non-null  float64
     7   living_area           21796 non-null  float64
     8   floor                 23699 non-null  int64  
     9   is_apartment          2775 non-null   object 
     10  studio                23699 non-null  bool   
     11  open_plan             23699 non-null  bool   
     12  kitchen_area          21421 non-null  float64
     13  balcony               12180 non-null  float64
     14  locality_name         23650 non-null  object 
     15  airports_nearest      18157 non-null  float64
     16  cityCenters_nearest   18180 non-null  float64
     17  parks_around3000      18181 non-null  float64
     18  parks_nearest         8079 non-null   float64
     19  ponds_around3000      18181 non-null  float64
     20  ponds_nearest         9110 non-null   float64
     21  days_exposition       20518 non-null  float64
    dtypes: bool(2), float64(14), int64(3), object(3)
    memory usage: 3.7+ MB





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>cityCenters_nearest</th>
      <th>parks_around3000</th>
      <th>parks_nearest</th>
      <th>ponds_around3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>13000000.0</td>
      <td>108.0</td>
      <td>2019-03-07T00:00:00</td>
      <td>3</td>
      <td>2.70</td>
      <td>16.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>NaN</td>
      <td>...</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>18863.0</td>
      <td>16028.0</td>
      <td>1.0</td>
      <td>482.0</td>
      <td>2.0</td>
      <td>755.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>3350000.0</td>
      <td>40.4</td>
      <td>2018-12-04T00:00:00</td>
      <td>1</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>18.6</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>посёлок Шушары</td>
      <td>12817.0</td>
      <td>18603.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>5196000.0</td>
      <td>56.0</td>
      <td>2015-08-20T00:00:00</td>
      <td>2</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>34.3</td>
      <td>4</td>
      <td>NaN</td>
      <td>...</td>
      <td>8.3</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>21741.0</td>
      <td>13933.0</td>
      <td>1.0</td>
      <td>90.0</td>
      <td>2.0</td>
      <td>574.0</td>
      <td>558.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>64900000.0</td>
      <td>159.0</td>
      <td>2015-07-24T00:00:00</td>
      <td>3</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>9</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>28098.0</td>
      <td>6800.0</td>
      <td>2.0</td>
      <td>84.0</td>
      <td>3.0</td>
      <td>234.0</td>
      <td>424.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>10000000.0</td>
      <td>100.0</td>
      <td>2018-06-19T00:00:00</td>
      <td>2</td>
      <td>3.03</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>13</td>
      <td>NaN</td>
      <td>...</td>
      <td>41.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>31856.0</td>
      <td>8098.0</td>
      <td>2.0</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>48.0</td>
      <td>121.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



We should check the dataframe for full duplicates:


```python
realty.duplicated().sum()
```




    0



### Preliminary insights on the data

The dataset contains 23699 rows and 22 columns. Some columns have values missing, since it's count is less than 23699. These values seem to be `NaN`s. 

Almost all column names meet PEP8 recommendations, with four notable exceptions to be corrected: `'cityCenters_nearest'`, `'parks_around3000'`, `'parks_nearest'`, `'ponds_around3000'`.

Some columns will need data type to be converted: e.g., it doesn't seem necessary to keep prices in `float` with one deicaml point: even if a seller prefers to be THIS precise, that kind of precision is not heplful for our purposes.

Obvious duplicates are not found, but the missing values are to deal with. `'is_apartment'`, `'parks_nearest'`, and `'ponds_nearest'` have lots of them, probably due to a lack of those in some neighborhoods and settlements. Some missing values will require data from other columns to fill the `'NaN'`s in.
 
After we define some functions, we rename the four aforemetioned columns according to the PEP8 and check every dataframe column a bit closer.

## Data prepocessing and EDA

### Defining analytical functions

We will need them for our convenience:


```python
def quant_dist(dataframe, column, plot_size=(18,6)):
    '''
    The function provides text and graphic means of distribution analysis
    for a dataframe column containing quantitative values.
    It takes three arguments:
    1. dataframe: dataframe name.
    2. column: column's name as a string (e.g. 'column').
    3. plot_size: plot size for both plots in inches (figsize from matplotlib).
       Default is 18 by 3 (width by heigt).
        
    The function prints pd.DataFrame.describe() method results and draws
    two plots: a boxplot and a distribution histogram.
    '''
    
    print('Feature:', column)
    print(dataframe[column].describe())
    fig = plt.figure(figsize=plot_size)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.set_title(f'Boxplot and distribution plot of {column} feature')
    dataframe.boxplot(column=column, vert=False, rot=90, ax=ax1, showmeans=True)
    dataframe.hist(column=column, bins='fd', yrot=90, ax=ax2)
    plt.show()
    print()
```


```python
def sample_criterion(column, value):
    '''
    The function displays dataframe rows satisfying a user-defined criterion: 
    a user-specified column contains a certain value. The function can be used
    to identify observations containing outliers and unexpected values.
    It takes two arguments:
    1. column: a dataframe column as a Series object (e.g. dataframe['column']).
    2. value: a value being searched in the specified dataframe column.
    
    The function returns a random sample with 5 dataframe rows satysfying
    the criterion. If less then 5 rows were found, an error is raised.
    '''
    sample = realty.query('@column == @value')
    try:
        return sample.sample(n=5)
    except:
        print('Less then 5 rows were found, make a direct slice!')
```


```python
def details(column, upper_limit):
    '''
    The function provides details on value distribution in a dataset column,
    if outliers are suspected. It takes two arguments:
    1. column: a dataframe column as a Series object (e.g. dataframe['column']).;
    2. upper_range: the upper value limit 
    
    The function draws a distribution histogram and prints pandas.describe() output.
    '''
    plt.figure(figsize=(8,4))
    column.hist(bins=100, range=(0, upper_limit))
    plt.title(f'{column.name} value distribution ranged from 0 to {upper_limit}')
    plt.show()
    return column.describe()
```

### Renaming the columns


```python
realty = clean_columns(realty)
realty.columns
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span> column names have been cleaned
</pre>






    Index(['total_images', 'last_price', 'total_area', 'first_day_exposition',
           'rooms', 'ceiling_height', 'floors_total', 'living_area', 'floor',
           'is_apartment', 'studio', 'open_plan', 'kitchen_area', 'balcony',
           'locality_name', 'airports_nearest', 'city_centers_nearest',
           'parks_around_3000', 'parks_nearest', 'ponds_around_3000',
           'ponds_nearest', 'days_exposition'],
          dtype='object')



All columns are named in a convenient way for further analysis.

### `'total_images'` — how many aparment photos an ad contains

We expect a column with discrete non-negative values. Let's check their spread and distribution with a boxplot and a histogram:


```python
quant_dist(realty, 'total_images')
```

    Feature: total_images
    count    23699.000000
    mean         9.858475
    std          5.682529
    min          0.000000
    25%          6.000000
    50%          9.000000
    75%         14.000000
    max         50.000000
    Name: total_images, dtype: float64



    
![png](output_25_1.png)
    


    


Most observations fit Poisson distribution, excepting outliers: zeroes and exactly 20 photos.

Probably we should check the latter obervations for anomalies. We start with 5 random ads without any images:


```python
sample_criterion(realty['total_images'], 0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16373</th>
      <td>0</td>
      <td>3990000.0</td>
      <td>58.7</td>
      <td>2018-02-08T00:00:00</td>
      <td>4</td>
      <td>2.94</td>
      <td>4.0</td>
      <td>36.6</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>Кронштадт</td>
      <td>69449.0</td>
      <td>51175.0</td>
      <td>2.0</td>
      <td>433.0</td>
      <td>3.0</td>
      <td>448.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>20063</th>
      <td>0</td>
      <td>4490000.0</td>
      <td>44.0</td>
      <td>2019-02-22T00:00:00</td>
      <td>2</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>32.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>32364.0</td>
      <td>9442.0</td>
      <td>1.0</td>
      <td>2416.0</td>
      <td>2.0</td>
      <td>488.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>23375</th>
      <td>0</td>
      <td>3150000.0</td>
      <td>46.0</td>
      <td>2017-11-03T00:00:00</td>
      <td>2</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Колпино</td>
      <td>25152.0</td>
      <td>30938.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>896.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>2110</th>
      <td>0</td>
      <td>2400000.0</td>
      <td>29.0</td>
      <td>2017-08-31T00:00:00</td>
      <td>1</td>
      <td>2.50</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Колпино</td>
      <td>26257.0</td>
      <td>32044.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>917.0</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>2897</th>
      <td>0</td>
      <td>11250000.0</td>
      <td>100.0</td>
      <td>2017-04-18T00:00:00</td>
      <td>3</td>
      <td>2.80</td>
      <td>24.0</td>
      <td>50.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>Санкт-Петербург</td>
      <td>49338.0</td>
      <td>16224.0</td>
      <td>1.0</td>
      <td>226.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>289.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Nothing strange so far. What about 20 images?


```python
sample_criterion(realty['total_images'], 20)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14841</th>
      <td>20</td>
      <td>5200000.0</td>
      <td>64.0</td>
      <td>2018-03-21T00:00:00</td>
      <td>3</td>
      <td>2.5</td>
      <td>9.0</td>
      <td>39.6</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>29439.0</td>
      <td>12990.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>5209</th>
      <td>20</td>
      <td>4300000.0</td>
      <td>60.2</td>
      <td>2017-02-16T00:00:00</td>
      <td>3</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>42.7</td>
      <td>4</td>
      <td>NaN</td>
      <td>...</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>Санкт-Петербург</td>
      <td>25780.0</td>
      <td>14493.0</td>
      <td>1.0</td>
      <td>685.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>10020</th>
      <td>20</td>
      <td>14000000.0</td>
      <td>106.0</td>
      <td>2016-04-05T00:00:00</td>
      <td>3</td>
      <td>2.8</td>
      <td>25.0</td>
      <td>60.0</td>
      <td>14</td>
      <td>NaN</td>
      <td>...</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>40584.0</td>
      <td>12398.0</td>
      <td>3.0</td>
      <td>447.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>438.0</td>
    </tr>
    <tr>
      <th>10529</th>
      <td>20</td>
      <td>2600000.0</td>
      <td>58.0</td>
      <td>2018-04-05T00:00:00</td>
      <td>3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>34.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>городской посёлок Мга</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>273.0</td>
    </tr>
    <tr>
      <th>19267</th>
      <td>20</td>
      <td>74000000.0</td>
      <td>153.6</td>
      <td>2019-04-29T00:00:00</td>
      <td>3</td>
      <td>3.3</td>
      <td>8.0</td>
      <td>56.0</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>50.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
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
<p>5 rows × 22 columns</p>
</div>



Alles gut hier. 50 photos?


```python
sample_criterion(realty['total_images'], 50)
```

    Less then 5 rows were found, make a direct slice!



```python
realty.query('total_images == 50')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9778</th>
      <td>50</td>
      <td>11000000.0</td>
      <td>87.0</td>
      <td>2017-10-25T00:00:00</td>
      <td>2</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>32.5</td>
      <td>11</td>
      <td>NaN</td>
      <td>...</td>
      <td>31.0</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>9586.0</td>
      <td>11649.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12667</th>
      <td>50</td>
      <td>20500000.0</td>
      <td>76.0</td>
      <td>2017-12-10T00:00:00</td>
      <td>3</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>47.0</td>
      <td>16</td>
      <td>NaN</td>
      <td>...</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>50812.0</td>
      <td>16141.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>17396</th>
      <td>50</td>
      <td>14500000.0</td>
      <td>119.7</td>
      <td>2017-12-02T00:00:00</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>87.5</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>13.5</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>24375.0</td>
      <td>2410.0</td>
      <td>1.0</td>
      <td>551.0</td>
      <td>2.0</td>
      <td>617.0</td>
      <td>106.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>



Nothing special again. These are normal ads with 50 apartment photos. Let's make an educatted guess. Most ad services put a limit on a number of pictures to upload. That explains the "20 images" anomaly. It's quite rare, that a user selects best 20 pictures, it's more of a bulk upload with a script limiting a number of files. But a script may fail. So, "50 pics" anomalies are probably caused by this kind of failure.

Since this feature is not critical in any analysis step, we leave these values as they were.

To optimize memory consumption it might be better to change the data type to `int8`:


```python
realty['total_images'] = realty['total_images'].astype('int8')
```

Let us set an additional goal: we test if changing data type affects overall memory consumption. To do that we'll change data types wherever it's possible: either to date-time or to a lower bit width. After we're done with data preprocessing, the conclusion will be made: was it worth the effort?

### `'last_price'` — price when an ad was taken out


```python
quant_dist(realty, 'last_price')
```

    Feature: last_price
    count    2.369900e+04
    mean     6.541549e+06
    std      1.088701e+07
    min      1.219000e+04
    25%      3.400000e+06
    50%      4.650000e+06
    75%      6.800000e+06
    max      7.630000e+08
    Name: last_price, dtype: float64



    
![png](output_37_1.png)
    


    


We observe serious outliers that distort the whole distribution picture. Taking a peek at a Top-10 of the most expensive realty in the dataset might provide us with some insights. For the sake simplicity we consider Euro to Russian ruble and US dollar to Russian ruble exchange rates even:


```python
realty.sort_values('last_price', ascending=False).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12971</th>
      <td>19</td>
      <td>763000000.0</td>
      <td>400.0</td>
      <td>2017-09-30T00:00:00</td>
      <td>7</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>250.0</td>
      <td>10</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>25108.0</td>
      <td>3956.0</td>
      <td>1.0</td>
      <td>530.0</td>
      <td>3.0</td>
      <td>756.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>19540</th>
      <td>8</td>
      <td>420000000.0</td>
      <td>900.0</td>
      <td>2017-12-06T00:00:00</td>
      <td>12</td>
      <td>2.80</td>
      <td>25.0</td>
      <td>409.7</td>
      <td>25</td>
      <td>NaN</td>
      <td>...</td>
      <td>112.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>30706.0</td>
      <td>7877.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>318.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>14706</th>
      <td>15</td>
      <td>401300000.0</td>
      <td>401.0</td>
      <td>2016-02-20T00:00:00</td>
      <td>5</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>204.0</td>
      <td>9</td>
      <td>False</td>
      <td>...</td>
      <td>24.0</td>
      <td>3.0</td>
      <td>Санкт-Петербург</td>
      <td>21912.0</td>
      <td>2389.0</td>
      <td>1.0</td>
      <td>545.0</td>
      <td>1.0</td>
      <td>478.0</td>
      <td>393.0</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>19</td>
      <td>330000000.0</td>
      <td>190.0</td>
      <td>2018-04-04T00:00:00</td>
      <td>3</td>
      <td>3.50</td>
      <td>7.0</td>
      <td>95.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>23011.0</td>
      <td>1197.0</td>
      <td>3.0</td>
      <td>519.0</td>
      <td>3.0</td>
      <td>285.0</td>
      <td>233.0</td>
    </tr>
    <tr>
      <th>15651</th>
      <td>20</td>
      <td>300000000.0</td>
      <td>618.0</td>
      <td>2017-12-18T00:00:00</td>
      <td>7</td>
      <td>3.40</td>
      <td>7.0</td>
      <td>258.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>70.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>32440.0</td>
      <td>5297.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>198.0</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>22831</th>
      <td>18</td>
      <td>289238400.0</td>
      <td>187.5</td>
      <td>2019-03-19T00:00:00</td>
      <td>2</td>
      <td>3.37</td>
      <td>6.0</td>
      <td>63.7</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>30.2</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>22494.0</td>
      <td>1073.0</td>
      <td>3.0</td>
      <td>386.0</td>
      <td>3.0</td>
      <td>188.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16461</th>
      <td>17</td>
      <td>245000000.0</td>
      <td>285.7</td>
      <td>2017-04-10T00:00:00</td>
      <td>6</td>
      <td>3.35</td>
      <td>7.0</td>
      <td>182.8</td>
      <td>4</td>
      <td>NaN</td>
      <td>...</td>
      <td>29.8</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>33143.0</td>
      <td>6235.0</td>
      <td>3.0</td>
      <td>400.0</td>
      <td>3.0</td>
      <td>140.0</td>
      <td>249.0</td>
    </tr>
    <tr>
      <th>13749</th>
      <td>7</td>
      <td>240000000.0</td>
      <td>410.0</td>
      <td>2017-04-01T00:00:00</td>
      <td>6</td>
      <td>3.40</td>
      <td>7.0</td>
      <td>218.0</td>
      <td>7</td>
      <td>NaN</td>
      <td>...</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>32440.0</td>
      <td>5297.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>198.0</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>5893</th>
      <td>3</td>
      <td>230000000.0</td>
      <td>500.0</td>
      <td>2017-05-31T00:00:00</td>
      <td>6</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>...</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>32440.0</td>
      <td>5297.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>198.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>8900</th>
      <td>13</td>
      <td>190870000.0</td>
      <td>268.0</td>
      <td>2016-03-25T00:00:00</td>
      <td>3</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>132.0</td>
      <td>7</td>
      <td>NaN</td>
      <td>...</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>32440.0</td>
      <td>5297.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>198.0</td>
      <td>901.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 22 columns</p>
</div>



Prices at hundreds of million rubles (i.e. millions of Euros/dollars) are not a sick fantasy. Although the priciest realty is ₽340M (€/$5.3M) ahead of other contenders, it seems realistic: these prices could exist. And the data type might be changed to `integer`:


```python
realty['last_price'] = realty['last_price'].astype('int32')
```

What about price distribution in a not so "hi-end" segment with a ₽20M upper limit?


```python
realty['last_price'].hist(bins=100, range=(0, 20000000))
plt.show()
```


    
![png](output_43_0.png)
    


₽3.5M (€/$ 55.5 thousand) peak divides the observations into two uneven groups:


```python
realty['last_price'].describe()
```




    count    2.369900e+04
    mean     6.541549e+06
    std      1.088701e+07
    min      1.219000e+04
    25%      3.400000e+06
    50%      4.650000e+06
    75%      6.800000e+06
    max      7.630000e+08
    Name: last_price, dtype: float64



That seems more realistic. The median shows much more balanced prices around ₽4.65M (€/$7 3.6 thousand), while the value of the upper 75% of realty prices is closer to ₽6.8M (€107.8 thousand).

Of course, there are much more expensive offers, including extremly expensive ones. It's to be expected for St.Petersburg being a city with populaion of 5 million.

Do we have abnormally low price values?


```python
details(realty['last_price'], 400000)
```


    
![png](output_48_0.png)
    





    count    2.369900e+04
    mean     6.541549e+06
    std      1.088701e+07
    min      1.219000e+04
    25%      3.400000e+06
    50%      4.650000e+06
    75%      6.800000e+06
    max      7.630000e+08
    Name: last_price, dtype: float64



Only one has been found:


```python
realty[realty['last_price'] < 400000]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8793</th>
      <td>7</td>
      <td>12190</td>
      <td>109.0</td>
      <td>2019-03-20T00:00:00</td>
      <td>2</td>
      <td>2.75</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>25</td>
      <td>NaN</td>
      <td>...</td>
      <td>40.5</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>36421.0</td>
      <td>9176.0</td>
      <td>1.0</td>
      <td>805.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>
</div>



₽12190 (€/$195) for 1 bedroom apartment with 109 meters square footage on a 25th floor in St.Petersburg is not a fairy tale, it's a data error. A user probably entered the sum in thousands of roubles instead of just rubles. We can correct it:


```python
realty.loc[8793, 'last_price'] *= 1000
```

### `'total_area'` — total apartment square footage (m²)


```python
quant_dist(realty, 'total_area')
```

    Feature: total_area
    count    23699.000000
    mean        60.348651
    std         35.654083
    min         12.000000
    25%         40.000000
    50%         52.000000
    75%         69.900000
    max        900.000000
    Name: total_area, dtype: float64



    
![png](output_54_1.png)
    


    


Once more we observe quite a diverse realty (in terms of square footage), including extremely spacy apartments. In order to get a better picture of the most salable properties a 200 m² limit is placed and the resulting distribution is visualized:


```python
details(realty['total_area'], 200)
```


    
![png](output_56_0.png)
    





    count    23699.000000
    mean        60.348651
    std         35.654083
    min         12.000000
    25%         40.000000
    50%         52.000000
    75%         69.900000
    max        900.000000
    Name: total_area, dtype: float64



75% of all observations are in a range up to 70 m². Other 25% are unevenly distibuted within the 70-200 m² range. The rest may be considered as extreme outliers.

### `'first_day_exposition'` — date of ad publication

Obviously teh data type is to be changed to date-time. No missing values are found, but we better to check the date range in order to avoid error values.


```python
realty['first_day_exposition'] = pd.to_datetime(realty['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')
```


```python
plt.figure(figsize=(12,8))
realty['first_day_exposition'].value_counts().plot(kind='line')
plt.title('Date distribution')
plt.show()
```


    
![png](output_61_0.png)
    


Everething looks fine. The end of 2014 (when first ads appeared) seemed to be a time the service was launched. After showing some growth in first half of 2016, the populartity somewhat declined; that could reflect a situation in the real estate market. Since 2017, the market started again to show an increase in supply with several peaks and a new collapse in the middle of 2018. The upper limit of observations is in April 2019.

### `'rooms'` — how many rooms an apartment has

This column contains no missing values. However, we should check it for observations that indicate "0 rooms" (if any): that may be true for commercial real estate (like hangars or some warehouses), but suspicious for residential housing.


```python
quant_dist(realty, 'rooms')
```

    Feature: rooms
    count    23699.000000
    mean         2.070636
    std          1.078405
    min          0.000000
    25%          1.000000
    50%          2.000000
    75%          3.000000
    max         19.000000
    Name: rooms, dtype: float64



    
![png](output_65_1.png)
    


    



```python
details(realty['rooms'], 7)
```


    
![png](output_66_0.png)
    





    count    23699.000000
    mean         2.070636
    std          1.078405
    min          0.000000
    25%          1.000000
    50%          2.000000
    75%          3.000000
    max         19.000000
    Name: rooms, dtype: float64



The vast majority of apartments have one or two rooms (**not bedrooms**: rooms in total). Three rooms apartments are rarer, but still present considerable share. A distribution "rule of thumb" is simple: the more rooms - the less ads.

Time to check those "zero rooms anomalies":


```python
realty.query('rooms == 0').plot(y='total_area', grid=True, kind='hist',bins=100)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](output_68_1.png)
    



```python
sample_criterion(realty['rooms'], 0)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23554</th>
      <td>15</td>
      <td>3350000</td>
      <td>26.00</td>
      <td>2018-09-07</td>
      <td>0</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>8</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>18090.0</td>
      <td>17092.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>112.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>23637</th>
      <td>8</td>
      <td>2350000</td>
      <td>26.00</td>
      <td>2018-06-26</td>
      <td>0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>17.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>посёлок Бугры</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>207.0</td>
    </tr>
    <tr>
      <th>18132</th>
      <td>6</td>
      <td>4800000</td>
      <td>32.50</td>
      <td>2019-04-09</td>
      <td>0</td>
      <td>2.8</td>
      <td>17.0</td>
      <td>22.9</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>43996.0</td>
      <td>13576.0</td>
      <td>1.0</td>
      <td>396.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22246</th>
      <td>6</td>
      <td>3100000</td>
      <td>27.30</td>
      <td>2018-05-29</td>
      <td>0</td>
      <td>2.7</td>
      <td>16.0</td>
      <td>23.3</td>
      <td>15</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Санкт-Петербург</td>
      <td>41935.0</td>
      <td>9551.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>2727</th>
      <td>4</td>
      <td>3670000</td>
      <td>26.49</td>
      <td>2018-07-26</td>
      <td>0</td>
      <td>2.6</td>
      <td>21.0</td>
      <td>19.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Санкт-Петербург</td>
      <td>36579.0</td>
      <td>9092.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>454.0</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



These anomalies include all kinds of property (in terms of square footage):


```python
realty.query('(rooms == 0) and (total_area > 50)').sort_values('total_area', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19392</th>
      <td>5</td>
      <td>71000000</td>
      <td>371.0</td>
      <td>2018-07-26</td>
      <td>0</td>
      <td>3.57</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>25257.0</td>
      <td>6060.0</td>
      <td>1.0</td>
      <td>761.0</td>
      <td>1.0</td>
      <td>584.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>20082</th>
      <td>10</td>
      <td>16300000</td>
      <td>98.4</td>
      <td>2017-11-08</td>
      <td>0</td>
      <td>3.10</td>
      <td>5.0</td>
      <td>60.5</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>26972.0</td>
      <td>5819.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>674.0</td>
      <td>537.0</td>
    </tr>
    <tr>
      <th>3458</th>
      <td>6</td>
      <td>7400000</td>
      <td>73.6</td>
      <td>2017-05-18</td>
      <td>0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>50.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>26581.0</td>
      <td>6085.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>348.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>21227</th>
      <td>0</td>
      <td>8200000</td>
      <td>71.0</td>
      <td>2017-07-21</td>
      <td>0</td>
      <td>5.80</td>
      <td>5.0</td>
      <td>68.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>20170.0</td>
      <td>1261.0</td>
      <td>2.0</td>
      <td>295.0</td>
      <td>3.0</td>
      <td>366.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>13613</th>
      <td>16</td>
      <td>8100000</td>
      <td>58.4</td>
      <td>2019-04-26</td>
      <td>0</td>
      <td>3.30</td>
      <td>7.0</td>
      <td>33.0</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>14509.0</td>
      <td>8288.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



The ad with the largest square footage does not have a *living* area, but shows large **total** area and an open layout. We can safely assume it being a commercial property, thus possibly having no rooms. The rest of the observations are studios.

Let's make test out a hypothesis; when filling out the ad form, some users get confused about how many rooms there are in a studio:


```python
realty.query('studio == True').pivot_table(
    index='rooms', values='total_area', aggfunc=('count'))
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_area</th>
    </tr>
    <tr>
      <th>rooms</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>138</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



Well, the hypothesis seems to be confirmed. What should we do?

Why don't we test another hypothesis: zero number of rooms is connected to facts of an apartment being a studio and/or having an open layout? "Studio - Open plan" comblination takes one of four possible values:


```python
for studio in (True, False):
    for open_plan in (True, False):
        query = realty.query('(rooms == 0) and (studio == @studio) and (open_plan == @open_plan)')['rooms'].count()
        print(f'Studio = {studio}, Open plan = {open_plan}:', query)
```

    Studio = True, Open plan = True: 0
    Studio = True, Open plan = False: 138
    Studio = False, Open plan = True: 59
    Studio = False, Open plan = False: 0


Ads with neither zero rooms AND an open layout, nor with zero rooms NOT being a studio AND NOT having an open layout have been found.

What has been found, are apartments with **yet** zero rooms AND NOT being studios.

Therefore, the hypothesis has been confirmed: there's a certain connection. If sellers define a property a studio, they never consider it having an open layout. The converse is alsi tru: if an apartment considered having an open layout, it's not called "a studio" **yet**.

So, whether a studio has a room or not is ambiguous. Realty sellers approach this question in two opposite manners. In order to resolve the ambiguity for our purposes we pronounce all studios having 1 room:


```python
realty['rooms'] = realty['rooms'].mask(
    (realty['rooms'] == 0) & (realty['studio'] == True) & (realty['open_plan'] == False), other=1)

realty['rooms'] = realty['rooms'].astype('int8')
```


```python
del studio, open_plan
```

### `'ceiling_height'` — ceiling height (meters)


```python
quant_dist(realty, 'ceiling_height')
```

    Feature: ceiling_height
    count    14504.000000
    mean         2.771499
    std          1.261056
    min          1.000000
    25%          2.520000
    50%          2.650000
    75%          2.800000
    max        100.000000
    Name: ceiling_height, dtype: float64



    
![png](output_80_1.png)
    


    


We seem be dealing with some data errors. Ceiling height over 10 meters? Possible in commercial buildings (shopping malls, office centers, gyms, workshops, warehouses etc.), but rather unlikely elsewere. Let's place a 5 meters limit:


```python
realty.query('ceiling_height > 5').head().sort_values('last_price', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1026</th>
      <td>20</td>
      <td>155000000</td>
      <td>310.0</td>
      <td>2018-10-12</td>
      <td>5</td>
      <td>5.3</td>
      <td>3.0</td>
      <td>190.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>63.0</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>24899.0</td>
      <td>4785.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>603.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>464</th>
      <td>15</td>
      <td>66571000</td>
      <td>280.3</td>
      <td>2015-06-11</td>
      <td>6</td>
      <td>5.2</td>
      <td>8.0</td>
      <td>159.5</td>
      <td>7</td>
      <td>NaN</td>
      <td>...</td>
      <td>21.1</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>26316.0</td>
      <td>6655.0</td>
      <td>3.0</td>
      <td>187.0</td>
      <td>1.0</td>
      <td>616.0</td>
      <td>578.0</td>
    </tr>
    <tr>
      <th>1388</th>
      <td>20</td>
      <td>59800000</td>
      <td>399.0</td>
      <td>2015-01-21</td>
      <td>5</td>
      <td>5.6</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>26204.0</td>
      <td>6934.0</td>
      <td>2.0</td>
      <td>149.0</td>
      <td>2.0</td>
      <td>577.0</td>
      <td>719.0</td>
    </tr>
    <tr>
      <th>355</th>
      <td>17</td>
      <td>3600000</td>
      <td>55.2</td>
      <td>2018-07-12</td>
      <td>2</td>
      <td>25.0</td>
      <td>5.0</td>
      <td>32.0</td>
      <td>2</td>
      <td>False</td>
      <td>...</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Гатчина</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>259.0</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>14</td>
      <td>2900000</td>
      <td>75.0</td>
      <td>2018-11-12</td>
      <td>3</td>
      <td>32.0</td>
      <td>3.0</td>
      <td>53.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>Волхов</td>
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
<p>5 rows × 22 columns</p>
</div>



Judging by the prices, all ads with ceiling heights from 5 to 6 meters are somewhat of luxury or commercial real estate. Anything above 6 meters is listed as residential property, pretty standard in everything but ceiling heights. Conclusion: all above 6 meters is a simple mistyping result. Suggested action: since there is no additional data to help in guessing correct heights, to replace errors with a median ceiling height, calculated for the entire dataset.


```python
realty['ceiling_height'] = realty['ceiling_height'].mask(
    (realty['ceiling_height'] > 6), other=realty['ceiling_height'].median())
```

The same approach is used for missing values:


```python
realty['ceiling_height'] = realty['ceiling_height'].fillna(
    realty['ceiling_height'].median())

realty['ceiling_height'] = realty['ceiling_height'].astype('float32')
```

### `'floors_total'` — how many floors are in a building


```python
quant_dist(realty, 'floors_total')
```

    Feature: floors_total
    count    23613.000000
    mean        10.673824
    std          6.597173
    min          1.000000
    25%          5.000000
    50%          9.000000
    75%         16.000000
    max         60.000000
    Name: floors_total, dtype: float64



    
![png](output_88_1.png)
    


    


That's a perfectly understandable distribution. Peaks at five and nine floors are quite expected, that's a common height for (post-)soviet apartment buldings. As for the lack of 11 and 13-floors buildings, there are virtually no buildings of those heights. Let's take a closer look at all observations above 25 floors:


```python
realty.query('floors_total > 25').sort_values('floors_total', ascending=False).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2253</th>
      <td>12</td>
      <td>3800000</td>
      <td>45.5</td>
      <td>2018-06-28</td>
      <td>2</td>
      <td>2.88</td>
      <td>60.0</td>
      <td>27.4</td>
      <td>4</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.40</td>
      <td>NaN</td>
      <td>Кронштадт</td>
      <td>67763.0</td>
      <td>49488.0</td>
      <td>2.0</td>
      <td>342.0</td>
      <td>3.0</td>
      <td>614.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>16731</th>
      <td>9</td>
      <td>3978000</td>
      <td>40.0</td>
      <td>2018-09-24</td>
      <td>1</td>
      <td>2.65</td>
      <td>52.0</td>
      <td>10.5</td>
      <td>18</td>
      <td>NaN</td>
      <td>...</td>
      <td>14.00</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>20728.0</td>
      <td>12978.0</td>
      <td>1.0</td>
      <td>793.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>16934</th>
      <td>5</td>
      <td>4100000</td>
      <td>40.0</td>
      <td>2017-10-17</td>
      <td>1</td>
      <td>1.75</td>
      <td>37.0</td>
      <td>17.4</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>8.34</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>5807</th>
      <td>17</td>
      <td>8150000</td>
      <td>80.0</td>
      <td>2019-01-09</td>
      <td>2</td>
      <td>2.65</td>
      <td>36.0</td>
      <td>41.0</td>
      <td>13</td>
      <td>NaN</td>
      <td>...</td>
      <td>12.00</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15</td>
      <td>5990000</td>
      <td>54.0</td>
      <td>2018-03-22</td>
      <td>2</td>
      <td>2.65</td>
      <td>36.0</td>
      <td>21.4</td>
      <td>28</td>
      <td>NaN</td>
      <td>...</td>
      <td>18.70</td>
      <td>1.0</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>11079</th>
      <td>16</td>
      <td>9200000</td>
      <td>75.0</td>
      <td>2019-02-22</td>
      <td>2</td>
      <td>2.70</td>
      <td>36.0</td>
      <td>40.0</td>
      <td>29</td>
      <td>NaN</td>
      <td>...</td>
      <td>12.00</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12888</th>
      <td>7</td>
      <td>7600000</td>
      <td>70.0</td>
      <td>2016-11-18</td>
      <td>3</td>
      <td>2.70</td>
      <td>35.0</td>
      <td>36.5</td>
      <td>27</td>
      <td>NaN</td>
      <td>...</td>
      <td>23.10</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>413.0</td>
    </tr>
    <tr>
      <th>22946</th>
      <td>14</td>
      <td>7690000</td>
      <td>75.0</td>
      <td>2018-03-27</td>
      <td>2</td>
      <td>2.65</td>
      <td>35.0</td>
      <td>40.0</td>
      <td>8</td>
      <td>NaN</td>
      <td>...</td>
      <td>15.00</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13975</th>
      <td>19</td>
      <td>6990000</td>
      <td>65.0</td>
      <td>2018-10-06</td>
      <td>2</td>
      <td>2.65</td>
      <td>35.0</td>
      <td>32.1</td>
      <td>23</td>
      <td>NaN</td>
      <td>...</td>
      <td>8.90</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>2966</th>
      <td>9</td>
      <td>4300000</td>
      <td>37.0</td>
      <td>2017-08-08</td>
      <td>1</td>
      <td>2.65</td>
      <td>35.0</td>
      <td>14.0</td>
      <td>15</td>
      <td>NaN</td>
      <td>...</td>
      <td>10.40</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>18732.0</td>
      <td>20444.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>80.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 22 columns</p>
</div>



The first entry is hardly true: there are no 60 floors buildings in Kronstadt (St.Petersburg suburb). Probably, it's just a typo: `6` floors should be listed instead.

The same doubts are raised by the second entry. In St. Petersburg, there are no buildings of such height, where realty would be sold even at the moment of conducting this study, and the cost of this offer is way below the "high-floor" segment. A typo is also likely here: instead of `25`, the author of the ad typed `52`.

Ads for buildings 37, 36 and 35 floors tall refer to buildings located side by side, judging by the columns `'airports_nearest'`, `'city_centers_nearest'` and `'parks_nearest'`. It's difficult to recover their true height, and probably unnecessary. We'll just take care of the first two outliers:


```python
realty.loc[2253, 'floors_total'] = 6
realty.loc[16731, 'floors_total'] = 25
```

Now we should deal with the missing values. This information may be important to us, so it is necessary to fill them in. Ii might be convenient to use median values from the entire sample. However, that way the situation is possible, when the `'floor'` value from the ad may be greater than the median value. So, we set a condition. If the property's floor value is greater than the median, we set the `'floors_total'` value equal to the `'floor'` value (as if the property was on the last floor). If it's less than the median, we set the median value for the `'floors_total'`. Let's define a function for that:


```python
def floors_total_fillna(row):
    '''
    The function fills missing values in the 'floors_total' column with a median value.
    It takes the only argument:
    1. row: dataframe row.

    The function checks if the row meets two conditions, and returns either
    a median value for the 'floors_total' column or a corresponding value
    from a 'floor' row. 
    '''
    floors_median = realty['floors_total'].median()
    if pd.isna(row['floors_total']) == True:
        if row['floor'] > floors_median:
            return row['floor']
        else:
            return floors_median
    else:
        return row['floors_total']
```


```python
print('Missing values before:', realty['floors_total'].isna().sum())
realty['floors_total'] = realty.apply(floors_total_fillna, axis=1)
print('Missing values after :', realty['floors_total'].isna().sum())
```

    Missing values before: 86
    Missing values after : 0


### `'living_area'` — livable area square footage (m²)


```python
quant_dist(realty, 'living_area')
```

    Feature: living_area
    count    21796.000000
    mean        34.457852
    std         22.030445
    min          2.000000
    25%         18.600000
    50%         30.000000
    75%         42.300000
    max        409.700000
    Name: living_area, dtype: float64



    
![png](output_97_1.png)
    


    


We observe lots of outliers again. Let's examine the ads with a square footage greater that 200 m²: 


```python
realty.query('living_area > 200').sort_values('floors_total', ascending=False).head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19540</th>
      <td>8</td>
      <td>420000000</td>
      <td>900.0</td>
      <td>2017-12-06</td>
      <td>12</td>
      <td>2.80</td>
      <td>25.0</td>
      <td>409.7</td>
      <td>25</td>
      <td>NaN</td>
      <td>...</td>
      <td>112.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>30706.0</td>
      <td>7877.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>318.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>6621</th>
      <td>20</td>
      <td>99000000</td>
      <td>488.0</td>
      <td>2017-04-09</td>
      <td>5</td>
      <td>2.95</td>
      <td>20.0</td>
      <td>216.0</td>
      <td>17</td>
      <td>NaN</td>
      <td>...</td>
      <td>50.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>27641.0</td>
      <td>4598.0</td>
      <td>1.0</td>
      <td>646.0</td>
      <td>1.0</td>
      <td>368.0</td>
      <td>351.0</td>
    </tr>
    <tr>
      <th>12971</th>
      <td>19</td>
      <td>763000000</td>
      <td>400.0</td>
      <td>2017-09-30</td>
      <td>7</td>
      <td>2.65</td>
      <td>10.0</td>
      <td>250.0</td>
      <td>10</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>25108.0</td>
      <td>3956.0</td>
      <td>1.0</td>
      <td>530.0</td>
      <td>3.0</td>
      <td>756.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>22494</th>
      <td>7</td>
      <td>91075000</td>
      <td>491.0</td>
      <td>2017-05-27</td>
      <td>5</td>
      <td>4.20</td>
      <td>9.0</td>
      <td>274.0</td>
      <td>9</td>
      <td>NaN</td>
      <td>...</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>25525.0</td>
      <td>5845.0</td>
      <td>2.0</td>
      <td>116.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>15701</th>
      <td>12</td>
      <td>83000000</td>
      <td>293.6</td>
      <td>2017-11-10</td>
      <td>4</td>
      <td>2.65</td>
      <td>9.0</td>
      <td>250.0</td>
      <td>7</td>
      <td>NaN</td>
      <td>...</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>25593.0</td>
      <td>5913.0</td>
      <td>2.0</td>
      <td>164.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14706</th>
      <td>15</td>
      <td>401300000</td>
      <td>401.0</td>
      <td>2016-02-20</td>
      <td>5</td>
      <td>2.65</td>
      <td>9.0</td>
      <td>204.0</td>
      <td>9</td>
      <td>False</td>
      <td>...</td>
      <td>24.0</td>
      <td>3.0</td>
      <td>Санкт-Петербург</td>
      <td>21912.0</td>
      <td>2389.0</td>
      <td>1.0</td>
      <td>545.0</td>
      <td>1.0</td>
      <td>478.0</td>
      <td>393.0</td>
    </tr>
    <tr>
      <th>21955</th>
      <td>19</td>
      <td>130000000</td>
      <td>431.0</td>
      <td>2017-10-02</td>
      <td>7</td>
      <td>3.70</td>
      <td>8.0</td>
      <td>220.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>...</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>161.0</td>
    </tr>
    <tr>
      <th>14088</th>
      <td>8</td>
      <td>51000000</td>
      <td>402.0</td>
      <td>2017-02-07</td>
      <td>6</td>
      <td>3.15</td>
      <td>8.0</td>
      <td>300.0</td>
      <td>6</td>
      <td>NaN</td>
      <td>...</td>
      <td>56.0</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>24484.0</td>
      <td>5052.0</td>
      <td>1.0</td>
      <td>253.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>7857</th>
      <td>11</td>
      <td>150000000</td>
      <td>230.0</td>
      <td>2017-10-25</td>
      <td>8</td>
      <td>2.65</td>
      <td>8.0</td>
      <td>220.0</td>
      <td>8</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>25525.0</td>
      <td>5845.0</td>
      <td>2.0</td>
      <td>116.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>12401</th>
      <td>20</td>
      <td>91500000</td>
      <td>495.0</td>
      <td>2017-06-19</td>
      <td>7</td>
      <td>4.65</td>
      <td>7.0</td>
      <td>347.5</td>
      <td>7</td>
      <td>NaN</td>
      <td>...</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>NaN</td>
      <td>5735.0</td>
      <td>2.0</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 22 columns</p>
</div>



Nothing outstanding. Just in case, we perform another check: we shouldn't find any observations, where a `'living_area'` value is greater that a `'total_area'` value:


```python
realty.query('living_area > total_area')['living_area'].count()
```




    0



The check came clear. Now we look closer to ads with square footage less than 100 m²: 


```python
details(realty['living_area'], 100)
```


    
![png](output_103_0.png)
    





    count    21796.000000
    mean        34.457852
    std         22.030445
    min          2.000000
    25%         18.600000
    50%         30.000000
    75%         42.300000
    max        409.700000
    Name: living_area, dtype: float64



The distribution brings no surprises. The most property is quite small, below 20 and from 28 to 34 squafe meters.

Let's investigate the missing values:


```python
realty[realty['living_area'].isna() == True].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>64900000</td>
      <td>159.0</td>
      <td>2015-07-24</td>
      <td>3</td>
      <td>2.65</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>9</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>28098.0</td>
      <td>6800.0</td>
      <td>2.0</td>
      <td>84.0</td>
      <td>3.0</td>
      <td>234.0</td>
      <td>424.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>7915000</td>
      <td>71.6</td>
      <td>2019-04-18</td>
      <td>2</td>
      <td>2.65</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>22</td>
      <td>NaN</td>
      <td>...</td>
      <td>18.9</td>
      <td>2.0</td>
      <td>Санкт-Петербург</td>
      <td>23982.0</td>
      <td>11634.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>12</td>
      <td>2200000</td>
      <td>32.8</td>
      <td>2018-02-19</td>
      <td>1</td>
      <td>2.65</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Коммунар</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>10</td>
      <td>1990000</td>
      <td>45.8</td>
      <td>2017-10-28</td>
      <td>2</td>
      <td>2.50</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>поселок городского типа Красный Бор</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>196.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>13</td>
      <td>5350000</td>
      <td>40.0</td>
      <td>2018-11-18</td>
      <td>1</td>
      <td>2.65</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Санкт-Петербург</td>
      <td>30471.0</td>
      <td>11603.0</td>
      <td>1.0</td>
      <td>620.0</td>
      <td>1.0</td>
      <td>1152.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



The subsample contains a variety of ads with different square footage and quantity of rooms. It's not that we can just fill the missing values with the median. Therefore, the following decision has been made: we copy the `'living_area'`, `'total_area'`, `'kitchen_area'`, and `'studio'` columns into a temporary dataset (the last two are needed a bit later, as the column `'kitchen_area'` also has missing values). A column for categorical values will also be created in that temporary dataset. Categorization will be carried according the square footage; then median values for a living area square footage will be calculated, and the missing values will be filled with these medians, depending on the category of the property according to the total square footage.


```python
area = realty[['total_area', 'living_area', 'kitchen_area', 'studio']].copy()
```


```python
def total_area_categories(total_area):
    '''
    The function catogorizes observations based on a 'total area' column value.
    It takes the only argument:
    1. total_area: total area value.

    The function returns one categorical value from this list:
    - '<20'
    - '20-35'
    - '35-45'
    - '45-60'
    - '60-80'
    - '80-100'
    - '100-200'
    - '>200'
    '''
    if total_area < 20:
        return '<20'
    if 20 <= total_area < 35:
        return '20-35'
    if 35 <= total_area < 45:
        return '35-45'
    if 45 <= total_area < 60:
        return '45-60'
    if 60 <= total_area < 80:
        return '60-80'
    if 80 <= total_area < 100:
        return '80-100'
    if 100 <= total_area < 200:
        return '100-200'
    if 200 <= total_area:
        return '>200'
```


```python
area.loc[:, 'total_area_category'] = area.loc[:, 'total_area'].apply(total_area_categories)
```

Now we calculate median values of the living area square footage for each category:


```python
total_area_medians = area.groupby('total_area_category')['total_area'].median()
print(total_area_medians)
```

    total_area_category
    100-200    120.0
    20-35       31.4
    35-45       40.0
    45-60       52.0
    60-80       67.4
    80-100      87.3
    <20         17.6
    >200       250.0
    Name: total_area, dtype: float64



```python
def living_area_fillna(row):
    """
    The function performs a check, if a 'living_area' column value is missing.
    It takes the only argument:
    1. row: a dataframe row to check the value of the 'living_area' column.

    If the check returns True, the function returns a median value
    for a corresponding category from a 'total_area_category' column. 
    """
    if pd.isna(row['living_area']) == True:
        if row['total_area_category'] == '<20':
            return total_area_medians['<20']
        elif row['total_area_category'] == '20-35':
            return total_area_medians['20-35']
        elif row['total_area_category'] == '35-45':
            return total_area_medians['35-45']
        elif row['total_area_category'] == '45-60':
            return total_area_medians['45-60']
        elif row['total_area_category'] == '60-80':
            return total_area_medians['60-80']
        elif row['total_area_category'] == '80-100':
            return total_area_medians['80-100']
        elif row['total_area_category'] == '100-200':
            return total_area_medians['100-200']
        elif row['total_area_category'] == '>200':
            return total_area_medians['>200']
    return row['living_area']
```


```python
print('Missing values before:', area['living_area'].isna().sum())
area.loc[:, 'living_area'] = area.apply(living_area_fillna, axis=1)
print('Missing values after :', area['living_area'].isna().sum())
```

    Missing values before: 1903
    Missing values after : 0


And we're done! The most likely cause of the missing values is that this field was omitted when the observation was created.

Now we should merge our results with the original dataset. However, we will need the temporary dataframe again when we analyze the kitchen square footage. So, we'll wait for a little longer with merging.


```python
del total_area_medians
```

### `'floor'` — floor


```python
quant_dist(realty, 'floor')
```

    Feature: floor
    count    23699.000000
    mean         5.892358
    std          4.885249
    min          1.000000
    25%          2.000000
    50%          4.000000
    75%          8.000000
    max         33.000000
    Name: floor, dtype: float64



    
![png](output_117_1.png)
    


    



```python
details(realty['floor'], 25)
```


    
![png](output_118_0.png)
    





    count    23699.000000
    mean         5.892358
    std          4.885249
    min          1.000000
    25%          2.000000
    50%          4.000000
    75%          8.000000
    max         33.000000
    Name: floor, dtype: float64



No surprises here. The most advertised property is on floors 1 to 5; that's the only thing for sure.

### `'is_apartment'` — is an apartment considered to serve strictly commercial purposes (bool)

We know this column to be of Boolean type. How are the data distributed?


```python
realty['is_apartment'].value_counts()
```




    False    2725
    True       50
    Name: is_apartment, dtype: int64



Apparently, not all property sellers bothered to indicate "the property is not an apartment". It seems ok to fill in all missing values with `'False'`. By the way, this column stores data as strings, which is hardly optimal in terms of memory usage:


```python
realty.loc[:, 'is_apartment'] = realty.loc[:, 'is_apartment'].fillna(False).astype('bool')
```


### `'studio'` — if an apartment is a studio (bool)


```python
realty['studio'].value_counts()
```




    False    23550
    True       149
    Name: studio, dtype: int64



Everything's in order!

### `'open_plan'` — if an appartment has an open layout (bool)


```python
realty['open_plan'].value_counts()
```




    False    23632
    True        67
    Name: open_plan, dtype: int64



Again, everything looks fine.

### `'kitchen_area'` — kitchen square footage (m²)


```python
quant_dist(realty, 'kitchen_area')
```

    Feature: kitchen_area
    count    21421.000000
    mean        10.569807
    std          5.905438
    min          1.300000
    25%          7.000000
    50%          9.100000
    75%         12.000000
    max        112.000000
    Name: kitchen_area, dtype: float64



    
![png](output_132_1.png)
    


    


How big a kitchen may be! It is possible, though, e.g., a dining room with a cooking zone. Anyway, let's have a closer look:


```python
realty.query('kitchen_area > 40')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>10000000</td>
      <td>100.0</td>
      <td>2018-06-19</td>
      <td>2</td>
      <td>3.03</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>13</td>
      <td>False</td>
      <td>...</td>
      <td>41.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>31856.0</td>
      <td>8098.0</td>
      <td>2.0</td>
      <td>112.0</td>
      <td>1.0</td>
      <td>48.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>7</td>
      <td>45000000</td>
      <td>161.0</td>
      <td>2017-10-17</td>
      <td>3</td>
      <td>3.20</td>
      <td>8.0</td>
      <td>38.0</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>50.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>32537.0</td>
      <td>6589.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>99.0</td>
      <td>541.0</td>
    </tr>
    <tr>
      <th>263</th>
      <td>12</td>
      <td>39900000</td>
      <td>140.6</td>
      <td>2016-11-19</td>
      <td>2</td>
      <td>3.50</td>
      <td>8.0</td>
      <td>39.8</td>
      <td>7</td>
      <td>False</td>
      <td>...</td>
      <td>49.2</td>
      <td>4.0</td>
      <td>Санкт-Петербург</td>
      <td>32537.0</td>
      <td>6589.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>99.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>492</th>
      <td>18</td>
      <td>95000000</td>
      <td>216.0</td>
      <td>2017-12-05</td>
      <td>4</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>86.0</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>77.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>21740.0</td>
      <td>436.0</td>
      <td>2.0</td>
      <td>138.0</td>
      <td>3.0</td>
      <td>620.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>511</th>
      <td>7</td>
      <td>5950000</td>
      <td>69.0</td>
      <td>2017-12-21</td>
      <td>1</td>
      <td>2.65</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>12</td>
      <td>False</td>
      <td>...</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>посёлок Мурино</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>56.0</td>
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
    </tr>
    <tr>
      <th>21923</th>
      <td>10</td>
      <td>115490000</td>
      <td>235.0</td>
      <td>2017-04-09</td>
      <td>5</td>
      <td>4.90</td>
      <td>5.0</td>
      <td>140.0</td>
      <td>5</td>
      <td>False</td>
      <td>...</td>
      <td>50.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>22777.0</td>
      <td>1328.0</td>
      <td>3.0</td>
      <td>652.0</td>
      <td>3.0</td>
      <td>253.0</td>
      <td>351.0</td>
    </tr>
    <tr>
      <th>22494</th>
      <td>7</td>
      <td>91075000</td>
      <td>491.0</td>
      <td>2017-05-27</td>
      <td>5</td>
      <td>4.20</td>
      <td>9.0</td>
      <td>274.0</td>
      <td>9</td>
      <td>False</td>
      <td>...</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>25525.0</td>
      <td>5845.0</td>
      <td>2.0</td>
      <td>116.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>22980</th>
      <td>19</td>
      <td>24500000</td>
      <td>155.4</td>
      <td>2017-10-10</td>
      <td>3</td>
      <td>3.00</td>
      <td>4.0</td>
      <td>72.0</td>
      <td>2</td>
      <td>False</td>
      <td>...</td>
      <td>65.0</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>43758.0</td>
      <td>15461.0</td>
      <td>1.0</td>
      <td>756.0</td>
      <td>2.0</td>
      <td>278.0</td>
      <td>325.0</td>
    </tr>
    <tr>
      <th>23327</th>
      <td>19</td>
      <td>34400000</td>
      <td>215.0</td>
      <td>2019-03-15</td>
      <td>5</td>
      <td>2.75</td>
      <td>4.0</td>
      <td>82.4</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>40.1</td>
      <td>NaN</td>
      <td>Санкт-Петербург</td>
      <td>37268.0</td>
      <td>15419.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23491</th>
      <td>20</td>
      <td>21800000</td>
      <td>250.0</td>
      <td>2017-09-16</td>
      <td>3</td>
      <td>2.65</td>
      <td>12.0</td>
      <td>104.0</td>
      <td>7</td>
      <td>False</td>
      <td>...</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Санкт-Петербург</td>
      <td>43558.0</td>
      <td>13138.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>49.0</td>
    </tr>
  </tbody>
</table>
<p>131 rows × 22 columns</p>
</div>



Something must be wrong. It's hard to imagine a 50 m² kitchen in a 69 m² apartment. In general, we can expect apartment and kitchen square footage values to be correlated:


```python
realty[['total_area', 'kitchen_area']].corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_area</th>
      <th>kitchen_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>total_area</th>
      <td>1.000000</td>
      <td>0.609121</td>
    </tr>
    <tr>
      <th>kitchen_area</th>
      <td>0.609121</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The linear correlation appears to be not exactly strong, but noteable. What if we find a kitchen to total square footage ratio in order to understand if the "50 m² kitchen in a 69 m² apartment" situation is normal?


```python
area['kitchen_to_total_ratio'] = area['kitchen_area'] / area['total_area']
area['kitchen_to_total_ratio'].describe()
```




    count    21421.000000
    mean         0.187355
    std          0.072968
    min          0.025381
    25%          0.133333
    50%          0.172414
    75%          0.232416
    max          0.787879
    Name: kitchen_to_total_ratio, dtype: float64



This ratio's average and median values indicate that the said situation is far from typical in the sample. Some data need to be corrected.
A careful review of the records where the ratio exceeds the IQR revealed the source of many errors. The dataset has numerous records where kitchen square footage exceeds living area square footage. Most likely, the property sellers mixed up the corresponding fields in a web form and put the kitchen square footage value into the living area square footage field and vice versa. We should identify these records and swap the living area and kitchen square footage values.


```python
area.query('kitchen_area > living_area')['kitchen_area'].count()
```




    355




```python
def living_swap(row):
    if row['kitchen_area'] > row['living_area']:
        living_area_swap = row['kitchen_area']
        return living_area_swap
    return row['living_area']

def kitchen_swap(row):
    if row['kitchen_area'] > row['living_area']:
        kitchen_area_swap = row['living_area']
        return kitchen_area_swap
    return row['kitchen_area']
```


```python
area['living_area'] = area.loc[:].apply(living_swap, axis=1)
area['kitchen_area'] = area.loc[:].apply(kitchen_swap, axis=1)
```


```python
area.query('kitchen_area > living_area')['kitchen_area'].count()
```




    0



Now let's fill in the missing values. Since we were able to calculate the average ratio of the kitchen to the total square footage, we could calculate the approximate square footage of the kitchen based on this average:


```python
def kitchen_area_fillna(row):
    '''
    The function fills missing values in the 'kitchen_area' column with a mean value.
    It takes the only argument:
    1. row: dataframe row.

    The function checks if a value is missing, and returns a new value based
    on kitchen area to total area ratio and a mean value of all kitchen areas.
    '''
    kitchen_mean = area['kitchen_to_total_ratio'].describe()['mean']
    if pd.isna(row['kitchen_area']) == True:
        kitchen_estimate = row['total_area'] * kitchen_mean
        return kitchen_estimate
    return row['kitchen_area']
```


```python
print('Missing values before:', area['kitchen_area'].isna().sum())
area.loc[:, 'kitchen_area'] = area.apply(kitchen_area_fillna, axis=1)
print('Missing values after :', area['kitchen_area'].isna().sum())
```

    Missing values before: 2278
    Missing values after : 0


Finally, we round the kitchen area values to one decimal place and transfer all the values of the `'living_area'` and `'kitchen_area'` columns that we made changes back from the auxiliary dataframe to the main one:


```python
area['kitchen_area'] = round(area['kitchen_area'], 1)

realty[['living_area', 'kitchen_area']] = area[['living_area', 'kitchen_area']]
```

Since we started to optimize the memory usage, we should delete the `'area'` dataframe. Using the `'.copy()'` method, we created a direct copy of it, not a symbolic link. However, it will still be helpful to us so we will do it later.

The last thing to do is to fill in the missinga values that the presence of `'NaN'` has left in the `'kitchen_to_total_ratio'` column. Let's repeat the action we've already done once again, rounding the values to the second decimal place for ease of viewing:


```python
area['kitchen_to_total_ratio'] = round((area['kitchen_area'] / area['total_area']), 2)
```

### `'balcony'` — how many balconies an apartment has


```python
quant_dist(realty, 'balcony')
```

    Feature: balcony
    count    12180.000000
    mean         1.150082
    std          1.071300
    min          0.000000
    25%          0.000000
    50%          1.000000
    75%          2.000000
    max          5.000000
    Name: balcony, dtype: float64



    
![png](output_152_1.png)
    


    


Let's find out which apartments have more than two balconies:


```python
sample_criterion(realty['balcony'], 3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8528</th>
      <td>20</td>
      <td>4700000</td>
      <td>56.0</td>
      <td>2016-04-27</td>
      <td>2</td>
      <td>2.75</td>
      <td>20.0</td>
      <td>30.70</td>
      <td>10</td>
      <td>False</td>
      <td>...</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>поселок Мурино</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>13289</th>
      <td>20</td>
      <td>8900000</td>
      <td>110.0</td>
      <td>2018-10-25</td>
      <td>5</td>
      <td>2.60</td>
      <td>9.0</td>
      <td>69.80</td>
      <td>8</td>
      <td>False</td>
      <td>...</td>
      <td>15.3</td>
      <td>3.0</td>
      <td>Санкт-Петербург</td>
      <td>32611.0</td>
      <td>14447.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9483</th>
      <td>20</td>
      <td>8800000</td>
      <td>87.1</td>
      <td>2019-04-02</td>
      <td>3</td>
      <td>2.60</td>
      <td>12.0</td>
      <td>48.70</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>11.1</td>
      <td>3.0</td>
      <td>Санкт-Петербург</td>
      <td>28036.0</td>
      <td>13755.0</td>
      <td>1.0</td>
      <td>526.0</td>
      <td>1.0</td>
      <td>161.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>13525</th>
      <td>8</td>
      <td>22450000</td>
      <td>211.0</td>
      <td>2017-09-27</td>
      <td>7</td>
      <td>3.47</td>
      <td>5.0</td>
      <td>143.00</td>
      <td>3</td>
      <td>False</td>
      <td>...</td>
      <td>26.8</td>
      <td>3.0</td>
      <td>Санкт-Петербург</td>
      <td>24431.0</td>
      <td>4263.0</td>
      <td>2.0</td>
      <td>366.0</td>
      <td>3.0</td>
      <td>329.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20397</th>
      <td>19</td>
      <td>7999000</td>
      <td>67.5</td>
      <td>2018-11-27</td>
      <td>2</td>
      <td>2.65</td>
      <td>19.0</td>
      <td>33.67</td>
      <td>13</td>
      <td>False</td>
      <td>...</td>
      <td>15.6</td>
      <td>3.0</td>
      <td>Санкт-Петербург</td>
      <td>16795.0</td>
      <td>15591.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>936.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
sample_criterion(realty['balcony'], 4)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19982</th>
      <td>14</td>
      <td>6700000</td>
      <td>45.8</td>
      <td>2018-01-10</td>
      <td>1</td>
      <td>3.00</td>
      <td>15.0</td>
      <td>15.9</td>
      <td>8</td>
      <td>False</td>
      <td>...</td>
      <td>14.9</td>
      <td>4.0</td>
      <td>Санкт-Петербург</td>
      <td>19040.0</td>
      <td>4450.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>871.0</td>
      <td>231.0</td>
    </tr>
    <tr>
      <th>18006</th>
      <td>19</td>
      <td>10999000</td>
      <td>96.0</td>
      <td>2017-10-20</td>
      <td>3</td>
      <td>2.76</td>
      <td>25.0</td>
      <td>58.8</td>
      <td>17</td>
      <td>False</td>
      <td>...</td>
      <td>12.6</td>
      <td>4.0</td>
      <td>Санкт-Петербург</td>
      <td>22751.0</td>
      <td>15365.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>439.0</td>
      <td>184.0</td>
    </tr>
    <tr>
      <th>17630</th>
      <td>8</td>
      <td>31490000</td>
      <td>150.0</td>
      <td>2017-11-21</td>
      <td>4</td>
      <td>2.80</td>
      <td>12.0</td>
      <td>74.0</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>Санкт-Петербург</td>
      <td>39946.0</td>
      <td>10720.0</td>
      <td>1.0</td>
      <td>2102.0</td>
      <td>3.0</td>
      <td>303.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>14537</th>
      <td>17</td>
      <td>14000000</td>
      <td>95.0</td>
      <td>2019-02-27</td>
      <td>3</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>76.0</td>
      <td>4</td>
      <td>False</td>
      <td>...</td>
      <td>17.8</td>
      <td>4.0</td>
      <td>Санкт-Петербург</td>
      <td>11633.0</td>
      <td>8872.0</td>
      <td>1.0</td>
      <td>116.0</td>
      <td>3.0</td>
      <td>277.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>788</th>
      <td>20</td>
      <td>10000000</td>
      <td>75.0</td>
      <td>2019-04-24</td>
      <td>4</td>
      <td>2.60</td>
      <td>16.0</td>
      <td>51.0</td>
      <td>5</td>
      <td>False</td>
      <td>...</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>Санкт-Петербург</td>
      <td>35614.0</td>
      <td>11971.0</td>
      <td>1.0</td>
      <td>536.0</td>
      <td>1.0</td>
      <td>229.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
sample_criterion(realty['balcony'], 5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>kitchen_area</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10684</th>
      <td>6</td>
      <td>32894076</td>
      <td>364.7</td>
      <td>2019-05-02</td>
      <td>7</td>
      <td>3.32</td>
      <td>3.0</td>
      <td>250.0</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>68.3</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20085</th>
      <td>9</td>
      <td>3400000</td>
      <td>53.0</td>
      <td>2017-09-28</td>
      <td>2</td>
      <td>2.40</td>
      <td>5.0</td>
      <td>29.0</td>
      <td>3</td>
      <td>False</td>
      <td>...</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>деревня Малое Верево</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>2613</th>
      <td>12</td>
      <td>6750000</td>
      <td>68.0</td>
      <td>2019-04-19</td>
      <td>3</td>
      <td>2.50</td>
      <td>6.0</td>
      <td>47.0</td>
      <td>5</td>
      <td>False</td>
      <td>...</td>
      <td>7.5</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>18363.0</td>
      <td>15820.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11925</th>
      <td>18</td>
      <td>9900000</td>
      <td>86.1</td>
      <td>2018-02-20</td>
      <td>3</td>
      <td>2.70</td>
      <td>24.0</td>
      <td>49.0</td>
      <td>2</td>
      <td>False</td>
      <td>...</td>
      <td>11.2</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>7145.0</td>
      <td>13360.0</td>
      <td>1.0</td>
      <td>1022.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>344.0</td>
    </tr>
    <tr>
      <th>2113</th>
      <td>6</td>
      <td>5500000</td>
      <td>58.0</td>
      <td>2018-03-06</td>
      <td>3</td>
      <td>2.65</td>
      <td>10.0</td>
      <td>52.0</td>
      <td>5</td>
      <td>False</td>
      <td>...</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>Санкт-Петербург</td>
      <td>33928.0</td>
      <td>12928.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



It isn't easy to conclude. There are some realistic options here: it is quite possible to imagine four balconies in a four-room apartment or at least four separate exits to connected balconies. However, imagining five balconies in a one-room apartment is pretty challenging.
Since access to a balcony from a kitchen is not rare, we'll set a limitation: there can be no more balconies in a property than "number of rooms +1". How many ads violate this limitation?


```python
realty.query('balcony > (rooms + 1)')['balcony'].count()
```




    383



There is more than one of these ads. Well, let's make a decision again. We'll consider the information about the number of balconies in such ads to be incorrect (the result of an intentional or unintentional user error) and correct it using the formula "number of balconies" = "number of rooms" + 1:


```python
for i in list(realty.query('balcony > (rooms + 1)').index):
    realty.loc[i, 'balcony'] = realty.loc[i, 'rooms'] + 1
```

Finally, let's get rid of the missing values. Here we'll assume that the missing values are the result of the ad authors being lazy to fill in this field in the absence of a balcony. Therefore a missing value is equivalent to a zero value:


```python
realty['balcony'] = realty['balcony'].fillna(0).astype('int32')
```


```python
del i
```

### `'locality_name'` — settlement name

First of all, we need to check, how the names of settlements are displayed. We can do it with `realty['locality_name'].unique()`. The output is omitted due to its massiveness.

The records are mostly formatted to reflect not only the name but also the type of settlement. This makes searching somewhat hard. Therefore, we will have to eliminate the types of settlements in the names. Also, the `'NaN'` values in the column will get in the way, so we first replace them with the `'unknown'` string values:


```python
realty.loc[:, 'locality_name'] = realty.loc[:, 'locality_name'].fillna('unknown')
```


```python
realty['locality_name'] = realty['locality_name'].str.lower()
locality_list = realty['locality_name'].unique()
```

Due to the length of the output, we'll describe what would be done next and give the commented-out code without output. To determine which settlement types are contained in the `'locality_name'` column, lemmatization is used:


```python
m = Mystem()

locality_lemmas = []
for locality in locality_list:
    locality_lemmas.extend(m.lemmatize(locality))
print(locality_lemmas[:30]) # limiting the output, shaping it into one continious list of values, not "one value - one string"
```

    ['санкт-петербург', '\n', 'поселок', ' ', 'шушары', '\n', 'городской', ' ', 'поселок', ' ', 'янино', '-', '1', '\n', 'поселок', ' ', 'парголовый', '\n', 'поселок', ' ', 'муриный', '\n', 'ломоносов', '\n', 'сертолово', '\n', 'петергоф', '\n', 'пушкин', '\n']


We assume that words related to a settlement type would occur in the dataset more often than actual names. Based on the word frequency, we make a reference list  with the most popular words describing the settlement type:


```python
locality_type = [
    'поселок ', 'городской ', 'тип ', 'деревня ',
    'садовый ', 'некоммерческий ', 'товарищество ', 'село ', 'станция ',
    'коттеджный ', 'при ', 'железнодорожный '
]
```

It's time to define a function to process the settlement names. They also need to be lemmatized, but with different purposes:
* to bring the words denoting the settlement type into the same dictionary form as in the reference list;
* to remove the cyrillic letter `ё`.


```python
def locality_name(row):
    lemma = m.lemmatize(row) # the method always returns '\n' as the last element
    lemma.pop() # deleting '\n'
    row = ''.join(lemma) # rejoining substrings to one string
    return row
```

Let's apply the function to the `'locality_name'` column:


```python
realty.loc[:, 'locality_name'] = realty.loc[:, 'locality_name'].apply(locality_name)
```

Now let's check each element of the `'locality_name'` column for the presence of words from the reference list of settlement types. Each found match is to be removed from the target column.


```python
pat = '|'.join(locality_type)
realty['locality_name'] = realty.locality_name.str.lower().replace(pat,'', regex=True)
```

Success! The output of `'realty['locality_name'].unique()'` is again omitted due to its length, though it has been checked. All column values contain only proper names in lowercase. That allows at least to group them.

We perfectly understand that the function will not work one hundred percent accurately. From the samples, it is clear that the name "Шушары" ("Shushary") turns into "шушары" ("shushary") and "Дружная горка" ("Druzhnaya Gorka") - into "дружный горка" ("druzhniy gorka"). But when analyzing large datasets, this is the least of the possible evils. The main thing is to be aware of these slight inaccuracies in the processed data.


```python
del m, locality_lemmas, locality_type, pat
```

### `'airports_nearest'` — distance to a nearest airport (meters)


```python
quant_dist(realty, 'airports_nearest')
```

    Feature: airports_nearest
    count    18157.000000
    mean     28793.672193
    std      12630.880622
    min          0.000000
    25%      18585.000000
    50%      26726.000000
    75%      37273.000000
    max      84869.000000
    Name: airports_nearest, dtype: float64



    
![png](output_181_1.png)
    


    


As we can see, the distance varies from 8 to 90 km. These data do not cause any suspicions.

The question is: how do we fill in the missing values? One of the ways is to try to find ads from the same settlement in which the distance to the airport is indicated. If there are any, it will be easy to deal with the missing values.

A small loop will search these records. It calculates for each settlement:
* a number of records containing a distance to the airport;
* a number of records where this value is missing.

If these numbers differ from each other or coincide, but there is at least one record with a distance to the airport, the loop displays the name of the settlement and both values.


```python
for locality in locality_list:
    no_airport = realty[(realty['locality_name'] == locality) & (realty['airports_nearest'].isna() == True)]['airports_nearest'].count()
    has_airport = realty[(realty['locality_name'] == locality) & (realty['airports_nearest'].isna() == False)]['airports_nearest'].count()
    if no_airport != has_airport or has_airport != 0:
        print(f"For {locality} {no_airport} records without distance to airport, {has_airport} - with distance to airport")

```

    For санкт-петербург 0 records without distance to airport, 15636 - with distance to airport
    For ломоносов 0 records without distance to airport, 132 - with distance to airport
    For петергоф 0 records without distance to airport, 201 - with distance to airport
    For пушкин 0 records without distance to airport, 369 - with distance to airport
    For колпино 0 records without distance to airport, 337 - with distance to airport
    For кронштадт 0 records without distance to airport, 95 - with distance to airport
    For павловск 0 records without distance to airport, 38 - with distance to airport
    For сестрорецк 0 records without distance to airport, 183 - with distance to airport
    For зеленогорск 0 records without distance to airport, 24 - with distance to airport
    For unknown 0 records without distance to airport, 41 - with distance to airport


We did not find such records. This means that we cannot determine the distance to the airport based on the data from the dataset and fill in the missing values with it. The median or mean will not work for geographic data. We'll have to leave the missing values as they were.


```python
del no_airport, has_airport
```

### `'city_centers_nearest'` — distance to a settlement center (meters)


```python
quant_dist(realty, 'city_centers_nearest')
```

    Feature: city_centers_nearest
    count    18180.000000
    mean     14191.277833
    std       8608.386210
    min        181.000000
    25%       9238.000000
    50%      13098.500000
    75%      16293.000000
    max      65968.000000
    Name: city_centers_nearest, dtype: float64



    
![png](output_187_1.png)
    


    


So far, there are no particular problems in the data. Individual "peaks" on the charts may correspond to listings of properties associated with the locality. At the same time, we know that these data have missing values. By analogy with the previous column, we can assume that the missing values occur when there's no way to calculate the distance to a city center automatically. Let's apply the same logic as to the distance to the airport: we see if we can fill in at least some of the missing values based on data from other ads from the same settlement, but with the specified distance to the center of St. Petersburg:


```python
for locality in locality_list:
    no_city_center = realty[(realty['locality_name'] == locality) & (realty['city_centers_nearest'].isna() == True)]['city_centers_nearest'].count()
    has_city_center = realty[(realty['locality_name'] == locality) & (realty['city_centers_nearest'].isna() == False)]['city_centers_nearest'].count()
    if no_city_center != has_city_center or has_city_center != 0:
        print(f"For {locality} {no_city_center} records without distance to city center, {has_city_center} - with distance to city center")

```

    For санкт-петербург 0 records without distance to city center, 15660 - with distance to city center
    For ломоносов 0 records without distance to city center, 132 - with distance to city center
    For петергоф 0 records without distance to city center, 201 - with distance to city center
    For пушкин 0 records without distance to city center, 368 - with distance to city center
    For колпино 0 records without distance to city center, 337 - with distance to city center
    For кронштадт 0 records without distance to city center, 95 - with distance to city center
    For павловск 0 records without distance to city center, 38 - with distance to city center
    For сестрорецк 0 records without distance to city center, 183 - with distance to city center
    For зеленогорск 0 records without distance to city center, 24 - with distance to city center
    For unknown 0 records without distance to city center, 41 - with distance to city center


The same conclusion here. Unfortunately, we do not have grounds to determine the distance to the city center based on data from the dataset and fill in missing values. The median or mean values are applicable. The missing values remain.


```python
del locality, locality_list, no_city_center, has_city_center
```

### `'parks_around_3000'` — how many parks are within 3 kilometers distance


```python
quant_dist(realty, 'parks_around_3000')
```

    Feature: parks_around_3000
    count    18181.000000
    mean         0.611408
    std          0.802074
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          1.000000
    max          3.000000
    Name: parks_around_3000, dtype: float64



    
![png](output_193_1.png)
    


    


Not every property has a park or even several within walking distance. An immediate assumption can be made: the absence of a value in this column can be interpreted as 0 parks within a radius of 3 km. So, we fill in the missing values with zero values:


```python
realty.loc[:, 'parks_around_3000'] = realty.loc[:, 'parks_around_3000'].fillna(0).astype('int8')
```

### `'parks_nearest'` — distance to a nearest park (meters)


```python
quant_dist(realty, 'parks_nearest')
```

    Feature: parks_nearest
    count    8079.000000
    mean      490.804555
    std       342.317995
    min         1.000000
    25%       288.000000
    50%       455.000000
    75%       612.000000
    max      3190.000000
    Name: parks_nearest, dtype: float64



    
![png](output_197_1.png)
    


    


It's great to have a park nearby! It's all the greater to know that most of the ads mentioning a park have it so close. We'll see if that affects the price later.

### `'ponds_around_3000'` — how many ponds or lakes are within 3 kilometers distance


```python
quant_dist(realty, 'ponds_around_3000')
```

    Feature: ponds_around_3000
    count    18181.000000
    mean         0.770255
    std          0.938346
    min          0.000000
    25%          0.000000
    50%          1.000000
    75%          1.000000
    max          3.000000
    Name: ponds_around_3000, dtype: float64



    
![png](output_200_1.png)
    


    


Nothing out of the ordinary either. Unfortunately, there are not many ponds, and not every ad can boast a pond nearby. Just like with parks, let's assume that a missing value means no pond nearby (users chose not to fill in this field):


```python
realty.loc[:, 'ponds_around_3000'] = realty.loc[:, 'ponds_around_3000'].fillna(0).astype('int8')
```


### `'ponds_nearest'` — distance to a nearest pond or lake (meters)


```python
quant_dist(realty, 'ponds_nearest')
```

    Feature: ponds_nearest
    count    9110.000000
    mean      517.980900
    std       277.720643
    min        13.000000
    25%       294.000000
    50%       502.000000
    75%       729.000000
    max      1344.000000
    Name: ponds_nearest, dtype: float64



    
![png](output_204_1.png)
    


    


Again, excellent, very even distribution. If one happens to have a pond within a radius of 3 kilometers - it will not be further than one and a half kilometers, so about a 20-minute walk.

### `'days_exposition'` — how long an ad stayed from publication to taking out (days)


```python
quant_dist(realty, 'days_exposition')
```

    Feature: days_exposition
    count    20518.000000
    mean       180.888634
    std        219.727988
    min          1.000000
    25%         45.000000
    50%         95.000000
    75%        232.000000
    max       1580.000000
    Name: days_exposition, dtype: float64



    
![png](output_207_1.png)
    


    


Bad news for users hoping that Yandex.Realty will help sell the property in a couple of weeks. The good news for more realistic users is that half of all ads are taken out (hopefully due to a secured deal) in 3 months and three-quarters of all properties in 7-8 months.

Judging by these data, "a quick selling of an apartment" goes up to a month and a half (the second quartile's a lower limit). In other words, if you sell an apartment in St. Petersburg and its suburbs - stock up on time from a quarter to six months, you have 2 out of three chances to sell it during this period.

And let's not forget about properties that take more than a year to be sold, especially with record holders of two years and above, including the primary record holder - a property that has been on sale for almost five years.

In addition, we have several missing values that should be filled with median values:


```python
realty.loc[:, 'days_exposition'] = realty.loc[:, 'days_exposition'].fillna(
    realty['days_exposition'].describe()['50%']).astype('int16')
```


Let's take a closer look at ads that stayed active for up to a year:


```python
realty.query('days_exposition < 365')['days_exposition'].hist(bins=365)
```




    <AxesSubplot:>




    
![png](output_211_1.png)
    


High peaks bring attention; they fall out of the general uniform distribution. Let's take a look at all the values of the number of days an ad was displayed, to which more than 500 ads correspond:


```python
realty.query('days_exposition < 365')['days_exposition'].value_counts().head(10)
```




    95    3245
    45     880
    60     538
    7      234
    30     208
    90     204
    4      176
    3      158
    5      152
    14     148
    Name: days_exposition, dtype: int64



The prominent peaks in the number of ads correspond to the "round" values of 45, 60, and 95 days. The following values are also "beautiful" - 7, 30, and 90 days. This tendency to round numbers is not a characteristic of any "natural" distribution. It can be assumed that these are traces of automatic or "manual" rounding of values (or assignment of missing values) at some stage of dataset formation or automatic removal of ads from renewal.

### Data preprocessing conclusions

We filled in missing values where possible and changed the data type in several columns. Somewhere we did that because it made it possible to work with data differently (for example, for the "date-time" type), and somewhere - as an experiment, to check whether this allows us to optimize at least memory usage. Time to check if our experiment is successful:


```python
realty.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23699 entries, 0 to 23698
    Data columns (total 22 columns):
     #   Column                Non-Null Count  Dtype         
    ---  ------                --------------  -----         
     0   total_images          23699 non-null  int8          
     1   last_price            23699 non-null  int32         
     2   total_area            23699 non-null  float64       
     3   first_day_exposition  23699 non-null  datetime64[ns]
     4   rooms                 23699 non-null  int8          
     5   ceiling_height        23699 non-null  float32       
     6   floors_total          23699 non-null  float64       
     7   living_area           23699 non-null  float64       
     8   floor                 23699 non-null  int64         
     9   is_apartment          23699 non-null  bool          
     10  studio                23699 non-null  bool          
     11  open_plan             23699 non-null  bool          
     12  kitchen_area          23699 non-null  float64       
     13  balcony               23699 non-null  int32         
     14  locality_name         23699 non-null  object        
     15  airports_nearest      18157 non-null  float64       
     16  city_centers_nearest  18180 non-null  float64       
     17  parks_around_3000     23699 non-null  int8          
     18  parks_nearest         8079 non-null   float64       
     19  ponds_around_3000     23699 non-null  int8          
     20  ponds_nearest         9110 non-null   float64       
     21  days_exposition       23699 non-null  int16         
    dtypes: bool(3), datetime64[ns](1), float32(1), float64(8), int16(1), int32(2), int64(1), int8(4), object(1)
    memory usage: 2.5+ MB


We saved approximately 1.1 MB of memory, which may seem like a minor success in absolute terms, but is almost a 30% optimization in relative terms. In highly loaded systems with large amounts of data, that optimization might mean gigabytes and terabytes of memory space and hours of machine time.

## Calculating synthetic features

We'll add to the dataset columns containing the following:

* square meter cost;
* day of the week, month, and year of an ad publication;
* apartment floor category: first, last, other;
* the ratio of living to total square footage, kitchen to total square footage.

### Square meter cost


```python
realty['square_meters'] = round(realty['last_price'] / realty['total_area']).astype('int32')
quant_dist(realty, 'square_meters')
```

    Feature: square_meters
    count    2.369900e+04
    mean     9.942637e+04
    std      5.030273e+04
    min      7.963000e+03
    25%      7.659950e+04
    50%      9.500000e+04
    75%      1.142560e+05
    max      1.907500e+06
    Name: square_meters, dtype: float64



    
![png](output_221_1.png)
    


    


Because of the outliers, the picture is very incomprehensible. We'll filter them by setting the upper-cost limit at ₽ 250 thousand (€ 3960) per m²:


```python
details(realty['square_meters'], 250000)
```


    
![png](output_223_0.png)
    





    count    2.369900e+04
    mean     9.942637e+04
    std      5.030273e+04
    min      7.963000e+03
    25%      7.659950e+04
    50%      9.500000e+04
    75%      1.142560e+05
    max      1.907500e+06
    Name: square_meters, dtype: float64



Well, that's quite an understandable cost distribution, not raising any questions. But outliers like more than a million rubles (€ 15850) per m² are of interest:


```python
realty[realty['square_meters'] > 1000000]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_images</th>
      <th>last_price</th>
      <th>total_area</th>
      <th>first_day_exposition</th>
      <th>rooms</th>
      <th>ceiling_height</th>
      <th>floors_total</th>
      <th>living_area</th>
      <th>floor</th>
      <th>is_apartment</th>
      <th>...</th>
      <th>balcony</th>
      <th>locality_name</th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
      <th>parks_around_3000</th>
      <th>parks_nearest</th>
      <th>ponds_around_3000</th>
      <th>ponds_nearest</th>
      <th>days_exposition</th>
      <th>square_meters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1436</th>
      <td>19</td>
      <td>330000000</td>
      <td>190.0</td>
      <td>2018-04-04</td>
      <td>3</td>
      <td>3.50</td>
      <td>7.0</td>
      <td>95.0</td>
      <td>5</td>
      <td>False</td>
      <td>...</td>
      <td>0</td>
      <td>санкт-петербург</td>
      <td>23011.0</td>
      <td>1197.0</td>
      <td>3</td>
      <td>519.0</td>
      <td>3</td>
      <td>285.0</td>
      <td>233</td>
      <td>1736842</td>
    </tr>
    <tr>
      <th>12971</th>
      <td>19</td>
      <td>763000000</td>
      <td>400.0</td>
      <td>2017-09-30</td>
      <td>7</td>
      <td>2.65</td>
      <td>10.0</td>
      <td>250.0</td>
      <td>10</td>
      <td>False</td>
      <td>...</td>
      <td>2</td>
      <td>санкт-петербург</td>
      <td>25108.0</td>
      <td>3956.0</td>
      <td>1</td>
      <td>530.0</td>
      <td>3</td>
      <td>756.0</td>
      <td>33</td>
      <td>1907500</td>
    </tr>
    <tr>
      <th>14706</th>
      <td>15</td>
      <td>401300000</td>
      <td>401.0</td>
      <td>2016-02-20</td>
      <td>5</td>
      <td>2.65</td>
      <td>9.0</td>
      <td>204.0</td>
      <td>9</td>
      <td>False</td>
      <td>...</td>
      <td>3</td>
      <td>санкт-петербург</td>
      <td>21912.0</td>
      <td>2389.0</td>
      <td>1</td>
      <td>545.0</td>
      <td>1</td>
      <td>478.0</td>
      <td>393</td>
      <td>1000748</td>
    </tr>
    <tr>
      <th>22831</th>
      <td>18</td>
      <td>289238400</td>
      <td>187.5</td>
      <td>2019-03-19</td>
      <td>2</td>
      <td>3.37</td>
      <td>6.0</td>
      <td>63.7</td>
      <td>6</td>
      <td>False</td>
      <td>...</td>
      <td>0</td>
      <td>санкт-петербург</td>
      <td>22494.0</td>
      <td>1073.0</td>
      <td>3</td>
      <td>386.0</td>
      <td>3</td>
      <td>188.0</td>
      <td>95</td>
      <td>1542605</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 23 columns</p>
</div>



However, there are no big surprises here — just an overpriced property — or one or two extra zeroes added to an ad by mistake.

### Day of the week, month, and year of ad publication


```python
realty.loc[:, 'day_exposition'] = pd.DatetimeIndex(realty.loc[:, 'first_day_exposition']).weekday
realty.loc[:, 'month_exposition'] = pd.DatetimeIndex(realty.loc[:, 'first_day_exposition']).month
realty.loc[:, 'year_exposition'] = pd.DatetimeIndex(realty.loc[:, 'first_day_exposition']).year
```


```python
def exposition_day(column):
    '''
    The function processes a date and provides its details.
    It takes the only argument:
    1. column: a dataframe column containing dates in the following
       format: realty.loc[row, 'column']
    
    The function returns day of week as string.
    '''
    days = {0: '0 - Monday', 1: '1 - Tuesday', 2: '2 - Wednesday',
            3: '3 - Thursday', 4: '4 - Friday', 5: '5 - Saturday', 6: '6 - Sunday'}
    return days[column]
```


```python
def exposition_month(column):
    '''
    The function processes a date and provides its details.
    It takes the only argument:
    1. column: a dataframe column containing dates in the following
       format: realty.loc[row, 'column']
    
    The function returns month as string.
    '''
    months = {1: '00 - January', 2: '01 - February', 3: '02 - March', 4: '03 - April',
              5: '04 - May', 6: '05 - June', 7: '06 - July', 8: '07 - August', 
              9: '08 - September', 10: '09 - October', 11: '10 - November', 12: '11 - December'}
    return months[column]
```


```python
realty.loc[:, 'day_exposition'] = realty.loc[:, 'day_exposition'].apply(exposition_day)
realty.loc[:, 'month_exposition'] = realty.loc[:, 'month_exposition'].apply(exposition_month)
```

Here's the result:


```python
print(realty['day_exposition'].value_counts())
print()
print(realty['month_exposition'].value_counts())
```

    3 - Thursday     4295
    1 - Tuesday      4183
    4 - Friday       4000
    2 - Wednesday    3974
    0 - Monday       3612
    5 - Saturday     1936
    6 - Sunday       1699
    Name: day_exposition, dtype: int64
    
    01 - February     2640
    02 - March        2587
    03 - April        2379
    10 - November     2371
    09 - October      2127
    08 - September    1981
    05 - June         1760
    07 - August       1744
    06 - July         1695
    11 - December     1641
    00 - January      1500
    04 - May          1274
    Name: month_exposition, dtype: int64


We need numbers in the names of days and months to arrange them in the charts in the same order they follow in a week and a year.

### Apartment floor category

Let's categorise the values:


```python
def floor_category(row):
    '''
    The function catogorizes observations based on a 'floor' column value.
    It takes the only argument:
    1. row: a dataframe row containing dates in the following
       format: realty.loc[row, 'column']

    The function returns one categorical value from the list:
    - 'Top floor'
    - 'Ground floor'
    - 'Other floor'
    '''
    if row['floor'] == row['floors_total']:
        return 'Top floor'
    elif row['floor'] == 1:
        return 'Ground floor'
    return 'Other floor'
```


```python
realty.loc[:, 'floor_category'] = realty.loc[:].apply(floor_category, axis=1)
realty['floor_category'].value_counts()
```




    Other floor     17404
    Top floor        3403
    Ground floor     2892
    Name: floor_category, dtype: int64



### Square footage ratios

Let's go back to the `'area'` auxiliary dataframe. We'll add one more column to it and then transfer both columns to the main dataframe:


```python
area['living_to_total_ratio'] = round((area['living_area'] / area['total_area']), 2)
realty[['living_to_total_ratio', 'kitchen_to_total_ratio']] = area[['living_to_total_ratio', 'kitchen_to_total_ratio']]
```

Finally, we'll remove the auxiliary dataframe we created earlier. It served us well, and now its watch is over ⚔


```python
del area
```

## Exploratory data analysis

### Correlations between variables

We'll check if there are linear and non-linear correlations between categorical and quantitative variables (automatically transformed into interval form). To do this, first, we'll check the data types of the dataset in its current state:


```python
realty.dtypes
```




    total_images                        int8
    last_price                         int32
    total_area                       float64
    first_day_exposition      datetime64[ns]
    rooms                               int8
    ceiling_height                   float32
    floors_total                     float64
    living_area                      float64
    floor                              int64
    is_apartment                        bool
    studio                              bool
    open_plan                           bool
    kitchen_area                     float64
    balcony                            int32
    locality_name                     object
    airports_nearest                 float64
    city_centers_nearest             float64
    parks_around_3000                   int8
    parks_nearest                    float64
    ponds_around_3000                   int8
    ponds_nearest                    float64
    days_exposition                    int16
    square_meters                      int32
    day_exposition                    object
    month_exposition                  object
    year_exposition                    int64
    floor_category                    object
    living_to_total_ratio            float64
    kitchen_to_total_ratio           float64
    dtype: object



We'll use the extended pandas data type for categorical variables:


```python
for col in ['locality_name', 'day_exposition', 'month_exposition', 'floor_category']:
    realty[col] = pd.Categorical(realty[col])
```

Here's a correlation matrix as a heatmap:


```python
collinearity = realty.phik_matrix(interval_cols=['total_images', 'last_price', 'total_area',
                                                 'rooms', 'ceiling_height', 'floors_total',
                                                 'living_area', 'floor', 'kitchen_area', 'balcony',
                                                 'airports_nearest', 'city_centers_nearest', 'parks_around_3000',
                                                 'parks_nearest', 'ponds_around_3000', 'ponds_nearest',
                                                 'days_exposition', 'square_meters', 'year_exposition',
                                                 'living_to_total_ratio', 'kitchen_to_total_ratio'])
plot_correlation_matrix(collinearity.values, 
                        x_labels=collinearity.columns, 
                        y_labels=collinearity.index, 
                        vmin=0, vmax=1, color_map="Greens", 
                        title=r"Correlation matrix $\phi_K$",
                        figsize=(16, 16))
plt.tight_layout()
```


    
![png](output_248_1.png)
    


Due to a large number of features in the matrix, we will focus only on features that demonstrate correlation coefficients of more than 0.55 (the average value for a "moderate" correlation on the Chaddock scale).


```python
collinearity = realty[['kitchen_to_total_ratio', 'living_to_total_ratio', 'year_exposition',
                       'month_exposition', 'day_exposition', 'square_meters', 'days_exposition',
                       'parks_nearest', 'parks_around_3000', 'ponds_around_3000', 'ponds_nearest',
                       'city_centers_nearest', 'airports_nearest', 'kitchen_area', 'floor', 'living_area',
                       'floors_total', 'rooms', 'first_day_exposition', 'total_area', 'last_price']].phik_matrix(
                           interval_cols=['last_price', 'total_area', 'rooms', 'floors_total',
                                          'living_area', 'floor', 'kitchen_area', 'airports_nearest',
                                          'city_centers_nearest', 'parks_around_3000', 'parks_nearest',
                                          'ponds_around_3000', 'ponds_nearest', 'days_exposition',
                                          'square_meters', 'living_to_total_ratio', 'kitchen_to_total_ratio'])
```


```python
plot_correlation_matrix(collinearity.values, 
                        x_labels=collinearity.columns, 
                        y_labels=collinearity.index, 
                        vmin=0, vmax=1, color_map="Greens", 
                        title=r"Correlation matrix $\phi_K$",
                        figsize=(16, 16))
plt.tight_layout()
```


    
![png](output_251_0.png)
    


It is expected (and not surprising at all) that there is a relatively strong correlation between individual parameters that somehow describe the square footage of the property: the total square footage, the number of rooms, the ​​living area square footage, the kitchen square footage, the ratio of the kitchen and living area to the total square footage. Due to multicollinearity concerns, most of these features wouldn't be used if we were developing a predictive real estate model.

In addition to the square footage ratios, the dataset contains several more features synthesized from each other, which explains the strong correlation between them:
- Date of placement of the announcement and duration of the demonstration of the announcement;
- Date of an ad publication and year, month, and day of the week when the ad was posted;
- The duration of how the ad has been posted and the day of the week when the ad was posted;
- Ad publication year and month.

A correlation was found between signs that are not direct derivatives of each other:
- The number of parks nearby and the distance to the nearest park. That's no surprise; these variables describe one fact "there is a park next to the property" in slightly different ways.
- The number of parks nearby and the number of ponds nearby. The probability of finding a pond in a park is much higher than finding it elsewhere.
- The distance to the city center and the distance to the airport. This correlation can be reversed (the phi coefficient does not provide information about the direction of the connection). Let's check it using the linear correlation coefficient:


```python
realty[['airports_nearest', 'city_centers_nearest']].corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airports_nearest</th>
      <th>city_centers_nearest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>airports_nearest</th>
      <td>1.000000</td>
      <td>0.272184</td>
    </tr>
    <tr>
      <th>city_centers_nearest</th>
      <td>0.272184</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



That is a weak correlation. These variables are connected non-linearly: the closer the object is to the center of St. Petersburg, the farther it is from the airport. If we talk about small settlements, the distances from objects to the center, for example, of Pushkin, can demonstrate a strong inverse relation to the distance to Pulkovo airport. The phita k coefficient can reveal such complex ties.

Finally, the main conclusion:
There is a moderately strong correlation between the apartment's square footage and price. Of course, other factors also affect the price (at least the location, type of property, transport accessibility, etc.). However, the square footage remains the basis for evaluating an apartment price.

That isn't exactly true for proximity to a city center. Historical residential buildings in city centers (especially in St. Petersburg) often do not fully meet modern standards of comfort, infrastructure, and communications quality. 


```python
realty.pivot_table(index='floor_category', values='last_price', aggfunc=('mean', 'median')).plot()
```




    <AxesSubplot:xlabel='floor_category'>




    
![png](output_255_1.png)
    


The first floor is not the leader in terms of real estate prices. And even though the most expensive property is located on the top floor, the leading share of "more expensive" properties (judging by the ratio of the median and average) is in the "other floor" category.

Finally, let's look at the dynamics of real estate prices depending on the date, month, and day of the week of the ad posting.


```python
decomp_median = seasonal_decompose(
    realty.pivot_table(index='first_day_exposition',
                   values='last_price', aggfunc='median'),
    period=365)

decomp_count = seasonal_decompose(
    realty.pivot_table(index='first_day_exposition',
                   values='last_price', aggfunc='count'),
    period=365)

plt.figure(figsize=(18, 5))
plt.plot(decomp_median.trend)
plt.title('Median price: trend')
plt.show()

plt.figure(figsize=(18, 5))
plt.plot(decomp_count.trend)
plt.title('Amount of apartments being sold: trend')
plt.show()
```


    
![png](output_257_0.png)
    



    
![png](output_257_1.png)
    


The most "fruitful" years for selling the realty were 2017-2018. In 2017, the most expensive property was put up for sale. At the same time, the median price remained not too high, which gives a reason to think that most ads advertised the "middle" (or even "economy") class property.

Trend charts for the entire period demonstrate how the law of supply and demand works in the real estate market: an increase in ads is accompanied by a decrease in the median price of real estate.


```python
plt.figure(figsize=(18, 5))
plt.plot(realty.pivot_table(index='month_exposition', values='last_price', aggfunc='median'))
plt.title('Median price by month')
plt.show()

plt.figure(figsize=(18, 5))
plt.plot(realty.pivot_table(index='month_exposition', values='last_price', aggfunc='count'))
plt.title('Amount of apartments being sold by month')
plt.show()
```


    
![png](output_259_0.png)
    



    
![png](output_259_1.png)
    


The median price stays about the same from month to month but is higher in April than in September, November, and December. Between these months, it drops. It is undoubtedly better to buy realty in St. Petersburg in summer.

The most "popular" months to decide to sell real estate (an indirect sign of which is posting an ad) are the end of winter - the beginning of spring (February, March, April), and November. In May, buying realty is more complicated: there will be few new offers on the market.


```python
plt.figure(figsize=(18, 5))
plt.plot(realty.pivot_table(index='day_exposition', values='last_price', aggfunc='median'))
plt.title('Median price by days of week')
plt.show()

plt.figure(figsize=(18, 5))
plt.plot(realty.pivot_table(index='day_exposition', values='last_price', aggfunc='count'))
plt.title('Amount of apartments being sold by days of week')
plt.show()
```


    
![png](output_261_0.png)
    



    
![png](output_261_1.png)
    


As for days of the week, ads are posted more often on weekdays and much less frequently on weekends. But the median cost is higher for Tuesdays and Wednesdays. The first observation is expected, and the second is just a funny and hardly explainable incident.


```python
del col, collinearity, decomp_count, decomp_median
```

### Settelements with most ads

Let's sort the settlements by the number of ads and print ten leaders:


```python
locality_majority = realty.groupby('locality_name')['last_price'].count().sort_values(ascending=False).head(10)
locality_majority.index
```




    CategoricalIndex(['санкт-петербург', 'муриный', 'кудрово', 'шушары',
                      'всеволожск', 'пушкин', 'колпино', 'парголовый', 'гатчина',
                      'выборг'],
                     categories=['unknown', 'агалатово', 'александровский', 'алексеевка', ..., 'ялгино', 'яльгелевый', 'яма-тесовый', 'янино-1'], ordered=False, dtype='category', name='locality_name')



It is obvious (and expected): the vast majority of ads are about real estate in St. Petersburg.

Here is an average square meter cost in settlements listed above:


```python
for locality in locality_majority.index:
    print(locality, ':',
        round(
            realty.query('locality_name == @locality')['square_meters'].mean()
        )
    )
```

    санкт-петербург : 114856
    муриный : 86088
    кудрово : 95325
    шушары : 78677
    всеволожск : 68654
    пушкин : 103126
    колпино : 75425
    парголовый : 90176
    гатчина : 68746
    выборг : 58142


The highest average square meter cost is in St. Petersburg. The smallest one is in Vyborg. Although we can expect a wide range of prices from St. Petersburg depending on the area and distance to the center (which will be discussed in the next section), the square meter cost in Pushkin suggests the name of the settlement is not the issue. Let's check the dynamics of the median square meter cost:


```python
for locality in locality_majority.index:
    print(locality, ':',
        round(
            realty.query('locality_name == @locality')['square_meters'].median()
        )
    )
```

    санкт-петербург : 104770
    муриный : 86176
    кудрово : 95676
    шушары : 76876
    всеволожск : 65789
    пушкин : 100000
    колпино : 74724
    парголовый : 91643
    гатчина : 67797
    выборг : 58158


The median square meter cost follows the same dynamics, which means a low contribution of high values (more than can be expected from St. Petersburg) for these indicators.


```python
del locality_majority, locality
```

### What is a "center," and how does the real estate price depend on its proximity?

We'll consider the distance from a property in St. Petersburg to the city center:


```python
spb_realty = realty.query('locality_name == "санкт-петербург"').copy()
del realty
```


```python
details(spb_realty['city_centers_nearest'], 40000) # somewhat random limit
```


    
![png](output_276_0.png)
    





    count    15660.000000
    mean     11601.291571
    std       4842.035279
    min        181.000000
    25%       8327.000000
    50%      12244.500000
    75%      14943.000000
    max      29493.000000
    Name: city_centers_nearest, dtype: float64



Based on several checks, the value of ~29.5 km to the city center was the maximum among the ads from St. Petersburg. The value count here is slightly less than the total number of ads from this city due to several missing values.

Let's round the values to kilometers, perform grouping, calculate the average price for each kilometer and build a line chart of the average price depending on the distance to the center.


```python
spb_realty.loc[:, 'city_centers_nearest'] = round(spb_realty.loc[:, 'city_centers_nearest'] / 1000)
```


```python
spb_realty.groupby(
    'city_centers_nearest')['last_price'].mean().plot(
        style='o-', grid=True, figsize=(12, 5))
```




    <AxesSubplot:xlabel='city_centers_nearest'>




    
![png](output_279_1.png)
    


The average price drops steadily over a distance of up to 3 kilometers and then rises again. It can be assumed that approximately a three-kilometer radius is the historical center of St. Petersburg. The reason for the price reduction within this radius may lie in the high density of historical buildings, which are not necessarily the property of high quality and comfort. On the contrary, only outside the historical center can property developers offer high-standard housing to consumers, which may explain the gradual increase in the average price in the interval from 3 to 7 kilometers from the historical center. The proximity to the center becomes less relevant, while the factor of high consumer standards gains relevance.

Let's take a closer look at the "segment of apartments in the center." We'll look at 5 kilometers radius of the center of St. Petersburg.


```python
spb_realty_center = spb_realty.query('city_centers_nearest <= 5')
del spb_realty
```


```python
spb_realty_center['total_area'].plot(kind='hist', grid=True, bins=100, title='Area (square meters)')
plt.show()
spb_realty_center['total_area'].describe()
```


    
![png](output_282_0.png)
    





    count    2487.000000
    mean       95.637021
    std        59.040476
    min        12.000000
    25%        60.050000
    50%        82.000000
    75%       112.100000
    max       631.200000
    Name: total_area, dtype: float64



So, the leading share of apartment square footage is approximately 50 to 100 m². At the same time, the value dispersion around the average value (slightly larger than the median) is quite large. That indicates a wide variance in square footage.

We already know that the range of real estate prices is extensive, so we'll set the upper price limit of ₽20 million (€/$ 322 000):


```python
spb_realty_center['last_price'].plot(kind='hist', grid=True, bins=50,
                                     title='Realty prices', range=(0, 20000000))
plt.show()
spb_realty_center['last_price'].describe()
```


    
![png](output_284_0.png)
    





    count    2.487000e+03
    mean     1.480580e+07
    std      2.611717e+07
    min      1.600000e+06
    25%      6.950000e+06
    50%      9.500000e+06
    75%      1.425000e+07
    max      7.630000e+08
    Name: last_price, dtype: float64



The real estate price is within a relatively wide range with large fluctuations, where the leading share of offers is within the interval from ₽ 4 to 15 million (€/$ 64 000 - 241 500).


```python
spb_realty_center['rooms'].plot(kind='hist', grid=True, bins=30, title='Rooms')
plt.show()
spb_realty_center['rooms'].describe()
```


    
![png](output_286_0.png)
    





    count    2487.000000
    mean        2.970245
    std         1.499504
    min         0.000000
    25%         2.000000
    50%         3.000000
    75%         4.000000
    max        19.000000
    Name: rooms, dtype: float64



Most offers are two to four rooms, with two/three room options far more widespread than any other.

Ceiling height values bring a little surprise:


```python
spb_realty_center['ceiling_height'].plot(kind='hist', grid=True, bins=50, title='Ceiling height')
plt.show()
spb_realty_center['ceiling_height'].describe()
```


    
![png](output_288_0.png)
    





    count    2487.000000
    mean        2.977857
    std         0.377418
    min         2.400000
    25%         2.650000
    50%         2.900000
    75%         3.200000
    max         5.800000
    Name: ceiling_height, dtype: float64



The center of St. Petersburg, if one conforms to a stereotype, is full of old housing with high ceilings. And there are many such buildings. Moreover, measures of central tendency suggest that most ads promote this type of property. However, the histogram shows that the share of real estate with "average" (up to 2.7 m) ceiling height is at least very noticeable.

Now let's see if there is a **connection between these and some other factors and real estate prices**.

Let's start with the floor where the property is located and its price:


```python
plt.figure(figsize=(12,5))
plt.plot(spb_realty_center.pivot_table(index='floor', values='last_price', aggfunc='median'))
plt.title('Realty price by floor in Saint-Petersburg center')
plt.show()
```


    
![png](output_291_0.png)
    


The distribution shows that the price increases depending on the floor, but this increase does not look very significant within the 1-6 floor range. In rare tall buildings in the center of St. Petersburg, apartments on high (7-10) floors are priced higher than apartments on lower floors.

In general, we can say that there's a slight linear trend: the median price of real estate increases with the floor. However, this connection is not very obvious.


```python
plt.figure(figsize=(12,5))
plt.plot(spb_realty_center.pivot_table(index='city_centers_nearest', values='last_price', aggfunc=('median')))
plt.title('Realty price by distance from Saint-Petersburg center')
plt.show()
```


    
![png](output_293_0.png)
    


The median price of real estate drops significantly with distance from the center, which is quite understandable - the closer to the center, the higher the prestige of owning realty, even not of excellent quality. Further, the median price grows as the share of the newer, more attractive realty increases. Therefore, it is difficult to talk about a linear correlation between the distance from the center and the realty price within the center of St. Petersburg, despite a visible decrease in the first three kilometers.

Now let's look at the price and the number of rooms together:


```python
plt.figure(figsize=(12,5))
plt.plot(spb_realty_center.pivot_table(index='rooms', values='last_price', aggfunc=('median')))
plt.title('Realty price by rooms in Saint-Petersburg center')
plt.show()
```


    
![png](output_295_0.png)
    


An apparent trend is seen. The number of rooms directly impacts realty price. The trend is noticeable on the whole dataset, even considering the 15-rooms outlier.

Finally, we observe the price changing depending on the year the ad was posted:


```python
plt.figure(figsize=(12,5))
plt.plot(spb_realty_center.pivot_table(index='year_exposition', values='last_price', aggfunc=('median')))
plt.title('Realty price by year in Saint-Petersburg center')
plt.show()
```


    
![png](output_297_0.png)
    


The price of real estate in the city center follows the general trend: the prices sharply declined until 2018 with a further slow increase. We have already observed this non-linear tendency for all ads in the dataset.


```python
del spb_realty_center
```

## Conclusions and discussion

We evaluated key parameters that determine the market value of realty in St. Petersburg and the Leningrad oblast (region). Data distributions by price, area, number of rooms, and ceiling heights have been studied.

A fairly reasonable hypothesis about a direct connection between the property price and its square footage (which, in turn, is related to the number of rooms) has been confirmed. For St. Petersburg, the hypothesis of a connection between real estate prices and the number of rooms in it is also confirmed by graphical analysis. At the same time, it has been shown that the distance to the center, the floor, and the year the ad was posted do not have an unambiguous correlation with the market value of the real estate, at least in the center of St. Petersburg. However, there is a moderate inverse correlation between the distance to the center and the price within the entire city.
When we consider all of Leningrad oblast's settlements, the picture changes slightly. It is possible to single out a specific year (2014) when the median property price was higher than in other years. Further, it decreased until 2018, then slightly increased in 2019. But the number of ads was maximum in 2017-2018, which provides a ground to assume why the prices dropped: it was the law of supply and demand. To make their property more attractive, sellers slightly decrease the price - again and again.

In addition, the distribution of the ads by different indicators provides grounds for some conclusions about the typical consumer behavior of real estate sellers:
- Ads are posted mainly on weekdays, which suggests that users sell their property probably during business hours and presumably from the workplace. However, this assumption needs to be verified (for example, using data on the time of ad posting).
- The most expensive (by median price) property is posted Tuesdays and Wednesdays.
- The most "expensive" month in real estate supply is April, and the cheapest is June.
- Generally, the period from February to April is the most active real estate sales.
