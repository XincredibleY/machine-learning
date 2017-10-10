

```python
# 导入组件包
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# retina screen
%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline
```


```python
# 读取杭州出租房数据（包括训练数据和测试数据）
all_data = pd.read_csv("./csv/beijing_all_data.csv")
# 检查数据列详情
all_data.columns
```




    Index([u'index_id', u'room_longitude', u'room_latitude', u'total_agent',
           u'floor_total', u'owner_gender', u'room_name', u'scope_name',
           u'estate_house_quantity', u'room_money', u'rf_kt', u'rf_src', u'rf_yg',
           u'rf_bx', u'rf_wsj', u'rf_cf', u'rf_rsq', u'rf_kd', u'rf_yt', u'rf_drc',
           u'rf_xzt', u'rf_pc', u'rf_xyj', u'rf_sf', u'rf_szt', u'rf_dsj',
           u'rf_unknow', u'zhuanzu', u'keduanzu', u'estate_name', u'puf_no',
           u'puf_bx', u'puf_xyj', u'puf_kd', u'puf_rsq', u'puf_cf', u'puf_wsj',
           u'puf_kt', u'puf_sf', u'puf_nq', u'puf_wbl', u'puf_kx', u'puf_dsj',
           u'agent_money', u'commission_price', u'area', u'fdxh_zxgl', u'fdxh_agj',
           u'fdxh_bxy', u'fdxh_bhj', u'fdxh_ds', u'fdxh_yh', u'fdxh_drz',
           u'fdxh_srz', u'fdxh_nvsheng', u'fdxh_nansheng', u'region_name',
           u'rent_type', u'only_girl', u'brand_type', u'subway_nums',
           u'subway_least_distance', u'decoration', u'floor', u'room_direction',
           u'hall_num', u'room_area', u'wei_num', u'business_type', u'is_zhongjie',
           u'room_num', u'pay_method', u'room_type'],
          dtype='object')




```python
#  去除ID
all_data.drop("index_id", axis = 1, inplace = True)
# 去除中介费、提成，因为这两项一般由出租房价格决定，不是自变量
all_data.drop("agent_money", axis = 1, inplace = True)
all_data.drop("commission_price", axis = 1, inplace = True)
# 去除含义不清的rf_unknow，puf_no
all_data.drop("rf_unknow", axis = 1, inplace = True)
all_data.drop("puf_no", axis = 1, inplace = True)

# 检查一下数据
all_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>room_longitude</th>
      <th>room_latitude</th>
      <th>total_agent</th>
      <th>floor_total</th>
      <th>owner_gender</th>
      <th>room_name</th>
      <th>scope_name</th>
      <th>estate_house_quantity</th>
      <th>room_money</th>
      <th>rf_kt</th>
      <th>...</th>
      <th>floor</th>
      <th>room_direction</th>
      <th>hall_num</th>
      <th>room_area</th>
      <th>wei_num</th>
      <th>business_type</th>
      <th>is_zhongjie</th>
      <th>room_num</th>
      <th>pay_method</th>
      <th>room_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>116.430926</td>
      <td>39.901161</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>252</td>
      <td>23</td>
      <td>3800.0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>15.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>116.419561</td>
      <td>40.063693</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>253</td>
      <td>8</td>
      <td>5000.0</td>
      <td>0</td>
      <td>...</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>75.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>116.559847</td>
      <td>39.862177</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>254</td>
      <td>13</td>
      <td>4200.0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>90.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>116.254537</td>
      <td>39.898568</td>
      <td>1</td>
      <td>20</td>
      <td>2</td>
      <td>0</td>
      <td>255</td>
      <td>3</td>
      <td>4000.0</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>116.240144</td>
      <td>39.926930</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>256</td>
      <td>20</td>
      <td>4000.0</td>
      <td>0</td>
      <td>...</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>52.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>




```python
#scatterplot
# 定量特征
analyse_columns = ['total_agent','floor_total','estate_house_quantity',
                    'area',
                       'subway_nums','subway_least_distance','floor','hall_num','room_area',
                       'wei_num','room_num']
analyse_columns += ['room_money']
sns.set()
sns.pairplot(all_data[analyse_columns], size = 2.5)
plt.show();
```


![png](output_3_0.png)



```python
# 观察租金的影响因素中，存在异常值的有：floor_total，area，subway_least_distance，floor，hall_num，room_area，room_num
```


```python
# 将上图中能看清的 outliers 去除

all_data = all_data[all_data.floor_total<50]
all_data = all_data[all_data.area<1000]
all_data = all_data[all_data.subway_least_distance<4000]
all_data = all_data[all_data.floor<2000]
all_data = all_data[all_data.hall_num<5]
all_data = all_data[all_data.room_area<500]

```


```python
# 还有floor看不清，小于2000层的楼还有许多
# check the outliers
plt.figure(figsize=(10,10))
plt.scatter(all_data.floor,all_data.room_money,c='blue',marker='s')
plt.xlabel('room floor')
plt.ylabel('room money')
plt.title('looking for outliers')

```




    <matplotlib.text.Text at 0x1130e3890>




![png](output_6_1.png)



```python
# remove

all_data = all_data[all_data.floor<200]



```


```python
# check the outliers again
plt.figure(figsize=(10,10))
plt.scatter(all_data.floor,all_data.room_money,c='blue',marker='s')
plt.xlabel('room floor')
plt.ylabel('room money')
plt.title('looking for outliers again')

```




    <matplotlib.text.Text at 0x11bdbfb10>




![png](output_8_1.png)



```python
# 还有room_num看不清
# check the outliers
plt.figure(figsize=(10,10))
plt.scatter(all_data.room_num,all_data.room_money,c='blue',marker='s')
plt.xlabel('room num')
plt.ylabel('room money')
plt.title('looking for outliers')

```




    <matplotlib.text.Text at 0x11d604490>




![png](output_9_1.png)



```python
# remove

all_data = all_data[all_data.room_num<15]



```


```python
# check the outliers again
plt.figure(figsize=(10,10))
plt.scatter(all_data.room_num,all_data.room_money,c='blue',marker='s')
plt.xlabel('room num')
plt.ylabel('room money')
plt.title('looking for outliers again')

```




    <matplotlib.text.Text at 0x11d624710>




![png](output_11_1.png)



```python
# 租金数据一览
all_data['room_money'].describe()
```




    count     34048.000000
    mean       4596.661860
    std        4666.400578
    min          10.000000
    25%        1800.000000
    50%        3300.000000
    75%        5700.000000
    max      100000.000000
    Name: room_money, dtype: float64




```python
# 竟然有100000一个月的房子！
```


```python
# check the outliers
plt.figure(figsize=(10,10))
plt.scatter(all_data.room_area,all_data.room_money,c='blue',marker='s')
plt.xlabel('room area')
plt.ylabel('room money')
plt.title('looking for outliers')

```




    <matplotlib.text.Text at 0x112ffd5d0>




![png](output_14_1.png)



```python
# remove the outliers

all_data = all_data[all_data.room_money<80000]
```


```python
# check the outliers again
plt.figure(figsize=(10,10))
plt.scatter(all_data.room_area,all_data.room_money,c='blue',marker='s')
plt.xlabel('room area')
plt.ylabel('room money')
plt.title('looking for outliers again')

```




    <matplotlib.text.Text at 0x11bee92d0>




![png](output_16_1.png)



```python
# 我们检查一下租金是否属于正态分布
# histogram
plt.figure(figsize=(10,6))
sns.distplot(all_data['room_money'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11d1b10d0>




![png](output_17_1.png)



```python
# 明显不符合正态分布
# 偏度和峰度
print("Skewness: %f" % all_data['room_money'].skew())
print("Kurtosis: %f" % all_data['room_money'].kurt())
```

    Skewness: 3.699799
    Kurtosis: 23.397908



```python
# 城区类型对租金的影响
plt.subplots(figsize=(12, 6))
fig = sns.boxplot(x='region_name', y="room_money", data=all_data)
```


![png](output_19_0.png)



```python
# 房东性别对租金的影响
plt.subplots(figsize=(12, 6))
fig = sns.boxplot(x='owner_gender', y="room_money", data=all_data)
```


![png](output_20_0.png)



```python
# 北京的女房东收钱更高
```


```python
# 装修类型对租金的影响
plt.subplots(figsize=(12, 6))
fig = sns.boxplot(x='decoration', y="room_money", data=all_data)
```


![png](output_22_0.png)



```python
# 显然装修越好，价格越贵，但毛坯房（1）均价高于普通装修（2）
```


```python
#correlation matrix
corrmat = all_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,cmap='YlGnBu');
```


![png](output_24_0.png)



相关性矩阵图基本上能反应出各种特征之间的关系，信息量非常大。<br>
可以发现图中明显有几块深色区域，说明这些特征之间有非常强的相关性：<br>
1、房间设施之间，比如电视机（rf_dsj）和冰箱（rf_bx）之间；<br>
2、公用设施之间，比如宽带（puf_kd）和空调（puf_kt）之间；<br>
3、房东喜欢类型之间，比如爱干净（fdxh_agj）和不喝酒（fdxh_bhj）之间。<br>

实际上，这些变量之间可能存在多重共线性（multicollinearity）。<br>


如果我们要分析租金和其他特征的相关性，还需要作如下处理：


```python
#我们查看与租金最相关的前10个因素
k = 11 # number of variables for heatmap
cols = corrmat.nlargest(k, 'room_money')['room_money'].index
cm = np.corrcoef(all_data[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap='YlGnBu')
```


![png](output_26_0.png)


影响出租房价格前10位因素分别是：<br>
<li> 房间面积：room_area<br>
<li>是否有中介：is_zhongjie<br>
<li>整租还是合租：rent_type<br>
<li> 房间类型：room_type<br>
<li> 整套总面积：area<br>
<li> 厅数量：hall_num<br>
<li>周边地铁站点数量：subway_nums<br>
<li>房东性别：owner_gender<br>
<li> 小区：estate_name<br>
<li> 区域名称：scope_name<br>


```python
# 以上分析的影响出租房价格前10位因素不仅包含定量（面积）特征，也包含定性特征（厨房）。我们只看定量特征：

numerical_columns = ['total_agent','floor_total','estate_house_quantity',
                    'area',
                       'subway_nums','subway_least_distance','floor','hall_num','room_area',
                       'wei_num','room_num']
# 加上room_money临时分析
numerical_columns += ['room_money']
print("Find most important features relative to target")
corr = all_data[numerical_columns].corr()
corr.sort_values(["room_money"], ascending = False, inplace = True)
print(corr.room_money)
```

    Find most important features relative to target
    room_money               1.000000
    room_area                0.782779
    area                     0.417062
    hall_num                 0.345006
    subway_nums              0.281140
    floor_total              0.140389
    wei_num                  0.088890
    total_agent              0.026334
    subway_least_distance    0.022735
    room_num                -0.043141
    estate_house_quantity   -0.118687
    floor                   -0.119390
    Name: room_money, dtype: float64


## 构建新的特征


```python
# room_area                0.782789
# area                     0.417068
# hall_num                 0.345044
# subway_nums              0.281103
# floor_total              0.140415
# wei_num                  0.088955
# total_agent              0.026331
# subway_least_distance    0.022794

# create new features
# 多项式
# 平方、立方、开方
all_data['room_area-s2'] = all_data['room_area']**2
all_data['room_area-s3'] = all_data['room_area']**3
all_data['room_area-sq'] = np.sqrt(all_data[all_data['room_area'] >= 0]['room_area']) 

all_data['area-s2'] = all_data['area']**2
all_data['area-s3'] = all_data['area']**3
all_data['area-sq'] = np.sqrt(all_data[all_data['area'] >= 0]['area']) 

all_data['hall_num-s2'] = all_data['hall_num']**2
all_data['hall_num-s3'] = all_data['hall_num']**3
all_data['hall_num-sq'] = np.sqrt(all_data[all_data['hall_num'] >= 0]['hall_num']) 

all_data['subway_nums-s2'] = all_data['subway_nums']**2
all_data['subway_nums-s3'] = all_data['subway_nums']**3
all_data['subway_nums-sq'] = np.sqrt(all_data[all_data['subway_nums'] >= 0]['subway_nums']) 


all_data['floor_total-s2'] = all_data['floor_total']**2
all_data['floor_total-s3'] = all_data['floor_total']**3
all_data['floor_total-sq'] = np.sqrt(all_data[all_data['floor_total'] >= 0]['floor_total']) 

all_data['wei_num-s2'] = all_data['wei_num']**2
all_data['wei_num-s3'] = all_data['wei_num']**3
all_data['wei_num-sq'] = np.sqrt(all_data[all_data['wei_num'] >= 0]['wei_num']) 


all_data['total_agent-s2'] = all_data['total_agent']**2
all_data['total_agent-s3'] = all_data['total_agent']**3
all_data['total_agent-sq'] = np.sqrt(all_data[all_data['total_agent'] >= 0]['total_agent']) 

# 将-1变成0
all_data.loc[all_data[all_data['subway_least_distance']==-1].index.tolist(),'subway_least_distance'] = 0
all_data['subway_least_distance-s2'] = all_data['subway_least_distance']**2
all_data['subway_least_distance-s3'] = all_data['subway_least_distance']**3
all_data['subway_least_distance-sq'] = np.sqrt(all_data[all_data['subway_least_distance'] >= 0]['subway_least_distance']) 


# 是否合租，1表示是
all_data["rentTogether"] = all_data.room_type.replace({ 1 : 1,
                                                        2 : 1,
                                                        3 : 1,
                                                        4 : 0,
                                                        5 : 0})

```


```python
# 定量特征
numerical_columns = ['total_agent','floor_total','estate_house_quantity',
                    'area',
                       'subway_nums','subway_least_distance','floor','hall_num','room_area',
                       'wei_num','room_num']

numerical_columns += ['room_area-s2','room_area-s3','room_area-sq',
                      'area-s2','area-s3','area-sq',
                      'hall_num-s2','hall_num-s3','hall_num-sq',
                      'subway_nums-s2','subway_nums-s3','subway_nums-sq',
                     'floor_total-s2','floor_total-s3','floor_total-sq',
                     'wei_num-s2','wei_num-s3','wei_num-sq',
                      'total_agent-s2','total_agent-s3','total_agent-sq',
                      'subway_least_distance-s2','subway_least_distance-s3','subway_least_distance-sq']

# 定性特征
categorical_columns = ['owner_gender','room_name','scope_name','rf_kt','rf_src',
                       'rf_yg','rf_bx','rf_wsj','rf_cf','rf_rsq',
                       'rf_kd','rf_yt','rf_drc','rf_xzt','rf_pc',
                      'rf_xyj','rf_sf','rf_szt','rf_dsj',
                      'zhuanzu','keduanzu','estate_name',
                      'puf_bx','puf_xyj','puf_kd','puf_rsq','puf_cf',
                       'puf_wsj','puf_kt','puf_sf','puf_nq','puf_wbl',
                      'puf_kx','puf_dsj','fdxh_zxgl','fdxh_agj','fdxh_bxy',
                      'fdxh_bhj','fdxh_ds','fdxh_yh','fdxh_drz','fdxh_srz',
                      'fdxh_nvsheng','fdxh_nansheng','region_name','rent_type','only_girl',
                      'brand_type','decoration','room_direction','business_type','is_zhongjie',
                      'pay_method','room_type']


categorical_columns += ['rentTogether']



print("Numerical features : " + str(len(numerical_columns)))
print("Categorical features : " + str(len(categorical_columns)))


all_data_num = all_data[numerical_columns]
all_data_cat = all_data[categorical_columns]

```

    Numerical features : 35
    Categorical features : 55



```python
# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(all_data_num.isnull().values.sum()))
all_data_num = all_data_num.fillna(all_data_num.median())
print("Remaining NAs for numerical features in train : " + str(all_data_num.isnull().values.sum()))
```

    NAs for numerical features in train : 0
    Remaining NAs for numerical features in train : 0



```python
# Log transform of the skewed numerical features to lessen impact of outliers
# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
# 检查偏度>0.5的定量特征
skewness = all_data_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index

# 查看分布图
f = pd.melt(all_data_num, value_vars=all_data_num[skewed_features])
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```

    30 skewed numerical features to log transform



![png](output_33_1.png)



```python
# 楼层存在负数，需要处理
all_data_num['floor'].describe()
```




    count    34042.000000
    mean         3.234005
    std          5.323775
    min         -9.000000
    25%          0.000000
    50%          0.000000
    75%          5.000000
    max         61.000000
    Name: floor, dtype: float64




```python
# 将每个值加上 9 
all_data_num['floor']+=9
```


```python
# 再看一下
all_data_num['floor'].describe()
```




    count    34042.000000
    mean        12.234005
    std          5.323775
    min          0.000000
    25%          9.000000
    50%          9.000000
    75%         14.000000
    max         70.000000
    Name: floor, dtype: float64




```python
# 对数转换
all_data_num[skewed_features] = np.log1p(all_data_num[skewed_features])

# 再检查一下分布图
f = pd.melt(all_data_num, value_vars=all_data_num[skewed_features])
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```


![png](output_37_0.png)



```python
# Create dummy features for categorical values via one-hot encoding
print("NAs for categorical features in train : " + str(all_data_cat.isnull().values.sum()))
all_data_cat = pd.get_dummies(all_data_cat)
print("Remaining NAs for categorical features in train : " + str(all_data_cat.isnull().values.sum()))
```

    NAs for categorical features in train : 0
    Remaining NAs for categorical features in train : 0



```python
#查看价格直方图


plt.figure(figsize=(10,6))
plt.title('price')
sns.distplot(all_data.room_money )

plt.figure(figsize=(10,6))
plt.title('log transformed price')
sns.distplot(np.log1p(all_data.room_money))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116df3e50>




![png](output_39_1.png)



![png](output_39_2.png)



```python
# 对租金进行对数转换
all_data.room_money = np.log1p(all_data.room_money)
y = all_data.room_money 
```


```python
# Join categorical and numerical features
all_data = pd.concat([all_data_num, all_data_cat], axis = 1)
print("New number of features : " + str(all_data.shape[1]))

# Partition the dataset in train + validation sets
X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
```

    New number of features : 90
    X_train : (23829, 90)
    X_test : (10213, 90)
    y_train : (23829,)
    y_test : (10213,)



```python
# Standardize numerical features
stdSc = StandardScaler()
X_train.loc[:, numerical_columns] = stdSc.fit_transform(X_train.loc[:, numerical_columns])
X_test.loc[:, numerical_columns] = stdSc.transform(X_test.loc[:, numerical_columns])
```

    /Users/leo/Develop/python_env/env_python2.7/lib/python2.7/site-packages/pandas/core/indexing.py:517: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



```python
# Define error measure for official scoring : RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)
```


```python
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Look at predictions on training and validation set
print("RMSE on Training set :", rmse_cv_train(lr).mean())
print("RMSE on Test set :", rmse_cv_test(lr).mean())
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Plot residuals
plt.figure(figsize=(16,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.figure(figsize=(16,8))
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
```

    /Users/leo/Develop/python_env/env_python2.7/lib/python2.7/site-packages/scipy/linalg/basic.py:1018: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      warnings.warn(mesg, RuntimeWarning)


    ('RMSE on Training set :', 0.32592474296235729)
    ('RMSE on Test set :', 0.32056622201579327)



![png](output_44_2.png)



![png](output_44_3.png)



```python
# 2* Ridge
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)


print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)

ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)



print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)

# Plot residuals
plt.figure(figsize=(16,8))
plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.figure(figsize=(16,8))
plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(ridge.coef_, index = X_train.columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
plt.figure(figsize=(16,8))
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()
```

    ('Best alpha :', 0.01)
    Try again for more precision with alphas centered around 0.01
    ('Best alpha :', 0.0060000000000000001)
    ('Ridge RMSE on Training set :', 0.32335748227932121)
    ('Ridge RMSE on Test set :', 0.32088423928390142)



![png](output_45_1.png)



![png](output_45_2.png)


    Ridge picked 89 features and eliminated the other 1 features



![png](output_45_4.png)



```python
ridge.coef_
```




    array([  8.67436017e-03,  -1.74854908e+00,   3.89414455e-03,
            -3.85881529e+00,  -1.65550685e+00,  -5.67459771e-02,
             6.25481009e-03,  -8.51986168e-02,  -2.65317333e+01,
            -8.12733616e-01,  -1.55648225e-02,   6.38234862e+00,
             6.98387468e+00,   1.36557233e+01,   2.99002233e+00,
             6.51644118e-01,   2.05329526e-01,  -6.81137553e-01,
             6.14062765e-01,   1.93004594e-01,   5.71111573e+00,
            -3.71336431e+00,  -3.46785606e-01,   4.22842241e+00,
            -2.39409804e+00,  -1.10725252e-01,  -2.14482945e-01,
             6.91308270e-01,   3.46524993e-01,  -5.11472372e-01,
             5.66021424e-01,  -5.89008107e-02,   5.91786309e-01,
            -7.44314272e-01,   4.20425168e-01,  -9.20515370e-03,
             7.51039492e-03,   3.24342775e-04,   4.37133567e-02,
             2.77683807e-02,  -1.49028393e-02,  -4.25718094e-02,
             4.79659244e-02,  -3.19473847e-02,  -6.32768999e-02,
             1.31999421e-02,   1.96619493e-02,  -6.06151774e-02,
            -1.64189315e-02,   1.83556735e-02,   5.49769823e-02,
             3.77305440e-02,   1.59029393e-03,   8.50878869e-02,
             2.11168236e-02,   0.00000000e+00,   2.29124748e-06,
             3.28192241e-02,  -3.66063476e-03,  -3.57437231e-04,
            -2.93185841e-02,   1.00567845e-02,  -7.40044862e-02,
             5.05018158e-02,  -6.03621371e-02,   5.43087583e-02,
             2.83227124e-03,   7.06724747e-03,   4.20947840e-03,
             4.83692283e-02,  -9.24507652e-02,   1.12549380e-01,
            -1.57813592e-01,  -1.08916646e-02,  -1.81026779e-02,
             6.38104149e-02,   6.11085199e-02,   6.96122474e-02,
            -4.72079584e-02,  -9.48686788e-04,   1.88335351e-01,
            -3.25360291e-02,   6.66735443e-03,   3.23268080e-02,
             8.47305149e-03,   1.05250203e-01,   3.36354499e-01,
            -5.00092854e-03,  -5.97365459e-02,  -1.88335335e-01])




```python
# 3* Lasso
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())
y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)

# Plot residuals
plt.figure(figsize=(16,8))
plt.scatter(y_train_las, y_train_las - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.figure(figsize=(16,8))
plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
plt.figure(figsize=(16,8))
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()
```

    /Users/leo/Develop/python_env/env_python2.7/lib/python2.7/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    ('Best alpha :', 0.0001)
    Try again for more precision with alphas centered around 0.0001
    ('Best alpha :', 6.0000000000000002e-05)
    ('Lasso RMSE on Training set :', 0.32492344810521118)
    ('Lasso RMSE on Test set :', 0.32346211335998659)



![png](output_47_2.png)



![png](output_47_3.png)


    Lasso picked 70 features and eliminated the other 20 features



![png](output_47_5.png)



```python
# 4* ElasticNet
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(X_train, y_train)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(X_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1    
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
      " and alpha centered around " + str(alpha))
elasticNet = ElasticNetCV(l1_ratio = ratio,
                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                    alpha * 1.35, alpha * 1.4], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(X_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1    
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())
print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())
y_train_ela = elasticNet.predict(X_train)
y_test_ela = elasticNet.predict(X_test)

# Plot residuals
plt.figure(figsize=(16,8))
plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.figure(figsize=(16,8))
plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(elasticNet.coef_, index = X_train.columns)
print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
plt.figure(figsize=(16,8))
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")
plt.show()
```

    ('Best l1_ratio :', 0.59999999999999998)
    ('Best alpha :', 0.0001)
    Try again for more precision with l1_ratio centered around 0.6
    ('Best l1_ratio :', 0.56999999999999995)
    ('Best alpha :', 0.0001)
    Now try again for more precision on alpha, with l1_ratio fixed at 0.57 and alpha centered around 0.0001
    ('Best l1_ratio :', 0.56999999999999995)
    ('Best alpha :', 6.0000000000000002e-05)
    ('ElasticNet RMSE on Training set :', 0.32509114220203178)
    ('ElasticNet RMSE on Test set :', 0.32349645660557413)



![png](output_48_1.png)



![png](output_48_2.png)


    ElasticNet picked 74 features and eliminated the other 16 features



![png](output_48_4.png)



```python

```
