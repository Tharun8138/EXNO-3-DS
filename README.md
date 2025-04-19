## EXNO-3-DS
## DATE : 19.04.2025
## REGISTER NO : 212224230290

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv('/content/Encoding Data.csv')
df
```

# OUTPUT:
![ex 3 shot 1](https://github.com/user-attachments/assets/10854d3c-15af-4ddd-88bf-8ae81a0272b6)


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

# OUTPUT:
![ex 3 shot 2](https://github.com/user-attachments/assets/cf82187e-a09c-489d-8943-2eb391593385)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
# OUTPUT:

![ex 3 shot 3](https://github.com/user-attachments/assets/b683d2b6-a96e-4fd7-8d90-6113a3aaef3d)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
# OUTPUT:

![ex 3 shot 4](https://github.com/user-attachments/assets/d75d1c95-b2c4-432a-bb38-0c00a9e85e2d)

```
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
# OUTPUT:

![ex 3 shot 5](https://github.com/user-attachments/assets/e4c32786-7ef2-43b8-bc28-e1a56865557a)

```
pd.get_dummies(df2,columns=["nom_0"])
```
# OUTPUT:

![ex 3 shot 6](https://github.com/user-attachments/assets/6f8a0be4-6b8c-419d-b78f-2db86822471e)

```
!pip install category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```

# OUTPUT:
![ex 3 shot 7](https://github.com/user-attachments/assets/6f5fa9bd-1f4c-4ce4-9907-e24bc21c65e4)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

# OUTPUT:
![ex 3 shot 8](https://github.com/user-attachments/assets/27d0eb59-0fbb-433f-854e-77f18297d743)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```

# OUTPUT:
![ex 3 shot 9](https://github.com/user-attachments/assets/314d9e6a-76ab-40de-929f-98019e7f2b4a)

```
df.skew()
```

# OUTPUT:

![ex 3 shot 10](https://github.com/user-attachments/assets/5c974e20-41fb-4121-8359-ecc85947f30c)

```
np.reciprocal(df["Moderate Positive Skew"])
```

# OUTPUT:

![ex 3 shot 11](https://github.com/user-attachments/assets/86b4c504-ae18-4e2f-ba6e-6a048dc50671)

```
np.sqrt(df["Highly Positive Skew"])
```

# OUTPUT:

![ex 3 shot 12](https://github.com/user-attachments/assets/4aebae2d-4f12-4f8c-9c0c-7475edfbb351)
```
np.square(df["Highly Positive Skew"])
```

# OUTPUT:

![ex 3 shot 13](https://github.com/user-attachments/assets/2f1fda4b-1b5e-4bb5-a16e-63b59d91d56d)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

# OUTPUT:
![ex 3 shot 14](https://github.com/user-attachments/assets/f84c9925-fe0c-4d0e-91dc-e6d4632ff549)
```
df.skew()
```

# OUTPUT:
![ex 3 shot 15](https://github.com/user-attachments/assets/bbed2103-f8f0-4add-9fc3-d7d415c603d1)

```
df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

# OUTPUT:
![ex 3 shot 16](https://github.com/user-attachments/assets/e04e4157-fe8b-4574-87d9-f10dc64c9c6e)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

# OUTPUT:
![ex 3 shot 17](https://github.com/user-attachments/assets/6c0a69ff-8222-4535-a74c-d8fa3030df23)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

# OUTPUT:
![ex 3 shot 18](https://github.com/user-attachments/assets/e42ab3bd-47ab-4106-bba6-cf28262fe856)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

# OUTPUT:
![ex 3 shot 19](https://github.com/user-attachments/assets/ef5dbe7d-9515-4afe-9d36-78148fdc07a8)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

# OUTPUT:
![ex 3 shot 20](https://github.com/user-attachments/assets/24911e28-fde9-4c42-abb2-253e849e21e6)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

# OUTPUT:
![ex 3 shot 21](https://github.com/user-attachments/assets/6d7e4b04-446b-4870-afea-f2d41922eccb)

```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

# OUTPUT:
![ex 3 shot 22](https://github.com/user-attachments/assets/ba74a552-a92b-4b1d-9f12-d2808d07bb14)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

# OUTPUT:
![ex 3 shot 23](https://github.com/user-attachments/assets/4c62fa29-5c1e-43db-a9cc-5eca7cb07796)

# RESULT:
```
       Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully
```

       
