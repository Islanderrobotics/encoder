import pandas as pd
data  = pd.read_csv("/Users/williammckeon/Sync/youtube videos/dataanalysis/housing.csv")
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
# encoder.fit_transform(data["ocean_proximity"])
encoder.fit_transform(data["ocean_proximity"].array.reshape(-1,1))
encoder.categories_
encoder.categories_[0][3]
data["ocean_proximity"]
new_data =encoder.fit_transform(data["ocean_proximity"].array.reshape(-1,1))
data.drop(columns = "ocean_proximity", inplace = True)
data["ocean_proximity"] = new_data
data.info()
data["ocean_proximity"]
encoder.categories_


# One Hot encoder
import pandas as pd
data  = pd.read_csv("/Users/williammckeon/Sync/youtube videos/dataanalysis/housing.csv")
x = "ocean_proximity"
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
new_data = encoder.fit_transform(data[x].array.reshape(-1,1))
new_data
new_data = encoder.fit_transform(data[x].array.reshape(-1,1)).toarray()
new_data
convert = pd.DataFrame(new_data,columns = encoder.categories_)
convert
data.drop(columns = x, inplace = True)
for i in convert.columns:
    data[i] = convert[i]
data
data.info()

import pandas as pd
data  = pd.read_csv("/Users/williammckeon/Sync/youtube videos/dataanalysis/housing.csv")
import IslanderDataPreprocessing as IR
encoder = IR.Encoder(df = data)
encoder.Check()
encoder.df.info()
#you can also uncomment the next line to reinitialize data to the corrected dataset
# data = encoder.df
# to use the ordinal encoder method you will need to set up IR.Encoder this way
encoder = IR.Encoder(df = data, type = "ordinalencoder")
encoder.Check()
encoder.df.info()
#  data = encoder.df