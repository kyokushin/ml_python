import pandas as pd
import utils
from nltk.corpus import stopwords

df = pd.read_csv('./movie_data.csv')

print(df.loc[0, 'review'][-50:])

print(utils.preprocessor(df.loc[0, 'review'][-50:]))

df.loc['review'] = df.loc['review'].apply(utils.preprocessor)

stop = stopwords.words('english')
