import pandas as pd
import glob
from tqdm import tqdm
# datapath = './collected/'
datapath = '/home/local/PSYCH-ADS/xuqian_chen/Projects/Twitter/collected/'
preprocessedpath = '/home/local/PSYCH-ADS/xuqian_chen/Github/twitter_entropy/data/preprocessed/'
# mkdir if not exist
import os
if not os.path.exists(preprocessedpath):
    os.makedirs(preprocessedpath)

usfiles = glob.glob(datapath + 'us*.csv')
jpfiles = glob.glob(datapath + 'jp*.csv')

# put us files together
usdf = pd.DataFrame()
for file in tqdm(usfiles):
    try:
        df = pd.read_csv(file)
        usdf = pd.concat([usdf, df])
    except:
        print(file)
usdf.to_csv(preprocessedpath + 'us.csv',index=False)

# put jp files together
jpdf = pd.DataFrame()
for file in tqdm(jpfiles):
    try:
        df = pd.read_csv(file)
        jpdf = pd.concat([jpdf, df])
    except:
        print(file)
jpdf.to_csv(preprocessedpath + 'jp.csv',index=False)
