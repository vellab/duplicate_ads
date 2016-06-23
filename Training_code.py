

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
from operator import itemgetter
import zipfile
from sklearn.metrics import roc_auc_score
import time
import io
from PIL import Image
random.seed(2016)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def run_default_test(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state
    }
    num_boost_round = 260
    early_stopping_rounds = 20
    test_size = 0.1

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_ntree_limit)
    score = roc_auc_score(X_valid[target].values, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('id,probability\n')
    total = 0
    for id in test['id']:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

    # print('Creating zip-file...')
    # z = zipfile.ZipFile(sub_file + ".zip", "w", zipfile.ZIP_DEFLATED)
    # z.write(sub_file)
    # z.close()


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('itemID_1')
    output.remove('itemID_2')
    return output


def prep_train():
    testing = 0
    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'isDuplicate': np.dtype(int),
        'generationMethod': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_train.csv")
    pairs = pd.read_csv("ItemPairs_train.csv", dtype=types1)
    # Add 'id' column for easy merge
    print("Load ItemInfo_train.csv")
    items = pd.read_csv("ItemInfo_train.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    location = pd.read_csv("Location.csv")
    category = pd.read_csv("Category.csv")

    train = pairs
    train = train.drop(['generationMethod'], axis=1)

    print('Add text features...')
    items['len_title'] = items['title'].str.len()
    items['len_description'] = items['description'].str.len()
    items['len_attrsJSON'] = items['attrsJSON'].str.len()

    print('Merge item 1...')
    item1 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon', 
    'len_title', 'len_description', 'len_attrsJSON']]
    item1 = pd.merge(item1, category, how='left', on='categoryID', left_index=True)
    item1 = pd.merge(item1, location, how='left', on='locationID', left_index=True)

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1',
            'len_title': 'len_title_1',
			'len_description': 'len_description_1',
			'len_attrsJSON': 'len_attrsJSON_1',
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon', 
    'len_title', 'len_description', 'len_attrsJSON']]
    item2 = pd.merge(item2, category, how='left', on='categoryID', left_index=True)
    item2 = pd.merge(item2, location, how='left', on='locationID', left_index=True)

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2',
            'len_title': 'len_title_2',
			'len_description': 'len_description_2',
			'len_attrsJSON': 'len_attrsJSON_2'
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)

    # Create same arrays
    print('Create same arrays')
    train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)
    train['locationID_same'] = np.equal(train['locationID_1'], train['locationID_2']).astype(np.int32)
    train['categoryID_same'] = np.equal(train['categoryID_1'], train['categoryID_2']).astype(np.int32)
    train['regionID_same'] = np.equal(train['regionID_1'], train['regionID_2']).astype(np.int32)
    train['metroID_same'] = np.equal(train['metroID_1'], train['metroID_2']).astype(np.int32)
    train['lat_same'] = np.equal(train['lat_1'], train['lat_2']).astype(np.int32)
    train['lon_same'] = np.equal(train['lon_1'], train['lon_2']).astype(np.int32)

    # print(train.describe())
    print('Create train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train


def prep_test():
    start_time = time.time()

    types1 = {
        'itemID_1': np.dtype(int),
        'itemID_2': np.dtype(int),
        'id': np.dtype(int),
    }

    types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }

    print("Load ItemPairs_test.csv")
    pairs = pd.read_csv("ItemPairs_test.csv", dtype=types1)
    print("Load ItemInfo_testcsv")
    items = pd.read_csv("ItemInfo_test.csv", dtype=types2)
    items.fillna(-1, inplace=True)
    location = pd.read_csv("Location.csv")
    category = pd.read_csv("Category.csv")

    train = pairs

    print('Add text features...')
    items['len_title'] = items['title'].str.len()
    items['len_description'] = items['description'].str.len()
    items['len_attrsJSON'] = items['attrsJSON'].str.len()
    
    print('Merge item 1...')
    item1 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon', 
    'len_title', 'len_description', 'len_attrsJSON']]
    item1 = pd.merge(item1, category, how='left', on='categoryID', left_index=True)
    item1 = pd.merge(item1, location, how='left', on='locationID', left_index=True)

    item1 = item1.rename(
        columns={
            'itemID': 'itemID_1',
            'categoryID': 'categoryID_1',
            'parentCategoryID': 'parentCategoryID_1',
            'price': 'price_1',
            'locationID': 'locationID_1',
            'regionID': 'regionID_1',
            'metroID': 'metroID_1',
            'lat': 'lat_1',
            'lon': 'lon_1',
            'len_title': 'len_title_1',
			'len_description': 'len_description_1',
			'len_attrsJSON': 'len_attrsJSON_1'
        }
    )

    # Add item 1 data
    train = pd.merge(train, item1, how='left', on='itemID_1', left_index=True)

    print('Merge item 2...')
    item2 = items[['itemID', 'categoryID', 'price', 'locationID', 'metroID', 'lat', 'lon',
    'len_title', 'len_description', 'len_attrsJSON']]
    item2 = pd.merge(item2, category, how='left', on='categoryID', left_index=True)
    item2 = pd.merge(item2, location, how='left', on='locationID', left_index=True)

    item2 = item2.rename(
        columns={
            'itemID': 'itemID_2',
            'categoryID': 'categoryID_2',
            'parentCategoryID': 'parentCategoryID_2',
            'price': 'price_2',
            'locationID': 'locationID_2',
            'regionID': 'regionID_2',
            'metroID': 'metroID_2',
            'lat': 'lat_2',
            'lon': 'lon_2',
            'len_title': 'len_title_2',
			'len_description': 'len_description_2',
			'len_attrsJSON': 'len_attrsJSON_2',
        }
    )

    # Add item 2 data
    train = pd.merge(train, item2, how='left', on='itemID_2', left_index=True)

    # Create same arrays
    print('Create same arrays')
    train['price_same'] = np.equal(train['price_1'], train['price_2']).astype(np.int32)
    train['locationID_same'] = np.equal(train['locationID_1'], train['locationID_2']).astype(np.int32)
    train['categoryID_same'] = np.equal(train['categoryID_1'], train['categoryID_2']).astype(np.int32)
    train['regionID_same'] = np.equal(train['regionID_1'], train['regionID_2']).astype(np.int32)
    train['metroID_same'] = np.equal(train['metroID_1'], train['metroID_2']).astype(np.int32)
    train['lat_same'] = np.equal(train['lat_1'], train['lat_2']).astype(np.int32)
    train['lon_same'] = np.equal(train['lon_1'], train['lon_2']).astype(np.int32)

    # print(train.describe())
    print('Create test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train


def read_test_train():
    train = prep_train()
    test = prep_test()
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    # Get only subset of data
    if 1:
        len_old = len(train.index)
        train = train.sample(frac=0.5)
        len_new = len(train.index)
        print('Reduce train from {} to {}'.format(len_old, len_new))
    features = get_features(train, test)
    return train, test, features
'''
def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    pixels = list(image.getdata())
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    
    return ''.join(hex_string)
    

for zip_counter in [0]:
    imgzipfile = zipfile.ZipFile('Images_'+str(zip_counter)+'.zip')
    print ('Doing zip file ' + str(zip_counter))
    #namelist = imgzipfile.namelist()
    # Comment this line below and uncomment the above line when you do for the whole set
    namelist = imgzipfile.namelist()[:10]
    print ('Total elements ' + str(len(namelist)))

    img_id_hash = []
    counter = 1
    for name in namelist:
        #print name
        try:
            imgdata = imgzipfile.read(name)
            if len(imgdata) >0:
                img_id = name[:-4]
                stream = io.BytesIO(imgdata)
                img = Image.open(stream)
                img_hash = dhash(img)
                img_id_hash.append([img_id,img_hash])
                counter+=1
            # Uncomment the lines below to get an idea of progress when you do for the whole set
            #if counter%10000==0:
            #    print 'Done ' + str(counter) , datetime.datetime.now()
        except:
            print ('Could not read ' + str(name) + ' in zip file ' + str(zip_counter))
    df = pd.DataFrame(img_id_hash,columns=['image_id','image_hash'])
    df.to_csv('image_hash_' + str(zip_counter) + '.csv')
'''
#To be removed    
    
train, test, features = read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_default_test(train, test, features, 'isDuplicate')
print('Real score = {}'.format(score))
create_submission(score, test, test_prediction)


