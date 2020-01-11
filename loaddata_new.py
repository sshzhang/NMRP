# -*- coding: utf-8 -*-
import json
from datetime import datetime

import pandas as pd
import numpy as np
import os
import pickle

users_id = []
items_id = []
ratings = []
reviews = []
review_times = []

np.random.seed(2017)

TPS_DIR = '../data/toys'

with open(os.path.join(TPS_DIR, "Toys_and_Games_5.json")) as fl:
    for line in fl:
        line = json.loads(line)
        review_time = datetime.strptime(line['reviewTime'], "%m %d, %Y")

        if str(line['reviewerID']) == 'unknown':
            print("error!")
            break

        if str(line['asin']) == 'unknown':
            print("error!")
            break

        users_id.append(str(line['reviewerID']))
        items_id.append(str(line['asin']))
        ratings.append(str(line['overall']))
        reviews.append(line['reviewText'])
        review_times.append(review_time)

data = pd.DataFrame({'user_id': pd.Series(users_id),
                     'item_id': pd.Series(items_id),
                     'ratings': pd.Series(ratings),
                     'reviews': pd.Series(reviews),
                     'review_times': pd.Series(review_times)})[
    ['user_id', 'item_id', 'ratings', 'reviews', 'review_times']]


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid = usercount.index
unique_sid = itemcount.index
# 获得{用户Id:索引号,用户Id:索引号,用户Id:索引号,用户Id:索引号} map
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))


def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    # 把用户和物品的id　替换成相应的索引号
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp


data = numerize(data)
tp_rating = data[['user_id', 'item_id', 'ratings']]

n_ratings = tp_rating.shape[0]

test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

# 测试数据集
tp_1 = tp_rating[test_idx]
# 训练数据集
tp_train = tp_rating[~test_idx]

data2 = data[test_idx]
data = data[~test_idx]

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

# 测试集合和验证集合各占一半
tp_test = tp_1[test_idx]

tp_valid = tp_1[~test_idx]

tp_train.to_csv(os.path.join(TPS_DIR, 'train_data.csv'), index=False, header=None, encoding='utf-8')

tp_valid.to_csv(os.path.join(TPS_DIR, 'valid_data.csv'), index=False, header=None, encoding='utf-8')

tp_test.to_csv(os.path.join(TPS_DIR, 'test_data.csv'), index=False, header=None, encoding='utf-8')


# 　{用户索引号:[评论,评论,评论,评论]}
user_reviews = {}
item_reviews = {}

user_reviews_date = {}
item_review_date = {}

# {用户索引号:[评论过物品索引号,评论过物品索引号,评论过物品索引号]}
user_rid = {}
item_rid = {}

for i in data.values:
    if i[0] in user_reviews:
        user_reviews_date[i[0]].append([i[4].day, i[4].month, i[4].year])
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]] = [i[1]]
        user_reviews_date[i[0]] = [[i[4].day, i[4].month, i[4].year]]
        user_reviews[i[0]] = [i[3]]
    if i[1] in item_reviews:
        item_review_date[i[1]].append([i[4].day, i[4].month, i[4].year])
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_review_date[i[1]] = [[i[4].day, i[4].month, i[4].year]]
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]] = [i[0]]



print(len(item_rid))

print(len(user_rid))


# 　填充训练集合中不存在的评论数据
for i in data2.values:
    if i[0] in user_reviews:
        pass
    else:
        user_rid[i[0]] = [0]
        user_reviews[i[0]] = ['0']
        user_reviews_date[i[0]]=[[6, 10, 2018]]
        # 验证集中存在，　但训练集中不存在
        print("error user_id %d is empty" % i[0])
    if i[1] in item_reviews:
        pass
    else:
        item_reviews[i[1]] = ['0']
        item_rid[i[1]] = [0]
        item_review_date[i[1]]=[[6, 10, 2018]]
        print("error item_id %d is empty" % i[1])

print(len(item_rid))
print(len(user_rid))

pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

pickle.dump(user_reviews_date, open(os.path.join(TPS_DIR, 'user_reviews_date'), 'wb'))
pickle.dump(item_review_date, open(os.path.join(TPS_DIR, 'item_review_date'), 'wb'))

pickle.dump(np.array([len(item_rid), len(user_rid)]), open(os.path.join(TPS_DIR, 'item_user_length'), 'wb'))





# playcount_groupby = data[['user_id']].groupby('user_id', as_index=False)
# sm = np.array(playcount_groupby.size().values)
# sm.sort()
# print(sm.sum())#  TODO 为什么不会填充数据呢
# print(len(sm))
# itemcount_groupby = data[['item_id']].groupby('item_id', as_index=False)
# sn = np.array(itemcount_groupby.size().values)
# sn.sort()
# print(sn.sum())
# print(len(sn))
