import itertools
import os
import pickle
import re
from collections import Counter

import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

TPS_DIR = '.'

tf.flags.DEFINE_string("valid_data", os.path.join(TPS_DIR, "valid_data.csv"), "Data for validation")
tf.flags.DEFINE_string("test_data", os.path.join(TPS_DIR, "test_data.csv"), "Data for testing")
tf.flags.DEFINE_string("train_data", os.path.join(TPS_DIR, "train_data.csv"), "Data for training")
tf.flags.DEFINE_string("user_review", os.path.join(TPS_DIR, "user_review"), "User's reviews")
tf.flags.DEFINE_string("item_review", os.path.join(TPS_DIR, "item_review"), "Item's reviews")
tf.flags.DEFINE_string("user_review_id", os.path.join(TPS_DIR, "user_rid"), "user_review_id")
tf.flags.DEFINE_string("item_review_id", os.path.join(TPS_DIR, "item_rid"), "item_review_id")
tf.flags.DEFINE_string("item_review_date", os.path.join(TPS_DIR, "item_review_date"), "item_review_date")
tf.flags.DEFINE_string("user_reviews_date", os.path.join(TPS_DIR, "user_reviews_date"), "user_reviews_date")


def clean_str(string):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 填充数据
#  u_len 表示每个用户或者商品的评论个数  u2_len 表示每条评论的长度
def pad_sentence(u_text, u_len, u2_len, user_reviews_date, padding_word="an"):
    review_num = u_len
    review_len = u2_len
    u_text2 = {}
    u_review_mark = {}
    u_text_review_length = {}

    # 用户索引id
    for i in u_text.keys():
        # 针对某个用户的所有评论
        u_reviews = u_text[i]
        u_review_mark[i] = []
        u_text_review_length[i] = []
        padded_u_train = []

        # 时间信息
        if len(u_reviews)>review_num:
            user_reviews_date[i] = user_reviews_date[i][:review_num]
        else:
            dis = review_num - len(u_reviews)
            for k in range(dis):
                user_reviews_date[i].append([6, 10, 2018])

        # 对于每一条评论
        for ri in range(review_num):
            # 实际集合中具有这条评论
            if ri < len(u_reviews):
                # 填充评论的长度
                sentence = u_reviews[ri]
                # 因为我赋予的是最大值
                # u_text_review_length[i].append(len(sentence))
                if review_len > len(sentence):
                    u_text_review_length[i].append(len(sentence))
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    u_text_review_length[i].append(review_len)
                    padded_u_train.append(new_sentence)

                is_stopwords_marks = []
                for sm in range(len(new_sentence)):

                    word = new_sentence[sm]
                    if word in list(set(stopwords.words('english'))):
                        is_stopwords_marks.append(0)
                    else:
                        is_stopwords_marks.append(1)

                u_review_mark[i].append(is_stopwords_marks)

            else:
                # user_reviews_date[i].append([6, 10, 2018])
                u_text_review_length[i].append(0)
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
                u_review_mark[i].append([0] * review_len)

        u_text2[i] = padded_u_train
    return u_text2, u_review_mark, u_text_review_length, user_reviews_date


# 填充评论的id
def pad_reviewid(u_train, u_valid, u_len, num):
    pad_u_train = []
    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    pad_u_valid = []

    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]

        pad_u_valid.append(x)

    return pad_u_train, pad_u_valid


def build_vocab(sentences1, sentences2):
    word_counts1 = Counter(itertools.chain(*sentences1))
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}
    word_counts2 = Counter(itertools.chain(*sentences2))
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


# 把相应的单词换成索引
def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u

    i_text2 = {}

    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i

    return u_text2, i_text2


def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid, item_review_date_path,
              user_reviews_date_path):
    u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num \
        , reid_user_train, reid_item_train, reid_user_valid, reid_item_valid, item_review_date, user_reviews_date = load_data_and_labels(
        train_data,
        valid_data,
        user_review,
        item_review,
        user_rid,
        item_rid,
        item_review_date_path,
        user_reviews_date_path)

    print("load data done")
    u_text, u_review_mark, u_text_review_length, user_reviews_date = pad_sentence(u_text, u_len, u2_len, user_reviews_date)
    reid_user_train, reid_user_valid = pad_reviewid(reid_user_train, reid_user_valid, u_len, item_num + 1)
    print("pad user done")
    i_text, i_review_mark, i_text_review_length, item_review_date = pad_sentence(i_text, i_len, i2_len, item_review_date)
    reid_item_train, reid_item_valid = pad_reviewid(reid_item_train, reid_item_valid, i_len, user_num + 1)
    print("pad item done")
    # 单词集合
    user_voc = [xx for x in u_text.values() for xx in x]
    item_voc = [xx for x in i_text.values() for xx in x]
    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print(len(vocabulary_user))
    print(len(vocabulary_item))
    # {用户索引:[[单词索引,单词索引,单词索引,单词索引,],[单词索引,单词索引,单词索引,],[....]]}
    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)

    return [u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item,
            vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train,
            reid_item_train, reid_user_valid, reid_item_valid, i_text_review_length, u_text_review_length,
            u_review_mark, i_review_mark, item_review_date, user_reviews_date]


def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, item_review_date_path,
                         user_reviews_date_path):
    f_train = open(train_data, 'rb')
    f1 = open(user_review, 'rb')
    f2 = open(item_review, 'rb')
    f3 = open(user_rid, 'rb')
    f4 = open(item_rid, 'rb')
    f5 = open(item_review_date_path, 'rb')
    f6 = open(user_reviews_date_path, 'rb')
    user_reviews = pickle.load(f1, encoding='utf-8')
    item_reviews = pickle.load(f2, encoding='utf-8')
    user_rids = pickle.load(f3, encoding='utf-8')
    item_rids = pickle.load(f4, encoding='utf-8')
    item_review_date = pickle.load(f5, encoding='utf-8')
    user_reviews_date = pickle.load(f6, encoding='utf-8')

    item_user_length=pickle.load(open(os.path.join(TPS_DIR, "item_user_length"), 'rb'), encoding='utf-8')


    # 用于用户评论过得商品id　list集合
    reid_user_train = []
    reid_item_train = []

    uid_train = []
    iid_train = []
    y_train = []
    u_text = {}

    u_rid = {}
    i_text = {}
    i_rid = {}
    i = 0

    for line in f_train:
        i = i + 1
        line = str(line, 'utf-8')
        line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if int(line[0]) in u_text:
            reid_user_train.append(u_rid[int(line[0])])

        else:
            u_text[int(line[0])] = []
            # 用户每个序列的长度
            # u_text_review_length[int(line[0])] = []

            for s in user_reviews[int(line[0])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                # u_text_review_length[int(line[0])].append(len(s1))
                u_text[int(line[0])].append(s1)
            u_rid[int(line[0])] = []

            for s in user_rids[int(line[0])]:
                u_rid[int(line[0])].append(s)
            reid_user_train.append(u_rid[int(line[0])])

        if int(line[1]) in i_text:
            reid_item_train.append(i_rid[int(line[1])])
        else:
            # i_text_review_length[int(line[1])] = []
            i_text[int(line[1])] = []
            for s in item_reviews[int(line[1])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                # i_text_review_length[int(line[1])].append(len(s1))
                i_text[int(line[1])].append(s1)

            i_rid[int(line[1])] = []
            for s in item_rids[int(line[1])]:
                i_rid[int(line[1])].append(s)

            reid_item_train.append(i_rid[int(line[1])])

        y_train.append(float(line[2]))

    print("valid")

    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data, encoding='utf-8')
    for line in f_valid:
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if int(line[0]) in u_text:
            reid_user_valid.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [['<PAD/>']]
            u_rid[int(line[0])] = [0]
            reid_user_valid.append(u_rid[int(line[0])])
            print("erro validate dataset !!!!!!")

        if int(line[1]) in i_text:
            reid_item_valid.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = [['<PAD/>']]
            i_rid[int(line[1])] = [0]
            reid_item_valid.append(i_rid[int(line[1])])
        y_valid.append(float(line[2]))
    print("len")

    # 每一个用户评论的个数
    review_num_u = np.array([len(x) for x in u_text.values()])
    x = np.sort(review_num_u)
    # 每个用户评论的个数
    # u_len = x[int(0.9 * len(review_num_u)) - 1]
    # 获取最大的评论个数
    u_len = x[int(0.9 * len(review_num_u)) - 1]
    review_len_u = np.array([len(j) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)

    # 每条评论的长度
    # u2_len = x2[int(0.9 * len(review_len_u)) - 1]
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text.values()])
    y = np.sort(review_num_i)
    # i_len = y[int(0.9 * len(review_num_i)) - 1]
    i_len = y[int(0.9 * len(review_num_i)) - 1]

    review_len_i = np.array([len(j) for i in i_text.values() for j in i])

    y2 = np.sort(review_len_i)
    # i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print("u_len: %d" % u_len)
    print("i_len: %d" % i_len)
    print("u2_len: %d" % u2_len)
    print("i2_len: %d" % i2_len)
    # user_num = len(u_text)
    # item_num = len(i_text)

    item_num=item_user_length[0]

    user_num=item_user_length[1]

    print("user_num: %d" % user_num)
    print("item_num: %d" % item_num)
    return [u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train,
            iid_train, uid_valid, iid_valid, user_num,
            item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid, item_review_date,
            user_reviews_date]


if __name__ == '__main__':
    TPS_DIR = './data'
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item, \
    vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train, reid_item_train, \
    reid_user_valid, reid_item_valid, i_text_review_length, u_text_review_length, u_review_mark, i_review_mark, item_review_date, user_reviews_date = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review, FLAGS.item_review, FLAGS.user_review_id,
                  FLAGS.item_review_id, FLAGS.item_review_date, FLAGS.user_reviews_date)

    np.random.seed(2018)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    # 新增加一维
    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train))
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    print('write begin')
    output = open(os.path.join(TPS_DIR, 'train_data'), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, 'test_data'), 'wb')
    pickle.dump(batches_test, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[1].shape[1]
    para['review_len_i'] = i_text[1].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['i_text_review_length'] = i_text_review_length
    para['u_text_review_length'] = u_text_review_length
    para['u_review_mark'] = u_review_mark
    para['i_review_mark'] = i_review_mark
    para['item_review_date'] = item_review_date
    para['user_reviews_date'] = user_reviews_date
    output = open(os.path.join(TPS_DIR, 'music.para'), 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(para, output)
