# Extracting features related to query search results from data
# The features are specified under the paper: http://research.microsoft.com/en-us/um/people/sdumais/SIGIR2012-fp497-bennett.pdf

import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests, json
from jsonpath_rw import jsonpath, parse
from pymongo import MongoClient
from bson.objectid import ObjectId
from collections import defaultdict, namedtuple
from pandas.tslib import Timedelta
import nltk
# from data_analysis import Analyser
from scipy import spatial
import csv



# # matplotlib settings
# pd.set_option('display.mpl_style', 'default')
# plt.rcParams['figure.figsize'] = (15, 10)
# plt.rcParams['font.family'] = 'sans-serif'

#####
# q: query
# d: document/results
# q_r: related query
# d_qr: document/result of a related query
# w_qr: relationship weight between the related query and the current query for this user
# u_I: is the set of the user past issued queries, the search results, and any behavioral interactions the user had with the results
# p(q_r): the number of queries in the time view (session, historic, aggregate). p(q_r) = 1 is the most recent previous query


def view(u_I, temporal='all'):
    # the temporal view on the users past search interactions
    # view can either be: session, history, aggregation=all, union
    # Input:
    #   u_I: data[data.index == user_id]
    #   temporal: temporal view of the user interaction
    temporals = ['all', 'session', 'history']
    assert temporal in temporals, "temporal \'%s\' isn't valid. use one of the following: %s" % (temporal, temporals)

    try:
        if temporal == 'all':
            # queries = u_I.query('eventaction == "Search"').eventitem.values
            return u_I

        # returns cuerrent session
        elif temporal == 'session':
            usrSessions = split2sessions(u_I)
            return usrSessions[-1]

        # returns all prior to current session
        elif temporal == 'history':
            usrSessions = split2sessions(u_I)
            return usrSessions[:-1]
    except Exception as ex:
        print ex


# TODO change s.t. session is not only for a single query
def split2sessions(df):
        index = 0
        first_date = None
        last_date = None
        last_id = None
        sessions = []
        session = pd.DataFrame(columns=df.columns)
        textual_search = 'search videos. text:'
        start = len(textual_search)
        searched = False
        threshold = Timedelta(minutes=10)  # TODO 30 min

        for idx, row in enumerate(df.iterrows()):
            try:
                id, datetime, geo, user_id, companyid, label, items = row[1].values

                if idx % 1000 == 0:
                    print 'passed line #%d in split2sessions, # of search seassions is: %d, for id:%d' % (idx, len(sessions), id)

                if label == 'search - search box' and textual_search in items and len(items) > start + 2:
                    searched = True

                # TODO fix time differences type, and check between FIRST interaction and CURRENT
                # if (last_date and last_date and datetime - last_date > threshold) or (last_id and last_id != user_id):
                if (first_date and datetime - first_date > threshold) or (last_id and last_id != user_id):
                    if searched:
                        sessions.append(session)
                    index = 0
                    session = pd.DataFrame(columns=df.columns)
                    searched = False
                    first_date = None

                session.loc[index] = row[1]
                index += 1
                last_date = datetime
                last_id = user_id
                if not first_date:
                    first_date = datetime

            except Exception as ex:
                print ex

        # last
        if len(session) > 0:
            sessions.append(session)

        print '# of sesseions' + str(len(sessions))
        return sessions


def get_usr_queries(data):
    # output:
    #       usrQueries: dict=<qId, query>
    try:
        usrQueries = defaultdict(int)
        allQueries = data[['qID','QUERY']].drop_duplicates().values
        for i in range(len(allQueries)):
            qid = allQueries[i][0]
            query = allQueries[i][1]
            usrQueries[qid] = query

        return usrQueries
    except Exception as ex:
        print ex

def get_usr_vids(data):
    return data.vid_id.drop_duplicates().values


# TODO word2vec it for queries similarity
def related(q, usrQueries, mode):
    # returns a list of tuples <q_r_ID, q_r>
    # can be either: 1) all q's using view(u_i)
    #                2) exactly matching to q
    #                3) subset of q (after omitting words from q)
    #                4) superset of q (after adding words to q)
    # input:
    #   q: the query in question (text)
    #   usrQueries: all usr queries in dict=<qID, query>

    relatedQueries = []
    modes = ['all', 'exact', 'subset', 'superset']
    assert mode in modes, "mode \'%s\' isn't valid. use one of the following: %s" % (mode, modes)

    if mode == 'all':
        relatedQueries = [(k, v) for k, v in usrQueries.iteritems()]

    elif mode == 'exact':
        if q in usrQueries.values():
            qids = [key for key, value in usrQueries.items() if value == q]
            for qid in qids:
                relatedQueries.append([qid, usrQueries[qid]])

    else:  # Query Generalization and Specialization
        qWords = set(nltk.word_tokenize(q))
        for qID, myQuery in usrQueries.items():
            myQWords = set(nltk.word_tokenize(myQuery))
            if mode == 'superset' and qWords > myQWords:  # query has more words
                relatedQueries.append([qID, myQuery])

            elif mode == 'subset' and qWords < myQWords:  # query has less words
                relatedQueries.append([qID, myQuery])

    return relatedQueries


def results(qData, labeledData):
    # returns documents d_qr related to query q
    # input:
    #   qData -  tuple <qID, query>
    #   labeledData - df=<qID, user_id, query, vid, score>
    try:
        vids = []
        qID, query = qData
        vids = labeledData.query('qID == @qID').vid_id.values

        return vids
    except Exception as ex:
        print ex


def get_query_data():
    pass


def get_url_rep(docID):
    # Assign 1 to phi(docID), whereas phi = a sparse vector
    pass


def phi(vid, func='topic'):
    # determine which representation to use for vid
    return get_topic_rep(vid)


def get_topic_rep(vid):
    # Uses mongo\elastic to search taxonomies etc
    ### DEBUG
    if True:
        rep = np.zeros(1000)
        locs = np.random.randint(1000, size=10)
        for loc in locs:
            rep[loc] = 1
        return rep
    else:
        try:
            rep = np.zeros_like(allTopics, dtype=int)
            field = "classificationResults.taxonomyCategories"
            entry = db.video.find({"_id": ObjectId(vid)}, {"_id": 0, field: 1})
            if not entry.count():  # didnt found any results
                print 'No topic representation for %s' % vid
                return rep

            for doc in entry:
                res = doc.get('classificationResults')[0].get('taxonomyCategories')
                vidCategories = []
                for i, val in enumerate(res):
                    vidCategories.extend(val.get('name').split('/'))

            # categories = defaultdict(int)
            # for category in filter(None, vidCategories):  # remove empty strings
            #     categories[category] += 1
            #
            # categories = sorted(categories, key=categories.get, reverse=True)[:3]
            for category in vidCategories:
                idx = allTopics.index(category)
                if idx:
                    rep[idx] = 1

            return rep

        except Exception as ex:
            print ex


def get_all_topics():
    try:
        params = {'apikey': '05aROO8HpkFj2GRb0voOrSMJcDcOLfSM'}
        url = 'http://api.relegence.com/taxobrowser/hierarchy/subjects'
        resp = requests.get(url, params)
        data = json.loads(resp.text)
        field = parse('$..name')
        topics = [match.value for match in field.find(data)]
        return topics
    except Exception as ex:
        print ex


def get_weight(q_id, q_r_id, labeledData, uniform=False):
    # returns wqr = w(q_r, q, u_I) = (c ^ { p(q_r)-1 }) or 1 (decay or uniform)
    # p(q_r) = rank(q_r) - rank(q): (queryNum_A - queryNum_B) TODO need +1 ?
    if uniform:
        return 1

    c = 0.95
    wqr = None
    try:
        # p_q = labeledData.query('qID == @q_id').qID.values[0]  # all entries for the same q_id has the same queryNum
        # p_qr = labeledData.query('qID == @q_r_id').qID.values[0]
        wqr = abs(int(q_id) - int(q_r_id))  # sarbiv, how far are the queries from each other, not only past ones
        return c ** (wqr-1)
    except Exception as ex:
        print ex


def action(q_r_id, d_qr, labeledData):
    # satisfied (SAT) clicks on the search engine result page
    # at our case, can be "grab code"\"add to post"
    try:
        action = int(labeledData.query('qID == @q_r_id & vid_id == @d_qr').label.values[0])
        return action
    except Exception as ex:
        print ex


def get_feature():
    # f(q,d,u_i) = sum_(q_r) | q_r in view(u) { wqr * sum_(d_qr) { sim(d,d_qr) * action(q_r, d_qr) } }
    #            = < phi(d), omega(q,d,u_I)>
    # where omega(q,d,u_I) = wqr * sum_(d_qr) { phi(d_qr) * action(q_r, d_qr) }

    # spatial.distance.cosine(phi(vid), omega)
    pass


def get_omega(qData, currViewQueries, labeledData, relatedMode, phiMode):
    omega = 0
    q_id, q = qData

    try:
        related_qs = related(q, currViewQueries, mode=relatedMode)
        for q_r_id, q_r in related_qs:
            # TODO after training, do only related from the past
            try:
                if int(q_r_id) not in labeledData.index.values or q_id == q_r_id:
                    continue
                w_qr = get_weight(q_id, q_r_id, labeledData)
                docs_qr = results([q_r_id, q_r], labeledData)
                try:
                    for d_qr in docs_qr:
                        action_d = action(q_r_id, d_qr, labeledData)

                        omega += w_qr * phi(d_qr, func=phiMode) * action_d

                    return omega

                except Exception as ex:
                    print ex

            except Exception as ex:
                print ex

    except Exception as ex:
        print ex


def vids_with_data(field="classificationResults.taxonomyCategories"):
    allVids = float(db.video.find({}).count())
    vidsWithField = float(db.video.find({field: {'$exists': True}}).count())

    return vidsWithField/allVids*100

SELECTIONS_ = ['video item - checkbox', 'video item - details button', 'add to selections']
SPREFIX_ = 'search videos. text:'
EXTRACT_ID_ = ['video item - details button', 'video item - checkbox']
EXTRACT_IDS_ = ['add to selections', 'add to post', 'grab code']
ADD_TO_POST_ = ['add to post', 'generate code', 'grab code']

# elastic search API, Relegence topics
elasticAPI = 'http://elasticsearch-master.vidible.aolcloud.net:9200/video/video/_search'
allTopics = get_all_topics()

# # Mongo connection
# client = MongoClient()
# db = client.maindb
# cursor = db.video.find()
# # percent = vids_with_data()

# get data with results (labels)
fileWithResults = '../data/usage_report.mid.2016.ds.txt'
labeledData = pd.read_csv(fileWithResults,
                          header=None,
                          # names=['queryNum', 'qID', 'user_id', 'query', 'vid_id', 'label', 'rank'],  # TODO change
                          names=['SESSION', 'qID', 'DATE', 'GEO', 'USER', 'COMPANY', 'QUERY', 'vid_id', 'label',
                                 'POSITION'])
                          # index_col='qID')


features = defaultdict(int)
for i, u_I in labeledData.groupby('USER'):
    user = u_I.USER.values[0]
    if '/' in user or '+' in user or '=' in user:
        continue
    fname = "../res/userFeatures_" + user + ".csv"
    print "Creating features for user:", user

    with open(fname, 'wb') as csvfile:
        writer = csv.writer(csvfile)

        allUsrQueries = get_usr_queries(u_I)  # all the use queries to analyze
        allUsrVids = get_usr_vids(u_I)

        # get a specific view and analyze by it  # TODO per temporal view
        temporalSelection = 'all'
        temporalView = view(u_I, temporalSelection)  # all the user data in a specific view
        currViewQueries = get_usr_queries(temporalView)

        for vid in allUsrVids:
            for qid, query in allUsrQueries.items():
                try:
                    relatedMode = 'subset'
                    phiMode = 'topic'

                    omega = get_omega([qid, query], currViewQueries, labeledData, relatedMode, phiMode)
                    if omega != None:
                        feat = spatial.distance.cosine(phi(vid), omega)

                        features[[user, qid, vid]].append(feat)
                except Exception as ex:
                    print ex

        for key, value in features.items():
            u, q, v = key
            writer.writerow([u, q, v, value])
