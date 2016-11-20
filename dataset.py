from gevent import monkey
#monkey.patch_all()

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import base64
import json
import logging
import pickle
from datetime import timedelta
import time
import pandas
import grequests
from bson import ObjectId
from pymongo import MongoClient
import requests
import csv



SELECTIONS_ = ['video item - checkbox', 'video item - details button', 'add to selections']
SPREFIX_ = 'search videos. text:'
EXTRACT_ID_ = ['video item - details button', 'video item - checkbox']
EXTRACT_IDS_ = ['add to selections', 'add to post', 'grab code']
ADD_TO_POST_ = ['add to post', 'generate code', 'grab code']
BY_QUERY = u'http://portal.vidible.tv/video/searchByQuery'
SEARCH_SERVICE = u'http://search.vidible.tv:8892/video/'

LABELS_ = ['video item - details button', 'video item - checkbox', 'add to selections', 'add to post', 'grab code']
logger = logging.getLogger()
hdlr = logging.FileHandler('run.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)

# These rules all independent, order of
# escaping doesn't matter
escapeRules = {'+': r'\+',
               '-': r'\-',
               '&': r'\&',
               '|': r'\|',
               '!': r'\!',
               '(': r'\(',
               ')': r'\)',
               '{': r'\{',
               '}': r'\}',
               '[': r'\[',
               ']': r'\]',
               '^': r'\^',
               '~': r'\~',
               '*': r'\*',
               '?': r'\?',
               ':': r'\:',
               '"': r'\"',
               ';': r'\;',
               ',': ''}


def escapedSeq(term):
    """ Yield the next string based on the
        next character (either this char
        or escaped version """
    for char in term:
        if char in escapeRules.keys():
            yield escapeRules[char]
        else:
            yield char


def escapeESArg(term):
    """ Apply escaping to the passed in query terms
        escaping special characters like : , etc"""
    term = term.replace('\\', r'\\')  # escape \ first
    return "".join([nextStr for nextStr in escapedSeq(term)])


def load_default():
    with open('/Users/baumatz/Documents/python/rerank/data/query.json') as f:
        dic = json.load(f)
        return dic


def generate_query(dictionary, q, date, company_id, sort):
    dic = dictionary.copy()
    l = [u'%s_owner' % company_id, u'%s_public' % company_id, u'public', u'%s' % company_id,
         u'%s_whitelabeler' % company_id]
    s = [u'%s_%s_vl' % (company_id, company_id), u'%s_vl' % company_id]

    dic[u'query'][u'function_score'][u'query'][u'filtered'][u'query'][u'query_string'][u'query'] = q
    dic[u'query'][u'function_score'][u'functions'][0][u'gauss'][u'creationdate'][u'origin'] = date
    dic[u'query'][u'function_score'][u'functions'][1][u'gauss'][u'creationdate'][u'origin'] = date
    dic[u'query'][u'function_score'][u'functions'][2][u'gauss'][u'creationdate'][u'origin'] = date
    dic[u'query'][u'function_score'][u'query'][u'filtered'][u'filter'][u'and'][u'filters'][0][u'and'][u'filters'][4][
        u'range'][u'publishdate'][u'to'] = date

    dic[u'query'][u'function_score'][u'query'][u'filtered'][u'filter'][u'and'][u'filters'][0][u'and'][u'filters'][2][
        u'or'][u'filters'][1][u'bool'][u'must'][u'terms'][u'acl.allow'] = l
    dic[u'query'][u'function_score'][u'query'][u'filtered'][u'filter'][u'and'][u'filters'][0][u'and'][u'filters'][2][
        u'or'][u'filters'][0][u'terms'][u'acl.allow'] = s
    dic[u'query'][u'function_score'][u'query'][u'filtered'][u'filter'][u'and'][u'filters'][0][u'and'][u'filters'][2][
        u'or'][u'filters'][1][u'bool'][u'must_not'][u'terms'][u'acl.deny'] = l

    if len(sort) > 0: dic[u'sort'] = [{sort: {"order": "desc", "ignore_unmapped": True}}]
    # dic['explain'] = True

    query = json.dumps(dic)
    return query


def guess_results(dictionary, q, date, company_id, sort,
                  uri='http://elasticsearch-master.vidible.aolcloud.net:9200//video/video/_search'):
    query = generate_query(dictionary, q, date, company_id, sort)

    response = requests.get(uri, data=query)
    results = json.loads(response.text)
    return results


# guess_results(dictionary, 'Jacqueline Bisset, 1969: A Look Back', '2016-02-08T16:12:19Z' ,'50d595ec0364e95588c77bd2', None)
def add_triplts(results, ids, rank, triplets, user_id):
    if all(id not in results for id in ids):
        # add positive triplets
        triplets.extend(map(lambda id: (user_id, id, rank), ids))
        return False

    # add negative triplets
    in_list = filter(lambda id: id in results, ids)
    bottom = max(map(lambda id: results.index(id), in_list))
    triplets.extend(map(lambda id: (user_id, id, 0), [id for id in results[:bottom + 2] if id not in ids]))

    # add positive triplets
    triplets.extend(map(lambda id: (user_id, id, rank), ids))
    return True


def add_viewed(ids, triplets, user_id):
    triplets.extend(map(lambda id: (user_id, id, 0), ids))

def get_keys(sessions):


     textual_search = 'search videos. text:'
     start = len(textual_search)

     last_query = None
     row_idx = 0
     keys = []

     # collect
     for s_idx, session in enumerate(sessions):

        if s_idx % 100 == 0: print 'passed %d sessions from %d sessions' % (s_idx, len(sessions))


        for idx, row in enumerate(session.iterrows()):

            row_idx += 1

            try:
                id, datetime, geo, user_id, company_id, label, extra_data = row[1].values

                # if label == 'sort dropdown':
                #     extra_data.split(':')[-1].strip()sort =

                if label == 'search - search box' and textual_search in extra_data and len(extra_data) > start + 2:

                    query = escapeESArg(extra_data[start:].strip())
                    if query != last_query:
                        # restart
                        splitted = str(datetime).split(' ')
                        date = '%sT%sZ' % (splitted[0], splitted[1])
                        key = '%s,%s,%s' % (query, date, company_id)
                        keys.append(key)

            except Exception as ex:
                logger.error('%d\t%s\t%s\n' % (row_idx, str(row), ex.message))

     return keys

def user2company(sessions):

    if isinstance(sessions, str):
        with open(sessions) as f:
            sessions = pickle.load(f)
    textual_search = 'search videos. text:'
    start = len(textual_search)

    last_query = None
    row_idx = 0
    mapper = {}

    # collect
    for s_idx, session in enumerate(sessions):

        if s_idx % 100 == 0: print 'passed %d sessions from %d sessions' % (s_idx, len(sessions))


        for idx, row in enumerate(session.iterrows()):

            row_idx += 1

            try:
                id, datetime, geo, user_id, company_id, label, extra_data = row[1].values
                if mapper.has_key(user_id) : continue

                mapper[user_id] = company_id
            except Exception as ex:
                print ex

    return mapper

def collect_url_queries(path, save):
    urls = []
    textual_search = 'search videos. text:'
    start = len(textual_search)
    df = pandas.read_csv(path)
    writer = open(save, 'w')
    last_query = None
    for idx, row in enumerate(df.iterrows()):
        try:
            _, id, datetime, geo, user_id, company_id, label, extra_data , _= row[1].values

            if label == 'search - search box' and textual_search in extra_data and len(extra_data) > start + 2:

                query = escapeESArg(extra_data[start:].strip())
                if query != last_query:
                    # restart
                    splitted = str(datetime).split(' ')
                    date = '%sT%sZ' % (splitted[0], splitted[1])
                    key = '%s,%s,%s' % (query, date, user_id)
                    urls.append(key)
                    writer.write(key.strip() + '\n')
                    # set
                    last_query = query

        except Exception as ex:
                print ex
    return urls


def collect_url_queries_from_sessions(sessions):

    if isinstance(sessions, str):
        with open(sessions) as f:
            sessions = pickle.load(f)
    textual_search = 'search videos. text:'
    start = len(textual_search)

    last_query = None
    row_idx = 0
    urls = []
    df = pd.load_csv()
    # collect
    for s_idx, session in enumerate(sessions):

        if s_idx % 100 == 0: print 'passed %d sessions from %d sessions' % (s_idx, len(sessions))
        temp = {}
        key = None

        for idx, row in enumerate(session.iterrows()):

            row_idx += 1

            try:
                id, datetime, geo, user_id, company_id, label, extra_data = row[1].values

                # if label == 'sort dropdown':
                #     extra_data.split(':')[-1].strip()sort =

                if label == 'search - search box' and textual_search in extra_data and len(extra_data) > start + 2:

                    query = escapeESArg(extra_data[start:].strip())
                    if query != last_query:
                        # restart
                        splitted = str(datetime).split(' ')
                        date = '%sT%sZ' % (splitted[0], splitted[1])
                        key = '%s,%s,%s' % (query, date, user_id)
                        urls.append(key)

                        # set
                        last_query = query

                # if label in LABELS_ and key :
                #     ids = get_vids(extra_data)
                #     if len(ids) > 0:
                #         selected = ids[0]
                #
                #         if mapper.has_key(key) and selected in mapper[key]:
                #             # print 'found relevancy ' + selected
                #             session.loc[temp[key]] = [id + 1, datetime + timedelta(seconds=1), geo, user_id, company_id, 'search results', '%s$$$%s' % (','.join(mapper[key]), 'relevancy')]
                #             key = None
                #         elif sort_by_date.has_key(key) and selected in sort_by_date[key]:
                #             # print 'found recency ' + selected
                #             session.loc[temp[key]] = [id + 1, datetime + timedelta(seconds=1), geo, user_id, company_id, 'search results', '%s$$$%s' % (','.join(sort_by_date[key]), 'recency')]
                #             key = None
            except Exception as ex:
                print('%d\t%s\t%s\n' % (row_idx, str(row), ex.message))

    return urls

def collect_queries(args):

    queries = dict()
    sessions, mapper = args

    dictionary = load_default()
    textual_search = 'search videos. text:'
    start = len(textual_search)

    last_query = None
    row_idx = 0

    # collect
    for s_idx, session in enumerate(sessions):

        if s_idx % 100 == 0: print 'passed %d sessions from %d sessions' % (s_idx, len(sessions))
        temp = {}
        key = None

        for idx, row in enumerate(session.iterrows()):

            row_idx += 1

            try:
                id, datetime, geo, user_id, company_id, label, extra_data = row[1].values

                # if label == 'sort dropdown':
                #     extra_data.split(':')[-1].strip()sort =

                if label == 'search - search box' and textual_search in extra_data and len(extra_data) > start + 2:

                    query = escapeESArg(extra_data[start:].strip())
                    if query != last_query:
                        # restart
                        splitted = str(datetime).split(' ')
                        date = '%sT%sZ' % (splitted[0], splitted[1])
                        key = '%s,%s' % (query, date)

                        if mapper and mapper.has_key(key): # insert row
                            temp[key] = row[0] + 1 #[id + 1, datetime + timedelta(seconds=1), geo, user_id, company_id, 'search results', ','.join(mapper[key])]
                        elif not mapper:
                            data = generate_query(dictionary, query, date, company_id, 'compositeStartDate')
                            # invoke_queries([data])
                            queries[key] = data

                        # set
                        last_query = query

                if label in LABELS_ and key :
                    ids = get_vids(extra_data)
                    if len(ids) > 0:
                        selected = ids[0]

                        if mapper.has_key(key) and selected in mapper[key]:
                            # print 'found relevancy ' + selected
                            session.loc[temp[key]] = [id + 1, datetime + timedelta(seconds=1), geo, user_id, company_id, 'search results', '%s$$$%s' % (','.join(mapper[key]), 'relevancy')]
                            key = None
            except Exception as ex:
                print('%d\t%s\t%s\n' % (row_idx, str(row), ex.message))
    if len(queries):
        return queries
    return sessions

def get_vids(data):

    if 'videoId:' in data:
        trimmed = data.split('videoId:')[-1].split(';')[0].strip()
        return trimmed.split(',') if len(trimmed) > 5  else []
    else:
        trimmed = data.split('videoIds:')[-1].split(';')[0].strip()
        return trimmed.split(',') if len(trimmed) > 5  else []

def get_ids(path):

    with open(path) as f:
        mapper = pickle.load(f)

    s = set()
    for ids in mapper.values():
        for id in ids: s.add(id)

    return s

def invoke_queries(queries):
    mapper = dict()
    batches = [queries[i:i + 20] for i in range(0, len(queries), 20)]
    for idx, batch in enumerate(batches):
        print 'passed batch %d from %d batches' % (idx, len(batches))
        rs = (grequests.post('http://elasticsearch-master.vidible.aolcloud.net:9200//video/video/_search',
                             data=params, timeout=10) for params in batch)
        for response in grequests.map(rs):
            if not response: continue
            try:
                rsp = json.loads(response.content)
                results = map(lambda x: x.get('_id'), rsp.get(u'hits').get(u'hits'))
                mapper[response.request.body] = results
            except Exception as ex:
                print ex

    return mapper

def encripet_mail():
    pass


import hashlib
def get_permissions(users_path, mail2token_path, token2mail_path, token2password_path):

    token2password = dict()
    mail2token = dict([ (line.strip().split('\t')) for line in open(mail2token_path) ])
    token2mail = dict([ (line.strip().split('\t')) for line in open(token2mail_path) ])
    lines = [line for line in open(token2password_path)]
    for i in range(0, len(lines) -1, 2):
        token2password[lines[i].strip()] = lines[i+1].strip()

    key='vdbRemember'
    mapper = dict()
    expiration_time = str(int(time.time()))
    client = MongoClient()
    db = client.maindb
    user_ids = [line.strip() for line in open(users_path)]
    for user_id in user_ids:
        try:
            # print user_id
            mail = user_id
            token = user_id

            if '@' in mail: token = mail2token[mail]
            else : mail = token2mail[token]

            #if not mail == 'ezer.karavani@teamaol.com': continue

            if token2password.has_key(token):
                password = token2password[token]
            else:
                continue

            expiration_time = "1479810318596"
            md = hashlib.md5(mail + ":" + expiration_time + ":" + password + ":" + key).hexdigest()
            coockie = base64.b64encode(mail + ":" + expiration_time + ":" +md)
            mapper[user_id] = coockie
        except Exception as ex:
            print ex

    return mapper



    #result example:
    # { "_id" : ObjectId("54aa8647e4b096441a81ece4"), "roleId" : ObjectId("50dd86fccab1c9e7056d4923") }

    # //step 2 find role by roleId
    # db.role.find({_id: ObjectId("50dd86fccab1c9e7056d4923")})
    #
    # NOTE if role.superAdmin == true, user has all permissions on _full_ in sids

    # base64(username + ":" + expirationTime + ":" +
    #          md5Hex(username + ":" + expirationTime + ":" password + ":" + key))

def get_very_short_key_mapper(path):

    with open(path) as f:
        mapper = pickle.load(f)
        print 'loaded'
        counter = 0
        short_mapper = {}
        for key, value in mapper.iteritems():
             try:
                 dic = json.loads(key)
                 query = dic[u'query'][u'function_score'][u'query'][u'filtered'][u'query'][u'query_string'][u'query']
                 date = dic[u'query'][u'function_score'][u'functions'][0][u'gauss'][u'creationdate'][u'origin']
                 # company_id = dic[u'query'][u'function_score'][u'query'][u'filtered'][u'filter'][u'and'][u'filters'][0][u'and'][u'filters'][2][u'or'][u'filters'][0][u'terms'][u'acl.allow'][0].split('_')[0]

                 short_key = '%s,%s' % (query, date)
                 short_mapper[short_key] = value
                 if counter % 100 == 0: print counter, query, date
                 counter +=1
             except Exception as ex:
                 print ex
    return short_mapper

def get_short_key_mapper(path):

    with open(path) as f:
        mapper = pickle.load(f)
        print 'loaded'
        counter = 0
        short_mapper = {}
        for key, value in mapper.iteritems():
             try:
                 dic = json.loads(key)
                 query = dic[u'query'][u'function_score'][u'query'][u'filtered'][u'query'][u'query_string'][u'query']
                 date = dic[u'query'][u'function_score'][u'functions'][0][u'gauss'][u'creationdate'][u'origin']
                 company_id = dic[u'query'][u'function_score'][u'query'][u'filtered'][u'filter'][u'and'][u'filters'][0][u'and'][u'filters'][2][u'or'][u'filters'][0][u'terms'][u'acl.allow'][0].split('_')[0]

                 short_key = '%s,%s,%s' % (query, date, company_id)
                 short_mapper[short_key] = value
                 if counter % 100 == 0: print counter, query, date, company_id
                 counter +=1
             except Exception as ex:
                 print ex
    return short_mapper

import pandas as pd
import numpy as np
import multiprocessing as mp
def run_in_parallel(func, args, iterable, processes, stop, step):


    pool = mp.Pool(processes = processes)
    runs = [ (iterable[i:i+step], args, i ) for i in range(0, stop, step)]
    print 'starting mp on %d runs, with step %d and stop at %d' % (len(runs), step, stop)

    results = pool.map(func, runs)

    # merg = {}
    # for res in results: merg.update(res)
    pool.close()
    pool.join()

    # flatten = [item for sublist in results for item in sublist]
    return results

# curl --cookie 'SPRING_SECURITY_REMEMBER_ME_COOKIE=YWRtaW5AdmlkaWJsZS50djoxNDc5NzMzNDY1ODcxOmIyZTEwZGJlZWUwYmQ5MjgzZTNhM2I1Yjk2ZmQ2YzMx' -H 'Content-Type: application/json' -X POST "http://portal.dev.vidible.tv/video/searchByQuery" -d '{"wideOwnershipVideoSearchCriteria":{"limit":1}}'
def call_api(path='temp_writer.test.1.txt'):
    limit = 20
    responses = {}
    writer = open(path, 'a')
    headers = {'Content-Type': 'application/json'}
    with open('../data/user2coockie.pkl') as f:
        mapper = pickle.load(f)

    with open('sessions.search.keys.txt') as f:
        lines = ['duke,2016-04-10T16:25:26Z,alexis.jackson@teamaol.com']#f.readlines()
    # group by user
    users = dict()
    last = None
    for lidx, mapping in enumerate(lines):

        query, date, user = mapping.strip().split(',')
        if last == query or query.startswith('keyword'): continue

        if not users.has_key(user) : users[user] = []
        users[user].append((query, date, user))

    schemas = {}
    skip = False
    for idx, user in enumerate(users.keys()):

        try:
            # if user == 'chris.dangelo@huffingtonpost.com': skip = False
            if skip : continue
            queries = users[user]
            print 'passed %d out of %d users. processing user %s, %d queries' % (idx, len(users.keys()), user, len(queries))

            coockie = mapper.get(user)
            data = json.dumps({"wideOwnershipVideoSearchCriteria": {"limit": 1, "q" : "" } })
            response = requests.post(BY_QUERY, data, cookies={'SPRING_SECURITY_REMEMBER_ME_COOKIE': coockie }, headers= headers)

            if not response : continue
            dic = json.loads(response.content)
            if not dic.has_key(u'searchCriteria') : continue
            critiria = dic[u'searchCriteria']
            data = json.dumps(critiria)
            datas = get_queries_data(data, queries)
            schemas[user] = datas


            reqs = [grequests.post(SEARCH_SERVICE, data=d, cookies= {'SPRING_SECURITY_REMEMBER_ME_COOKIE': coockie}, headers=headers) for d in datas]
            batches = [reqs[i: i+ limit] for i in range(0, len(reqs), limit)]
            print 'requests: %d, batches: %d' % (len(reqs) * limit, len(batches))

            for idx, batch in enumerate(batches):
                print 'executin bacth %d from %d' % (idx, len(batches))
                for response in grequests.map(r for r in batch):
                    try:
                        if response:
                            dic = json.loads(response.content)
                            ids = [ d for d in dic['data']]
                            body = json.loads(response.request.body)
                            d = body[u'wideOwnershipVideoSearchCriteria'][u'dateRange'][u'endDate']
                            q = body[u'wideOwnershipVideoSearchCriteria'][u'query']
                            key = '%s,%s,%s' % (q, d, user)
                            writer.write('%s : %s\n' % ( key, ','.join(ids)))

                    except Exception as ex:
                           print ex
        except Exception as ex:
            print ex

    writer.close()
    with open('user_schemas.pkl', 'w') as f:
        pickle.dump(schemas, f)


def get_queries_data(data, queries):
    datas = []
    for q in queries:
        query, date, user = q
        dateJson = '"dateRange": {"dateRangeField": "creationdate", "endDate": "%s", "startDate": "2000-01-01T00:00:00Z"},' % date
        temp = data.replace('"rawQuery": ""', ' %s "query": "%s"' % (dateJson, query)).replace('"limit": 1', '"limit": %d' % 1000)
        temp = temp.replace('includeMyCompanyGroup": false', 'includeMyCompanyGroup": true').\
            replace('"includeHidden": false','"includeHidden": true' ).replace('"includeMyWhitelabeler": false', '"includeMyWhitelabeler": true').\
            replace('"includeNotOwned": false', '"includeNotOwned": true').replace('"includeMyCompanyGroup": false', '"includeMyCompanyGroup": true')

        datas.append(temp)

    return datas



def filter_results(queries='very.short.queries.wo.permissions.clean.csv', ligit_keys='temp_writer.test.txt'):

    filterd = {}
    ligit = {}
    mapper ={}
    for line in open(queries):
        key, value = line.strip().split('\t')
        value = value[1:-1]
        l = []
        for id in value.split(','):
            l.append(id.strip()[2:-1])
        mapper[key] = l

    # mapper = dict([ (line.strip().split('\t')[0], [ id.strip() for id in line.strip().split('\t')[1].split(',')]) for line in open(mapper)])
    print mapper.iteritems().next()
    for line in open(ligit_keys):
        try:
            if ' : ' in line:
                splitted = line.strip().split(' : ')
                if len(splitted) != 2:
                    #print line - no results ...
                    continue
                key, value= splitted
            else:
                key, value= line.strip().split('\t')

            splitted = key.strip().split(',')
            key = splitted[0].strip() + ',' + splitted[1].strip()
            ligit[key] = value.split(',')
        except Exception as ex:
            print ex
    print ligit.iteritems().next()


    mutual = set(mapper.keys()).intersection(ligit.keys())
    diff = set(mapper.keys()).difference(ligit.keys())
    print len(mapper), len(ligit), len(mutual)

    miss = 0
    match = 0
    for idx, key in enumerate(mutual):
        s = set(ligit[key])
        l = [id for id in mapper[key] if id.strip() in s]
        if len(l) > 0 :
            match+=1
            filterd[key] = l
        else:
            miss+=1

        if idx % 1000 == 0 :
            print ligit[key]
            print mapper[key]
            print 'passed %d out of %d. current key: %s, nonfilterd=%d, filterd=%d' % (idx, len(mutual), key, len( mapper[key]), len(l))

    print miss, match
    return filterd, diff

def inject_search_results(args):

    sessions, key2ids, run_id = args

    textual_search = 'search videos. text:'
    start = len(textual_search)
    last_query = None
    last_session = None
    modified = pd.DataFrame(columns = sessions.columns)
    l = []
    found = False
    for idx, row in enumerate(sessions.iterrows()):
        l.append(row[1])
        if idx % 5000 == 0: print 'passed %d rows from %d in run %d' % (idx, len(sessions), run_id)
        try:
            datetime, geo, user_id, company_id, label, extra_data, s_index = row[1].values
            if s_index != last_session:
                if found:
                    for r in l: modified.loc[len(modified.index)] = r
                found = False
                l = []

            last_session = s_index
            if label == 'search - search box' and textual_search in extra_data and len(extra_data) > start + 2:

                query = escapeESArg(extra_data[start:].strip())
                if query != last_query:

                    # restart
                    splitted = str(datetime).split(' ')
                    date = '%sT%sZ' % (splitted[0], splitted[1])
                    key = '%s,%s' % (query, date)

                    if key2ids.has_key(key) : # insert row
                        found = True
                        # modified.loc[len(modified.index)]= [datetime + timedelta(seconds=1), geo, user_id, company_id, 'search results', key2ids.ix[key],s_index]
                        l.append([datetime + timedelta(seconds=1), geo, user_id, company_id, 'search results', key2ids[key],s_index])

                    # set
                    last_query = query

        except Exception as ex:
            print('%d\t%s\t%s\n' % (idx, str(row), ex.message))

    return modified

def get_data_for_ids(path):

    writer = open('vid2data.csv', 'w')
    client = MongoClient()
    db = client.maindb

    size = 20
    fallback = lambda dic, key: dic[key][0] if dic.has_key(key) else None

    with open(path) as f:
        lines = f.readlines()

    batches = [lines[i:i+size] for i in range(0, len(lines), size)]
    for idx, batch in enumerate(batches):
        try:
            if idx % 1000 : print 'passed %d ids from %d' % (idx * size, len(batches) * size)
            ids = [ {'_id' : ObjectId(id.strip())} for id in batch]
            data = db.video.find({'$or' : ids} , {u'companyId': 1, u'creationDate': 1, u'modificationDate': 1, u'category':1 }) #
            for dp in data:
                writer.write('%s,%s,%s,%s,%s\n' % (dp[u'_id'], dp[u'companyId'], dp[u'creationDate'], dp[u'modificationDate'], fallback(dp, u'category')))
                # df.loc[len(df.index)] = [dp[u'_id'], dp[u'companyId'], dp[u'creationDate'], dp[u'modificationDate'], fallback(dp, u'category')]
        except Exception as ex:
            print ex


    writer.close()
    # df.to_csv('vid2data.csv', index=False)

def get_dataset(path):

    sessions = pd.read_csv(path, index_col=False)
    # session_stats = pd.DataFrame(index = [0], columns=['SESSION', 'USER', 'ROWS', 'RFINEMENTS', 'SCROLL', 'ADD_TO_POST', 'START', 'END', 'SELECTED', 'POSITION' ])
    labels = {}
    query = None
    query_id = 1
    last_row = None
    with open('../data/dataset/usage_report.mid.2016.ds.txt', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['SESSION', 'QUERY_ID', 'DATE', 'GEO' 'USER', 'COMPANY' 'QUERY', 'VIDEO', 'LABEL', 'POSITION'])

    misses = open('misses.txt', 'w')

    last_query = None
    ids = []

    for idx, row in enumerate(sessions.iterrows()):

        try:

            if idx % 10000 == 0: print 'passed %d sessions from %ds sessions'% (idx, len(sessions))

            datetime, geo, user_id, company_id, label, data, session = row[1].values

            if query and label == 'search results':
                ids.extend(id[1:-1] for id in data[1:-1].split(', '))

            if label == 'search - search box':
                query = data[len(SPREFIX_):]
                if last_query and last_query != query:

                    for key, value in labels.iteritems():  writer.writerow([session, query_id, last_row[0:4],last_query, key, value, ids.index(key)])
                    query_id +=1

                    ids = []
                    labels = {}
                last_query = query
                last_row = row[1].values


            if len(ids) > 1 and label in SELECTIONS_:
                selected = get_ids_from_data(label, data)
                for touched in selected:
                    found = change_labels(touched, ids, labels, 0, 1)
                    if misses and not found:
                        splitted = str(datetime).split(' ')
                        date = '%sT%sZ' % (splitted[0], splitted[1])
                        misses.write('%s,%s\n' % (query, date))

            if len(ids) > 1 and label in ADD_TO_POST_:
                selected = get_ids_from_data(label, data)
                for touched in selected:
                    found = change_labels(touched, ids, labels, 0, 2)
                    if misses and not found:
                        splitted = str(datetime).split(' ')
                        date = '%sT%sZ' % (splitted[0], splitted[1])
                        misses.write('%s,%s\n' % (query, date))

        except Exception as ex:
            print ex

    writer.close()
    misses.close()

def write_lables(labels, query, query_id, datetime, user_id, writer):
    for key, value in labels.iteritems():
        try:
            writer.write('%d,%s,%s,%s,%s,%s\n' % (query_id, datetime, user_id, query, key, value))
        except:
            pass

def extract_new_schema(self, ids, labels, query, row):
    user_id, eventaction, eventitem, datetime = row[1].values
    if query and eventaction.lower() == 'search results':
        vids = eventitem.split(',')
        if self.is_valid_result(query, vids):
            ids.extend(vids)
    if eventaction.lower() == 'search' and self.is_text_query(eventitem):
        query = eventitem
    if len(ids) > 0 and (eventaction.lower() == 'select' or eventaction.lower() == 'preview video'):
        selected = eventitem.split(',')
        for touched in selected:
            self.change_labels(touched, ids, labels, 0, 1)
    if len(ids) > 0 and (eventaction.lower() == 'add to post' or eventaction.lower() == 'generate code'):
        selected = eventitem.split(',')
        for touched in selected:
            self.change_labels(touched, ids, labels, 0, 2)

    return query, user_id

def extract_old_schema(ids, labels, query, row, misses):

    id, datetime, geo, user_id, company_id, label, data = row[1].values

    if query and label == 'search results':
        ids.extend(data.strip().split(','))

    if label == 'search - search box':
        query = data[len(SPREFIX_):]

    if len(ids) > 1 and label in SELECTIONS_:
        selected = get_ids_from_data(label, data)
        for touched in selected:
            found = change_labels(touched, ids, labels, 0, 1)
            if misses and not found:
                 splitted = str(datetime).split(' ')
                 date = '%sT%sZ' % (splitted[0], splitted[1])
                 misses.write('%s,%s\n' % (query, date))
    if len(ids) > 1 and label in ADD_TO_POST_:
        selected = get_ids_from_data(label, data)
        for touched in selected:
            found = change_labels(touched, ids, labels, 0, 2)
            if misses and not found:
                splitted = str(datetime).split(' ')
                date = '%sT%sZ' % (splitted[0], splitted[1])
                misses.write('%s,%s\n' % (query, date))

    return query, user_id

def get_ids_from_data( label, data):

    if label in EXTRACT_ID_:
        trimmed = data.split('videoId:')[-1].split(';')[0].strip()
        return trimmed.split(',') if len(trimmed) > 5  else []
    elif label in EXTRACT_IDS_:
        trimmed = data.split('videoIds:')[-1].split(';')[0].strip()
        return trimmed.split(',') if len(trimmed) > 5  else []

def change_labels(selected, ids, labels, up, score):
    # from selected to top + one under
    if selected in ids:
        idx = min(ids.index(selected) + 2, len(ids))
        for i in range(0, idx):
            id = ids[i]
            if not labels.has_key(id) : labels[id]  = up

        labels[selected] = score
        return True
    return False

def inject_search():
    step = 10000
    sessions = pd.read_csv('../data/sessions.2016.1-6.csv'
                           ,header=None, names = [ 'datetime', 'geo', 'user_id', 'company_id', 'label', 'extra_data', 'session_id'],
                           usecols=[2, 3,4,5,6,7, 8], index_col=False, dtype={8:np.int32}, skiprows=1, parse_dates=[0])

    key2ids = {}
    for line in open('short.key2ids.csv'):
        key, value = line.strip().split('\t')
        key2ids[key] = value.strip().split(',')

    flatten = run_in_parallel(inject_search_results, key2ids, sessions, mp.cpu_count() - 1, len(sessions), step)
    df = pd.concat(flatten)
    df.to_csv('../data/sessions.2016.1-6.only.with.results.csv', index=False)

def split_by_time(mapper, df):

    selected = df[df.label ==2]
    for row in selected.iterrows():pass




get_dataset('../data/sessions/sessions.2016.1-6.only.with.results.csv')

# import ast
# mapper = {}
# df = pd.read_csv('vid2data.csv', header=None, index_col=False, error_bad_lines = False, parse_dates=[2])
#print sum( 1 if row[1][4] else 0  for row in df.iterrows()), len(df.index)


# for row in df.iterrows():
#
#     key = row[1].values[0]
#     mapper[key]= (row[1].values[2],  row[1].values[4][0] if row[1].values[4] else None)
#
# df = pd.read_csv('../data/dataset/usage_report.mid.2016.ds.temp.txt', header=None, parse_dates=1, index_col=False, names=['q_id', 'datetime', 'user_id', 'query', 'v_id', 'label'])
# selected = df[df.label ==2]
#
#
# for g in df.groupby('q_id'):
#
#     print g
#
#
# split_by_time(mapper, df)


# print mapper







# key2ids = {}
# for line in open('short.key2ids.csv'):
#     key, value = line.strip().split('\t')
#     key2ids[key] = value.strip().split(',')
# # key2ids = pd.read_csv('short.key2ids.csv', sep='\t', index_col=0)#.apply(lambda row: (row[0].split(',')), axis=1)
# inject_search_results((sessions,key2ids,1))
# inject_search()
# df = pd.read_csv('../data/sessions.2016.1-6.w.results.csv')
# print df

# sessions = inject_search_results(sessions, key2ids)
# sessions.to_csv('sessions.2016.1-6.w.results.csv')


# filtered,diff = filter_results()
# with open('short.key2ids.csv', 'w') as f:
#     for key, value in filtered.iteritems():
#         f.write('%s\t%s\n' % (key, ','.join(value)))
#
# with open('diff.2..csv', 'w') as f:
#     for key in diff:
#         f.write('%s\n' % (key) )



# call_api()
# mapper = filter_results()
# with open('short.key2ids.csv', 'w') as f:
#     for key, value in mapper.iteritems():
#         f.write('%s\t%s\n' % (key, ','.join(value)))

# call_api()
# last = None
# with open('../data/urls.keys.clean.txt', 'w') as f:
#     need = set([line.strip() for line in open('../data/temp.txt')])
#     here = set([line.strip() for line in open('ligit.keys.txt')])
#     rn = need.difference(here)
#     for item in rn : f.write(item + '\n')




#filter()
# with open('very.short.queries.wo.permissions.pkl') as f:
#     u2c = pickle.load(f)
#
# last = ''
# # for line in open('ligit.keys.txt'):
#
#
# keys = set([ line.split(',')[0] + ',' + line.split(',')[1] for line in open('ligit.keys.txt')])
# count = 0
# all =0
# with open('very.short.queries.wo.permissions.csv', 'w') as f:
#     for key, value in u2c.iteritems():
#
#         all+=1
#         f.write('%s\t%s\n' % (key.strip(), value))
#         if key not in keys:
#             print key
#             count +=1
#
# print count,all

