__author__ = 'reihane'
import os
import igraph
import sys
import numpy as np
from nltk.parse import stanford
import nltk
import louvain
import pickle
from CASSIM import Cassim
import copy
import csv
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
os.environ['STANFORD_PARSER'] = 'jars/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'jars/stanford-parser-3.5.2-models.jar'
parser = stanford.StanfordParser(model_path="jars/englishPCFG.ser.gz")


class Document(object):
    name = ""
    group = ""
    content = ""

    def __init__(self, name, group, content):
        self.name = name
        self.group = group
        self.content = content


def graph_creation_new():  # 1st col: name of the document, 2nd col: group it belongs to, 3rd col: sentence/s
    myCassim = Cassim()
    f = open(sys.argv[1], 'rb')
    dataset = csv.reader(f)
    mydocuments = []
    cassimdocs = []
    mygraph_list = []
    for datapoint in dataset:
        mydocuments.append(Document(datapoint[0],datapoint[1],datapoint[2].decode('ascii', 'ignore')))
        cassimdocs.append(datapoint[2].decode('ascii', 'ignore'))
    cassimres = myCassim.syntax_similarity_one_list(cassimdocs)
    for i in range(len(mydocuments)):
        for j in range(i+1, len(mydocuments)):
            mygraph_list.append((mydocuments[i].name, mydocuments[j].name, (cassimres[(mydocuments[i].content, mydocuments[j].content)])))
    mygraph = igraph.Graph.TupleList(mygraph_list, weights=True)
    outputgraph= open("myGraph_"+sys.argv[1],'wb')
    pickle.dump(mygraph,outputgraph)


def most_common(myclusters):
    mycounter =np.zeros(len(myclusters))
    for i in range(len(myclusters)):
        for j in range(len(myclusters)):
            if (myclusters[i]==myclusters[j]).all():
                mycounter[i]+=1
    return myclusters[np.argmax(mycounter)]


def find_core(mygraph, minnodes):
    if len(mygraph.vs) <= minnodes:
        return mygraph
    totalweight = np.zeros(len(mygraph.vs))
    for v in mygraph.vs:
        for r in mygraph.es[mygraph.incident(v)]:
            totalweight[v.index]+= r['weight']
    to_delete = totalweight.argmin()
    mygraph.delete_vertices(to_delete)
    return find_core(mygraph,minnodes)


def clustering_LDA_type_new(mygraph, mygroups, algoname, corecalculation):  # algoname is to specify which algorithm the function should use, corecalculation is wether to calcualte core documents or not
    counter_matrices = []
    for k in range(1):  # clustering is a greedy algorithm, run it 100 times and find the most frequent answer among all
        if algoname==0:
            myclusters = mygraph.community_fastgreedy(weights="weight").as_clustering(3)  # cluster based on max weight
        elif algoname==1:
            myclusters = louvain.find_partition(mygraph, method='Modularity', weight='weight', resolution_parameter=1)  # smaller resolution, smaller number of clusters #best
        elif algoname==2:
            myclusters= louvain.find_partition(mygraph, method='RBConfiguration', weight='weight', resolution_parameter=1)
        elif algoname ==3:
            myclusters=louvain.find_partition(mygraph, method='RBER', weight='weight', resolution_parameter=1)  # not working very well
        elif algoname==4:
            myclusters=louvain.find_partition(mygraph, method='CPM', weight='weight', resolution_parameter=0.6)  # not working very well
        cluster_list = list(myclusters)
        counter_matrix = np.zeros((len(cluster_list), len(mygroups)))
        cluster_subgraphs =[]
        for i in range(len(cluster_list)):
            cluster_subgraphs.insert(i, mygraph.subgraph(cluster_list[i]))  # creating clusters subgraph
            for v in cluster_list[i]:  # calculating matrices
                for j in range(len(mygroups)):
                    if mygraph.vs[v]['name'] in mygroups[j]:
                        counter_matrix[i][j] +=1
        counter_matrices.insert(k, counter_matrix)
    counter_matrix = most_common(counter_matrices)  # counter matrix (rows are clusters and columns are groups)
    cluster_cores =[]
    if corecalculation==1:
        for i in range(len(cluster_subgraphs)):  # finding core of each cluster subgraph
            cluster_cores.insert(i,find_core(cluster_subgraphs[i], 3))  # 3 means return the three center documents, you can set it to whatever number you wish
            print cluster_cores[i].vs()['name']
    group_matrix = counter_matrix/counter_matrix.sum(axis=0)[None,:]  # probability of each cluster belonging to each group
    cluster_matrix = counter_matrix/counter_matrix.sum(axis=1)[:,None]  # probability of each group belonging to each cluster
    return cluster_matrix, cluster_list, cluster_cores


def calculate_node_cluster_louvain_package(mygraph, mygraph_cluster_list, test_vertex):
    avgweight = np.zeros(len(mygraph_cluster_list))
    totalweight = 2 * np.sum(mygraph.es()['weight'])  # total_weight
    kin = np.sum(mygraph.es[mygraph.incident(test_vertex.index)]['weight'])
    kout = np.sum(mygraph.es[mygraph.incident(test_vertex.index)]['weight'])
    for i in range(len(mygraph_cluster_list)):
        winoutnew = 2 * np.sum(mygraph.es.select(_between = ([test_vertex.index], [mygraph_cluster_list[i][k].index for k in range(len(mygraph_cluster_list[i]))]))['weight']) # w_to_new + w_from_new
        kinnew = 0
        for k in range(len(mygraph_cluster_list[i])):
            kinnew += np.sum(mygraph.es[mygraph.incident(k)]['weight'])
        kinnew+=kin
        koutnew = kinnew
        avgweight[i]= (winoutnew-(kout*kinnew/totalweight)-(kin*koutnew/totalweight))

    return avgweight



def clustering_accuracy_loo(mygraph, mygroups, algoname, corecalculation):
    corrects =0.0
    for tv in mygraph.vs():  # looping over vertices and leave one them out for testing
        training_subgraph = copy.deepcopy(mygraph)
        training_subgraph.delete_vertices(tv)
        cluster_matrix, cluster_list, cluster_cores = clustering_LDA_type_new(training_subgraph, mygroups, algoname, corecalculation)
        mygraph_cluster_list=[]
        for i in range(len(cluster_list)):  # convert list of numbers(cluster_list) to list of vertices (mygraph_cluster_list)
            temp=[]
            for j in range(len(cluster_list[i])):
                temp.append(mygraph.vs.find(name = training_subgraph.vs()[cluster_list[i][j]]['name']))
            mygraph_cluster_list.insert(i, temp)
        avgweight = calculate_node_cluster_louvain_package(mygraph, mygraph_cluster_list, tv)
        assigned_cluster = avgweight.argmax()  # find cluster which has higher louvain number
        for i in range(len(mygroups)):
            if tv['name'] in mygroups[i]:
                if cluster_matrix[assigned_cluster][i] > 0.5:  # if the probability of text_vertx's group is higher in the assigned cluster, then increase corrects.
                    corrects+=1
    print corrects/(len(mygraph.vs()))


def clustering_analysis():
    inputgraph = open(sys.argv[1],'rb')
    mygraph = pickle.load(inputgraph)
    inputgroup = csv.reader(open(sys.argv[2], "rb"))  # 1st col= name of doc, 2nd col= group it belongs
    mygroups={}
    for l in inputgroup:  # read from csv to store group with document names belonging to the group
        name, group = l[0], l[1]
        if group not in mygroups:
            mygroups.setdefault(group, [])
        mygroups[group].append(name)  # for the whole doc
    temp, i = {}, 0
    for g in mygroups:  # change group names to group indices
        temp[i] = mygroups[g]
        i+=1
    mygroups = temp
    # clustering_LDA_type_new(mygraph, mygroups, 1 , 1)  # if it's just a clustering and you are not interested in the accuracy
    clustering_accuracy_loo(mygraph, mygroups, 1 , 0)


clustering_analysis()

