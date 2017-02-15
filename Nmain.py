import pickle
import pandas as pd
from neural_network import Neural
#'''
data_expclass = pd.read_csv("D:/tmp/Evaluation/dataclass_learning.csv",encoding="shift-jis")
neu = Neural(data_expclass)
neu.oversample()
Nvector, NS = neu.word_vector(u'名詞クラス',u'深層格')
Vvector, VS = neu.word_vector(u'動詞クラス',u'深層格')
Pvector, PS = neu.word_vector(u'助詞',u'深層格')
Xvector = [Nvector, Vvector, Pvector]
XS = [NS, VS, PS]
dummylist, Ddummy = neu.dummy()
net, ds = neu.neural_data(dummylist[0],dummylist[1],dummylist[2],Ddummy)
net = neu.neural_learn(net, ds)
file = open('D:/tmp/Evaluation/neural_network/neuron1/Trained.Network','w')
pickle.dump(net, file)
file.close()
file = open('D:/tmp/Evaluation/neural_network/neuron1/dummylist.Word','w')
pickle.dump(dummylist, file)
file.close()
#file = open('D:/tmp/Evaluation/Xvector.Word','w')
#pickle.dump(Xvector, file)
#file.close()
#file = open('D:/tmp/Evaluation/XS.Word','w')
#pickle.dump(XS, file)
#file.close()

precision_perD, recall_perD, resultlist, correctlist = neu.netresult(net, ds)
print precision_perD
#'''
data_expclass = pd.read_csv("C:/tmp/Evaluation/dataclass_predict.csv",encoding="shift-jis")
neu2 = Neural(data_expclass)
neu2.data_oversam = neu2.data_expclass
dummylist2, Ddummy2 = neu2.dummy()
Ddummy2.columns = [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"]

file = open('C:/tmp/Evaluation/neural_network/neuron7/Trained.Network')
net = pickle.load(file)
file.close()
file = open('C:/tmp/Evaluation/neural_network/neuron7/dummylist.Word')
dummylist = pickle.load(file)
file.close()
#file = open('D:/tmp/Evaluation/Xvector.Word')
#Xvector = pickle.load(file)
#file.close()
#file = open('D:/tmp/Evaluation/XS.Word')
#XS = pickle.load(file)
#file.close()


net2, ds2 = neu2.neural_data(dummylist[0],dummylist[1],dummylist[2],Ddummy2)

precision_perD2, recall_perD2, resultlist, correctlist = neu2.netresult(net, ds2)
print precision_perD2

data_result = neu2.data_oversam
data_result[u'result'] = resultlist
data_result[u'correct'] = correctlist
precision2_perD = neu2.neteval(data_result)
print precision2_perD
data_result.to_csv("C:/tmp/Evaluation/data_result.csv",encoding="shift-jis")

neu2.resultplot(precision_perD2, precision2_perD, Ddummy2)

data_expclass = pd.read_csv("C:/tmp/Evaluation/data_expclass.csv",encoding="shift-jis")
data_expclass = data_expclass[data_expclass[u"名詞クラス"].isnull()]
neu3 = Neural(data_expclass)
neu3.data_oversam = neu3.data_expclass
dummylist3, Ddummy3 = neu3.dummy()
Ddummy3.columns = [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"]

net3, ds3 = neu3.neural_data(dummylist[0],dummylist[1],dummylist[2],Ddummy3)
precision_perD3, recall_perD3, resultlist, correctlist = neu3.netresult(net, ds3)
print precision_perD3
data_result = neu3.data_oversam
data_result[u'correct'] = correctlist
precision2_perD3 = neu3.neteval(data_result)
print precision2_perD3

neu3.resultplot(precision_perD3, precision2_perD3, Ddummy3)


#'''