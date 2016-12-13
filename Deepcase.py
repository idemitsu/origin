# -*- coding: utf-8 -*-
from Word_class import Word_class
import pickle
import sys
import datetime
import pandas as pd

class Deepcase:
    def __init__(self, netpath, dummylistpath, NV_classpath):
        file = open(netpath.decode("shift-jis").encode("utf-8"))
        net = pickle.load(file)
        file.close()
        file = open(dummylistpath.decode("shift-jis").encode("utf-8"))
        dummylist = pickle.load(file)
        file.close()
        file = open(NV_classpath.decode("shift-jis").encode("utf-8"))
        NV_class = pickle.load(file)
        file.close()
        self.net = net
        self.dummylist = dummylist
        self.NV_class = NV_class
        self.DeepCaseList = [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"]
    
    def unregistered_words(self, unNoun_path, unVerb_path):
        unNoun_dict = pd.read_csv(unNoun_path, encoding='shift-jis', header=None)
        unVerb_dict = pd.read_csv(unVerb_path, encoding='shift-jis', header=None)
        unNoun_dict = unNoun_dict[unNoun_dict[1].notnull()]
        unVerb_dict = unVerb_dict[unVerb_dict[1].notnull()]
        for unNoun, unNoun_class in zip(unNoun_dict[0], unNoun_dict[1]):
            self.NV_class[0][unNoun] = [unNoun_class]
        for unVerb, unVerb_class in zip(unVerb_dict[0], unVerb_dict[1]):
            self.NV_class[1][unVerb] = [unVerb_class]    
    
    def buruidb_verbs(self, bunruidb_path, bun_Vthe_path):
        bun_verbs = pd.read_csv(bunruidb_path, encoding='shift-jis', header=None)
        bun_Vthe_verbs = pd.read_csv(bun_Vthe_path, encoding='shift-jis', header=None)
        bun_verbs = bun_verbs[(bun_verbs[3]==u"用") | (bun_verbs[3]==u"相")]
              
        bun_Vthe_dict = {}
        for bun in bun_Vthe_verbs[0].drop_duplicates():
            Vthelist = []
            for Vthe in bun_Vthe_verbs[bun_Vthe_verbs[0]==bun][1]:
                Vthelist.append(Vthe)
            bun_Vthe_dict[bun] = Vthelist
            
        for bun_vclass, vkey in zip(bun_verbs[5],bun_verbs[12]):
            if vkey not in self.NV_class[1]:               
                self.NV_class[1][vkey] = bun_Vthe_dict[bun_vclass]


        
    def predict(self, Noun, Particle, Verb):
        #print Noun,Particle, Verb
        #print "Start time", datetime.datetime.now()
        Wc = Word_class(self.NV_class)
        #NV = Wc.to_class(Noun.decode("shift-jis"), Verb.decode("shift-jis"))
        NV = Wc.to_class(Noun, Verb)
        X = self.getX(NV, Particle)
        #inputlist =[]
        resultlist =[]
        for NVrecord ,Xrecord in zip(NV,X):
            #inputlist.append((NVrecord[0], NVrecord[1], Particle))
            resultlist.append([(Noun, Verb, NVrecord[0], NVrecord[1], Particle),tuple(self.net.activate(Xrecord))])
        return resultlist                    
        
        
    def getX(self, NV, Particle):
        X=[]
        for record in NV:
            Xrecord=[]
            for Ninput in self.dummylist[0].columns:
                if record[0]==Ninput:
                    Xrecord.append(1.0)
                else:
                    Xrecord.append(0.0)
            for Vinput in self.dummylist[1].columns:
                if record[1]==Vinput:
                    Xrecord.append(1.0)
                else:
                    Xrecord.append(0.0)
            for Pinput in self.dummylist[2].columns:
                if Particle==Pinput:
                    Xrecord.append(1.0)
                else:
                    Xrecord.append(0.0)
            X.append(Xrecord)
        return X

    def output(self,result):
        for i in result: 
            for j in i:
                for k in j:
                    if type(k) is unicode:
                        print k,
                    else:
                        print k,
            print

    def identify(self,result):

        max_value = -1.0
        for output_values in result: 
            if max_value < max(output_values[1]):
                max_value = max(output_values[1])
                DeepCase_unique = self.DeepCaseList[output_values[1].index(max(output_values[1]))]
        return DeepCase_unique

        

    if __name__ == "__main__":
        from Deepcase import Deepcase
        netpath="D:/tmp/Evaluation/neural_network/neuron7/Trained.Network"
        dummylistpath = "D:/tmp/Evaluation/dummylist.Word"
        NV_classpath="D:/tmp/Evaluation/NV_class.Word"
        print u"名詞, 動詞, 名詞クラス, 助詞, 主体, 起点, 対象, 状況, 着点, 手段, 関係"
        Dc = Deepcase(netpath, dummylistpath, NV_classpath)
        
        
        Result=Dc.predict(u"摩耗粒子",u"が",u"生じる")

        Dc.output(Result)
