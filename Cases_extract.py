# -*- coding: utf-8 -*-
'''
import sys
sys.path.append("C:\Users\ide\Dropbox\python")

sys.modules.pop('neural_network') #アンインポート
'''

from Treport import Treport
from Language import Language
from pandas import DataFrame
from TWordclass import TWordclass
import pandas as pd

'''
class Cases_extract:
    def __init__(self):
        pass
'''        
def Triple_extract(path):
        TR = Treport(path)
        triplelist = {}
        for i in range(1, TR.s.nrows):
            #if i>TR.s.nrows/2:
            if i>100:
                break
            print i
            noenc = TR.delete_unnecc(i)
            print noenc
            Lan = Language(noenc)
            cabocha_xml = Lan.cabocha_command()
            chunkinfo, tokinfo, sentence_tok = Lan.chunk_structured(cabocha_xml)
            triple_perR = []
            for chunk in chunkinfo:
                compnoun_tail_id = -1
                for tok_id, tokinfo_mor in enumerate(tokinfo[int(chunk[u"id"])]):
                    #print tok_id, compnoun_tail_id
                    if tok_id <= compnoun_tail_id:
                        continue
                    sentence_tok_set = sentence_tok[int(chunk[u"id"])]                     
                    if tokinfo_mor[0]==u"名詞" or tokinfo_mor[0]==u"連体詞":
                        Noun = sentence_tok_set[tok_id]
                        compnoun_tail_id = tok_id
                        for tok_id_noun in range(tok_id+1, len(tokinfo[int(chunk[u"id"])])-1):                            
                            if tokinfo[int(chunk[u"id"])][tok_id_noun][0]==u"名詞" :
                                Noun += sentence_tok[int(chunk[u"id"])][tok_id_noun]
                                compnoun_tail_id = tok_id_noun
                            else: 
                                break
                        
                        if compnoun_tail_id+1 == len(tokinfo[int(chunk[u"id"])]):
                            continue
                        chunk_id_from = int(chunk[u"id"])
                       

                        for i_from in reversed(range((int(chunk["id"])+1)*-1, 0)):
                            if int(chunkinfo[int(chunk[u"id"])+i_from]["link"])==chunk_id_from:                           
                                chunk_id_from -= 1
                                from_tail_tok= tokinfo[int(chunk[u"id"])+i_from][len(tokinfo[int(chunk[u"id"])+i_from])-1]
                                if from_tail_tok[1]==u"連体化" or from_tail_tok[0]==u"形容詞" or from_tail_tok[0]==u"連体詞" or from_tail_tok[0]==u"助動詞" or [1]==u"並立助詞" or from_tail_tok[0]==u"接続詞" or from_tail_tok[1]==u"読点" or from_tail_tok[1]==u"接続助詞":
                                    for sentence_tok_from in reversed(list(sentence_tok[int(chunkinfo[int(chunk[u"id"])+i_from]["id"])])):                                                                            
                                        Noun = sentence_tok_from +Noun
                                        
                                else:
                                    break
                            else:
                                break

                            
                        if tokinfo[int(chunk[u"id"])][compnoun_tail_id+1][0]==u"助詞" and tokinfo[int(chunk[u"id"])][compnoun_tail_id+1][1]!=u"接続助詞":
                            Particle = tokinfo[int(chunk[u"id"])][compnoun_tail_id+1][6]

                            Noun_suru = u""
                            for tok_id_link, tok_link_mor in enumerate(tokinfo[int(chunk[u"link"])]):
                                if tok_link_mor[0]==u"名詞" and tok_link_mor[1]!=u"形容動詞語幹":
                                    Noun_suru += sentence_tok[int(chunk[u"link"])][tok_id_link]
                                    continue
                                if tok_link_mor[0]==u"動詞" or tok_link_mor[0]==u"形容詞" or tok_link_mor[1]==u"形容動詞語幹":                                                                     
                                    if tok_link_mor[1]!=u"末尾":
                                        Verb = u""
                                        if tok_link_mor[6]==u"する":
                                            Verb = Noun_suru+u"する"
                                        else:
                                            Verb = tok_link_mor[6]   

                                        Verb_id = int(chunk[u"link"])
                                        triple_perR.append((Noun, Particle, Verb, Verb_id))
                                        print Noun, Particle, Verb
                                        break

            triplelist[i] = triple_perR
        return triplelist

def TNoun_extract(tripleFrame, NVClass):
    TN_class = TWordclass()
    triple_Treport = [[]] * len(tripleFrame.columns)
    for id, noun, particle, verb, verb_id in zip(tripleFrame.head(5).index, tripleFrame.head(5)[u"名詞"],
                                                 tripleFrame.head(5)[u"助詞"], tripleFrame.head(5)[u"動詞"],
                                                 tripleFrame.head(5)[u"動詞_id"]):
        Lan = Language(noun)
        outList = Lan.getMorpheme()
        Tneed = False

        for nounMor in outList:
            if nounMor[0] in Dc.NV_class[0].keys():
                print nounMor[0],
                for Nclass in Dc.NV_class[0][nounMor[0]]:
                    print Nclass,
                    if Nclass in TN_class.Nounclass_all:
                        Tneed = True
                    elif Nclass in TN_class.TNounclass_Nopart.keys():
                        Tneed = True
                    elif Nclass in TN_class.TNounclass_part.keys():
                        Tneed = True
                    else:
                        continue
        if Tneed:
            triple_Treport[0].append(id)
            triple_Treport[1].append(noun)
            triple_Treport[2].append(particle)
            triple_Treport[3].append(verb)
            triple_Treport[4].append(verb_id)
            print
        triple_Treportdict = {
            u"id": triple_Treport[0],
            tripleFrame.columns[0]: triple_Treport[1],
            tripleFrame.columns[1]: triple_Treport[2],
            tripleFrame.columns[2]: triple_Treport[3],
            tripleFrame.columns[3]: triple_Treport[4],
        }
    triple_TreportFrame = DataFrame(triple_Treportdict,
                                    columns=[u"id", tripleFrame.columns[0], tripleFrame.columns[1],
                                             tripleFrame.columns[2], tripleFrame.columns[3],tripleFrame.columns[4]])

if __name__ =="__main__":
        path = u'C:/Users/ide/Desktop/データ関連/診断考察/report_data_ver4_1.xlsx'
        path = u'D:/研究/データ/report_data_ver4_1.xlsx'
        triplelist = Triple_extract(path)
        Nounlist = []
        Particlelist = []
        Verblist = []
        Verb_idlist = []
        idlist = []
        for Rindex in triplelist:
            for triple in triplelist[Rindex]:
                #print triple[0], triple[1], triple[2]
                Nounlist.append(triple[0])
                Particlelist.append(triple[1])
                Verblist.append(triple[2])
                Verb_idlist.append(triple[3])
                idlist.append(Rindex)
        tripleFrame = DataFrame({u"名詞":Nounlist, u"助詞":Particlelist, u"動詞":Verblist, u"動詞_id":Verb_idlist, u"id":idlist}, columns=[u"id", u"名詞", u"助詞", u"動詞", u"動詞_id"])
        tripleFrame.set_index(u"id", inplace=True)
        tripleFrame.to_csv("D:/tmp/Treport/Triple.csv", encoding='shift-jis')
                
        from Deepcase import Deepcase
        netpath="D:/tmp/Evaluation/neural_network/neuron7/Trained.Network"
        dummylistpath = "D:/tmp/Evaluation/dummylist.Word"
        NV_classpath="D:/tmp/Evaluation/NV_class.Word"
        Dc = Deepcase(netpath, dummylistpath, NV_classpath)
        '''
        unNoun_path = u"C:/tmp/Treport/unregistered_NounsClass.csv"
        unVerb_path = u"C:/tmp/Treport/unregistered_Verbs(bunruidb)Class.csv"
        Dc.unregistered_words(unNoun_path, unVerb_path)
        
        #bunruidb_path = u"C:/研究/bunruidb.txt"
        bunruidb_path = u"C:/Users/ide/Desktop/データ関連/bunruidb/bunruidb.txt"
        bun_Vthe_path = u"C:/tmp/Treport/bunruidb_Vthesaurus.csv"
        Dc.buruidb_verbs(bunruidb_path, bun_Vthe_path)
        '''    

        Result_input = []
        Result_output = []
        DeepCaseList = [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"]

        DeepCase_Noun_perV = ["\0", "\0", "\0", "\0", "\0", "\0", "\0"]
        SurfaceCase_Noun_perV = ["\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0"]

        DeepCase_Noun = [[], [], [], [], [], [], []]
        SurfaceCase_Noun = [[], [], [], [], [], [], [], [], [], [], [], []]
        Verb_target = []
        Verb_target_id = [[],[]]

        for Report_id in range(1, tripleFrame.index[len(tripleFrame)-1]+1):

            tripleFrame_sort = tripleFrame.ix[Report_id,:].sort(u"動詞_id")
            Verb_id_pre = tripleFrame_sort.head(1)[u"動詞_id"].values[0]
            for index_perR, (Noun, Verb, Particle, Verb_id) in enumerate(zip(tripleFrame_sort[u"名詞"], tripleFrame_sort[u"動詞"], tripleFrame_sort[u"助詞"], tripleFrame_sort[u"動詞_id"])):

                print Report_id, Verb_id
                Result = Dc.predict(Noun, Particle, Verb)
                DeepCase_unique = Dc.identify(Result)
                print Noun, Particle, Verb, DeepCase_unique

                if Verb_id != Verb_id_pre or index_perR == len(tripleFrame_sort):
                    Verb_target.append(Verb)
                    Verb_target_id[0].append(Report_id)
                    Verb_target_id[1].append(Verb_id_pre)
                    for Di in range(len(DeepCaseList)):
                        DeepCase_Noun[Di].append(DeepCase_Noun_perV[Di])
                    for Si in range(len(Dc.dummylist[2].columns)):
                        SurfaceCase_Noun[Si].append(SurfaceCase_Noun_perV[Si])

                    DeepCase_Noun_perV = ["\0", "\0", "\0", "\0", "\0", "\0", "\0"]
                    SurfaceCase_Noun_perV = ["\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0"]


                DeepCase_Noun_perV[DeepCaseList.index(DeepCase_unique)] = Noun
                if Particle in list(Dc.dummylist[2].columns):
                    SurfaceCase_Noun_perV[list(Dc.dummylist[2].columns).index(Particle)] = Noun

                Verb_id_pre = Verb_id

        case_frame={
            u"報告書_id": Verb_target_id[0], u"動詞_id": Verb_target_id[1], u"動詞": Verb_target,
                DeepCaseList[0]: DeepCase_Noun[0], DeepCaseList[1]: DeepCase_Noun[1], DeepCaseList[2]: DeepCase_Noun[2], DeepCaseList[3]: DeepCase_Noun[3], DeepCaseList[4]: DeepCase_Noun[4], DeepCaseList[5]:  DeepCase_Noun[5],DeepCaseList[6]: DeepCase_Noun[6],
                Dc.dummylist[2].columns[0]: SurfaceCase_Noun[0], Dc.dummylist[2].columns[1]: SurfaceCase_Noun[1], Dc.dummylist[2].columns[2]: SurfaceCase_Noun[2], Dc.dummylist[2].columns[3]: SurfaceCase_Noun[3], Dc.dummylist[2].columns[4]: SurfaceCase_Noun[4], Dc.dummylist[2].columns[5]: SurfaceCase_Noun[5], Dc.dummylist[2].columns[6]: SurfaceCase_Noun[6], Dc.dummylist[2].columns[7]: SurfaceCase_Noun[7], Dc.dummylist[2].columns[8]: SurfaceCase_Noun[8], Dc.dummylist[2].columns[9]: SurfaceCase_Noun[9], Dc.dummylist[2].columns[10]: SurfaceCase_Noun[10], Dc.dummylist[2].columns[11]: SurfaceCase_Noun[11]
                    }

        case_df = DataFrame(case_frame,
        columns=[
        u"報告書_id", u"動詞_id", u"動詞",
        DeepCaseList[0], DeepCaseList[1], DeepCaseList[2], DeepCaseList[3], DeepCaseList[4], DeepCaseList[5],DeepCaseList[6],
        Dc.dummylist[2].columns[0], Dc.dummylist[2].columns[1], Dc.dummylist[2].columns[2], Dc.dummylist[2].columns[3], Dc.dummylist[2].columns[4], Dc.dummylist[2].columns[5], Dc.dummylist[2].columns[6], Dc.dummylist[2].columns[7], Dc.dummylist[2].columns[8], Dc.dummylist[2].columns[9], Dc.dummylist[2].columns[10], Dc.dummylist[2].columns[11]
        ])

        case_df.to_csv(u"C:/tmp/Treport/caseframe.csv", encoding='shift-jis', index=False)

        '''
                for r in Result:
                    Result_input.append(r[0]+(Report_id, Verb_id))
                    Result_output.append(r[1])

        inputFrame = DataFrame(Result_input, columns=[u"名詞", u"動詞", u"名詞クラス", u"動詞クラス", u"助詞", u"id", u"動詞_id"])
        outputFrame = DataFrame(Result_output, columns=[u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"])
        inputFrame.to_csv(u"D:/tmp/Treport/inputFrame.csv",encoding='shift-jis')
        outputFrame.to_csv(u"D:/tmp/Treport/outputFrame.csv",encoding='shift-jis')
        '''