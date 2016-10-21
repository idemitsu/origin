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

def TNoun_extract(tripleFrame, NV_class):
    TW = TWordclass()
    unify_particle= lambda x: TW.Particle_to.get(x, x)
    tripleFrame[u"助詞"] = tripleFrame[u"助詞"].map(unify_particle)

    triple_Treport = [[] for i in range(len(tripleFrame.columns) + 1)]
    for id, noun, particle, verb, verb_id in zip(tripleFrame.index, tripleFrame[u"名詞"],
                                                 tripleFrame[u"助詞"], tripleFrame[u"動詞"],
                                                 tripleFrame[u"動詞_id"]):
        print "Extract triple_Treport:", id, verb_id
        Lan = Language(noun)
        outList = Lan.getMorpheme()
        Mor_1 = [outList[i][1] for i in range(len(outList))]
        Mor_2 = [outList[i][2] for i in range(len(outList))]

        TNneed = False
        TVneed = False

        #トライボロジーに関係する名詞か判定
        for nounMor in outList:
            if nounMor[2] == u"代名詞" or nounMor[1]==u"連体詞":
                TNneed = True
                break
            if nounMor[0] in NV_class[0].keys():
                for Nclass in NV_class[0][nounMor[0]]:
                    if Nclass in TW.TNounclass_all:
                        TNneed = True
                        break
                    elif Nclass in TW.TNounclass_Nopart.keys():
                        TNneed = True
                        for TNoun_Nopart in TW.TNounclass_Nopart[Nclass]:
                            if TNoun_Nopart in nounMor[0]:
                                TNneed = False
                            elif Nclass==u"様相" and nounMor[2]==u"形容動詞語幹":
                                TNneed = False

                    elif Nclass in TW.TNounclass_part.keys():
                        for TNoun_Nopart in TW.TNounclass_part[Nclass]:
                            if TNoun_Nopart in nounMor[0]:
                                TNneed = True
                                break
                    else:
                        continue
        #トライボロジーに関係する動詞か判定

        if TNneed:
            if verb in NV_class[1].keys():
                for Vclass in NV_class[1][verb]:
                    if Vclass in TW.TVerbclass_all:
                        TVneed = True
                        break
                    elif Vclass in TW.TVerbclass_Nopart.keys():
                        TVneed = True
                        for TVerb_Nopart in TW.TVerbclass_Nopart[Vclass]:
                            if TVerb_Nopart in Verb:
                                TVneed = False

                    elif Vclass in TW.TVerbclass_part.keys():
                        for TVerb_Nopart in TW.TVerbclass_part[Vclass]:
                            if TVerb_Nopart in verb:
                                TVneed = True
                                break
                    else:
                        continue





        if TNneed and TVneed:
            triple_Treport[0].append(id)
            triple_Treport[1].append(noun)
            triple_Treport[2].append(particle)
            triple_Treport[3].append(verb)
            triple_Treport[4].append(verb_id)

    triple_Treportdict = {
        u"id": triple_Treport[0],
        tripleFrame.columns[0]: triple_Treport[1],
        tripleFrame.columns[1]: triple_Treport[2],
        tripleFrame.columns[2]: triple_Treport[3],
        tripleFrame.columns[3]: triple_Treport[4],
    }
    tripleFrame_Treport = DataFrame(triple_Treportdict,
                                    columns=[u"id", tripleFrame.columns[0], tripleFrame.columns[1],
                                             tripleFrame.columns[2], tripleFrame.columns[3]])

    return tripleFrame_Treport


if __name__ =="__main__":
        #全トリプルの抽出
        '''
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
        '''
        #トリプルから事象の抽出

        from Deepcase import Deepcase
        netpath="C:/tmp/Evaluation/neural_network/neuron7/Trained.Network"
        dummylistpath = "C:/tmp/Evaluation/dummylist.Word"
        NV_classpath="C:/tmp/Evaluation/NV_class.Word"
        Dc = Deepcase(netpath, dummylistpath, NV_classpath)
        #'''
        tripleFrame = pd.read_csv("C:/tmp/Treport/Triple.csv", encoding='shift-jis', index_col=u'id')
        tripleFrame_Treport = TNoun_extract(tripleFrame, Dc.NV_class)
        tripleFrame_Treport.set_index(u"id", inplace=True)
        tripleFrame_Treport.to_csv("C:/tmp/Treport/Triple_Treport.csv", encoding='shift-jis')
        #'''
        #未登録語の登録
        '''
        unNoun_path = u"C:/tmp/Treport/unregistered_NounsClass.csv"
        unVerb_path = u"C:/tmp/Treport/unregistered_Verbs(bunruidb)Class.csv"
        Dc.unregistered_words(unNoun_path, unVerb_path)
        
        #bunruidb_path = u"C:/研究/bunruidb.txt"
        bunruidb_path = u"C:/Users/ide/Desktop/データ関連/bunruidb/bunruidb.txt"
        bun_Vthe_path = u"C:/tmp/Treport/bunruidb_Vthesaurus.csv"
        Dc.buruidb_verbs(bunruidb_path, bun_Vthe_path)
        '''    
        #格フレームの構築
        tripleFrame_Treport = pd.read_csv("C:/tmp/Treport/Triple_Treport.csv", encoding='shift-jis', index_col=u'id')
        Result_input = []
        Result_output = []
        DeepCaseList = [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"]

        DeepCase_Noun_perV = ["\0" for i in range(len(DeepCaseList)+1)]
        SurfaceCase_Noun_perV = ["\0" for i in range(len(Dc.dummylist[2].columns)+1)]

        DeepCase_Noun = [[] for i in range(len(DeepCaseList)+1)]
        SurfaceCase_Noun = [[] for i in range(len(Dc.dummylist[2].columns)+1)]
        Verb_target = []
        Verb_target_id = [[], []]

        for Report_id in range(1, tripleFrame_Treport.index[len(tripleFrame_Treport)-1]+1):

            tripleFrame_Treport_sort = tripleFrame_Treport.ix[Report_id,:].sort(u"動詞_id")
            tripleFrame_Treport_sort.reset_index(inplace=True)
            Verb_id_pre = tripleFrame_Treport_sort.head(1)[u"動詞_id"].values[0]
            Verb_id_first = tripleFrame_Treport_sort.head(2)[u"動詞_id"].values[1]
            for index_perR, (Noun, Verb, Particle, Verb_id) in enumerate(zip(tripleFrame_Treport_sort[u"名詞"], tripleFrame_Treport_sort[u"動詞"], tripleFrame_Treport_sort[u"助詞"], tripleFrame_Treport_sort[u"動詞_id"])):
                print Report_id, Verb_id
                Result = Dc.predict(Noun, Particle, Verb)
                DeepCase_unique = Dc.identify(Result)
                print Noun, Particle, Verb, DeepCase_unique

                DeepCase_Noun_perV[DeepCaseList.index(DeepCase_unique)] = Noun
                if Particle in list(Dc.dummylist[2].columns):
                    SurfaceCase_Noun_perV[list(Dc.dummylist[2].columns).index(Particle)] = Noun

                if index_perR < len(tripleFrame_Treport_sort)-1:
                    if (Verb_id < tripleFrame_Treport_sort.ix[index_perR+1, u"動詞_id"]) or (index_perR == 0 and Verb_id_pre != Verb_id_first):
                        Verb_target.append(Verb)
                        Verb_target_id[0].append(Report_id)
                        Verb_target_id[1].append(Verb_id)
                        for Di in range(len(DeepCaseList)):
                            DeepCase_Noun[Di].append(DeepCase_Noun_perV[Di])
                        for Si in range(len(Dc.dummylist[2].columns)):
                            SurfaceCase_Noun[Si].append(SurfaceCase_Noun_perV[Si])

                        DeepCase_Noun_perV = ["\0" for i in range(len(DeepCaseList))]
                        SurfaceCase_Noun_perV = ["\0" for i in range(len(Dc.dummylist[2].columns))]
                else:
                    Verb_target.append(Verb)
                    Verb_target_id[0].append(Report_id)
                    Verb_target_id[1].append(Verb_id)
                    for Di in range(len(DeepCaseList)):
                        DeepCase_Noun[Di].append(DeepCase_Noun_perV[Di])
                    for Si in range(len(Dc.dummylist[2].columns)):
                        SurfaceCase_Noun[Si].append(SurfaceCase_Noun_perV[Si])

                    DeepCase_Noun_perV = ["\0" for i in range(len(DeepCaseList))]
                    SurfaceCase_Noun_perV = ["\0" for i in range(len(Dc.dummylist[2].columns))]

                Verb_id_pre = Verb_id

        cf_list =[
        (u"報告書_id", Verb_target_id[0]), (u"動詞_id", Verb_target_id[1]), (u"動詞", Verb_target)
        ]
        cf_list.extend([(DeepCaseList[i], DeepCase_Noun[i]) for i in range(len(DeepCaseList))])
        cf_list.extend([(Dc.dummylist[2].columns[i], SurfaceCase_Noun[i]) for i in range(len(Dc.dummylist[2].columns))])

        case_frame=dict(cf_list)

        cd_columns = [u"報告書_id", u"動詞_id", u"動詞"]
        cd_columns.extend([DeepCaseList[i] for i in range(len(DeepCaseList))])
        cd_columns.extend([Dc.dummylist[2].columns[i] for i in range(len(Dc.dummylist[2].columns))])

        case_df = DataFrame(case_frame, columns=cd_columns)

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