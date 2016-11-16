# -*- coding: utf-8 -*-
'''
import sys
sys.path.append("D:\Users\ide\Dropbox\python")

sys.modules.pop('neural_network') #アンインポート
'''

from Treport import Treport
from Language import Language
from pandas import DataFrame, Series
from TWordclass import TWordclass
import pandas as pd
import nltk
import itertools
from collections import Counter
import math
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
            if i>1000:
                break
            print i
            noenc = TR.delete_unnecc(i)
           # TR.s.cell_value(i, 2)
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
                                if sentence_tok[int(chunk[u"id"])][tok_id_noun] == u"濃度":
                                    continue
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
                                from_tail_tok = tokinfo[int(chunk[u"id"])+i_from][len(tokinfo[int(chunk[u"id"])+i_from])-1]
                                if from_tail_tok[1]==u"連体化" or from_tail_tok[0]==u"形容詞" or from_tail_tok[0]==u"連体詞" or from_tail_tok[0]==u"助動詞" or [1]==u"並立助詞" or from_tail_tok[0]==u"接続詞" or from_tail_tok[1]==u"読点" or from_tail_tok[1]==u"接続助詞":
                                    for sentence_tok_from in reversed(list(sentence_tok[int(chunkinfo[int(chunk[u"id"])+i_from]["id"])])):                                                                            
                                        Noun = sentence_tok_from + Noun
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
                                        if tok_link_mor[6]==u"する" or tok_link_mor[6]==u"できる":
                                            Verb = Noun_suru+u"する"
                                        else:
                                            Verb = tok_link_mor[6]   

                                        Verb_id = int(chunk[u"link"])

                                        triple_perR.append((Noun, Particle, Verb, Verb_id))
                                        #print Noun, Particle, Verb
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
        if id >500: break

        #print noun, particle, verb
        print "Extract triple_Treport:", id, verb_id
        Lan = Language(noun)
        outList = Lan.getMorpheme()
        Mor_1 = [outList[i][1] for i in range(len(outList))]
        Mor_2 = [outList[i][2] for i in range(len(outList))]
        noun_comp_tmp = u""
        noun_comp = []
        noun_tail = []
        noun_Pos1 = []
        noun_Pos2 = []
        for mi, Pos in enumerate(Mor_1):
            if mi==len(Mor_1)-1:
                noun_comp.append(noun_comp_tmp + outList[mi][0])
                noun_tail.append(outList[mi][0])
                noun_Pos1.append(outList[mi][1])
                noun_Pos2.append(outList[mi][2])
                break
            if Pos==u"名詞":
                if Mor_1[mi+1]==u"名詞":
                    noun_comp_tmp += outList[mi][0]
                else:
                    noun_comp.append(noun_comp_tmp+outList[mi][0])
                    noun_tail.append(outList[mi][0])
                    noun_Pos1.append(outList[mi][1])
                    noun_Pos2.append(outList[mi][2])
                    noun_comp_tmp = u""


        TNneed = False
        TVneed = False

        #トライボロジーに関係する名詞か判定
        for cni, nounMor in enumerate(noun_comp):
            if noun_Pos2[cni] == u"代名詞" or noun_Pos1[cni]==u"連体詞":
                TNneed = True
                break
            if nounMor in NV_class[0].keys():
                noun_target = nounMor
            elif noun_tail[cni] in NV_class[0].keys():
                noun_target = noun_tail[cni]
            else:
                continue

            for Nclass in NV_class[0][noun_target]:
                if Nclass in TW.TNounclass_all:
                    TNneed = True
                    break
                elif Nclass in TW.TNounclass_Nopart.keys():
                    TNneed = True
                    for TNoun_Nopart in TW.TNounclass_Nopart[Nclass]:
                        if TNoun_Nopart in noun_target:
                            TNneed = False

                        elif Nclass==u"様相" and noun_Pos2[cni]==u"形容動詞語幹":
                            TNneed = False


                elif Nclass in TW.TNounclass_part.keys():
                    for TNoun_part in TW.TNounclass_part[Nclass]:
                        if TNoun_part in noun_target:
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
                            if TVerb_Nopart in verb:
                                TVneed = False

                    elif Vclass in TW.TVerbclass_part.keys():
                        for TVerb_part in TW.TVerbclass_part[Vclass]:
                            if TVerb_part in verb:
                                TVneed = True
                                break
                    else:
                        continue

        if TNneed and TVneed:
            #並列している名詞の分解
            Mor_connect = [[u"接続詞"],[u"読点", u"並立助詞", u"接続助詞"]]
            No_con = True
            for conList in Mor_connect:
                for con in conList:
                    if (con in Mor_1) or (con in Mor_2):
                        noun_con = u""
                        for oi, out in enumerate(outList):
                            if (outList[oi][1] != con) and (outList[oi][2] != con):
                                if out[0]!=u"等":
                                    noun_con += out[0]
                            else:
                                triple_Treport[0].append(id)
                                triple_Treport[1].append(noun_con)
                                triple_Treport[2].append(particle)
                                triple_Treport[3].append(verb)
                                triple_Treport[4].append(verb_id)
                                noun_con = u""
                                No_con = False
                            if oi==len(outList)-1:
                                triple_Treport[0].append(id)
                                triple_Treport[1].append(noun_con)
                                triple_Treport[2].append(particle)
                                triple_Treport[3].append(verb)
                                triple_Treport[4].append(verb_id)
                                noun_con = u""
                                No_con = False

            if No_con:
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

    fvu = lambda x: TW.Verb_unify.get(x, x)
    tripleFrame_Treport[u"動詞"] = tripleFrame_Treport[u"動詞"].map(fvu)
    return tripleFrame_Treport

def extract_terms(self, case_df):
    Noun_comp = u""
    wakachi = u""
    preR_id = 1
    terms = []
    documents = []
    for (Report_id, frame) in zip(case_df[u"報告書_id"], case_df.loc[:, [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係", u"動詞"]].values):
        if preR_id != Report_id:
            documents.append(wakachi)
            print wakachi
            wakachi = u""
        if frame[7][-2:] != u"する":
            wakachi += frame[7] + u" "
            if frame[7] not in terms: terms.append(frame[7])
        else:
            wakachi += frame[7][:-2] + u" "
            if frame[7][:-2] not in terms: terms.append(frame[7][:-2])
        for i in range(0, 7):
            if frame[i] == u' ':
                continue
            Lan = Language(frame[i])
            outList = Lan.getMorpheme()
            Mor_1 = [outList[i][1] for i in range(len(outList))]
            # if (u"接続詞" in Mor_1) | (u"記号" in Mor_1):
            #    continue
            for mi, Mor in enumerate(outList):
                if Mor_1[mi] == u"名詞" and Mor[2] != u"形容動詞語幹":
                    Noun_comp += Mor[0]
                    if mi < len(Mor_1) - 1:
                        if Mor_1[mi + 1] != u"名詞":
                            wakachi += Noun_comp + u" "
                            if Noun_comp not in terms: terms.append(Noun_comp)
                            Noun_comp = u""
                    else:
                        wakachi += Noun_comp + u" "
                        if Noun_comp not in terms: terms.append(Noun_comp)
                        Noun_comp = u""
                elif Mor_1[mi] != u"助詞" and Mor_1[mi] != u"助動詞" and Mor[5] != u"サ変・スル" and Mor[2] != u"接尾":
                    wakachi += Mor[0] + u" "
                    if Mor[0] not in terms: terms.append(Mor[0])

        preR_id += 1
    documents.append(wakachi)
    return terms, documents

def tf(self, terms, document):
    #TF値の計算。単語リストと文章を渡す
    tf_values = [document.count(term) for term in terms]
    return list(map(lambda x: x/sum(tf_values), tf_values))

def idf(self, terms, documents):
    #IDF値の計算。単語リストと全文章を渡す
    return [math.log10(len(documents)/sum([bool(term in document) for document in documents])) for term in terms]

def tf_idf(self, terms, documents):
    #TF-IDF値を計算。文章毎にTF-IDF値を計算
    return [[_tf*_idf for _tf, _idf in zip(tf(terms, document), idf(terms, documents))] for document in documents]


def bunrui_frame(self, case_df, terms, idf_Treport):
    MorList = []
    Noun_comp = u""
    Noun_weight = 2.0
    case_df[u"事象"] = case_df[u"主体"] + case_df[u"起点"] + case_df[u"対象"] + case_df[u"状況"] + case_df[u"着点"] + case_df[
        u"手段"] + case_df[u"関係"] + case_df[u"動詞"]

    for frame in case_df.loc[:, [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係", u"動詞"]].head(
            500).drop_duplicates().values:
        if frame[7][-2:] != u"する":
            MorList_tmp = {frame[7]: idf_Treport[terms.index(frame[7])]}
        else:
            MorList_tmp = {frame[7][:-2]: idf_Treport[terms.index(frame[7][:-2])]}
        for i in range(0, 7):
            if frame[i] == u' ':
                continue
            Lan = Language(frame[i])
            outList = Lan.getMorpheme()
            Mor_1 = [outList[i][1] for i in range(len(outList))]
            # if (u"接続詞" in Mor_1) | (u"記号" in Mor_1):
            #    continue
            Nocomp = True
            for mi, Mor in enumerate(outList):
                if Mor_1[mi] == u"名詞" and Mor[2] != u"形容動詞語幹":
                    Noun_comp += Mor[0]
                    if mi < len(Mor_1) - 1:
                        if Mor_1[mi + 1] != u"名詞":
                            MorList_tmp[Noun_comp] = idf_Treport[terms.index(Noun_comp)] * Noun_weight
                            Noun_comp = u""
                            Nocomp=True

                        else:
                            Nocomp =False
                    else:
                        if Nocomp and Mor[2] == u"サ変接続":
                            MorList_tmp[Noun_comp] = idf_Treport[terms.index(Noun_comp)]
                            Noun_comp = u""
                        else:
                            MorList_tmp[Noun_comp] = idf_Treport[terms.index(Noun_comp)] * Noun_weight
                            Noun_comp = u""

                elif Mor_1[mi] != u"助詞" and Mor_1[mi] != u"助動詞" and Mor[5] != u"サ変・スル" and Mor[2] != u"接尾":
                    MorList_tmp[Mor[0]] = idf_Treport[terms.index(Mor[0])]

        MorList.append(MorList_tmp)

    #caseFrame = case_df[u"主体"] + u" " + case_df[u"起点"] + u" " + case_df[u"対象"] + u" " + case_df[u"状況"] + u" " + case_df[
     #   u"着点"] + u" " + case_df[u"手段"] + u" " + case_df[u"関係"] + u" " + case_df[u"動詞"]
    cf = [i for i in case_df[u"事象"].drop_duplicates()]
    '''
    #包含関係による統一辞書の作成
    unifyList = {}
    for i, x in enumerate(MorList):
        for j, y in enumerate(MorList[i + 1:]):
            j = i + j + 1
            if (x.keys() in unifyList.keys()) | (y.keys() in unifyList.keys()):
                continue
            if bool(set(x.keys()).intersection(set(y.keys()))) and cf[i] != cf[j]:
                sym = set(x.keys()).symmetric_difference(set(y.keys()))
                if len(sym) < 1:
                    # print "%d:%s" %(i,cf[i]), "%d:%s" %(j,cf[j])
                    # print outList[len(outList)-1][0], outList[len(outList)-1][1]
                    continue
                Lan = Language(list(sym)[len(sym) - 1])
                outList = Lan.getMorpheme()
                if (outList[len(outList) - 1][1] != u"名詞" or outList[len(outList) - 1][2] == u"サ変接続" or
                            outList[len(outList) - 1][2] == u"形容動詞語幹") and len(outList) == 1:
                    if set(x.keys()).issubset(set(y.keys())):
                        unifyList[cf[j]] = cf[i]
                        print "%d:%s" % (i, cf[i]), "%d:%s" % (j, cf[j])
                        print outList[len(outList) - 1][0], outList[len(outList) - 1][1], outList[len(outList) - 1][2]

                    elif set(y.keys()).issubset(set(x.keys())):
                        unifyList[cf[i]] = cf[j]
                        print "%d:%s" % (i, cf[i]), "%d:%s" % (j, cf[j])
                        print outList[len(outList) - 1][0], outList[len(outList) - 1][1], outList[len(outList) - 1][2]

                        # print "%s<>%s:%f" % (cf[i],cf[j],nltk.distance.jaccard_distance(set(s),set(t)))
    '''
    #Jaccard係数による統一辞書の作成
    Wdist_index = []
    Wdist_column = []
    Wdist = []
    unifyList = {}
    Case_freq = Counter(case_df[u"事象"])
    # 各文の動詞が反対語リストに含まれていれば分類しない
    oppositeList = [u"良好", u"正常", u"低下する"]

    for i, x in enumerate(MorList):
        for j, y in enumerate(MorList[i + 1:]):
            j = i + j + 1
            if bool(set(oppositeList).intersection(set(x.keys()))) and not(bool(set(y.keys()).issuperset(set(oppositeList).intersection(set(x.keys()))))):
                continue
            elif bool(set(oppositeList).intersection(set(y.keys()))) and not(bool(set(x.keys()).issuperset(set(oppositeList).intersection(set(y.keys()))))):
                continue

            if (x.keys() in unifyList.keys()) | (y.keys() in unifyList.keys()):
                continue
            if bool(set(x.keys()).intersection(set(y.keys()))) and cf[i] != cf[j]:
                sym = set(x.keys()).symmetric_difference(set(y.keys()))
                if len(sym) < 1:
                    # print "%d:%s" %(i,cf[i]), "%d:%s" %(j,cf[j])
                    # print outList[len(outList)-1][0], outList[len(outList)-1][1]
                    continue
                Lan = Language(list(sym)[len(sym) - 1])
                outList = Lan.getMorpheme()
                if (outList[len(outList) - 1][1] != u"名詞" or outList[len(outList) - 1][2] == u"サ変接続" or
                            outList[len(outList) - 1][2] == u"形容動詞語幹") and len(outList) == 1:
                    xy_set = dict(x.items() + y.items())

                    xy_insec = set(x.keys()).intersection(set(y.keys()))
                    w_all = 0.00
                    for mor_val in xy_set.values():
                        w_all += mor_val
                    w_insec = 0.00
                    for mor_val in xy_insec:
                        w_insec += xy_set[mor_val]
                    dist_str = w_insec / w_all
                    if dist_str > 0.3:
                        #もしくは、頻度が高い格フレーム
                        if Case_freq[cf[j]] <= Case_freq[cf[i]] and cf[i] not in unifyList.keys():
                            unifyList[cf[i]] = cf[j]
                        elif Case_freq[cf[j]] > Case_freq[cf[i]] and cf[j] not in unifyList.keys():
                            unifyList[cf[j]] = cf[i]
                        '''
                        #形態素数が少ない格フレーム：形態素数が多い格フレーム
                        if len(set(x.keys())) < len(set(y.keys())) and cf[i] not in unifyList.keys():
                            unifyList[cf[i]] = cf[j]
                        elif len(set(x.keys())) > len(set(y.keys())) and cf[j] not in unifyList.keys():
                            unifyList[cf[j]] = cf[i]
                        '''
                        print "%d:%s" % (i, cf[i]), "%d:%s" % (j, cf[j]), dist_str, w_insec, w_all
                        print outList[len(outList) - 1][0], outList[len(outList) - 1][1], outList[len(outList) - 1][2]
                        Wdist_index.append(cf[i])
                        Wdist_column.append(cf[j])
                        Wdist.append(dist_str)

    Wdist_mat = Series(Wdist, index=[Wdist_index, Wdist_column]).unstack()
    Wdist = DataFrame(Wdist, index=[Wdist_index, Wdist_column], columns=[u"Similarity"])

    fnc = lambda x: unifyList.get(x, x)
    case_df[u"事象"] = case_df[u"事象"].map(fnc)
    return case_df


if __name__ =="__main__":
        #全トリプルの抽出
        '''
        path = u'D:/Users/ide/Desktop/データ関連/診断考察/report_data_ver4_1.xlsx'
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
        netpath="D:/tmp/Evaluation/neural_network/neuron7/Trained.Network"
        dummylistpath = "D:/tmp/Evaluation/dummylist.Word"
        NV_classpath="D:/tmp/Evaluation/NV_class.Word"
        Dc = Deepcase(netpath, dummylistpath, NV_classpath)
        '''
        tripleFrame = pd.read_csv("D:/tmp/Treport/Triple.csv", encoding='shift-jis', index_col=u'id')
        tripleFrame_Treport = TNoun_extract(tripleFrame, Dc.NV_class)
        tripleFrame_Treport.set_index(u"id", inplace=True)
        tripleFrame_Treport.to_csv("D:/tmp/Treport/Triple_Treport.csv", encoding='shift-jis')
        '''
        #未登録語の登録
        '''
        unNoun_path = u"D:/tmp/Treport/unregistered_NounsClass.csv"
        unVerb_path = u"D:/tmp/Treport/unregistered_Verbs(bunruidb)Class.csv"
        Dc.unregistered_words(unNoun_path, unVerb_path)
        
        #bunruidb_path = u"D:/研究/bunruidb.txt"
        bunruidb_path = u"D:/Users/ide/Desktop/データ関連/bunruidb/bunruidb.txt"
        bun_Vthe_path = u"D:/tmp/Treport/bunruidb_Vthesaurus.csv"
        Dc.buruidb_verbs(bunruidb_path, bun_Vthe_path)
        '''
        #格フレームの構築
        tripleFrame_Treport = pd.read_csv("D:/tmp/Treport/Triple_Treport.csv", encoding='shift-jis', index_col=u'id')
        Result_input = []
        Result_output = []
        DeepCaseList = [u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"]

        DeepCase_Noun_perV = [[] for i in range(len(DeepCaseList)+1)]
        #SurfaceCase_Noun_perV = [[] for i in range(len(Dc.dummylist[2].columns)+1)]

        DeepCase_Noun = [[] for i in range(len(DeepCaseList)+1)]
        #SurfaceCase_Noun = [[] for i in range(len(Dc.dummylist[2].columns)+1)]
        Verb_target = []
        Verb_target_id = [[], []]

        for Report_id in range(1, tripleFrame_Treport.index[len(tripleFrame_Treport)-1]+1):
            if list(tripleFrame_Treport.index).count(Report_id)<=1:
                continue
            tripleFrame_Treport_sort = tripleFrame_Treport.ix[Report_id,:].sort(u"動詞_id")
            tripleFrame_Treport_sort.reset_index(inplace=True)
            Verb_id_pre = tripleFrame_Treport_sort.head(1)[u"動詞_id"].values[0]
            Verb_id_first = tripleFrame_Treport_sort.head(2)[u"動詞_id"].values[1]
            for index_perR, (Noun, Verb, Particle, Verb_id) in enumerate(zip(tripleFrame_Treport_sort[u"名詞"], tripleFrame_Treport_sort[u"動詞"], tripleFrame_Treport_sort[u"助詞"], tripleFrame_Treport_sort[u"動詞_id"])):

                if Noun!=Noun:
                    continue
                print Report_id, Verb_id
                Result = Dc.predict(Noun, Particle, Verb)
                DeepCase_unique = Dc.identify(Result)
                print Noun, Particle, Verb, DeepCase_unique

                DeepCase_Noun_perV[DeepCaseList.index(DeepCase_unique)].append(Noun)
                #if Particle in list(Dc.dummylist[2].columns):
                #    SurfaceCase_Noun_perV[list(Dc.dummylist[2].columns).index(Particle)] = Noun

                if index_perR < len(tripleFrame_Treport_sort)-1:
                    if (Verb_id < tripleFrame_Treport_sort.ix[index_perR+1, u"動詞_id"]) or (index_perR == 0 and Verb_id_pre != Verb_id_first):
                        while [] in DeepCase_Noun_perV:
                            DeepCase_Noun_perV[DeepCase_Noun_perV.index([])]=[u" "]

                        for DeepCase_Noun_tmp in list(itertools.product(DeepCase_Noun_perV[0], DeepCase_Noun_perV[1], DeepCase_Noun_perV[2], DeepCase_Noun_perV[3], DeepCase_Noun_perV[4], DeepCase_Noun_perV[5], DeepCase_Noun_perV[6])):
                            Verb_target.append(Verb)
                            Verb_target_id[0].append(Report_id)
                            Verb_target_id[1].append(Verb_id)
                            for Di in range(len(DeepCaseList)):
                                DeepCase_Noun[Di].append(DeepCase_Noun_tmp[Di])

                            DeepCase_Noun_perV = [[] for i in range(len(DeepCaseList))]
                            #SurfaceCase_Noun_perV = [[] for i in range(len(Dc.dummylist[2].columns))]
                else:
                    while [] in DeepCase_Noun_perV:
                        DeepCase_Noun_perV[DeepCase_Noun_perV.index([])] = [u" "]

                    for DeepCase_Noun_tmp in list(itertools.product(DeepCase_Noun_perV[0], DeepCase_Noun_perV[1], DeepCase_Noun_perV[2], DeepCase_Noun_perV[3], DeepCase_Noun_perV[4], DeepCase_Noun_perV[5], DeepCase_Noun_perV[6])):
                        Verb_target.append(Verb)
                        Verb_target_id[0].append(Report_id)
                        Verb_target_id[1].append(Verb_id)
                        for Di in range(len(DeepCaseList)):
                            DeepCase_Noun[Di].append(DeepCase_Noun_tmp[Di])
                        #for Si in range(len(Dc.dummylist[2].columns)):
                            #SurfaceCase_Noun[Si].append(SurfaceCase_Noun_perV[Si])

                        DeepCase_Noun_perV = [[] for i in range(len(DeepCaseList))]
                        #SurfaceCase_Noun_perV = [[] for i in range(len(Dc.dummylist[2].columns))]

                Verb_id_pre = Verb_id

        cf_list =[
        (u"報告書_id", Verb_target_id[0]), (u"動詞_id", Verb_target_id[1]), (u"動詞", Verb_target)
        ]
        cf_list.extend([(DeepCaseList[i], DeepCase_Noun[i]) for i in range(len(DeepCaseList))])
        #cf_list.extend([(Dc.dummylist[2].columns[i], SurfaceCase_Noun[i]) for i in range(len(Dc.dummylist[2].columns))])

        case_frame=dict(cf_list)

        cd_columns = [u"報告書_id", u"動詞_id", u"動詞"]
        cd_columns.extend([DeepCaseList[i] for i in range(len(DeepCaseList))])
        #cd_columns.extend([Dc.dummylist[2].columns[i] for i in range(len(Dc.dummylist[2].columns))])

        case_df = DataFrame(case_frame, columns=cd_columns)
        case_df.to_csv(u"D:/tmp/Treport/caseframe.csv", encoding='shift-jis', index=False)

        terms, documents = extract_terms(case_df)
        idf_Treport = idf(terms, documents)

        '''
                for r in Result:
                    Result_input.append(r[0]+(Report_id, Verb_id))
                    Result_output.append(r[1])

        inputFrame = DataFrame(Result_input, columns=[u"名詞", u"動詞", u"名詞クラス", u"動詞クラス", u"助詞", u"id", u"動詞_id"])
        outputFrame = DataFrame(Result_output, columns=[u"主体", u"起点", u"対象", u"状況", u"着点", u"手段", u"関係"])
        inputFrame.to_csv(u"D:/tmp/Treport/inputFrame.csv",encoding='shift-jis')
        outputFrame.to_csv(u"D:/tmp/Treport/outputFrame.csv",encoding='shift-jis')
        '''