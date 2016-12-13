# -*- coding: utf-8 -*-

from Cases_extract import Cases_extract
from pandas import DataFrame
from pandas import Series
import pandas as pd
import itertools
import pickle
import math
from Deepcase import Deepcase
import re

def tf(terms, document):
    # TF値の計算。単語リストと文章を渡す
    tf_values = [document.count(term) for term in terms]
    return list(map(lambda x: x / sum(tf_values), tf_values))


def idf(terms, documents):
    # IDF値の計算。単語リストと全文章を渡す
    return [math.log10(len(documents) / sum([bool(term in document) for document in documents])) for term in terms]


def tf_idf(terms, documents):
    # TF-IDF値を計算。文章毎にTF-IDF値を計算
    return [[_tf * _idf for _tf, _idf in zip(tf(terms, document), idf(terms, documents))] for document in documents]


if __name__ == "__main__":
    # 全トリプルの抽出

    netpath = "D:/tmp/Evaluation/neural_network/neuron7/Trained.Network"
    dummylistpath = "D:/tmp/Evaluation/dummylist.Word"
    NV_classpath = "D:/tmp/Evaluation/NV_class.Word"
    Dc = Deepcase(netpath, dummylistpath, NV_classpath)

    Ce= Cases_extract(Dc)

    #'''
    path = u'D:/Users/ide/Desktop/データ関連/診断考察/report_data_ver4_2.xlsx'
    path = u'D:/研究/データ/report_data_ver4_2.xlsx'
    triplelist = Ce.Triple_extract(path)
    Nounlist = []
    Particlelist = []
    Verblist = []
    Ridlist = []
    Sidlist = []
    Vidlist = []
    for T_keys, T_values in zip(triplelist.keys(), triplelist.values()):
        for tri_tmp in T_values:
            Nounlist.append(tri_tmp[0])
            Particlelist.append(tri_tmp[1])
            Verblist.append(tri_tmp[2])
            Ridlist.append(T_keys[0])
            Sidlist.append(T_keys[1])
            Vidlist.append(T_keys[2])
    tripleFrame = DataFrame({u"報告書_id":Ridlist, u"文_id":Sidlist, u"動詞_id":Vidlist, u"名詞":Nounlist, u"助詞":Particlelist, u"動詞":Verblist}, columns=[u"報告書_id", u"文_id", u"動詞_id", u"名詞", u"助詞", u"動詞"])
    tripleFrame.sort_index(by=[u"報告書_id", u"文_id", u"動詞_id"], inplace=True)
    tripleFrame[u"報告書_id"] = [int(i) for i in tripleFrame[u"報告書_id"]]
    tripleFrame.to_csv("D:/tmp/Treport/Triple.csv", index=False, encoding='shift-jis')
    #'''
    # トリプルから事象の抽出

    #'''
    tripleFrame = pd.read_csv("D:/tmp/Treport/Triple.csv", encoding='shift-jis')
    tripleFrame_Treport = Ce.TNoun_extract(tripleFrame, Dc.NV_class)
    tripleFrame_Treport.sort_index(by=[u"報告書_id", u"文_id", u"動詞_id"], inplace=True)
    tripleFrame_Treport.to_csv("D:/tmp/Treport/Triple_Treport.csv", index=False, encoding='shift-jis')
    #'''
    # 未登録語の登録
    '''
    unNoun_path = u"D:/tmp/Treport/unregistered_NounsClass.csv"
    unVerb_path = u"D:/tmp/Treport/unregistered_Verbs(bunruidb)Class.csv"
    Dc.unregistered_words(unNoun_path, unVerb_path)

    #bunruidb_path = u"D:/研究/bunruidb.txt"
    bunruidb_path = u"D:/Users/ide/Desktop/データ関連/bunruidb/bunruidb.txt"
    bun_Vthe_path = u"D:/tmp/Treport/bunruidb_Vthesaurus.csv"
    Dc.buruidb_verbs(bunruidb_path, bun_Vthe_path)
    '''
    #'''
    #格フレームの構築
    tripleFrame_Treport = pd.read_csv("D:/tmp/Treport/Triple_Treport.csv", encoding='shift-jis')
    case_df = Ce.create_caseframe(tripleFrame_Treport)
    case_df.to_csv(u"D:/tmp/Treport/caseframe.csv", encoding='shift-jis', index=False)
    #'''
    case_df = pd.read_csv("D:/tmp/Treport/caseframe.csv", encoding='shift-jis')
    #'''
    terms, documents = Ce.extract_terms(case_df)

    idf_Treport = idf(terms, documents)

    file = open('D:/tmp/Treport/terms.list', 'w')
    pickle.dump(terms, file)
    file.close()
    file = open('D:/tmp/Treport/documents.list', 'w')
    pickle.dump(documents, file)
    file.close()
    file = open('D:/tmp/Treport/idf_Treport.list', 'w')
    pickle.dump(idf_Treport, file)
    file.close()
    #'''

    file = open('D:/tmp/Treport/terms.list')
    terms = pickle.load(file)
    file.close()
    file = open('D:/tmp/Treport/documents.list')
    documents = pickle.load(file)
    file.close()
    file = open('D:/tmp/Treport/idf_Treport.list')
    idf_Treport = pickle.load(file)
    file.close()

    book = u"D:/Users/ide/Desktop/データ関連/Tcluster.xls"
    book = u"D:\研究\データ\Tcluster.xls"
    sheet = u"Sheet1"  # 読み込むシート名
    EXL = pd.ExcelFile(book)  # xlsxファイルをPython上で開く
    Tcluster = EXL.parse(sheet)
    # case_df_Tcluster = case_df.ix[case_df[u"報告書_id"].map(lambda x: x in list(Tcluster[u"分析NO"].drop_duplicates())), :]
    # case_df_Tcluster.to_csv(u"D:/tmp/Treport/caseframe_Tcluster.csv", encoding='shift-jis', index=False)

    case_df_Tcluster = pd.read_csv("D:/tmp/Treport/caseframe_Tcluster.csv", encoding='shift-jis')

    #'''
    VC_list = case_df_Tcluster[u"動詞"].map(lambda x: Dc.NV_class[1].get(x, ""))
    Fre_VC_Dc = dict()
    for i, VC_perV in zip(VC_list.keys(), VC_list.values):
        Dc_count_tmp = [int(x) for x in case_df_Tcluster.ix[i, Dc.DeepCaseList] != u" "]
        for VC in VC_perV:
            if VC in Fre_VC_Dc.keys():
                Fre_VC_Dc[VC] = [x + y for x, y in zip(Fre_VC_Dc[VC], Dc_count_tmp)]
            else:
                Fre_VC_Dc[VC] = Dc_count_tmp
    times = 1.5
    VC_Dc = dict()
    for i in Fre_VC_Dc.keys():
        TW = sum(Fre_VC_Dc[i]) * (1 / float(len(Fre_VC_Dc[i]))) * times
        Dc_perVc = [Dc.DeepCaseList[Fre_VC_Dc[i].index(x)] for x in Fre_VC_Dc[i] if x > TW]
        VC_Dc[i] = Dc_perVc
    file = open('D:/tmp/Treport/VC_Dc.list', 'w')
    pickle.dump(VC_Dc, file)
    file.close()
    #'''


    file = open('D:/tmp/Treport/VC_Dc.list')
    VC_Dc = pickle.load(file)
    file.close()


    dist_method = u"Jaccard"
    # dist_method = u"Simpson"
    threshould_dist = 0.4

    # 確認用のためデータ削減
    N = 10
    ExNlist = list(case_df_Tcluster[u"報告書_id"].drop_duplicates()[:N])
    case_df_Tcluster = case_df_Tcluster.ix[case_df_Tcluster[u"報告書_id"].map(lambda x: x in ExNlist), :]

    case_df_Tcluster = Ce.Section_div(case_df_Tcluster)
    case_df_unified, Wdist = Ce.bunrui_frame(case_df_Tcluster, terms, idf_Treport, dist_method, threshould_dist)
    Wdist.sort_values(by=u"Similarity", ascending=False).to_csv("D:/tmp/Treport/Wdist.csv", encoding='shift-jis')

    for cluster in Tcluster[u"$T1-TwoStep"].drop_duplicates():
        if cluster != u"-1":
            id_perC_tmp = set(Tcluster[Tcluster[u"$T1-TwoStep"] == cluster][u"分析NO"]).intersection(
                set(case_df_unified[u"報告書_id"]))
            if len(id_perC_tmp) != 0:
                print cluster
                path = u"D:/tmp/Treport/caseframe_cluster/case_frame_%s.csv" % cluster
                case_df_unified.ix[case_df_unified[u"報告書_id"].map(lambda x: x in id_perC_tmp), :].to_csv(path,
                                                                                                         encoding="shift-jis",
                                                                                                         index=False)

    # 照応解析等による因果関係idの割り当て



    Wdist = pd.read_csv("D:/tmp/Treport/Wdist.csv", encoding='shift-jis')
    case_mat = pd.pivot_table(case_df_unified.loc[:, [u"レコード_id", u"事象"]], index=u"レコード_id", columns=u"事象",
                              aggfunc=len).fillna(0)
    case_mat[case_mat > 1] = 1
    case_mat.to_csv("D:/tmp/Treport/case_mat.csv", encoding='shift-jis', index=False)
    #出現頻度1の事象を抽出
    Fre1_cases = case_mat.columns[case_mat.sum() == 1]
    #出現頻度1以上の事象を抽出
    indexer = case_mat.sum() > 1
    case_mat2 = case_mat[indexer.index[indexer]]
    columner = case_mat2.sum(1) > 0
    case_mat2 = case_mat2.ix[columner, :]
    case_mat2.to_csv("D:/tmp/Treport/case_mat2.csv", encoding='shift-jis', index=False)