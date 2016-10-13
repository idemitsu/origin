#代名詞は無条件に抽出

#全て抽出するクラス
TNounclass_all=[
        u"食料",
        u"資材",
        u"空間",
        u"物質",
        u"物品",
        u"機関",
        u"機械",
        u"形",
        u"力",
        u"作用",
]

#部分一致でなければ抽出
TNounclass_Nopart ={
        u"存在": [u"結果"],
        u"住居": [u"間"],
        u"道具": [u"SOAP", u"面", u"的"],
        u"衣料": [u"面"],
        u"事業": [u"診断"],
        u"自然": [u"日"],
        u"様相": [u"可能性", u"一般", u"傾向", u"このよう", u"内容", u"現状", u"通常", u"特長", u"特長", u"調整", u"実態", u"一致", u"都合", u"単純"]
        #様相は形容動詞語幹でも判定し、複合名詞の場合は前の名詞のクラスに依存
}

#部分一致であれば抽出
TNounclass_part={
        u"仲間": [u"同士"],
        u"類": [u"等", u"類", u"系", u"系統", u"種", u"メタルコンタクト"],
        u"言語": [u"音"],
        u"事柄": [u"現象", u"クリープ", u"キャビテーション"],
        u"身体": [u"口"],
        u"生命": [u"寿命"],
        u"量": [u"値", u"度", u"量", u"粘度", u"酸価", u"塩基価", u"温", u"荷重", u"数", u"率", u"積", u"径", u"速"],
        u"行為": [u"センサーエラー", u"負荷"],
        u"成員": [u"ランナー", u"シリンダヘッド", u"キャップ"],
        u"天地": [u"電食"]
}

#抽出しないクラス
TNounclass_unnec=[
        u"経済",
        u"社会",
        u"生物",
        u"生活",
        u"植物",
        u"時間",
        u"心",
        u"待遇",
        u"家族",
        u"土地利用",
        u"動物",
        u"公私",
        u"人間",
        u"人物",
        u"交わり",
        u"芸術",
        u"未登録"
]