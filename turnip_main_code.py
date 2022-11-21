import pandas as pd
from pandas_datareader import data as pdr
import os
import turnip_parameter as p
import time
import requests
from csv import reader, writer
import json
import pandas as pd
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pydotplus
import io
from six import StringIO
import datetime
import statistics
import math
import joblib
import yfinance as yfin
yfin.pdr_override()

COL_NAME_Open   = p.COL_NAME_Open
COL_NAME_High   = p.COL_NAME_High
COL_NAME_Low    = p.COL_NAME_Low
COL_NAME_Close  = p.COL_NAME_Close
COL_NAME_Volume = p.COL_NAME_Volume

def generate_dfex(df_origin, judge_days, judge_border):
    df = df_origin.copy()
    
    # Volumeが0の行を削除
    df = df[df[COL_NAME_Volume] != 0]

    # カラムの作成
    df["終値差額"]=df[COL_NAME_Close].diff()
    df['終値差額率']=df['終値差額']/df[COL_NAME_Close].shift(1)
    df["高値安値差額"]=df[COL_NAME_High]-df[COL_NAME_Low]
    df["高値終値差額"]=df[COL_NAME_High]-df[COL_NAME_Close]
    df["高値終値差額率"]=df["高値終値差額"]/df[COL_NAME_Close]
    df["安値終値差額"]=df[COL_NAME_Low]-df[COL_NAME_Close]
    df["安値終値差額率"]=df["安値終値差額"]/df[COL_NAME_Close]
    df["終値5日平均"]=df[COL_NAME_Close].rolling(window=5).mean()
    df["終値5日平均乖離率"]=(df[COL_NAME_Close]-df["終値5日平均"])/df["終値5日平均"]
    df["終値25日平均"]=df[COL_NAME_Close].rolling(window=25).mean()
    df["終値25日平均乖離率"]=(df[COL_NAME_Close]-df["終値25日平均"])/df["終値25日平均"]
    df["終値50日平均"]=df[COL_NAME_Close].rolling(window=50).mean()
    df["終値50日平均乖離率"]=(df[COL_NAME_Close]-df["終値50日平均"])/df["終値50日平均"]
    df["終値75日平均"]=df[COL_NAME_Close].rolling(window=75).mean()
    df["終値75日平均乖離率"]=(df[COL_NAME_Close]-df["終値75日平均"])/df["終値75日平均"]
    df['出来高変動数']=df[COL_NAME_Volume].diff()
    df['出来高変動率']=df[COL_NAME_Volume]/df[COL_NAME_Volume].shift(1)
    df["出来高5日平均"]=df[COL_NAME_Volume].rolling(window=5).mean()
    df["出来高5日平均乖離率"]=(df[COL_NAME_Volume]-df["出来高5日平均"])/df["出来高5日平均"]
    df["出来高25日平均"]=df[COL_NAME_Volume].rolling(window=25).mean()
    df["出来高25日平均乖離率"]=(df[COL_NAME_Volume]-df["出来高25日平均"])/df["出来高25日平均"]

    # 上げ値, 下げ値
    df.loc[df['終値差額'] > 0, '上げ値'] = df['終値差額']
    df.loc[df['終値差額'] < 0, '上げ値'] = 0
    df.loc[df['終値差額'] < 0, '下げ値'] = -df['終値差額']
    df.loc[df['終値差額'] > 0, '下げ値'] = 0
    df=df.fillna(0) # NAfill

    # 上昇平均, RSI
    df["上昇平均14d"] = df["上げ値"].rolling(14).mean()
    df["下降平均14d"] = df["下げ値"].rolling(14).mean()
    df["RSI"] = df["上昇平均14d"]/(df["上昇平均14d"]+df["下降平均14d"])*100
    df=df.fillna(0) # NAfill

    # EMA12, EMA26 （EMA : 指数平滑移動平均）
    indexs = df.index.values

    # EMA12の計算
    df["ema12d"] = df[COL_NAME_Close].rolling(12).apply(lambda x: x.sum()/12)   
    for yesterday, today in list(zip(indexs[11:], indexs[12:])) :
        df.loc[today, "ema12d"] = df.loc[yesterday, "ema12d"] + (2/(12+1))*(df.loc[today, COL_NAME_Close] - df.loc[yesterday, "ema12d"])

    # EMA26の計算
    df["ema26d"] = df[COL_NAME_Close].rolling(26).apply(lambda x: x.sum()/26)
    for yesterday, today in list(zip(indexs[25:], indexs[26:])) :
        df.loc[today, "ema26d"] = df.loc[yesterday, "ema26d"] + (2/(26+1))*(df.loc[today, COL_NAME_Close] - df.loc[yesterday, "ema26d"])
        
    df = df.fillna(0) # NAfill
    

    # MACDの計算
    df["macd"] = df["ema12d"] - df["ema26d"]
    df.loc[df.index[:25], "macd"] = np.nan

    df["macd_signal"] = df["macd"].rolling(9).apply(lambda x: x.sum()/9)
    for yesterday, today in list(zip(indexs[33:], indexs[34:])) :
        df.loc[today, "macd_signal"] = df.loc[yesterday, "macd_signal"] + (2/(26+1))*(df.loc[today, "macd"] - df.loc[yesterday, "macd_signal"])
    df = df.fillna(0) # NAfill    

    # MACDのカラム作成
    df["MACD上昇・下落偏差(5日)"] = df["macd"]-df["macd"].shift(5)
    df["MACDシグナル上昇・下落偏差(5日)"] = df["macd_signal"]-df["macd_signal"].shift(5)
    df["MACD-MACDシグナル"] = df["macd"]-df["macd_signal"]
    df["MACD-MACDシグナル(5日標準偏差)"] = df["MACD上昇・下落偏差(5日)"]-df["MACDシグナル上昇・下落偏差(5日)"]

    # 取引量の計算
    df["取引量(出来高*株価((高値＋終値)/2))"] = df[COL_NAME_Volume]*((df[COL_NAME_High]+df[COL_NAME_Low])/2)
    df["取引量(出来高*株価(終値))前日との変動率"] = 0

    endvalue25d_top  = df.loc[indexs[0], "取引量(出来高*株価((高値＋終値)/2))"]
    endvalue25d_next = df.loc[indexs[1], "取引量(出来高*株価((高値＋終値)/2))"]
    
    for today, tomorrow in list(zip(indexs[1:], indexs[2:])) :
        df.loc[today, "取引量(出来高*株価(終値))前日との変動率"] = (endvalue25d_next-endvalue25d_top)/endvalue25d_top
        endvalue25d_top  = df.loc[today,    "取引量(出来高*株価((高値＋終値)/2))"]
        endvalue25d_next = df.loc[tomorrow, "取引量(出来高*株価((高値＋終値)/2))"]

    # 取引量変動率のカラム作成
    df["取引量(出来高*株価(終値))5日平均"] = df["取引量(出来高*株価((高値＋終値)/2))"].rolling(5).mean()
    df["取引量(出来高*株価(終値))5日平均乖離率"] = (df["取引量(出来高*株価((高値＋終値)/2))"]-df["取引量(出来高*株価(終値))5日平均"])/df["取引量(出来高*株価(終値))5日平均"]
    df["取引量(出来高*株価(終値))25日平均"] = df["取引量(出来高*株価((高値＋終値)/2))"].rolling(25).mean()
    df["取引量(出来高*株価(終値))25日平均乖離率"] = (df["取引量(出来高*株価((高値＋終値)/2))"] - df["取引量(出来高*株価(終値))25日平均"])/ df["取引量(出来高*株価(終値))25日平均"]

    # 上昇・下落判定の追加
    df.loc[  df[COL_NAME_Close].rolling(judge_days).max().shift(-1*judge_days) > df[COL_NAME_Open].shift(-1)*(1+judge_border/100),  "上昇判定結果"] = 1
    df.loc[~(df[COL_NAME_Close].rolling(judge_days).max().shift(-1*judge_days) > df[COL_NAME_Open].shift(-1)*(1+judge_border/100)), "上昇判定結果"] = 0
    df.loc[  df[COL_NAME_Close].rolling(judge_days).min().shift(-1*judge_days) < df[COL_NAME_Open].shift(-1)*(1-judge_border/100),  "下落判定結果"] = 1
    df.loc[~(df[COL_NAME_Close].rolling(judge_days).min().shift(-1*judge_days) < df[COL_NAME_Open].shift(-1)*(1-judge_border/100)), "下落判定結果"] = 0
    
    df=df.fillna(0) # NAfill
    return df

class Stock:
    def __init__(self, dotd, sh, hb, pb, j):
        self.open         = dotd[COL_NAME_Open]
        self.close        = dotd[COL_NAME_Close]
        self.high         = dotd[COL_NAME_High]
        self.low          = dotd[COL_NAME_Low]
        self.volume       = dotd[COL_NAME_Volume]
        self.date_earlier  = sh["date_earlier"]
        self.price_earlier = sh["price_earlier"]
        self.holding_days = sh["holding_days"]
        self.node_index   = sh["node_index"]
        
        self.is_selling   = False
        self.date_later    = 0
        self.price_later   = 0
        
        self.holding_border = hb
        self.profit_border  = pb if j == 'rise' else pb[::-1]
    
    @property
    def price_plus5(self):
        return self.price_earlier * ( 1 + self.profit_border[0] / 100 ) # 売却基準額（購入額の+5%) 
    
    @property
    def price_minus5(self):
        return self.price_earlier * ( 1 - self.profit_border[1] / 100 ) # 売却基準額（購入額の-5%)
    
    # 購入したものを売却するかどうかの判定
    def selling_condition(self, date):
        # -プラスとマイナスの両値をつけた場合は無効
        if( self.high > self.price_plus5 and self.price_minus5 > self.low):
            self.is_selling, self.date_later, self.price_later = True, date, self.price_earlier

        # -購入価格から5%の利益で売却（判定は最高値で判定）（S高）S安の場合は5%で処理
        elif( self.high > self.price_plus5):
            self.is_selling, self.date_later, self.price_later = True, date, self.price_plus5

        # -購入価格から-5%で損切（判定は最安値で判定）
        elif( self.price_minus5 > self.low):
            self.is_selling, self.date_later, self.price_later = True, date, self.price_minus5

        # -最長保持期間は5日（5日で5%に達しない場合は終値で売却）
        elif( self.holding_days >= self.holding_border):
            self.is_selling, self.date_later, self.price_later = True, date, self.close

        # -上記の条件に引っ掛からなかった場合
        else :
            self.is_selling = False

        return self
    
    def output_to_tuple(self):
        return (self.date_earlier, self.date_later, self.price_earlier, self.price_later, self.node_index, self.open, self.high, self.low, self.close, self.volume)
    
    def output_open_volume(self):
        return (self.open, self.volume)
    
def simulation(judge, nodes, df_valid, holding_border, profit_border) :
    stocks_having = [] # 購入株リスト　(購入日, 購入額, 保持日数)
    stocks_sold   = [] # 売却株リスト　(購入日, 売却日, 購入額, 売却額, ...etc)
    is_rising  = False
    is_holding = False
    
    for column_name, data_of_the_day in df_valid.iterrows():
        # break
        date = data_of_the_day["Date"] # ToDo: 本当はDateを入れる（現在は一意に定まりそうな出来高で代用）

        # 前日の上昇判定がtrueなら、始値で購入
        if(is_rising and not is_holding):
            is_holding = True
            price = data_of_the_day[COL_NAME_Open]
            stocks_having.append({"date_earlier": date, "price_earlier": price, "holding_days": 0, "node_index": node_index})

        # この日の上昇判定で上書き
        is_rising, node_index = get_is_rising(data_of_the_day, nodes)

        # それぞれの売却の判定
        for stock_having in stocks_having[:]:
            stock_having["holding_days"] += 1 # 有効な日のみ、保持日数のインクリメント
            stock  = Stock(dotd=data_of_the_day, sh=stock_having, hb=holding_border, pb=profit_border, j=judge)
            result = stock.selling_condition(date=date)

            if(result.is_selling):
                # 購入済み株リストから売却済み株リストに移動
                is_holding = False
                stocks_having.remove(stock_having)
                stocks_sold.append(result)
                
    return stocks_sold

def get_is_rising(data_of_the_day, nodes):
    for index, values in nodes.items() :
        if all([ data_of_the_day[col] <= float(num) if eq == "≦" else data_of_the_day[col] > float(num) for [col, eq, num] in [v.split(" ") for v in values["conditions"]]]) :
            return True, index
    return False, -1

def summarize_stock(code, stocks, criteria, entry, trans_vol, filepath):
    
    if  (criteria == "RISE"):
        profit_array = [x.price_later - x.price_earlier for x in stocks]
        profit_ave   = 0 if len(stocks) == 0 else sum(profit_array) / sum(x.price_earlier for x in stocks) * 100
    elif(criteria == "FALL"):
        profit_array = [x.price_earlier - x.price_later for x in stocks]
        profit_ave   = 0 if len(stocks) == 0 else sum(profit_array) / sum(x.price_later for x in stocks) * 100
    
    floor  = lambda x, n: math.floor(x * 10 ** n) / (10 ** n)
    zeroif = lambda l, a: 0 if len(l) == 0 else a
    
    stock_summary = {
        "code"             : code,
        "criteria"         : criteria,
        "count_transaction": len(stocks),
        "count_rose"       : sum(x.price_later-x.price_earlier > 0 for x in stocks),
        "count_fell"       : sum(x.price_later-x.price_earlier < 0 for x in stocks),
        "sum_price_earlier": sum(x.price_earlier for x in stocks),
        "sum_price_later"  : sum(x.price_later for x in stocks),
        "sum_profit"       : sum(profit_array),
        "profit_ave"       : floor(profit_ave, 3),
        "days_ave"         : 0 if len(stocks) == 0 else floor(sum(stock.holding_days for stock in stocks) / len(stocks), 2),
        "profit_median"    : 0 if len(stocks) == 0 else statistics.median(profit_array),
        "profit_mode"      : 0 if len(stocks) == 0 else statistics.mode(profit_array),
        "gross_profit_earn": len(stocks) * profit_ave,
        "valid_exponent"   : len(stocks) ** 2 * profit_ave ** 5,
        "trans_vol"        : trans_vol,
        "entry_judge"      : "1" if entry else "0"
    }
    row_list = ["code", "criteria", "count_transaction", "count_rose", "count_fell", "profit_ave", "days_ave", "gross_profit_earn", "valid_exponent", "trans_vol", "entry_judge"]
    row = [stock_summary[r] for r in row_list]
    
    if stock_summary["days_ave"] and stock_summary["profit_ave"] > 2 :
        with open(filepath, 'a', encoding="utf_8_sig") as f: writer(f).writerow(row)

    return stock_summary

def print_summary(stock_summary):
    
    print(f'\n{stock_summary["code"]}-{stock_summary["criteria"]} -- 平均利益率：{stock_summary["profit_ave"]:.3f}%, 平均保持日数：{stock_summary["days_ave"]:.2f}日')
    if  (stock_summary["criteria"] == "RISE"):
        print("  取　　引　　数：{:>5,}\t 買った際の総額：{:>10,.0f}".format(stock_summary["count_transaction"], stock_summary["sum_price_earlier"]))
        print("  上昇した取引数：{:>5,}\t 売った際の総額：{:>10,.0f}".format(stock_summary["count_rose"]       , stock_summary["sum_price_later"]))
        print("  下落した取引数：{:>5,}\t 利　　　　　益：{:>10,.0f}".format(stock_summary["count_fell"]       , stock_summary["sum_profit"]))
    elif(stock_summary["criteria"] == "FALL"):
        print("  取　　引　　数：{:>5,}\t 空売り時の総額：{:>10,.0f}".format(stock_summary["count_transaction"], stock_summary["sum_price_earlier"]))
        print("  上昇した取引数：{:>5,}\t 買った時の総額：{:>10,.0f}".format(stock_summary["count_rose"]       , stock_summary["sum_price_later"]))
        print("  下落した取引数：{:>5,}\t 利　　　　　益：{:>10,.0f}".format(stock_summary["count_fell"]       , stock_summary["sum_profit"]))

def output_csv(df, tuning_name, sim_num, name):
    dfs_path = f"dfs"
    if not os.path.isdir(dfs_path) : os.makedirs(dfs_path)
    file_name = f'{dfs_path}/df_{tuning_name}_{sim_num}_{name.split(".")[0]}.csv'
    df.to_csv(path_or_buf=file_name, encoding="utf_8_sig")

# --初期値  階層数: 12階層以内, sample数: 10以上
def make_decision_tree(df, judgeColList, removeColList, target, min_samples_leaf, max_depth) :
    X = df.iloc[:,[i for i,v in enumerate(df.columns.tolist()) if v not in judgeColList + removeColList]]
    y = df.loc[:,[target]]

    clf = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    clf = clf.fit(X, y)
    return clf

def visualiz_decision_tree(clf, df_column) :
    dot_data = StringIO() #dotファイル情報の格納先
    export_graphviz(
        clf,
        out_file = dot_data,
        feature_names = df_column,
        class_names = ["False","True"], #編集するのはここ
        node_ids = True,
        filled = True,
        rounded = True,  
        special_characters=True
    )
    return dot_data

def get_good_nodes(dotfile, criteria) :
    dotfile = [i for i in dotfile.getvalue().split("\n") if i.split()[0].isnumeric()]

    nodeNum = int(dotfile[-2].split()[0]) + 1
    leaves = [i for i in range(nodeNum) if i not in [int(j.split()[0]) for j in dotfile if "->" in j]]
    nodes = {int(line.split()[0]):line.split("<br/>")[1] if int(line.split()[0]) not in leaves else line.split("<br/>")[3] for line in dotfile if "->" not in line}
    branches = [[int(i.split()[0]),int(i.split()[2])] for i in dotfile if "->" in i]
    good_nodes = {}

    for leaf in leaves:
        [samples_false, samples_true] = [int(nodes[leaf].split()[2][1:-1]), int(nodes[leaf].split()[3][:-1])]
        percent = samples_true/(samples_false+samples_true)*100 

        if percent < criteria : continue
        good_nodes.update({leaf:{"percent":percent, "conditions":[]}})
        target = leaf
        for [parent_nodeNum, child_nodeNum] in reversed(branches):
            if target == child_nodeNum :
                good_nodes[leaf]["conditions"].append( nodes[parent_nodeNum].replace("&le;","≦") if parent_nodeNum+1 == child_nodeNum else nodes[parent_nodeNum].replace("&le;",">>") )
                target = parent_nodeNum
    return good_nodes

def generate_algorythm(
        code, tuning_name, learn_index, learn_num_min,
        judge_days, judge_border, min_samples_leaf, max_depth, criteria,
        judgeColList, removeColList, renameColList,
        CSV_PATH, NODES_PATH, RESULT_PATH ) :
    
    nodes_path = f'{NODES_PATH}/nodes{tuning_name}'
    if not os.path.isdir(nodes_path) : os.makedirs(nodes_path)
          
    df = pd.read_csv(f'{CSV_PATH}/{code}.csv', names=["Date",COL_NAME_Open,COL_NAME_High,COL_NAME_Low,"Close_origin",COL_NAME_Volume,COL_NAME_Close])
    df = df.iloc[-1:learn_index:-1] if df.iloc[1]["Date"] > df.iloc[-1]["Date"] else df.iloc[1:(-1*learn_index):]
    for after, befores in renameColList.items():
        for before in befores:
            if before in df.columns:
                df = df.rename(columns={before : after})
    
    unlisted_day = [(datetime.datetime.strptime(yesterday, '%Y/%m/%d') - datetime.datetime.strptime(today, '%Y/%m/%d')).days > 366 for yesterday, today in list(zip(df.Date[0:], df.Date[1:]))]
    unlisted_result = unlisted_day.index(True) if any(unlisted_day) else -1

    # 再上場後のみのデータで学習
    if unlisted_result >= learn_num_min : df = df.iloc[:unlisted_result+1,:]

    # 未上場期間１年以上 && 再上場後の件数不足
    if learn_num_min > unlisted_result > -1 : return -1

    # 学習データが確保できない場合はスキップする
    if len(df) < learn_num_min: return -1

    # 終値と調整後終値が違う場合は、その比率に基づいて他の値（始値、高値、低値、出来高）を修正する
    for row in df.itertuples():
        if float(row.Close) / float(row.Close_origin) != 1.0 :
            ratio = float(row.Close) / float(row.Close_origin)
            df.loc[row.Index, COL_NAME_Open]   = float(row.Open)   * ratio
            df.loc[row.Index, COL_NAME_High]   = float(row.High)   * ratio
            df.loc[row.Index, COL_NAME_Low]    = float(row.Low)    * ratio
            df.loc[row.Index, COL_NAME_Volume] = float(row.Volume) / ratio

    df = df.iloc[:,[i for i,v in enumerate(df.columns.tolist()) if v not in removeColList]]

    df[COL_NAME_Open]   = pd.to_numeric(df[COL_NAME_Open],   errors='coerce').fillna(0).astype(int)
    df[COL_NAME_High]   = pd.to_numeric(df[COL_NAME_High],   errors='coerce').fillna(0).astype(int)
    df[COL_NAME_Low]    = pd.to_numeric(df[COL_NAME_Low],    errors='coerce').fillna(0).astype(int)
    df[COL_NAME_Volume] = pd.to_numeric(df[COL_NAME_Volume], errors='coerce').fillna(0).astype(int)
    df[COL_NAME_Close]  = pd.to_numeric(df[COL_NAME_Close],  errors='coerce').fillna(0).astype(int)

    df = df.rename(columns=lambda s: s.replace(' ', ''), index=lambda s: s)
    df = generate_dfex(df, judge_days, judge_border)
    df_column = [v for v in df.columns.tolist() if v not in judgeColList]
    
    for target in judgeColList :
        try :
            clf      = make_decision_tree(df, judgeColList, removeColList, target, min_samples_leaf, max_depth)
            dot_data = visualiz_decision_tree(clf, df_column)
            nodes    = get_good_nodes(dot_data, criteria)
        except: pass
        else :
            node_path = f'{nodes_path}/{code}_{"fall" if judgeColList.index(target) else "rise" }.json'
            with open(node_path, mode='w') as f: f.write(json.dumps(nodes))
                
def step01(cdlst, paths, parameters, lists, overwrite, newdir, dlonly) :
    for path in directory_paths: if not os.path.isdir(paths) : os.makedirs(paths)

    for_start = time.time()
    
    for code in cdlst:

        # プログレスバーの表示
        total = len(cdlst)
        ind = cdlst.index(code) + 1
        bar = math.floor(ind / total * 50)
        sec = datetime.timedelta(seconds=(time.time()-for_start))
        print(f"\r{ind:>4}/{total}: [{''.join(['-' for _ in range(bar)]):<50}] {ind/total*100:>6.2f}%  {sec}", end='')

        # newdir - T: ディレクトリ名に今日の日付を追加, F: そのまま
        date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date().strftime("%Y%m%d") if newdir else ""
        csvs_path = f'{paths["CSV_PATH"]}{date}'

        if not os.path.isdir(csvs_path) : os.makedirs(csvs_path)
        csv_path = f'{csvs_path}/{code}.csv'
    
        start = time.time()
        # overwrite - T: 既にcsvファイルがあれば上書きする, F: そのまま
        if not os.path.exists(csv_path) or overwrite :
            csv_url = f'https://finance.yahoo.co.jp/quote/{code}.T/history/download'
#           csv_url = f'https://download.finance.yahoo.co.jp/common/history/{code}.T.csv'
            cookies = { "_n" : p._n, "T" : p.T, "Y" : p.Y } # T, Yとは別に　_nも必要になった
            csv = requests.get(csv_url, cookies=cookies)
            with open(csv_path, mode='w') as f: f.write(csv.text)
        else :
            start = -5
            
        if not dlonly :
            _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(generate_algorythm)(**parameter, **paths, **lists, code=code) for parameter in parameters)

        sleep = 5-(time.time()-start)
        if sleep > 0 : time.sleep(sleep)

def step02(tuning_name, holding_border, profit_border, learn_index, sim_num, judge_days, judge_border, NODES_PATH, CSV_PATH, RESULT_PATH) :
    if not os.path.exists(RESULT_PATH) : os.makedirs(RESULT_PATH)

    for simnum in sim_num :
        filepath = f'{RESULT_PATH}/step02_result_{tuning_name}_{simnum}.csv'
        with open(filepath, 'w', encoding="utf_8_sig") as f:
#             writer(f).writerow(["銘柄", "上昇/下落", "取引数", "上昇", "下落", "平均利益率", "保持日数", "獲得総利益", "有効指数", "25日取引量平均"])
            writer(f).writerow(["code", "criteria", "count_transaction", "count_rose", "count_fell", "profit_ave", "days_ave", "gross_profit_earn", "valid_exponent", "trans_vol_25daysAve", "entry"])

    nodes_path = f'{NODES_PATH}/nodes{tuning_name}'
    files = os.listdir(nodes_path)
    cdlst = sorted([f for f in files if os.path.isfile(os.path.join(nodes_path, f)) if "json" in f ])
#     cdlst=[ "3681_rise.json", "3110_rise.json", "6197_rise.json", "3921_rise.json", "6191_rise.json", "1712_rise.json" ]
    
    for node_filename in cdlst :

        path = f'{nodes_path}/{node_filename}'
        code = node_filename.split("_")[0]

        if 'json' not in path : continue
        if not os.path.exists(path) : continue
        with open(path, mode='r', encoding="utf8") as f:
            nodes = json.load(f)
        if nodes is None : continue
        if not len(nodes) : continue

        df = pd.read_csv(f'{CSV_PATH}/{code}.csv', names=["Date",COL_NAME_Open,COL_NAME_High,COL_NAME_Low,"Close_origin",COL_NAME_Volume,COL_NAME_Close])
        df = df.iloc[max(sim_num)+100:0:-1] if df.iloc[1]["Date"] > df.iloc[-1]["Date"] else df.iloc[-1*(max(sim_num)+100)::]
        
        # 終値と調整後終値が違う場合は、その比率に基づいて他の値（始値、高値、低値、出来高）を修正する
        for row in df.itertuples():
            ratio = float(row.Close) / float(row.Close_origin)
            if ratio != 1.0 :                
                df.loc[row.Index, COL_NAME_Open]   = float(row.Open)   * ratio
                df.loc[row.Index, COL_NAME_High]   = float(row.High)   * ratio
                df.loc[row.Index, COL_NAME_Low]    = float(row.Low)    * ratio
                df.loc[row.Index, COL_NAME_Volume] = float(row.Volume) / ratio

        df[COL_NAME_Open]   = pd.to_numeric(df[COL_NAME_Open],   errors='coerce').fillna(0).astype(int)
        df[COL_NAME_High]   = pd.to_numeric(df[COL_NAME_High],   errors='coerce').fillna(0).astype(int)
        df[COL_NAME_Low]    = pd.to_numeric(df[COL_NAME_Low],    errors='coerce').fillna(0).astype(int)
        df[COL_NAME_Volume] = pd.to_numeric(df[COL_NAME_Volume], errors='coerce').fillna(0).astype(int)
        df[COL_NAME_Close]  = pd.to_numeric(df[COL_NAME_Close],  errors='coerce').fillna(0).astype(int)

        df = df.rename(columns=lambda s: s.replace(' ', ''), index=lambda s: s)
        df = generate_dfex(df, judge_days, judge_border)
 
        for simnum in sim_num :
            judge = node_filename.split("_")[1][:4]
            df_sim = df.iloc[-1*simnum::]
            entry, _ = get_is_rising(df.iloc[-1], nodes)
            stocks   = simulation(judge, nodes, df_sim, holding_border, profit_border)
            if not len(stocks) : continue
            trans_vol = df.at[df.index[-1], "取引量(出来高*株価(終値))25日平均"]
            filepath = f'{RESULT_PATH}/step02_result_{tuning_name}_{simnum}.csv'
            stock_summary = summarize_stock(code, stocks, judge.upper(), entry, trans_vol, filepath)    

def reformat_df(df, judgeColList, removeColList, renameColList) :

    # Dateカラムがなく、indexにDate情報が含まれていれば、indexをDateカラムとして %Y/%m/%d で保存
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Date'})
    df['Date'] = df['Date'].astype(str).map(lambda date: date.replace('-','/'))
    strptime = datetime.datetime.strptime
    
    # カラム名で置き換えがあれば実行
    for after, befores in renameColList.items():
        for before in befores:
            if before in df.columns:
                df = df.rename(columns={before : after})

    # 終値と調整後終値が違う場合は、その比率に基づいて他の値（始値、高値、低値、出来高）を修正する
    for row in df.itertuples():
        if float(row.Close) / float(row.Close_origin) != 1.0 :
            ratio = float(row.Close) / float(row.Close_origin)
            df.loc[row.Index, COL_NAME_Open]   = float(row.Open)   * ratio
            df.loc[row.Index, COL_NAME_High]   = float(row.High)   * ratio
            df.loc[row.Index, COL_NAME_Low]    = float(row.Low)    * ratio
            df.loc[row.Index, COL_NAME_Volume] = float(row.Volume) / ratio
    df = df.iloc[:,[i for i,v in enumerate(df.columns.tolist()) if v != 'Close_origin']]
    
    df[COL_NAME_Open]   = pd.to_numeric(df[COL_NAME_Open],   errors='coerce').fillna(0).astype(int)
    df[COL_NAME_High]   = pd.to_numeric(df[COL_NAME_High],   errors='coerce').fillna(0).astype(int)
    df[COL_NAME_Low]    = pd.to_numeric(df[COL_NAME_Low],    errors='coerce').fillna(0).astype(int)
    df[COL_NAME_Volume] = pd.to_numeric(df[COL_NAME_Volume], errors='coerce').fillna(0).astype(int)
    df[COL_NAME_Close]  = pd.to_numeric(df[COL_NAME_Close],  errors='coerce').fillna(0).astype(int)
    
    return df

def unlisted_check() :
    # 未上場の期間が１年以上あるか判定 
    unlisted_day = [(strptime(yesterday, '%Y/%m/%d') - strptime(today, '%Y/%m/%d')).days > 366 for yesterday, today in list(zip(df.Date[0:], df.Date[1:]))]
    unlisted_result = unlisted_day.index(True) if any(unlisted_day) else -1

    # 再上場後のみのデータで学習
    if unlisted_result >= learn_num_min : df = df.iloc[:unlisted_result+1,:]

    # 未上場期間１年以上 && 再上場後の件数不足
    if learn_num_min > unlisted_result > -1 : return {}

    # 学習データが確保できない場合はスキップする
    if len(df) < learn_num_min: return {}

    df = df.iloc[:,[i for i,v in enumerate(df.columns.tolist()) if v != 'Date']]
    df = df.rename(columns=lambda s: s.replace(' ', ''), index=lambda s: s)
    
    
def generate_algorythm_useapi(
        code, df, tuning_name, learn_index, learn_num_min,
        judge_days, judge_border, min_samples_leaf, max_depth, criteria,
        judgeColList, removeColList, renameColList ) :

    strptime = datetime.datetime.strptime
    
    # 日付で昇順に
    df = df.iloc[-1:learn_index:-1] if strptime(df.iloc[1]["Date"], '%Y/%m/%d') > strptime(df.iloc[-1]["Date"], '%Y/%m/%d') else df.iloc[1:(-1*learn_index):]
    
    # 未上場の期間が１年以上あるか判定 
    unlisted_day = [(strptime(yesterday, '%Y/%m/%d') - strptime(today, '%Y/%m/%d')).days > 366 for yesterday, today in list(zip(df.Date[0:], df.Date[1:]))]
    unlisted_result = unlisted_day.index(True) if any(unlisted_day) else -1

    # 再上場後のみのデータで学習
    if unlisted_result >= learn_num_min : df = df.iloc[:unlisted_result+1,:]

    # 未上場期間１年以上 && 再上場後の件数不足
    if learn_num_min > unlisted_result > -1 : return {}

    # 学習データが確保できない場合はスキップする
    if len(df) < learn_num_min: return {}

    df = df.iloc[:,[i for i,v in enumerate(df.columns.tolist()) if v != 'Date']]
    df = df.rename(columns=lambda s: s.replace(' ', ''), index=lambda s: s)
    df = generate_dfex(df, judge_days, judge_border)
    df_column = [v for v in df.columns.tolist() if v not in judgeColList]
    
    nodess = {}
    for target in judgeColList :
        try :
            clf      = make_decision_tree(df, judgeColList, removeColList, target, min_samples_leaf, max_depth)
            dot_data = visualiz_decision_tree(clf, df_column)
            nodes    = get_good_nodes(dot_data, criteria)
        except: pass
        else  :
            target = "fall" if judgeColList.index(target) else "rise"
            if nodes != {} : nodess.update({target : nodes})
    return nodess

def step01_useapi(cdlst, parameters, lists, overwrite, newdir, dlonly) :

    for_start = time.time()
    nodesss = { p["tuning_name"] : {} for p in parameters }
    cdlst = ["3916", "4516", "7915"]
    for code in cdlst:

        df = pdr.get_data_yahoo(f'{code}.T')
        print("\r")
    
        # -- 計測開始 --
        start = time.time()
        
        # プログレスバーの表示
        total = len(cdlst)
        ind = cdlst.index(code) + 1
        bar = math.floor(ind / total * 50)
        sec = datetime.timedelta(seconds=(time.time()-for_start))
#         print(f"\r{ind:>4}/{total}: [{''.join(['-' for _ in range(bar)]):<50}] {ind/total*100:>6.2f}%  {sec}", end='')

        df = reformat_df(df, **lists)
        nodess = joblib.Parallel(n_jobs=-1)(joblib.delayed(generate_algorythm_useapi)(**parameter, **lists, code=code, df=df) for parameter in parameters)

        if any([ len(nodes) for nodes in nodess]) :
            for p in parameters :
                nodesss[p["tuning_name"]].update({ code : nodes for nodes in nodess if len(nodes) > 0 })
                nodesss[p["tuning_name"]][code].update({ 'df' : df })

        # -- 計測終了 --
        sleep = 5-(time.time()-start)
        if sleep > 0 : time.sleep(sleep)
    return nodesss


def step02_useapi(tuning_name, holding_border, profit_border, learn_index, sim_num, judge_days, judge_border, RESULT_PATH, nodes) :
    if not os.path.exists(RESULT_PATH) : os.makedirs(RESULT_PATH)

    for simnum in sim_num :
        filepath = f'{RESULT_PATH}/step02_result_{tuning_name}_{simnum}.csv'
        with open(filepath, 'w', encoding="utf_8_sig") as f:
            writer(f).writerow(["銘柄", "上昇/下落", "取引数", "上昇", "下落", "購入", "売却", "差額", "平均利益率", "保持日数", "獲得総利益", "有効指数", "エントリー"])

    cdlst = sorted([f for f in files if os.path.isfile(os.path.join(nodes_path, f)) if "json" in f ])
    
    for node_filename in cdlst :

        path = f'{nodes_path}/{node_filename}'
        code = node_filename.split("_")[0]

        df = pd.read_csv(f'{CSV_PATH}/{code}.csv', names=["Date",COL_NAME_Open,COL_NAME_High,COL_NAME_Low,"Close_origin",COL_NAME_Volume,COL_NAME_Close])
        df = df.iloc[max(sim_num)+100:0:-1] if df.iloc[1]["Date"] > df.iloc[-1]["Date"] else df.iloc[-1*(max(sim_num)+100)::]
        
        # 終値と調整後終値が違う場合は、その比率に基づいて他の値（始値、高値、低値、出来高）を修正する
        for row in df.itertuples():
            ratio = float(row.Close) / float(row.Close_origin)
            if ratio != 1.0 :                
                df.loc[row.Index, COL_NAME_Open]   = float(row.Open)   * ratio
                df.loc[row.Index, COL_NAME_High]   = float(row.High)   * ratio
                df.loc[row.Index, COL_NAME_Low]    = float(row.Low)    * ratio
                df.loc[row.Index, COL_NAME_Volume] = float(row.Volume) / ratio

        df[COL_NAME_Open]   = pd.to_numeric(df[COL_NAME_Open],   errors='coerce').fillna(0).astype(int)
        df[COL_NAME_High]   = pd.to_numeric(df[COL_NAME_High],   errors='coerce').fillna(0).astype(int)
        df[COL_NAME_Low]    = pd.to_numeric(df[COL_NAME_Low],    errors='coerce').fillna(0).astype(int)
        df[COL_NAME_Volume] = pd.to_numeric(df[COL_NAME_Volume], errors='coerce').fillna(0).astype(int)
        df[COL_NAME_Close]  = pd.to_numeric(df[COL_NAME_Close],  errors='coerce').fillna(0).astype(int)

        df = df.rename(columns=lambda s: s.replace(' ', ''), index=lambda s: s)
        df = generate_dfex(df, judge_days, judge_border)
 
        for simnum in sim_num :
            df = df.iloc[-1*simnum::]
            entry, _ = get_is_rising(df.iloc[-1], nodes)
            stocks   = simulation(nodes, df, holding_border, profit_border)
            if not len(stocks) : continue
            stock_summary = summarize_stock(code, stocks, node_filename.split("_")[1][:4].upper(), entry, filepath)