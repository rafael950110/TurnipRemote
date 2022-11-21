from csv import reader

parameters = [
    {
        "tuning_name"      : "01",
        "learn_index"      : 200,
        "learn_num_min"    : 1000,
        "sim_num"          : [50, 100, 200],
        "judge_days"       : 5,
        "judge_border"     : 7.5,
        "min_samples_leaf" : 10,
        "max_depth"        : 12,
        "criteria"         : 70,
        "holding_border"   : 5,
        "profit_border"    : [5, 4]
    },
#     {
#         "tuning_name"      : "02",
#         "learn_index"      : 200,
#         "learn_num_min"    : 1000,
#         "sim_num"          : [50, 100, 200],
#         "judge_days"       : 5,
#         "judge_border"     : 7.5,
#         "min_samples_leaf" : 10,
#         "max_depth"        : 12,
#         "criteria"         : 70,
#         "holding_border"   : 5,
#         "profit_border"    : [4, 3]
#     },
#     {
#         "tuning_name"      : "03",
#         "learn_index"      : 200,
#         "learn_num_min"    : 1000,
#         "sim_num"          : [50, 100, 200],
#         "judge_days"       : 5,
#         "judge_border"     : 5,
#         "min_samples_leaf" : 10,
#         "max_depth"        : 15,
#         "criteria"         : 70,
#         "holding_border"   : 5,
#         "profit_border"    : [4, 3]
#     },
    {
        "tuning_name"      : "05",
        "learn_index"      : 200,
        "learn_num_min"    : 1000,
        "sim_num"          : [50, 100, 200],
        "judge_days"       : 3,
        "judge_border"     : 4,
        "min_samples_leaf" : 10,
        "max_depth"        : 12,
        "criteria"         : 70,
        "holding_border"   : 3,
        "profit_border"    : [4, 3]
    },
    {
        "tuning_name"      : "07",
        "learn_index"      : 200,
        "learn_num_min"    : 1000,
        "sim_num"          : [50, 100, 200],
        "judge_days"       : 4,
        "judge_border"     : 5,
        "min_samples_leaf" : 10,
        "max_depth"        : 12,
        "criteria"         : 70,
        "holding_border"   : 4,
        "profit_border"    : [5, 4]
    }
]

lists = {
    "judgeColList"  : ["上昇判定結果", "下落判定結果"],
    "removeColList" : ["Date", "Close_origin"],
    "renameColList" : { "Close_origin" : ["Adj Close"] }
}

paths = {
    "NODES_PATH"  : "./nodes",
    "CSV_PATH"    : "./csvs",
    "RESULT_PATH" : "./result"
}

step01_path_names = ["NODES_PATH", "CSV_PATH"]
step02_path_names = ["NODES_PATH", "CSV_PATH", "RESULT_PATH"]
step03_path_names = ["NODES_PATH", "CSV_PATH"]

step01_parameter_names = ["tuning_name", "learn_index", "learn_num_min", "judge_days", "judge_border", "min_samples_leaf", "max_depth", "criteria"]
step02_parameter_names = ["tuning_name", "learn_index", "sim_num", "judge_days", "judge_border", "holding_border", "profit_border"]
step03_parameter_names = ["tuning_name", "data_num"]

step01Paths = { key: value for key, value in paths.items() if key in step01_path_names }
step02Paths = { key: value for key, value in paths.items() if key in step02_path_names }
step03Paths = { key: value for key, value in paths.items() if key in step03_path_names }

step01Parameters = [{ key: value for key, value in dic.items() if key in step01_parameter_names } for dic in parameters]
step02Parameters = [{ key: value for key, value in dic.items() if key in step02_parameter_names } for dic in parameters]
step03Parameters = [{ key: value for key, value in dic.items() if key in step03_parameter_names } for dic in parameters]

# スクレイピング用クッキー
Y = "v=1&n=7or81vjtjipcr&l=5kc8o0_c8o0d8i78/o&p=m2nvvjpa52000g00&ig=00mi2&r=i0&lg=ja-JP&intl=jp"
T = "z=8f/6iB8HOEjBlvxc0XWJy73Tzc3NwY1MDI2MU8wNk8-&sk=DAAP0Z5QpfXaAJ&ks=EAAfAMLiKrPW4wHJ2pMXFI91g--~F&kt=EAAvDhHbq2fX7fqCj0BzJ_rsg--~E&ku=FAAMEUCIAJlNNmXWsEx5gqmIpgdsvVyv2HiaEj0fVLrP4NqSknIAiEArinaZopv5sMluoDjvsuvxsySmlMsI_4k9I.QbjpTE8k-~B&d=dGlwATBKRkpGQwFhAVFBRQFnAVQzNElTSVlNN1JPM09TUlg1WTZTSVNJVTZBAXNsAU9EQXdNQUV5TnpVeE5qZzNNVGctAXNjAWZpbmFuY2UBenoBOGYvNmlCQTJK"

# カラム
COL_NAME_Open="Open"
COL_NAME_High="High"
COL_NAME_Low="Low"
COL_NAME_Close="Close"
COL_NAME_Volume="Volume"
COL_NAME_Close_origin="Close_origin"

with open('./code.csv') as csv_file: cdlst = sorted([x[0] for x in list(reader(csv_file))])
# cdlst=[ "3681", "3110", "6197", "3921", "6191", "1712" ]