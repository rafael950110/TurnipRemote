#!/usr/bin/env python
# coding: utf-8

# ## ライブラリインポート

# In[1]:


import joblib
import turnip_main_code as turnip
import turnip_parameter as p


# In[2]:


directory_paths = {
    "RESULT_PATH" : "./result",
    "NODES_PATH"  : "./nodes",
    "CSV_PATH"    : "./csvs"
}


# ### Step01. Generate Algorythm

# In[5]:


turnip.step01(cdlst=p.cdlst, paths=directory_paths, parameters=p.step01Parameters, lists=p.lists, overwrite=True, newdir=False, dlonly=False)


# ### Step02. Simulation

# ### Step03. csv

# In[8]:


import csv
import requests
import json
import datetime

dt_now = datetime.datetime.now()
text = f'{dt_now.year}年{dt_now.month}月{dt_now.day}日実行分です'

tuning_name = [t['tuning_name'] for t in p.parameters]
turnip_spuit = []
#     f'　　 *3315*  3   0   3 5.263 2.66  15.789  36347.497      363,482,188 01  50'
colmn = f'{"上/下":<4}{"銘柄":^8}{"取引数(上下)"}{"平均利益率":^8}{"保持日数":^6}{"獲得村利益":^10}{"有効指数":^4}{"25日取引量平均":>14}{"ロジック":>5} {"シミュレーション数":^7}\n'

block  = '[ { "type" : "section", "text" : { "text" : "' + text + '", "type" : "mrkdwn" } }, { "type" : "divider" }'
# block  = '{ "blocks": [ { "type" : "section", "text" : { "text" : "本日分です", "type" : "mrkdwn" } }, { "type" : "divider" }'

for tuniname in tuning_name :
    for simnum in [200, 100, 50] :
        with open(f'{step02Paths["RESULT_PATH"]}/step02_result_{tuniname}_{simnum}.csv') as f:
            reader = csv.reader(f)
            for line in [row for row in reader][1:] :
                if( int(line[10]) > 0 and int(line[2]) > 1 and float(line[5]) >= 2.8 and float(line[9]) > 150000000.0 ):
                    block += ',{ "type": "context","elements": [{"type": "mrkdwn","text":'
                    block += ' "🔼' if line[1] == 'RISE' else ' "🔽'
                    block += f'{"*"+line[0]+"*":>8}'
                    block += f'{line[2]:>3}'
                    block += f'{line[3]:>3}'
                    block += f'{line[4]:>3}'
                    block += f'{float(line[5]):>7.3f}'
                    block += f'{float(line[6]):>6.2f}'
                    block += f'{float(line[7]):>8.3f}'
                    block += f'{float(line[8]):>12.3f}'
                    block += f'{round(float(line[9])):>26,}'
                    block += f'{tuniname:>10}'
                    block += f'{simnum:>6}'
                    block += '"}]}'
                    line.append(tuniname)
                    line.append(simnum)
                    turnip_spuit.append(line)
                    
block += ',{"type": "divider"}]'
# block += ',{"type": "divider"}]}'
turnip_spuit.insert(0, ["code", "criteria", "count_transaction", "count_rose", "count_fell", "profit_ave", "days_ave", "gross_profit_earn", "valid_exponent", "trans_vol_25daysAve", "entry", "tuning_name", "simuration_num"])

with open(f'{step02Paths["RESULT_PATH"]}/turnip_spuit.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(turnip_spuit)


# In[9]:


def slack_notify(block, url) :
    file_data = open(f'{step02Paths["RESULT_PATH"]}/turnip_spuit.csv', 'rb').read()
    requests.post(url, json={
        "channel"    : "turnip_python勉強会",
        "icon_emoji" : ":turnip:",
        "username"   : "TURNIP",
        "blocks"     : json.loads(block)
    })
    
# IBM_webhookURL    = 'https://hooks.slack.com/services/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'
Turnip_webhookURL = 'https://hooks.slack.com/services/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'

# slack_notify(block, IBM_webhookURL)
# slack_notify(block, Turnip_webhookURL)


# In[ ]:




