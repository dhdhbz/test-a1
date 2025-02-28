import pandas as pd
import requests
from tqdm import tqdm

# 高德地图API配置
API_KEY = 'f64657dd69c4a79cf1b5e4648209a211'  # 替换为实际API密钥
API_URL = 'https://restapi.amap.com/v3/geocode/geo?parameters'

def get_geocode(address):
    """通过高德API获取地址经纬度"""
    params = {
        'key': API_KEY,
        'address': address,
        'output': 'json'
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        data = response.json()
        if data['status'] == '1' and data['count'] != '0':
            location = data['geocodes'][0]['location']
            lng, lat = location.split(',')
            return f"{float(lat):.2f}, {float(lng):.2f}"  # 格式：纬度,经度
        return ""
    except Exception as e:
        print(f"Error for {address}: {str(e)}")
        return ""

# 读取Excel文件
df = pd.read_excel('地址.xlsx', sheet_name='Sheet1')

# 拼接地址列（根据实际列名调整）
df['完整地址'] = df['省份'].fillna('') + df['城市'].fillna('') + df['区'].fillna('') + df['镇'].fillna('')

# 获取经纬度并填充
tqdm.pandas(desc="获取经纬度")
df['经纬度'] = df['完整地址'].progress_apply(lambda x: get_geocode(x) if x else "")

# 保存结果到新文件
df.to_excel('地址_带经纬度.xlsx', index=False)
print("处理完成，结果已保存到 地址_带经纬度.xlsx")