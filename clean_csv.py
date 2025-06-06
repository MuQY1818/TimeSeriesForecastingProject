import pandas as pd
import numpy as np

def clean_csv(input_file='total.csv', output_file='total_cleaned.csv'):
    print('开始读取CSV文件...')
    try:
        # 尝试使用gbk编码读取CSV文件
        df = pd.read_csv(input_file, encoding='gbk')
        
        print(f'原始数据形状: {df.shape}')
        print(f'原始列名: {df.columns.tolist()}')
        
        # 找到并删除重复的标题行
        header_mask = df.iloc[:, 0] == '日期'
        if header_mask.any():
            print(f'\n发现{header_mask.sum()}个重复的标题行，正在删除...')
            df = df[~header_mask]
        
        # 只保留前两列
        df = df.iloc[:, :2]
        
        # 确保列名正确
        df.columns = ['日期', '成交商品件数']
        
        # 检查重复日期
        duplicates = df['日期'].duplicated()
        if duplicates.any():
            print(f'\n发现{duplicates.sum()}个重复的日期，正在处理...')
            print('重复的日期:')
            print(df[df['日期'].duplicated(keep=False)].sort_values('日期'))
            
            # 对于重复的日期，保留成交量较大的记录
            df = df.sort_values('成交商品件数', ascending=False).drop_duplicates('日期')
            
        # 确保日期格式正确
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 按日期排序
        df = df.sort_values('日期')
        
        # 确保数据类型正确
        df['成交商品件数'] = pd.to_numeric(df['成交商品件数'], errors='coerce')
        df = df.dropna()  # 删除无效数据
        
        print('\n清理后数据预览:')
        print(df.head())
        print('\n基本统计信息:')
        print(df.describe())
        
        # 检查日期连续性
        date_range = pd.date_range(start=df['日期'].min(), end=df['日期'].max())
        missing_dates = set(date_range) - set(df['日期'])
        if missing_dates:
            print(f'\n发现{len(missing_dates)}个缺失日期')
            print('缺失日期示例（前5个）:')
            print(sorted(list(missing_dates))[:5])
        
        # 保存清理后的CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f'\n已保存清理后的文件到: {output_file}')
        print(f'最终数据形状: {df.shape}')
        
    except Exception as e:
        print(f'处理CSV文件时出错: {str(e)}')

if __name__ == '__main__':
    clean_csv() 