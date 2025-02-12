import pandas as pd
import re
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

def adjust_column_width(worksheet):
    """动态调整列宽"""
    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter  # 获取列字母
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column].width = adjusted_width

def process_result_file(input_file, output_file):
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按###分割不同模型的结果
    sections = content.split('###')[1:]  # 跳过第一个空部分
    
    # 准备数据存储
    data = []
    columns = ['Model']
    
    for section in sections:
        lines = section.strip().split('\n')
        model_name = lines[0].strip()
        
        # 解析指标行
        metrics = {}
        for line in lines[1:]:
            if 'Total time cost' in line:
                metrics['time_cost'] = float(re.search(r'Total time cost: ([\d.]+)', line).group(1))
                continue
            # 使用正则表达式提取指标名和值
            matches = re.findall(r'(\w+) ([\d.]+)', line)
            for metric, value in matches:
                if metric not in columns[1:]:
                    columns.append(metric)
                metrics[metric] = float(value)
        
        # 添加到数据列表
        row = [model_name] + [metrics.get(col, '') for col in columns[1:]]
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 保存到Excel，并设置格式
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        
        # 获取工作表
        worksheet = writer.sheets['Results']
        
        # 调整列宽
        adjust_column_width(worksheet)
        
        # 设置标题行格式
        for cell in worksheet[1]:
            cell.style = 'Headline 3'
    
    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    input_file = "result.md"
    output_file = "results_summary.xlsx"
    process_result_file(input_file, output_file)
