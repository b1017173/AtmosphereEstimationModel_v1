from FeatureExtraction.src.speechState import SpeechState
from os import read
import sys
import pandas as pd
import openpyxl as op

def readSTExcel():
    speak_time_excel_path = sys.argv[1]
    futures_attr_path = sys.argv[2]

    wb = pd.ExcelFile(speak_time_excel_path, engine='openpyxl') # ワークブックの読み込み
    shs = []
    for sh_name in wb.sheet_names:                              # 全シートループ
        sh = wb.parse(sh_name, usecols='B:IG')                  # シートの読み込み
        shs.append(SpeechState(sh))
    
    return shs

def main():
    shs = readSTExcel()

if __name__ == '__main__':
    main()