import sys
import pandas as pd

from speechState import SpeechState

def readSTExcel(speak_time_excel_path):
    wb = pd.ExcelFile(speak_time_excel_path, engine='openpyxl') # ワークブックの読み込み
    shs = []
    for sh_name in wb.sheet_names:                              # 全シートループ
        print('--- {0} ---'.format(sh_name))
        sh = wb.parse(sh_name, usecols='B:IG')                  # シートの読み込み
        shs.append(SpeechState(sh))
    
    return shs

def writeArff(futures_arff_path, shs:list[SpeechState]):
    with open(futures_arff_path, mode='w') as ar:
        pass

def main():
    # speak_time_excel_path = sys.argv[1]
    # futures_arff_path = sys.argv[2]
    speak_time_excel_path = 'data/voiceLine_v2.xlsx'
    shs = readSTExcel(speak_time_excel_path)

if __name__ == '__main__':
    main()