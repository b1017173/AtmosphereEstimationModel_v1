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

def writeArff(futures_arff_path, shs:list[SpeechState], estimation:str):
    with open(futures_arff_path, mode='w') as ar:
        ar.write('@relation {0}\n'.format(estimation))
        for i in range(len(shs[0].futures.futures)):
            ar.write('@attribute f{0} numeric\n'.format(i))
        ar.write('@attribute class {Positive, Negative}\n')
        ar.write('@data\n')
        for sh in shs:
            ar.write(sh.writeData(estimation))

def main():
    # speak_time_excel_path = sys.argv[1]
    # futures_arff_path = sys.argv[2]
    # estimation = svs.argv[3]
    estimation = 'excitement'
    speak_time_excel_path = 'data/voiceLine_v2.xlsx'
    futures_arff_path = 'out/{0}.arff'.format(estimation)
    shs = readSTExcel(speak_time_excel_path)
    writeArff(futures_arff_path, shs, estimation)

if __name__ == '__main__':
    main()