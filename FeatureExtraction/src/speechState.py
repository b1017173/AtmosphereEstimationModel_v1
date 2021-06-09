from futures import Futures
import pandas as pd
import numpy as np

class SpeechState:
    def __init__(self, sh:pd.DataFrame):
        _speech_recording = sh[0:4].fillna(0).astype(int).to_numpy()
        _speech_recording = self.sortSpeakingTime(_speech_recording)
        self.classificationSpeechStates(_speech_recording)
        self.futures = Futures(self.unspeakline, self.speaklines, self.simultaneous_speaklines)

        _speechAtomosphere = sh[7:8].astype(float).to_numpy()
        self.excitement = _speechAtomosphere[0, 0]
        self.seriousness = _speechAtomosphere[0, 1]
        self.cheerfulness = _speechAtomosphere[0, 2]
        self.comfortable = _speechAtomosphere[0, 3]
    
    # 発話量が多い順に並び替える
    def sortSpeakingTime(self, speech_recording:np.ndarray):
        if np.size(speech_recording, axis=0) < 2:
            return speech_recording
        
        _sum_rows = np.argsort(np.sum(speech_recording, axis=1))
        _sorted = speech_recording[_sum_rows]
        return _sorted[::-1]
  
    # 各発話状態の抽出
    def classificationSpeechStates(self, speech_recording:np.ndarray):
        self.unspeakline:np.ndarray = []                            # 非発話状態
        self.speaklines:list = [[], [], [], []]               # 各話者の発話状態
        self.simultaneous_speaklines:list = [[], [], [], []]   # 同時発話状態

        _prestatus = np.full(4, -1)
        for i in range(np.size(speech_recording, axis=1)):
            _half_second = speech_recording[:, i]
            if _half_second.sum() == 0:
                if _prestatus.sum() == 0:
                    self.unspeakline[len(self.unspeakline) - 1] += 0.5
                else:
                    self.unspeakline = np.append(self.unspeakline, 0.5)
            
            for j, value in enumerate(_half_second):
                if value > 0:
                    if _prestatus[j] > 0:
                        self.speaklines[j][len(self.speaklines[j]) - 1] += 0.5
                    else:
                        self.speaklines[j] = np.append(self.speaklines[j], 0.5)
                    
                    if _half_second.sum() > 1:
                        if _prestatus[j] > 0 and _prestatus.sum() > 1:
                            self.simultaneous_speaklines[j][len(self.simultaneous_speaklines[j]) - 1] += 0.5
                        else:
                            self.simultaneous_speaklines[j] = np.append(self.simultaneous_speaklines[j], 0.5)

            _prestatus = _half_second

        # 無音状態の最適化
        if speech_recording.sum(axis=0)[0] == 0:
            self.unspeakline = np.delete(self.unspeakline, [0])
        if speech_recording.sum(axis=0)[len(speech_recording.sum(axis=0)) -1] == 0:
            self.unspeakline = np.delete(self.unspeakline, [len(self.unspeakline) - 1])
        
        print('非発話状態: ', self.unspeakline)
        print('発話状態: ', self.speaklines)
        print('同時発話状態: ', self.simultaneous_speaklines)
