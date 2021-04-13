from FeatureExtraction.src.speechFutures import Futures
import pandas as pd
import numpy as np
import openpyxl as op

class SpeechState:
    def __init__(self, sh:pd.DataFrame):
        self.speechRecording = sh[0:4].fillna(0).astype(int).to_numpy
        self.speakerA = self.judgeSpeakerA(self.speechRecording)
        self.classificationSpeechStates(self.speechRecording, self.speakerA)
        self.futures = Futures(self.unspeakline, self.speakerAvoiceline, self.speakerBvoiceline, self.simultaneousVoiceline)

    def judgeSpeakerA(self, speechRecording:np.ndarray):
        _speakerA = 0
        for i, value in enumerate(speechRecording):
            if speechRecording[_speakerA].sum() < value.sum():
                _speakerA = i

        # print('A話者:', _speakerA)
        return _speakerA
    
    def classificationSpeechStates(self, speechRecording:np.ndarray, speakerA:int):
        self.unspeakline:np.ndarray = []                       # 非発話状態
        self.speakerAvoiceline:np.ndarray = []                 # A話者単独発話状態
        self.speakerBvoiceline:np.ndarray = [[],[],[],[]]      # B話者単独発話状態
        self.simultaneousVoiceline:np.ndarray = []             # A話者同時発話状態

        _prestatus = -1

        for i, sum in enumerate(speechRecording.sum(axis=0)):
            if sum == 0:
                self.unspeakline, _prestatus = self.addValueToVoiceline(self.unspeakline, _prestatus, 5)
            elif sum == 1:
                for j, value in enumerate(sum):
                    if value[i] > 0:
                        if speakerA == j:
                            self.speakerAvoiceline, _prestatus = self.addValueToVoiceline(self.speakerAvoiceline, _prestatus, j)
                        else:
                            self.speakerBvoiceline[j], _prestatus = self.addValueToVoiceline(self.speakerBvoiceline[j], _prestatus, j)
                        break
            else:
                self.simultaneousVoiceline, _prestatus = self.addValueToVoiceline(self.simultaneousVoiceline, _prestatus, 4)
        
        # 無音状態の最適化
        if speechRecording.sum(axis=0)[0] == 0:
            self.unspeakline = np.delete(self.unspeakline, [0])
        if speechRecording.sum(axis=0)[len(speechRecording.sum(axis=0)) -1] == 0:
            self.unspeakline = np.delete(self.unspeakline, [len(self.unspeakline) - 1])
        
        # B単独発話の最適化
        if sum(len(v) for v in self.speakerBvoiceline) != 0:
            self.speakerBvoiceline = [i for i in self.speakerBvoiceline if len(i) != 0]
        else:
            self.speakerBvoiceline = []

        print('非発話状態:', self.unspeakline)
        print('A話者単独発話状態:', self.speakerAvoiceline)
        print('B話者単独発話状態:', self.speakerBvoiceline)
        print('同時発話状態:', self.simultaneousVoiceline)
        
    def addValueToVoiceline(self, voiceline:np.ndarray, pre_value, now_value):
        # print('pre:', pre_value, 'now', now_value)
        if pre_value == now_value:
            # print('len(voiceline):', len(voiceline))
            voiceline[len(voiceline) - 1] += 0.5
        else:
            voiceline = np.append(voiceline, 0.5)
            # print('voiceline:', voiceline)
        pre_value = now_value
        
        return voiceline, pre_value