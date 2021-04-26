import numpy as np
import openpyxl as op

class Futures:
    def __init__(self, unspeakline:np.ndarray, speakerAvoiceline:np.ndarray, speakerBvoiceline:list, simultaneousVoiceline:np.ndarray):
        # データの読み込み・整備
        self.unspeakline:np.ndarray = unspeakline
        if len(self.unspeakline) == 0:
            self.unspeakline = np.empty(0)
        self.speakerAvoiceline:np.ndarray = speakerAvoiceline
        self.speakerBvoiceline:list = speakerBvoiceline
        self.simultaneousVoiceline:np.ndarray = simultaneousVoiceline
        if len(self.simultaneousVoiceline) == 0:
            self.simultaneousVoiceline = np.empty(0)
        self.voiceline = np.append(np.append(self.unspeakline, self.simultaneousVoiceline), self.speakerAvoiceline)
        self.speakersVoiceline = self.speakerAvoiceline
        for i in self.speakerBvoiceline:
            if len(i) > 0:
                for j in i:
                    self.voiceline = np.append(self.voiceline, j)
                    self.speakersVoiceline = np.append(self.speakersVoiceline, j)
        print('voiceline:', self.voiceline)
        self.speakTime = self.unspeakline.sum() + self.speakerAvoiceline.sum() + sum([sum(i) for i in self.speakerBvoiceline]) + self.simultaneousVoiceline.sum()
        # print('speakTime:', self.speakTime)

        # 特徴量抽出
        self.calFutures()
    
    def calFutures(self):
        self.f1 = np.mean(self.speakerAvoiceline)
        self.f2 = np.var(self.speakerAvoiceline)
        self.f3 = np.min(self.speakerAvoiceline) if (len(self.speakerAvoiceline) != 0) else np.nan
        self.f4 = np.max(self.speakerAvoiceline) if (len(self.speakerAvoiceline) != 0) else np.nan
        self.f5 = len(self.speakerAvoiceline)
        self.f6 = self.speakerAvoiceline.sum() / self.speakTime
        print('F1:', self.f1, 'F2:', self.f2, 'F3:', self.f3, 'F4:', self.f4, 'F5:', self.f5, 'F6:', self.f6)

        self.f7 = [np.mean(i) for i in self.speakerBvoiceline]
        self.f8 = [np.var(i) for i in self.speakerBvoiceline]
        self.f9 = [np.min(i) for i in self.speakerBvoiceline] if (len(self.speakerBvoiceline) != 0) else [np.nan]
        self.f10 = [np.max(i) for i in self.speakerBvoiceline] if (len(self.speakerBvoiceline) != 0) else [np.nan]
        self.f11 = [len(i) for i in self.speakerBvoiceline]
        self.f12 = [(sum(i) / self.speakTime) for i in self.speakerBvoiceline]
        print('F7:', self.f7, 'F8:', self.f8, 'F9:', self.f9, 'F10:', self.f10, 'F11:', self.f11, 'F12:', self.f12)

        self.f13 = np.mean(self.simultaneousVoiceline)
        self.f14 = np.var(self.simultaneousVoiceline)
        self.f15 = np.min(self.simultaneousVoiceline) if (len(self.simultaneousVoiceline) != 0) else np.nan
        self.f16 = np.max(self.simultaneousVoiceline) if (len(self.simultaneousVoiceline) != 0) else np.nan
        self.f17 = len(self.simultaneousVoiceline)
        self.f18 = self.simultaneousVoiceline.sum() / self.speakTime
        print('F13:', self.f13, 'F14:', self.f14, 'F15:', self.f15, 'F16:', self.f16, 'F17:', self.f17, 'F18:', self.f18)

        self.f19 = np.mean(self.unspeakline)
        self.f20 = np.var(self.unspeakline)
        self.f21 = np.min(self.unspeakline) if (len(self.unspeakline) != 0) else np.nan
        self.f22 = np.max(self.unspeakline) if (len(self.unspeakline) != 0) else np.nan
        self.f23 = len(self.unspeakline)
        self.f24 = self.unspeakline.sum() / self.speakTime
        print('F19:', self.f19, 'F20:', self.f20, 'F21:', self.f21, 'F22:', self.f22, 'F23:', self.f23, 'F24:', self.f24)

        self.f25 = [[(self.f1 / i) for i in self.f7], [self.f7[i] / self.f7[j] for i in range(len(self.f7)) for j in range(len(self.f7)) if i < j]]
        self.f26 = [[(self.f2 / i) for i in self.f8], [self.f8[i] / self.f8[j] for i in range(len(self.f8)) for j in range(len(self.f8)) if i < j]]
        self.f27 = [[(self.f3 / i) for i in self.f9], [self.f9[i] / self.f9[j] for i in range(len(self.f9)) for j in range(len(self.f9)) if i < j]]
        self.f28 = [[(self.f4 / i) for i in self.f10], [self.f10[i] / self.f10[j] for i in range(len(self.f10)) for j in range(len(self.f10)) if i < j]]
        self.f29 = [[(self.f5 / i) for i in self.f11], [self.f11[i] / self.f11[j] for i in range(len(self.f11)) for j in range(len(self.f11)) if i < j]]
        self.f30 = [[(self.f6 / i) for i in self.f12], [self.f12[i] / self.f12[j] for i in range(len(self.f12)) for j in range(len(self.f12)) if i < j]]
        print('F25:', self.f25, 'F26:', self.f26, 'F27:', self.f27, 'F28:', self.f28, 'F29:', self.f29, 'F30:', self.f30)

        # F31~48は無意味な特徴量だったので省略

        self.f49 = [[self.f19 / np.mean(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f19 / np.mean(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f50 = [[self.f20 / np.var(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f20 / np.var(np.append(i, self.simultaneousVoiceline)) if np.var(np.append(i, self.simultaneousVoiceline))!= 0 else np.nan for i in self.speakerBvoiceline]]
        self.f51 = [[self.f21 / np.min(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f21 / np.min(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f52 = [[self.f22 / np.max(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f22 / np.max(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f53 = [[self.f23 / len(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f23 / len(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f54 = [[self.f24 / (self.speakerAvoiceline.sum() + self.simultaneousVoiceline.sum() / self.speakTime)], [self.f24 / (i.sum() + self.simultaneousVoiceline.sum() / self.speakTime) for i in self.speakerBvoiceline]]
        print('F49:', self.f49, 'F50:', self.f50, 'F51:', self.f51, 'F52:', self.f52, 'F53:', self.f53, 'F54:', self.f54)

        self.f55 = [[self.f13 / self.f1], [self.f13 / i for i in self.f7]]
        self.f56 = [[self.f14 / self.f2], [self.f14 / i for i in self.f8]]
        self.f57 = [[self.f15 / self.f3], [self.f15 / i for i in self.f9]]
        self.f58 = [[self.f16 / self.f4], [self.f16 / i for i in self.f10]]
        self.f59 = [[self.f17 / self.f5], [self.f17 / i for i in self.f11]]
        self.f60 = [[self.f18 / self.f6], [self.f18 / i for i in self.f12]]
        print('F55:', self.f55, 'F56:', self.f56, 'F57:', self.f57, 'F58:', self.f58, 'F59:', self.f59, 'F60:', self.f60)

        self.f61 = [[self.f13 / np.mean(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f13 / np.mean(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f62 = [[self.f14 / np.var(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f14 / np.var(np.append(i, self.simultaneousVoiceline)) if np.var(np.append(i, self.simultaneousVoiceline))!= 0 else np.nan for i in self.speakerBvoiceline]]
        self.f63 = [[self.f15 / np.min(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f15 / np.min(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f64 = [[self.f16 / np.max(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f16 / np.max(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f65 = [[self.f17 / len(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [self.f17 / len(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f66 = [[self.f18 / (self.speakerAvoiceline.sum() + self.simultaneousVoiceline.sum() / self.speakTime)], [self.f18 / (i.sum() + self.simultaneousVoiceline.sum() / self.speakTime) for i in self.speakerBvoiceline]]
        print('F61:', self.f61, 'F62:', self.f62, 'F63:', self.f63, 'F64:', self.f64, 'F65:', self.f65, 'F66:', self.f66)

        self.f67 = [[self.f1 / np.mean(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [np.mean(i) / np.mean(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f68 = [[self.f2 / np.var(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [np.var(i) / np.var(np.append(i, self.simultaneousVoiceline)) if np.var(np.append(i, self.simultaneousVoiceline))!= 0 else np.nan for i in self.speakerBvoiceline]]
        self.f69 = [[self.f3 / np.min(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [np.min(i) / np.min(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f70 = [[self.f4 / np.max(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [np.max(i) / np.max(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f71 = [[self.f5 / len(np.append(self.speakerAvoiceline, self.simultaneousVoiceline))], [len(i) / len(np.append(i, self.simultaneousVoiceline)) for i in self.speakerBvoiceline]]
        self.f72 = [[self.f6 / (self.speakerAvoiceline.sum() + self.simultaneousVoiceline.sum() / self.speakTime)], [i.sum() / (i.sum() + self.simultaneousVoiceline.sum()) for i in self.speakerBvoiceline]]
        print('F67:', self.f67, 'F68:', self.f68, 'F69:', self.f69, 'F70:', self.f70, 'F71:', self.f71, 'F72:', self.f72)

        self.f73 = self.f19 / np.mean(self.voiceline)
        self.f74 = self.f20 / np.var(self.voiceline)
        self.f75 = self.f21 / np.min(self.voiceline)
        self.f76 = self.f22 / np.max(self.voiceline)
        self.f77 = self.f23 / len(self.voiceline)
        self.f78 = self.f24 / (self.voiceline.sum() / self.speakTime)
        print('F73:', self.f73, 'F74:', self.f74, 'F75:', self.f75, 'F76:', self.f76, 'F77:', self.f77, 'F78:', self.f78)

        self.f79 = self.f13 / np.mean(self.speakersVoiceline)
        self.f80 = self.f14 / np.var(self.speakersVoiceline)
        self.f81 = self.f15 / np.min(self.speakersVoiceline)
        self.f82 = self.f16 / np.max(self.speakersVoiceline)
        self.f83 = self.f17 / len(self.speakersVoiceline)
        self.f84 = self.f18 / (self.speakersVoiceline.sum() / self.speakTime)
        print('F79:', self.f79, 'F80:', self.f80, 'F81:', self.f81, 'F82:', self.f82, 'F83:', self.f83, 'F84:', self.f84)