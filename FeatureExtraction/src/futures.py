import numpy as np

class Futures:
    def __init__(self, unspeakline:np.ndarray, speaklines:np.ndarray, simultaneous_speaklines:np.ndarray) -> None:
        # 各発話状態の代入
        _unspeakline = unspeakline
        _speaklines = speaklines
        _simultaneous_speaklines = simultaneous_speaklines
        self.futures = self.calFutures(_unspeakline, _speaklines, _simultaneous_speaklines)
        # print('特徴量: ', self.futures)
    
    def calFutures(self, unspeakline:np.ndarray, speaklines:list, simultaneous_speaklines:list) -> np.ndarray:
        _futures = []   # 計算した特徴量を格納していく

        # 全集合(発話・非発話)の生成
        _allstate = unspeakline                    # 一次元の全集合
        for speakline in speaklines:
            _allstate = np.append(_allstate, speakline)
        _two_dimen_allstate:list = [unspeakline]
        for speakline in speaklines:               # 二次元の全集合
            _two_dimen_allstate.append(speakline)
        
        # 全発話集合の生成(一次元)
        _all_speakline:np.ndarray = []
        for speakline in speaklines:
            _all_speakline = np.append(_all_speakline, speakline)

        # 発話時間統計特徴の算出
        ## 非発話状態
        _futures = np.append(_futures, self.calBasicStats(unspeakline, _allstate))
        ## 単独発話状態
        for speakline in speaklines:
            _futures = np.append(_futures, self.calBasicStats(speakline, _allstate))
        ## 同時発話状態
        for simultaneous_speakline in simultaneous_speaklines:
            _futures = np.append(_futures, self.calBasicStats(simultaneous_speakline, _allstate))
        ## 全集合
        _futures = np.append(_futures, self.calBasicStats(_allstate, _allstate))
        ## 全発話集合
        _futures = np.append(_futures, self.calBasicStats(_all_speakline, _allstate))

        # 発話時間比較特徴
        ## 話者間単独発話比較
        for i in range(len(_two_dimen_allstate) - 1):
            for j in range(i + 1, len(_two_dimen_allstate)):
                _futures = np.append(_futures, self.calComparison(_two_dimen_allstate[i], _two_dimen_allstate[j], _allstate))
        ## 話者間同時発話比較
        for i in range(len(speaklines) - 1):
            for j in range(i + 1, len(speaklines)):
                _futures = np.append(_futures, self.calComparison(speaklines[i], speaklines[j], _allstate))
        ## 単独注目発話比較(1:その他の比較)
        for i in range(len(speaklines)):
            _other_speakline:list = []
            for j, speakline in enumerate(speaklines):
                if i == j:
                    continue
                _other_speakline = np.append(_other_speakline, speakline)
            _futures = np.append(_futures, self.calComparison(speaklines[i], _other_speakline, _allstate))
        ## 話者内発話比較
        for i in range(len(speaklines)):
            _futures = np.append(_futures, self.calComparison(speaklines[i], simultaneous_speaklines[i], _allstate))

        return _futures

    # 基本統計量の算出
    def calBasicStats(self, array:np.ndarray, allstate:np.ndarray) -> np.ndarray:
        if np.size(array) == 0:
            return np.full(6, np.nan)
        _mean = np.mean(array)
        _std = np.std(array)
        _max = np.max(array)
        _size = np.size(array)
        _sum = np.sum(array)
        _share = np.sum(array) / np.sum(allstate)

        return np.array([_mean, _std, _max, _size, _sum, _share])
    
    # 比較量の算出
    def calComparison(self, first_array:np.ndarray, second_array:np.ndarray, allstate:np.ndarray):
        _firstBS = self.calBasicStats(first_array, allstate)
        _secondBS = self.calBasicStats(second_array, allstate)
        _comparison:np.ndarray = []

        for i in range(np.size(_firstBS)):
            if _firstBS[i] == np.nan or _firstBS[i] == 0 or _secondBS[i] == np.nan:
                _comparison = np.append(_comparison, np.nan)
            else:
                _comparison = np.append(_comparison, _secondBS[i] / _firstBS[i])
        
        return _comparison
    
    # 書き出し用に文字列変換
    def writeFutures(self):
        futures_str = ''
        for future in self.futures:
            futures_str += '{0},'.format(str(future) if ~np.isnan(future) else '?')

        return futures_str