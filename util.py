import numpy as np
def sortedpeaks(score):
    peaks = []
    for i in range(1, len(score) - 2):
        if not np.isnan(score[i]):
            if np.isnan(score[i-1]) or score[i] > score[i-1]:
                if np.isnan(score[i+1]) or score[i] > score[i+1]:
                    peaks.append(i)
    return sorted(peaks, key=lambda x: score[x], reverse=True)
