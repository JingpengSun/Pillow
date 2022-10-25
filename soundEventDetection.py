import numpy as np


def get_splited_event(feature, onset_frame, offset_frame, threshold1):
    # 对于呼吸暂停造成的呼吸急促，事件之间的间隔会变短，因此需要针对这种情况减少对最短静音帧数的要求
    # 同时提高特征阈值以便能够很好地将连在一起的事件分开
    threshold1 = 1.5*threshold1
    threshold2 = 3*threshold1
    max_frame_silence = 5   # 静音时长达到5帧即认为是有效静音
    
    onset_list, offset_list = sed_param1D(feature, max_frame_silence, threshold1, threshold2
                                          
                                          
    return splited_onset, splited_offset
    
    


def 




threshold1 = np.mean(energy)
threshold2 = 1.2*threshold1

max_event_duration = 4    # max duration seconds threshold  
onset_list = dfs.onset.tolist()
offset_list = dfs.offset.tolist()

# 对于时长超过阈值的连续片段
for i in range(len(onset_list)):
    if offset[i] - onset[i]>=max_event_duration:   # 说明此时有可能存在呼吸急促造成的事件连在了一起，需要进行细分
        onset_frame = time2frame(onset[i])
        offset_frame = time2frame(offset[i])
        splited_event_onset, splited_event_offset = get_splited_event(feature, onset_frame, offset_frame, threshold1)

        
