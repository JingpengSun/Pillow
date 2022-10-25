import numpy as np


def get_splited_event(feature, onset_frame, offset_frame, threshold1):
    # 对于呼吸暂停造成的呼吸急促，事件之间的间隔会变短，因此需要针对这种情况减少对最短静音帧数的要求
    # 同时提高特征阈值以便能够很好地将连在一起的事件分开
    threshold1 = 1.5*threshold1
    threshold2 = 3*threshold1
    max_frame_silence = 5   # 静音时长达到5帧即认为是有效静音
    
    onset_list, offset_list = sed_param1D(feature, max_frame_silence, threshold1, threshold2
                                          
                                          
    return splited_onset, splited_offset
    
    
def time2frame(event_time, hop_length):
    frame_index = int(event_time//hop_length)
    return frame_index

def regression_onset_and_offset(feature, onset, offset, neighbor_duration):
    # onset, 起始时刻，in seconds
    # offset, 结束时刻
    # neighbor_duration, 当前位置与相距neighbor_duration的点进行对比
    # 具体来说，onset与其前后各neighbor_duration之间的关系，offset也是如此
    onset_frame_index = time2frame(onset)
    offset_frame_index = time2frame(offset)
    hop_duration = neighbor_duration/2
    nb_frame_num = time2frame(neighbor_duration)
    feature[onset_frame_index-nb_frame_num]
    
    
    




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

        
