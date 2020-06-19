import numpy as np
import torch

def get_onset_list(df, thres):
    result = [[df['start'][0], df['pitch'][0]]]

    for i in range(1, df.shape[0]):
        if df['start'][i] - df['end'][i - 1] > thres:
            result.append([
                df['end'][i - 1],
                0
            ])

            result.append([
                df['start'][i],
                df['pitch'][i]]
            )
        else:
            result.append([
                (df['start'][i] + df['end'][i - 1]) / 2,
                df['pitch'][i]
            ])

    result.append([df['end'].iloc[-1], 0])

    return np.array(result)

def make_target_tensor(onset_list, start_time, time_length, length):
    time_per_frame = time_length / length

    tensor = torch.zeros(length)

    for onset in onset_list:
        frame = int((onset - start_time) // time_per_frame)
        tensor[frame] = 1
        
        if frame > 0:
            tensor[frame - 1] = 0

        if frame < length - 1:
            tensor[frame + 1] = 0

    return tensor

def make_pitch_tensor(pitch_list, start_time, time_length, length):
    try:
        time_per_frame = time_length / length
   
        tensor = []
        onset_idx = 0

        if pitch_list[0][0] < start_time:
            while (len(tensor) + 1) * time_per_frame + start_time < pitch_list[1][0]:
                tensor.append(pitch_list[0][1])

            onset_idx = 1

        for i in range(len(tensor), length):
            if onset_idx + 1 < len(pitch_list):
                t = (i + 1) * time_per_frame + start_time

                if t >= pitch_list[onset_idx + 1][0]:
                    onset_idx += 1

            tensor.append(pitch_list[onset_idx][1])

        return torch.tensor(tensor).float()
    except:
        return torch.zeros(length).float()
