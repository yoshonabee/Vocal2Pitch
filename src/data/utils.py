import torch
from audiolazy.lazy_midi import freq2midi

def get_onset_list(df, thres):
    result = [df['start'][0]]

    for i in range(1, df.shape[0]):
        if df['start'][i] - df['end'][i - 1] > thres:
            result.append(df['end'][i - 1])
            result.append(df['start'][i])
        else:
            result.append((df['start'][i] + df['end'][i - 1]) / 2)

    result.append(df['end'].iloc[-1])

    return result

def make_target_tensor(onset_list, start_time, time_length, length):
    time_per_frame = time_length / length

    tensor = torch.zeros(length)

    for onset in onset_list:
        frame = int((onset - start_time) // time_per_frame)
        try:
            tensor[frame - 1] = 0.6
            tensor[frame - 2] = 0.2
        except:
            pass

        try:
            tensor[frame + 1] = 0.6
            tensor[frame + 2] = 0.2
        except:
            pass

        tensor[frame] = 1

    return tensor

