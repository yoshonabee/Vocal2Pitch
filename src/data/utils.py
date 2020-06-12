import torch

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

def make_target_tensor(onset_list, length):
    time_per_frame = 0.02

    target = torch.zeros(length).float()

    for onset in onset_list:
        frame = int(onset // time_per_frame)
        target[frame] = 1

    return target
