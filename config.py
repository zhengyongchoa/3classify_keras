workspace = "D:\code\PYcode\AgeRecognition\Agekeras"

# path
wav_dir = 'wav'
feat_dir ='feature '
out_model_dir = 'modelcheck '


# config
sample_rate = 16000
n_window = 1024
n_overlap = 512      # ensure 240 frames in 10 seconds
max_len = 240        # sequence max length is 10 s, 240 frames. 
step_time_in_sec = float(n_window - n_overlap) / sample_rate

          
# idx_to_id = {index: id for index, id in enumerate(ids)}
# id_to_idx = {id: index for index, id in enumerate(ids)}
# idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
# lb_to_idx = {lb: index for index, lb in enumerate(lbs)}


num_classes = 3
n_time = 21
n_freq = 63


