
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

task_name = 'flow_rate_allreduce_50m_hpcc_10us'
# task_name = 'flow_rate_allreduce_50m_dcqcn_20us'


def get_all_data():
    print('handling data...')
    rate_list_all = []
    with open(f'logs/{task_name}.txt', 'r') as f:
        lines = f.readlines()
    f.close()
    for line in tqdm(lines):
        parts = line.split(' ')
        time, src, dst, rate = parts
        time = int(time) / 1e3
        rate = float(rate)
        rate_list_all.append((src, dst, time, rate))
    print(f'frames amount: {len(rate_list_all)}')

    device_names = set()
    for item in rate_list_all:
        src, dst = item[0], item[1]
        device_names.add(src)
        device_names.add(dst)
    
    device_names = list(device_names)
    device_names.sort()

    data_all = {}
    for src_name in device_names:
        for dst_name in device_names:
            if src_name == dst_name:
                continue    
            data_all[(src_name, dst_name)] = []
    for item in tqdm(rate_list_all):
        data_all[(item[0], item[1])].append((item[2], item[3]))
    
    del_keys = []
    for name_index in data_all.keys():
        if len(data_all[name_index]) == 0:
            del_keys.append(name_index)
        else:
            data_all[name_index].sort(key=lambda p: p[0])
    for name_index in del_keys:
        del data_all[name_index]

    print(f'comm: {len(data_all.keys())}')

    fct_all = []
    fct_name = '_'.join(task_name.replace('flow_rate', 'fct').split('_')[:-1])
    with open(f'logs/{fct_name}.txt', 'r') as f:
        lines = f.readlines()
    f.close()
    for line in tqdm(lines):
        parts = line.split(' ')
        sip, dip, sport, dport = parts[:4]
        fst, fct = (int(parts[-3])) / 1e3, (int(parts[-3]) + int(parts[-2])) / 1e3
        fct_all.append((sip, dip, sport, dport, fst, fct))
    print(len(fct_all))
    return data_all, fct_all

def analyse_all(data_all, fct_all, method='rel', n_process=1, draw=False):
    k = 10
    if draw:
        threshold_list = [5]
    elif method == 'abs':
        # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5]
        threshold_list = [0.1, 0.5, 1, 2, 5, 10]
        
    elif method == 'rel':
        threshold_list = [0.1, 0.5, 1, 2, 5, 7, 10, 20, 30, 40, 50]
        # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5]

    args_params = []    
    for threshold in threshold_list:
        for name_index in data_all.keys():
            src, dst = name_index
            data_single_flow = data_all[(src, dst)]
            # if src != '0b000001':
            #     continue
            args_params.append((data_single_flow, fct_all, src, dst, k, threshold, method, draw))

    print(f'args combs: {len(args_params)}')
    n_process = min(len(args_params), 32)
    with mp.Pool(processes=n_process) as pool:
        results = pool.map(analyse_single, args_params)

    all_results = analyse_results_stat(results, k, threshold_list, method)
    if not draw:
        draw_visual(all_results, method)
    return all_results

def analyse_single(args_param):
    data_single_flow, fct_all, src_name, dst_name, k, threshold, method, draw = args_param
    # print(src_name, dst_name, len(data_single_flow))
    stable_list, stable_S_gt, stable_S_approx, cnt_enter_stable = get_stable_ground_truth(data_single_flow, k, threshold, method)
    
    st_time, ed_time = 0, 8000

    data_single_flow_ = [item for item in data_single_flow if st_time <= item[0] <= ed_time]
    rate_list = [item[1] for item in data_single_flow_]
    max_rate, min_rate = max(rate_list), min(rate_list)
    
    time_stable = [item[0] for i, item in enumerate(data_single_flow) if stable_list[i] and st_time <= item[0] <= ed_time]
    rate_stable = [item[1] for i, item in enumerate(data_single_flow) if stable_list[i] and st_time <= item[0] <= ed_time]
    time_unstable = [item[0] for i, item in enumerate(data_single_flow) if (not stable_list[i]) and st_time <= item[0] <= ed_time]
    rate_unstable = [item[1] for i, item in enumerate(data_single_flow) if (not stable_list[i]) and st_time <= item[0] <= ed_time]


    # time_stable = [item[0] for i, item in enumerate(data_single_flow) if stable_list[i]]
    # rate_stable = [item[1] for i, item in enumerate(data_single_flow) if stable_list[i]]
    # time_unstable = [item[0] for i, item in enumerate(data_single_flow) if not stable_list[i]]
    # rate_unstable = [item[1] for i, item in enumerate(data_single_flow) if not stable_list[i]]
    total_count = len(stable_list)
    stable_count = sum(stable_list)
    unstable_count = total_count - stable_count
    
    if draw:
        # plt.figure(figsize=(10, 6))
        plt.scatter(time_unstable, rate_unstable, color='b', label='rate for flow unstable', s=1)
        plt.scatter(time_stable, rate_stable, color='r', label='rate for flow stable', s=1)
        # plt.plot(time_unstable, rate_unstable, linestyle='-', color='b', label='rate for flow unstable', lw=1)
        # plt.plot(time_stable, rate_stable, linestyle='-', color='r', label='rate for flow stable', lw=1)
        
        for item in fct_all:
            sip, dip, sport, dport, fst, fct = item
            if st_time <= fst <= ed_time:
                plt.plot([fst, fst], [max_rate, min_rate], linestyle='-', color='gray', lw=1)
            if st_time <= fct <= ed_time:
                plt.plot([fct, fct], [max_rate, min_rate], linestyle='-', color='gray', lw=1)


        # time_rate_pair = [(time, rate) for time, rate in zip(time_stable + time_unstable, rate_stable + rate_unstable)]
        # time_rate_pair.sort(key=lambda p: p[0])
        # time_all, rate_all = [item[0] for item in time_rate_pair], [item[1] for item in time_rate_pair]
        # # plt.plot(time_stable+time_unstable, rate_stable+rate_unstable, linestyle='-', color='b', label='all rate', lw=1, marker='o', markersize=1)
        # plt.plot(time_all, rate_all, linestyle='-', color='b', label='all rate', lw=1)
        # plt.scatter(time_stable, rate_stable, color='r', label='stable', s=1)
        
        plt.xlabel("time(us)")
        plt.ylabel("rate(Gbps)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        # print('-'*40)
        # print(f'src: {src_name}, dst: {dst_name}')
        # print(f'k: {k}; threshold: {threshold}')
        # print(f'total: {total_count}, stable: {stable_count}, unstable: {unstable_count}, enter stable: {cnt_enter_stable}')
        # print(f'acceleratable_rate: {100 * stable_count / total_count:.3f}%, acc_speed: {total_count / max(1, unstable_count):.3f}')
        # print(f'error: {abs(100 * stable_S_approx / stable_S_gt - 100):.3f}%')
        if stable_S_gt == 0:
            stable_S_gt = 1
        plt.title(f'task: {task_name}, src: {src_name}, dst: {dst_name}, k: {k}, threshold: {threshold}, method: {method}\ntotal: {total_count}, stable: {100 * stable_count / total_count:.3f}%, err: {abs(100 * stable_S_approx / stable_S_gt - 100):.3f}%, switch ctx: {cnt_enter_stable}')
        plt.savefig(f'figs/single/{task_name}_{k}_{threshold}_{src_name}_{dst_name}_{method}_stable.png', bbox_inches='tight')
        
        plt.cla()
    # print(src_name, dst_name, len(data_single_flow))
    return total_count, stable_count, stable_S_gt, stable_S_approx, cnt_enter_stable, k, threshold, method

def analyse_results_stat(results, k_tgt, threshold_list, method_tgt):

    all_results = []
    for threshold_tgt in threshold_list:
        total_frames, stable_frames, stable_S_gt_total, stable_S_approx_total, cnt_ctx = 0, 0, 0, 0, 0
        for result in results:
            total_count, stable_count, stable_S_gt, stable_S_approx, cnt_enter_stable, k, threshold, method = result
            if threshold_tgt == threshold and method == method_tgt:
                total_frames += total_count
                stable_frames += stable_count
                stable_S_gt_total += stable_S_gt
                stable_S_approx_total += stable_S_approx
                cnt_ctx += cnt_enter_stable
        accele_rate = 100 * stable_frames / total_frames
        if stable_S_gt_total == 0:
            err = 0
        else:
            err = abs(100 * stable_S_approx_total / stable_S_gt_total - 100)
        print(f'running args k: {k_tgt}, threshold: {threshold_tgt}, method: {method_tgt}, accelerate_rate: {accele_rate:.3f}%, error: {err:.3f}%, change context: {cnt_ctx}')
        all_results.append((k_tgt, threshold_tgt, method_tgt, accele_rate, err, cnt_ctx))
    return all_results

def get_stable_ground_truth_continuous(rate_list, k, threshold, method):
    # unstable: False
    # stable: True
    list_len = len(rate_list)
    buffer = rate_list[:k]
    st_stable_time, st_stable_rate, in_stable = 0, 0, False
    stable_S_gt, stable_S_approx = 0, 0
    # return None
    buffer = [(item[0], item[1]) for item in rate_list[: k - 1]]
    buffer.insert(0, (0, 0))
    stable_list = [False] * (k - 1)
    cnt_enter_stable = 0
    # for idx in tqdm(range(k - 1, list_len)):
    for idx in range(k - 1, list_len):  
        cur_time, cur_rate = rate_list[idx]
        # update buffer
        buffer.append((cur_time, cur_rate))
        buffer = buffer[1:]
        cur_stable = judge_stable(buffer, threshold, method)
        stable_list.append(cur_stable)
        if cur_stable:
            if in_stable:
                # continue stable
                stable_S_gt += (cur_rate + buffer[-2][1]) * (cur_time - buffer[-2][0]) / 2
            else:
                # print(f'Enter Stable at time:{cur_time}')
                # enter stable
                pred_rate = predict_stable_rate(buffer)
                cnt_enter_stable += 1
                st_stable_time, st_stable_rate = buffer[0]
                for i in range(1, k):
                    # stable_list[- i - 1] = True # ?
                    stable_S_gt += (buffer[i - 1][1] + buffer[i][1]) * (buffer[i][0] - buffer[i-1][0]) / 2
                in_stable = True
        else:
            if in_stable:
                # print(f'Exit Stable at time:{cur_time}')
                # exit stable
                ed_stable_time, ed_stable_rate = buffer[-2] # TODO: change ed_rate?
                # st_stable_time, st_stable_rate, in_stable = 0, 0, False
                in_stable = False
                stable_S_approx += pred_rate * (ed_stable_time - st_stable_time)
            else:
                # continue unstable
                continue
    if stable_list[-1] == True:
        stable_S_approx += pred_rate * (rate_list[-1][0] - st_stable_time)
        
    return stable_list, stable_S_gt, stable_S_approx, cnt_enter_stable

def get_stable_ground_truth(rate_list, k, threshold, method):
    rate_list_subs = []
    list_len = len(rate_list)
    last_start = 0
    for idx in range(1, list_len):
        cur_time, last_time = rate_list[idx][0], rate_list[idx - 1][0]
        if cur_time - last_time > 1000:
            rate_list_subs.append(rate_list[last_start: idx])
            last_start = idx
    rate_list_subs.append(rate_list[last_start:])
    stable_list_all, stable_S_gt_all, stable_S_approx_all, cnt_enter_stable_all = [], 0, 0, 0
    for rate_list_sub in rate_list_subs:
        stable_list, stable_S_gt, stable_S_approx, cnt_enter_stable = get_stable_ground_truth_continuous(rate_list_sub, k, threshold, method)
        stable_list_all.extend(stable_list)
        stable_S_gt_all += stable_S_gt
        stable_S_approx_all += stable_S_approx
        cnt_enter_stable_all += cnt_enter_stable        

    return stable_list_all, stable_S_gt_all, stable_S_approx_all, cnt_enter_stable_all

def judge_stable(buffer, threshold, method):    
    max_rate = max(buffer, key=lambda p: p[1])[1]
    min_rate = min(buffer, key=lambda p: p[1])[1]
    if method == 'abs':
        return max_rate - min_rate <= threshold
    elif method == 'rel':
        return (max_rate - min_rate) * len(buffer) / sum([item[1] for item in buffer]) * 100 <= threshold
    else:
        raise NotImplementedError

def predict_stable_rate(buffer):
    n = len(buffer)
    # weights = [i for i in range(n)]
    weights = [1 for i in range(n)]
    
    weighted_rate = sum([item[1] * weight for item, weight in zip(buffer, weights)]) / sum(weights)
    return weighted_rate
    # return buffer[0][1]
    


def draw_visual(all_results, method_tgt):
    k_tgt = 10
    
    x_threshold = []
    y_acc, y_err = [], []
    for item in all_results:
        k, threshold, method, accele_rate, err, cnt_ctx = item
        if k == k_tgt and method == method_tgt:
            x_threshold.append(threshold)
            y_acc.append(accele_rate)
            y_err.append(err)

    fig, ax1 = plt.subplots()
    ax1.plot([i for i in range(len(x_threshold))], y_acc, linestyle='-', color='b', label='acceleratable rate', lw=1, marker='o', markersize=5)
    ax2 = ax1.twinx()
    ax2.plot([i for i in range(len(x_threshold))], y_err, linestyle='-', color='r', label='error rate total', lw=1, marker='o', markersize=5)
    x_label = 'threshold(Gbps)' if method_tgt == 'abs' else 'threshold(%)'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('acc_rate(%)', color='b')
    ax2.set_ylabel('error(%)', color='r')
    plt.xticks([i for i in range(len(x_threshold))], x_threshold, size=15)

    plt.grid(True)
    ax1.xaxis.grid(True)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # plt.tight_layout()
    plt.title(f'acc and err on {task_name} with k={k_tgt} and method={method_tgt}')
    plt.savefig(f'figs/{task_name}_threshold_k={k_tgt}_method={method_tgt}.png', bbox_inches='tight')
    plt.cla()

if __name__ == '__main__':
    # k = 10
    # threshold = 1
    data_all, fct_all = get_all_data()
    # analyse_all(data_all, n_process=32, draw=False)
    # analyse_all(data_all, fct_all, method='abs', n_process=32, draw=False)
    analyse_all(data_all, fct_all, method='rel', n_process=32, draw=False)
