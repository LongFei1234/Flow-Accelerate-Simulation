
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

from get_data import get_all_data

task_name = 'flow_rate_allreduce_50m_dcqcn_40us'

def analyse_all(data_all, fct_all, st=0, ed=8000, n_process=1, flag_break=False):

    args_params = []    
    for name_index in data_all.keys():
        src, dst = name_index
        data_single_flow = data_all[(src, dst)]
        if src != '0b000001' and src != '0b000101':
            continue
        # if src != '0b000001':
            # continue
        
        args_params.append((data_single_flow, fct_all, src, dst, st, ed, flag_break))
        # print(len(data_single_flow))

    print(f'args combs: {len(args_params)}')
    n_process = min(len(args_params), 32)
    with mp.Pool(processes=n_process) as pool:
        results = pool.map(analyse_single, args_params)


def analyse_single(args_param):
    data_single_flow, fct_all, src_name, dst_name, st_time, ed_time, flag_break = args_param
    data_single_flow_ = [item for item in data_single_flow if st_time <= item[0] <= ed_time]
    time_list = [item[0] for i, item in enumerate(data_single_flow_)]
    rate_list = [item[1] for i, item in enumerate(data_single_flow_)]

    max_rate, min_rate = max(rate_list), min(rate_list)

    time_list_total, rate_list_total = [], []
    time_list_tmp, rate_list_tmp = [], []
    last_time = time_list[0]
    for time, rate in zip(time_list, rate_list):
        if time - last_time >= 100: # 1ms
            time_list_total.append(time_list_tmp)
            rate_list_total.append(rate_list_tmp)
            time_list_tmp, rate_list_tmp = [], []
        time_list_tmp.append(time)
        rate_list_tmp.append(rate)
        last_time = time
    
    time_list_total.append(time_list_tmp)
    rate_list_total.append(rate_list_tmp)  
    
    for time_list, rate_list in zip(time_list_total, rate_list_total):
        plt.plot(time_list, rate_list, linestyle='-', color='b', label='flow rate', lw=1)
        # plt.scatter(time_list, rate_list, color='b', label='flow rate', s=1)
    
    if flag_break:
        for item in fct_all:
            sip, dip, sport, dport, fst, fct = item
            if st_time <= fst <= ed_time:
                plt.plot([fst, fst], [max_rate, min_rate], linestyle='-', color='gray', lw=1)
            if st_time <= fct <= ed_time:
                plt.plot([fct, fct], [max_rate, min_rate], linestyle='-', color='gray', lw=1)

    plt.xlabel("time(us)")
    plt.ylabel("rate(Gbps)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.title(f'task: {task_name}, src: {src_name}, dst: {dst_name}')
    plt.savefig(f'figs/single_us/{task_name}_{src_name}_{dst_name}.png', bbox_inches='tight')
    plt.cla()
    return None


if __name__ == '__main__':
    # st, ed = 0, 8000
    # st, ed = 0, 2000
    st, ed = 1800, 5100

    data_all, fct_all = get_all_data()
    analyse_all(data_all, fct_all, st, ed, n_process=32, flag_break=False)
