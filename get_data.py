from tqdm import tqdm

# [flow_node_{receiver} {clock}] {src_ip} {src_port} {dst_ip} {dst_port} {pkt_type} {pkt_size}

def get_all_data(task_name):
    print('handling data...')
    data_all = {}
    with open(f'{task_name}', 'r') as f:
        lines = f.readlines()
    f.close()
    for line in tqdm(lines):
        line = line.replace('[', '').replace(']', '').split(' ')
        pkt_type = int(line[-2])
        if pkt_type != 0: continue
        receiver = int(line[0].replace('flow_node_', ''))
        src, dst = int(line[2]), int(line[4])
        if receiver == src: continue
        pkt_size = int(line[-1])
        time = float(line[1])
        if data_all[(src, dst)] is None:
            data_all[(src, dst)] = [(time, pkt_size)]
        else:
            data_all[(src, dst)].append((time, pkt_size))
        
    # fct ?
    
    return data_all
    