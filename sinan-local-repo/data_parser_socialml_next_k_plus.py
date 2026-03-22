import numpy as np
import sys
import os
import re
import scipy.misc
import math
import argparse

CnnTimeSteps = 5

LookForward = 5
Upsample = False
QoS = 500
TargetViolatioRatio = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', type=str, dest='log_dir', required=True)
parser.add_argument('--save-dir', type=str, dest='save_dir', default='')
parser.add_argument('--look-forward', type=int, default=5, dest='look_forward')
parser.add_argument('--upsample', action='store_true', dest='upsample')
parser.add_argument('--mode', type=str, default='original', choices=['original', 'plus'],
                    help='original=6 channels, plus=12 doubled channels for Sinan+')
args = parser.parse_args()

LookForward = args.look_forward
LogDir = args.log_dir
SaveDir = args.save_dir
Upsample = args.upsample
Mode = args.mode

Services = ['compose-post-redis',
            'compose-post-service',
            'home-timeline-redis',
            'home-timeline-service',
            'nginx-thrift',
            'post-storage-memcached',
            'post-storage-mongodb',
            'post-storage-service',
            'social-graph-mongodb',
            'social-graph-redis',
            'social-graph-service',
            'text-service',
            'text-filter-service',
            'unique-id-service',
            'url-shorten-service',
            'media-service',
            'media-filter-service',
            'user-mention-service',
            'user-memcached',
            'user-mongodb',
            'user-service',
            'user-timeline-mongodb',
            'user-timeline-redis',
            'user-timeline-service',
            'write-home-timeline-service',
            'write-home-timeline-rabbitmq',
            'write-user-timeline-service',
            'write-user-timeline-rabbitmq']

DockerMetrics = ['cpu_usage', 'rss', 'cache_mem', 'page_faults',
                 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes',
                 'io_serviced', 'io_bytes']

Stats = ['mean', 'max', 'min', 'std']

def upsample(sys_data, lat_data, next_k_info, lat_next_k_label):
    global QoS, LookForward
    label_nxt_k = np.squeeze(lat_next_k_label[:, -2, :])
    label_nxt_k = np.greater_equal(label_nxt_k, QoS)
    if LookForward > 1:
        label_nxt_k = np.sum(label_nxt_k, axis=1)
    final_label_k = np.greater_equal(label_nxt_k, 1)
    sat_idx  = np.where(final_label_k == 0)[0]
    viol_idx = np.where(final_label_k == 1)[0]
    viol_sat_ratio = len(viol_idx)*1.0/(len(sat_idx) + len(viol_idx))
    print('#viol/#total = %.4f' %(viol_sat_ratio))
    if len(viol_idx) == 0:
        print('no viol in this run')
    elif viol_sat_ratio < TargetViolatioRatio:
        sys_data_sat  = np.take(sys_data, indices=sat_idx, axis=0)
        sys_data_viol = np.take(sys_data, indices=viol_idx, axis=0)
        lat_data_sat  = np.take(lat_data, indices=sat_idx, axis=0)
        lat_data_viol = np.take(lat_data, indices=viol_idx, axis=0)
        next_k_info_sat  = np.take(next_k_info, indices=sat_idx, axis=0)
        next_k_info_viol = np.take(next_k_info, indices=viol_idx, axis=0)
        lat_next_k_label_sat  = np.take(lat_next_k_label, indices=sat_idx, axis=0)
        lat_next_k_label_viol = np.take(lat_next_k_label, indices=viol_idx, axis=0)
        sample_time = int(math.ceil(TargetViolatioRatio/(1-TargetViolatioRatio)*len(sat_idx)*1.0/len(viol_idx)))
        sys_data = sys_data_sat
        lat_data = lat_data_sat
        next_k_info = next_k_info_sat
        lat_next_k_label = lat_next_k_label_sat
        for i in range(0, sample_time):
            sys_data = np.concatenate((sys_data, sys_data_viol), axis=0)
            next_k_info = np.concatenate((next_k_info, next_k_info_viol), axis=0)
            lat_data = np.concatenate((lat_data, lat_data_viol), axis=0)
            lat_next_k_label = np.concatenate((lat_next_k_label, lat_next_k_label_viol), axis=0)
    else:
        print('Upsample not needed')
    return sys_data, lat_data, next_k_info, lat_next_k_label

def get_metric_stat(file_name):
    metric = ''
    stat = ''
    for m in DockerMetrics:
        if m in file_name and len(m) > len(metric):
            metric = m
    for s in Stats:
        if s in file_name.replace(metric, '') and len(s) > len(stat):
            stat = s
    if metric == '' or stat == '':
        return None
    else:
        return metric + '_' + stat

def compose_sys_data_channel(raw_data, metric):
    if metric != 'rps':
        for i, service in enumerate(Services):
            if i == 0:
                metric_data = np.array(raw_data[metric][service])
            else:
                metric_data = np.vstack((metric_data, raw_data[metric][service]))
    else:
        for i, service in enumerate(Services):
            if i == 0:
                metric_data = np.array(raw_data[metric])
            else:
                metric_data = np.vstack((metric_data, raw_data[metric]))
    for i in range(0, metric_data.shape[1] - CnnTimeSteps - LookForward + 1):
        if i == 0:
            channel_data = metric_data[:, i:i+CnnTimeSteps].reshape([1, metric_data.shape[0], CnnTimeSteps])
        else:
            channel_data = np.vstack((channel_data,
                metric_data[:, i:i+CnnTimeSteps].reshape([1, metric_data.shape[0], CnnTimeSteps])))
    channel_data = channel_data.reshape(
        [channel_data.shape[0], 1, channel_data.shape[1], channel_data.shape[2]])
    return channel_data

def compose_next_k_data_channel(raw_data, metric):
    if metric != 'rps':
        for i, service in enumerate(Services):
            if i == 0:
                metric_data = np.array(raw_data[metric][service])
            else:
                metric_data = np.vstack((metric_data, raw_data[metric][service]))
    else:
        for i, service in enumerate(Services):
            if i == 0:
                metric_data = np.array(raw_data[metric])
            else:
                metric_data = np.vstack((metric_data, raw_data[metric]))
    for i in range(CnnTimeSteps, metric_data.shape[1]-LookForward+1):
        if i == CnnTimeSteps:
            next_k_channel = metric_data[:, i: i+LookForward].reshape(
                [1, metric_data.shape[0], LookForward])
        else:
            next_k_channel = np.vstack((next_k_channel,
                metric_data[:, i: i+LookForward].reshape([1, metric_data.shape[0], LookForward])))
    next_k_channel = next_k_channel.reshape(
        [next_k_channel.shape[0], 1, next_k_channel.shape[1], next_k_channel.shape[2]])
    return next_k_channel

def parse_subdir(log_dir, call_type=0):
    print('\nprocessing %s (call_type=%d)' % (log_dir, call_type))
    raw_data = {}
    raw_data['latency'] = {}
    raw_data['replica'] = {}
    raw_data['cpu_limit'] = {}
    for metric in DockerMetrics:
        for stat in Stats:
            raw_data[metric + '_' + stat] = {}

    for file in os.listdir(log_dir):
        if file == 'rps.txt':
            raw_data['rps'] = np.loadtxt(log_dir+'/'+file, dtype=float)
            continue
        elif file.startswith('e2e'):
            percent = re.sub('e2e_lat_', '', file)
            percent = re.sub('.txt', '', percent)
            raw_data['latency'][percent] = np.loadtxt(log_dir+'/'+file, dtype=float)
            continue
        elif file.startswith('cpu_limit'):
            service = file.replace('cpu_limit_', '').replace('.txt', '')
            assert service in Services
            raw_data['cpu_limit'][service] = np.loadtxt(log_dir+'/'+file, dtype=float)
            continue
        elif file.startswith('replica') and 'replica_cpu_limit' not in file:
            service = file.replace('replica_', '').replace('.txt', '')
            assert service in Services
            raw_data['replica'][service] = np.loadtxt(log_dir+'/'+file, dtype=float)
            continue
        metric_stat = get_metric_stat(file)
        if metric_stat is None:
            continue
        service = file.replace(metric_stat + '_', '').replace('.txt', '')
        assert service in Services
        raw_data[metric_stat][service] = np.loadtxt(log_dir+'/'+file, dtype=float)

    assert 'rps' in raw_data

    rps_data          = compose_sys_data_channel(raw_data, 'rps')
    replica_data      = compose_sys_data_channel(raw_data, 'replica')
    cpu_limit_data    = compose_sys_data_channel(raw_data, 'cpu_limit')
    cpu_usage_mean_data = compose_sys_data_channel(raw_data, 'cpu_usage_mean')
    rss_mean_data     = compose_sys_data_channel(raw_data, 'rss_mean')
    cache_mem_mean_data = compose_sys_data_channel(raw_data, 'cache_mem_mean')

    channels = [rps_data, replica_data, cpu_limit_data,
                cpu_usage_mean_data, rss_mean_data, cache_mem_mean_data]

    if Mode == 'original':
        sys_data = np.concatenate(channels, axis=1)  # shape: (batch, 6, 28, 5)
    else:
        # Sinan+: split each channel into seq and fanout versions
        call_mask = np.full_like(rps_data, fill_value=call_type, dtype=np.float32)
        seq_channels = [c * (1 - call_mask) for c in channels]
        fan_channels = [c * call_mask       for c in channels]
        sys_data = np.concatenate(seq_channels + fan_channels, axis=1)  # shape: (batch, 12, 28, 5)

    cpu_limit_next_k = compose_next_k_data_channel(raw_data, 'cpu_limit')
    next_k_info = np.squeeze(cpu_limit_next_k)

    for i, p in enumerate(['90.0', '95.0', '98.0', '99.0', '99.9']):
        if i == 0:
            lat = np.array(raw_data['latency'][p])
        else:
            lat = np.vstack((lat, raw_data['latency'][p]))

    for i in range(0, lat.shape[1] - CnnTimeSteps - LookForward + 1):
        if i == 0:
            lat_data = lat[:, i:i+CnnTimeSteps].reshape([1, lat.shape[0], CnnTimeSteps])
        else:
            lat_data = np.vstack((lat_data, lat[:, i:i+CnnTimeSteps].reshape([1, lat.shape[0], CnnTimeSteps])))

    for i in range(CnnTimeSteps, lat.shape[1]-LookForward+1):
        if i == CnnTimeSteps:
            lat_next_k_label = lat[:, i:i+LookForward].reshape([1, lat.shape[0], LookForward])
        else:
            lat_next_k_label = np.vstack((lat_next_k_label,
                lat[:, i:i+LookForward].reshape([1, lat.shape[0], LookForward])))

    shuffle_in_unison([sys_data, lat_data, next_k_info, lat_next_k_label])
    print('sys_data.shape =', sys_data.shape)
    print('lat_data.shape =', lat_data.shape)
    print('next_k_info.shape =', next_k_info.shape)
    print('lat_next_k_label.shape =', lat_next_k_label.shape)

    num_val = int(lat_next_k_label.shape[0] * 0.1)
    return [sys_data[num_val:], lat_data[num_val:], next_k_info[num_val:], lat_next_k_label[num_val:],
            sys_data[:num_val], lat_data[:num_val], next_k_info[:num_val], lat_next_k_label[:num_val]]

def shuffle_in_unison(arr):
    rnd_state = np.random.get_state()
    for a in arr:
        np.random.set_state(rnd_state)
        np.random.shuffle(a)
        np.random.set_state(rnd_state)

def main():
    count = 0
    for subdir in os.listdir(LogDir):
        if ("diurnal" in subdir) or ("users" in subdir):
            if len(os.listdir(LogDir+'/'+subdir)) == 0:
                continue
            call_type = 1 if 'fanout' in LogDir else 0
            [sys_data_t, lat_data_t, next_k_info_t, lat_next_k_label_t,
             sys_data_v, lat_data_v, next_k_info_v, lat_next_k_label_v] = parse_subdir(LogDir+'/'+subdir+'/', call_type)
            if count == 0:
                glob_sys_data_train = sys_data_t
                glob_lat_data_train = lat_data_t
                glob_next_k_info_train = next_k_info_t
                glob_lat_next_k_label_train = lat_next_k_label_t
                glob_sys_data_valid = sys_data_v
                glob_lat_data_valid = lat_data_v
                glob_next_k_info_valid = next_k_info_v
                glob_lat_next_k_label_valid = lat_next_k_label_v
            else:
                glob_sys_data_train = np.concatenate((glob_sys_data_train, sys_data_t), axis=0)
                glob_lat_data_train = np.concatenate((glob_lat_data_train, lat_data_t), axis=0)
                glob_next_k_info_train = np.concatenate((glob_next_k_info_train, next_k_info_t), axis=0)
                glob_lat_next_k_label_train = np.concatenate((glob_lat_next_k_label_train, lat_next_k_label_t), axis=0)
                glob_sys_data_valid = np.concatenate((glob_sys_data_valid, sys_data_v), axis=0)
                glob_lat_data_valid = np.concatenate((glob_lat_data_valid, lat_data_v), axis=0)
                glob_next_k_info_valid = np.concatenate((glob_next_k_info_valid, next_k_info_v), axis=0)
                glob_lat_next_k_label_valid = np.concatenate((glob_lat_next_k_label_valid, lat_next_k_label_v), axis=0)
            count += 1

    print('glob_sys_data_train.shape =', glob_sys_data_train.shape)

    save_dir = SaveDir if SaveDir != '' else './data_next_' + str(LookForward) + 's/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    np.save(save_dir + "/sys_data_train", glob_sys_data_train)
    np.save(save_dir + "/sys_data_valid", glob_sys_data_valid)
    np.save(save_dir + "/lat_data_train", glob_lat_data_train)
    np.save(save_dir + "/lat_data_valid", glob_lat_data_valid)
    np.save(save_dir + "/nxt_k_data_train", glob_next_k_info_train)
    np.save(save_dir + "/nxt_k_data_valid", glob_next_k_info_valid)
    np.save(save_dir + "/nxt_k_train_label", glob_lat_next_k_label_train)
    np.save(save_dir + "/nxt_k_valid_label", glob_lat_next_k_label_valid)

if __name__ == '__main__':
    main()
