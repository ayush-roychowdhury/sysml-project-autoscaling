import mxnet as mx
import numpy as np
import argparse
import logging
from importlib import import_module

def _load_model(model_prefix, load_epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch)
    return sym, arg_params, aux_params

def evaluate(data_dir, model_prefix, load_epoch, n_channels=6, batch_size=256):
    sys_data = np.load(data_dir + '/sys_data_valid.npy')
    lat_data = np.load(data_dir + '/lat_data_valid.npy')
    nxt_data = np.squeeze(np.load(data_dir + '/nxt_k_data_valid.npy')[:,:,0])
    label    = np.squeeze(np.load(data_dir + '/nxt_k_valid_label.npy')[:,:,0])

    print(f'Evaluating on {data_dir}: {sys_data.shape[0]} samples, sys_data shape: {sys_data.shape}')

    sym, arg_params, aux_params = _load_model(model_prefix, load_epoch)
    all_layers = sym.get_internals()
    latency_sym = all_layers['latency_output']

    Model = mx.mod.Module(
        context=mx.cpu(),
        symbol=latency_sym,
        data_names=('data1', 'data2', 'data3'),
    )
    Model.bind(for_training=False,
        data_shapes=[('data1', (batch_size, n_channels, 28, 5)),
                     ('data2', (batch_size, 5, 5)),
                     ('data3', (batch_size, 28))])
    Model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    all_preds = []
    all_labels = []

    n = sys_data.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = end - start

        s = sys_data[start:end]
        l = lat_data[start:end]
        x = nxt_data[start:end]
        lb = label[start:end]

        # pad to batch_size if needed
        if chunk < batch_size:
            s  = np.pad(s,  [(0, batch_size-chunk),(0,0),(0,0),(0,0)])
            l  = np.pad(l,  [(0, batch_size-chunk),(0,0),(0,0)])
            x  = np.pad(x,  [(0, batch_size-chunk),(0,0)])

        batch = mx.io.DataBatch(
            data=[mx.nd.array(s), mx.nd.array(l), mx.nd.array(x)])
        Model.forward(batch, is_train=False)
        pred = Model.get_outputs()[0].asnumpy()

        all_preds.append(pred[:chunk])
        all_labels.append(lb)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # p99 is column index 3 (90,95,98,99,99.9)
    rmse = np.sqrt(np.mean(np.square(all_preds[:,3] - all_labels[:,3])))
    mae  = np.mean(np.abs(all_preds[:,3] - all_labels[:,3]))
    print(f'  p99 RMSE: {rmse:.2f}ms  MAE: {mae:.2f}ms')
    return rmse, mae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model-prefix', type=str, required=True)
    parser.add_argument('--load-epoch', type=int, default=200)
    parser.add_argument('--n-channels', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_prefix, args.load_epoch, args.n_channels, args.batch_size)
