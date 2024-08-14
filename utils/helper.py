import os
from collections import OrderedDict

import torch
from dgl import function as fn


# DGL: dgl-cu101(0.4.3)
# We will get different results when using the function `fn.sum`, and the randomness is due to `atomicAdd`.
# Use custom function to ensure deterministic behavior.
def edge_softmax_fix(graph, score):

    def reduce_sum(nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1) + 1e-6
        return {'out_sum': accum}

    graph = graph.local_var()
    graph.edata['out'] = score
    graph.edata['out'] = torch.exp(graph.edata['out'])
    graph.update_all(fn.copy_e('out', 'temp'), reduce_sum)
    graph.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
    out = graph.edata['out']
    return out


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]              # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model




