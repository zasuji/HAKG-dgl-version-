import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run My.")
    # 预训练相关 #
    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?',
                        default='trained_model/KGAT_K20/model_epoch520.pth',
                        help='Path of stored model.')
    # 训练相关  #
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--epoch", type=int, default=5000, help='number of epochs')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--entity_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')

    parser.add_argument('--aggregation_type', nargs='?', default='bi-interaction',  # 聚合器
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--margin', type=float, default=0.7, help='the margin of contrastive_loss')

    # 数据相关  #
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba-fashion]")
    parser.add_argument("--data_path", nargs="?", default="datasets/", help="Input data path.")
    parser.add_argument("--link_nei", type=int, default=0, help="number of users' neighbor to items")
    parser.add_argument('--num_neg_sample', type=int, default=400, help='the number of negative sample')

    # 测试相关  #
    parser.add_argument('--K', type=int, default=20, help='@K')
    parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Output sizes of every layer')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='test batch size')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    # 早停与保存相关  #
    parser.add_argument('--stopping_steps', type=int, default=10, help='Number of epoch for early stopping')
    parser.add_argument('--save_dir', type=str, default='trained_model/KGAT/')
    return parser.parse_args()
