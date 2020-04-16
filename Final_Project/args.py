# from BUBO QA

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Joint Prediction")
    parser.add_argument('--entity_detection_path', type=str, default='entity_detection')
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default="EntityDetection")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dev_every', type=int, default=2000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--specify_prefix', type=str, default='id1')
    parser.add_argument('--words_dim', type=int, default=768)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--rnn_fc_dropout', type=float, default=0.3)
    parser.add_argument('--input_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--weight_decay',type=float, default=0)
    parser.add_argument('--fix_embed', action='store_false', dest='train_embed')
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--max_length', type=int, default = 64)
    # added for testing
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data/processed_simplequestions_dataset')
    parser.add_argument('--results_path', type=str, default='query_text')
    args = parser.parse_args()
    return args
