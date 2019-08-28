import argparse
import os
import time

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

from listops_model import ListOpsModel
from listops_data import load_data_and_embeddings, LABEL_MAP, PADDING_TOKEN, get_batch


def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        # torch.save([model, optimizer], f)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
        }, f)


def model_load(fn):
    global model, optimizer
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        test_loss = checkpoint['loss']


###############################################################################
# Training code
###############################################################################

@torch.no_grad()
def evaluate(data_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    total_datapoints = 0
    for batch, data in enumerate(data_iter):
        batch_data = get_batch(data)
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch_data

        X_batch = torch.from_numpy(X_batch).long().to('cuda' if args.cuda else 'cpu')
        y_batch = torch.from_numpy(y_batch).long().to('cuda' if args.cuda else 'cpu')

        lin_output = model(X_batch)
        count = y_batch.shape[0]
        total_loss += torch.sum(
            torch.argmax(lin_output, dim=1) == y_batch
        ).float().data
        total_datapoints += count

    return total_loss.item() / total_datapoints


def train():
    # Turn on training mode which enables dropout.
    model.train()

    total_loss = 0
    total_acc = 0
    start_time = time.time()
    for batch, data in enumerate(training_data_iter):
        # print(data)
        # batch_data = get_batch(next(training_data_iter))
        data, n_batches = data
        batch_data = get_batch(data)
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch_data

        X_batch = torch.from_numpy(X_batch).long().to('cuda' if args.cuda else 'cpu')
        y_batch = torch.from_numpy(y_batch).long().to('cuda' if args.cuda else 'cpu')

        optimizer.zero_grad()

        lin_output = model(X_batch)
        loss = model.cost(lin_output, y_batch)
        acc = torch.mean(
            (torch.argmax(lin_output, dim=1) == y_batch).float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.detach().data
        total_acc += acc.detach().data
        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} '
                '| {:5d}/ {:5d} batches '
                '| lr {:05.5f} | ms/batch {:5.2f} '
                '| loss {:5.2f} | acc {:0.2f}'.format(
                    epoch,
                    batch,
                    n_batches,
                    optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    total_loss.item() / args.log_interval,
                    total_acc.item() / args.log_interval))
            total_loss = 0
            total_acc = 0
            start_time = time.time()
        ###
        batch += 1
        if batch >= n_batches:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data', type=str, default='./data/listops',
                        help='location of the data corpus')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirection model')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='max sequence length')
    parser.add_argument('--seq_len_test', type=int, default=1000,
                        help='max sequence length')
    parser.add_argument('--no-smart-batching', action='store_true', #reverse
                        help='batch based on length')
    parser.add_argument('--no-use_peano', action='store_true',
                        help='batch based on length')
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='the size of each chunk')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.1,
                        help='dropout applied to weight (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropouto', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only')

    parser.add_argument('--logdir', type=str, default='./output',
                        help='path to save outputs')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--name', type=str, default=randomhash,
                        help='exp name')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--std', action='store_true',
                        help='use standard LSTM')
    parser.add_argument('--philly', action='store_true',
                        help='Use philly cluster')
    args = parser.parse_args()

    args.smart_batching = not args.no_smart_batching
    args.use_peano = not args.no_use_peano

    if not os.path.exists(os.path.join(args.logdir, args.name)):
        os.makedirs(os.path.join(args.logdir, args.name))

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    train_data_path = os.path.join(args.data, 'train_d20s.tsv')
    test_data_path = os.path.join(args.data, 'test_d20s.tsv')
    vocabulary, initial_embeddings, training_data_iter, eval_iterator, training_data_length, raw_eval_data \
        = load_data_and_embeddings(args, train_data_path, test_data_path)
    dictionary = {}
    for k, v in vocabulary.items():
        dictionary[v] = k
    # make iterator for splits
    vocab_size = len(vocabulary)
    num_classes = len(set(LABEL_MAP.values()))
    args.__dict__.update({'ntoken': vocab_size,
                          'ninp': args.emsize,
                          'nout': num_classes,
                          'padding_idx': vocabulary[PADDING_TOKEN]})

    model = ListOpsModel(args)

    if args.cuda:
        model = model.cuda()

    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1]
                       if len(x.size()) > 1 else x.size()[0]
                       for x in params if x.size())
    total_params_sanity = sum(np.prod(x.size()) for x in model.parameters())
    assert total_params == total_params_sanity
    print("TOTAL PARAMS: %d" % sum(np.prod(x.size()) for x in model.parameters()))
    print('Args:', args)
    print('Model total parameters:', total_params)

    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr,
                                 betas=(0, 0.999),
                                 eps=1e-9,
                                 weight_decay=args.wdecay)


    if not args.test_only:
        # Loop over epochs.
        lr = args.lr
        stored_loss = 0.

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=2, threshold=0)
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train()
                test_loss = evaluate(eval_iterator)

                print('-' * 89)
                print(
                    '| end of epoch {:3d} '
                    '| time: {:5.2f}s '
                    '| test acc: {:.4f} '
                    '|\n'.format(
                        epoch,
                        (time.time() - epoch_start_time),
                        test_loss
                    )
                )

                if test_loss > stored_loss:
                    model_save(os.path.join(args.logdir, args.name, 'model.pt'))
                    print('Saving model (new best validation)')
                    # model.save(os.path.join(args.logdir, args.name, 'model.ckpt'))
                    stored_loss = test_loss
                print('-' * 89)

                scheduler.step(test_loss)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # Load the best saved model.
    # model = SSTClassifier.load_model(os.path.join(args.logdir, args.name, 'model.ckpt'))
    # if args.cuda:
    #     model = model.cuda()

    model_load(os.path.join(args.logdir, args.name, 'model.pt'))
    test_loss = evaluate(eval_iterator)
    data = {'args': args.__dict__,
            'parameters': total_params,
            'test_acc': test_loss}
    print('-' * 89)
    print(
        '| test acc: {:.4f} '
        '|\n'.format(
            test_loss
        )
    )