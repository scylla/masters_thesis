#!/usr/bin/python
import ConfigParser
import time
import sys
import os
import numpy as np
import pdb
import theanets
import climate
import theano as T
import random
import _util

logging = climate.get_logger(__name__)

climate.enable_default_logging()

           
if __name__ == '__main__':

    cf = ConfigParser.ConfigParser()
    if len(sys.argv) < 2:
        logging.info('Usage: %s <conf_fn>',sys.argv[0])
        sys.exit()

    cf.read(sys.argv[1])
    h_size=int(cf.get('INPUT','h_size'))
    e_size=int(cf.get('INPUT','e_size'))
    h_l1=float(cf.get('INPUT','h_l1'))
    h_l2=float(cf.get('INPUT','h_l2'))
    l1=float(cf.get('INPUT','l1'))
    l2=float(cf.get('INPUT','l2'))
    model_fn = cf.get('INPUT', 'model_fn')
    batch_size = int(cf.get('INPUT', 'batch_size'))

    src_vocab_fn = cf.get('INPUT', 'src_vocab_fn')
    src_train_fn = cf.get('INPUT', 'src_train_fn')
    src_val_fn = cf.get('INPUT', 'src_val_fn')

    dst_vocab_fn = cf.get('INPUT', 'dst_vocab_fn')
    dst_train_fn = cf.get('INPUT', 'dst_train_fn')
    dst_val_fn = cf.get('INPUT', 'dst_val_fn')

    dropout = float(cf.get('INPUT', 'dropout'))

    save_dir=cf.get('OUTPUT', 'save_dir')

    # NOw, we can load the vocab and the fea.
    src_w2ix, src_ix2w = load_vocab(src_vocab_fn)
    dst_w2ix, dst_ix2w = load_vocab_dst(dst_vocab_fn)

    src_train_np = load_split(src_w2ix, src_train_fn)
    src_val_np = load_split(src_w2ix, src_val_fn)
    

    dst_train_np = load_split(dst_w2ix, dst_train_fn)
    dst_val_np = load_split(dst_w2ix, dst_val_fn)

    train_range = range(src_train_np.shape[0])
    def batch_train():
        random.shuffle(train_range)

        src = np.zeros((src_train_np.shape[1], batch_size, len(src_w2ix)), dtype = 'float32')
        #src_mask = np.zeros((src_train_np.shape[1], batch_size), dtype = 'int32')
        src_mask = np.zeros((src_train_np.shape[1], batch_size), dtype = 'float32')

        dst = np.zeros((dst_train_np.shape[1] + 2, batch_size, len(dst_w2ix)), dtype = 'float32')
        label = np.zeros((dst_train_np.shape[1] + 2, batch_size), dtype = 'int32')
        mask = np.zeros((dst_train_np.shape[1] + 2, batch_size), dtype = 'float32')
        #dst_mask = np.zeros((dst_train_np.shape[1], batch_size, len(dst_w2ix)), dtype = 'float32')
        #for i in range(batch_size):
        i = 0
        idx = 0
        indices = []
        while True:
            src_i = src_train_np[train_range[i],:]
            i += 1

            num_el = 0
            pos_s = [ pos for pos in src_i if pos >= 0 ]
            num_el = len(pos_s)

            if num_el < 6:
                continue
            
            for j,pos in enumerate(src_i):
                if pos < 0:
                    break
                src[j, idx, pos] = 1
                src_mask[j, idx] = 1
            indices.append(i)
            idx += 1
            if idx == batch_size:
                break

        for i in range(batch_size):
            dst_i = dst_train_np[train_range[indices[i]],:]
            dst_np = np.zeros((dst_i.size + 2,), dtype = 'int32')
            dst_np[:] = -1

            seq_len = [ el for el in dst_i if el >= 0 ]
            seq_len = len(seq_len)

            if dst_i[0] != dst_w2ix[start_tok]:
                dst_np[0] = dst_w2ix[start_tok]
                dst_np[1:1+dst_i.size] = dst_i
                if dst_i[-1] != dst_w2ix[end_tok]:
                    dst_np[1+seq_len] = dst_w2ix[end_tok]
            else:
                dst_np[0:dst_i.size] = dst_i
                if dst_i[-1] != dst_w2ix[end_tok]:
                    dst_np[seq_len] = dst_w2ix[end_tok]
            for j,pos in enumerate(dst_np):
                if pos < 0:
                    break
                dst[j, i, pos] = 1
                if j >= 1:
                    # this is the prediction.
                    label[j-1,i] = pos
                    mask[j-1,i] = 1

        print (src_mask.sum(axis = 0))
        print (mask.sum(axis = 0))
        return src, src_mask, dst, label, mask

    val_range = range(src_val_np.shape[0])
    def batch_val():
        random.shuffle(val_range)

        src = np.zeros((src_val_np.shape[1], batch_size, len(src_w2ix)), dtype = 'float32')
        #src_mask = np.zeros((src_val_np.shape[1], batch_size), dtype = 'int32')
        src_mask = np.zeros((src_val_np.shape[1], batch_size), dtype = 'float32')

        dst = np.zeros((dst_val_np.shape[1] + 2, batch_size, len(dst_w2ix)), dtype = 'float32')
        label = np.zeros((dst_val_np.shape[1] + 2, batch_size), dtype = 'int32')
        mask = np.zeros((dst_val_np.shape[1] + 2, batch_size), dtype = 'float32')
 
        #for i in range(batch_size):
        i = 0
        idx = 0
        indices = []
        while True:
            src_i = src_val_np[val_range[i],:]
            i += 1

            num_el = 0
            pos_s = [ pos for pos in src_i if pos >= 0 ]
            num_el = len(pos_s)

            if num_el < 6:
                continue
 
            for j,pos in enumerate(src_i):
                if pos < 0:
                    break
                src[j, idx, pos] = 1
                src_mask[j, idx] = 1
            indices.append(i)
            idx += 1
            if idx == batch_size:
                break

        for i in range(batch_size):
            dst_i = dst_val_np[val_range[indices[i]],:]
            dst_np = np.zeros((dst_i.size + 2,), dtype = 'int32')
            dst_np[:] = -1

            seq_len = [ el for el in dst_i if el >= 0 ]
            seq_len = len(seq_len)

            if dst_i[0] != dst_w2ix[start_tok]:
                dst_np[0] = dst_w2ix[start_tok]
                dst_np[1:1+dst_i.size] = dst_i
                if dst_i[-1] != dst_w2ix[end_tok]:
                    dst_np[1+seq_len] = dst_w2ix[end_tok]
            else:
                dst_np[0:dst_i.size] = dst_i
                if dst_i[-1] != dst_w2ix[end_tok]:
                    dst_np[seq_len] = dst_w2ix[end_tok]
            for j,pos in enumerate(dst_np):
                if pos < 0:
                    break
                dst[j, i, pos] = 1
                if j >= 1:
                    # this is the prediction.
                    label[j-1,i] = pos
                    mask[j-1,i] = 1
        if mask.sum() == 0:
            logging.error('Should not happen')
        print (src_mask.sum(axis = 0))
        print (mask.sum(axis = 0))
        return src, src_mask, dst, label, mask

    def layer_input_encdec(src_size, dst_size, emb_size):
        return dict(src_size = src_size, dst_size = dst_size, emb_size = emb_size)

    def layer_lstm(n):
        return dict(form = 'lstm', size = n)
    def layer_lstm_enc(n):
        return dict(form = 'lstmenc', size = n)
    def layer_lstm_dec(n):
        return dict(form = 'lstmdec', size = n)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    time_str = time.strftime("%d-%b-%Y-%H%M%S", time.gmtime())
    save_prefix = os.path.join(save_dir, os.path.splitext(os.path.basename(sys.argv[1]))[0])
    save_fn = save_prefix + '_' + time_str + '.pkl'
    logging.info('will save model to %s', save_fn)
    
    if os.path.isfile(model_fn):
        e = theanets.Experiment(model_fn)
    else:
        e = theanets.Experiment(
            theanets.recurrent.Classifier,
            layers=(layer_input_encdec(len(src_w2ix), len(dst_w2ix), e_size),
            layer_lstm_enc(h_size),
            layer_lstm_dec(h_size),
            (len(dst_w2ix), 'softmax')),
            weighted=True,
            encdec = True
        )
        e.train(
            batch_train,
            batch_val,
            algorithm='rmsprop',
            patience=100,
            min_improvement=0.2,
            #algorithm='adam',
            learning_rate=0.0001,
            momentum=0.9,
            max_gradient_elem=1,
            input_noise=0.0,
            train_batches=1,
            valid_batches=1,
            hidden_l1 = h_l1,
            hidden_l2 = h_l2,
            weight_l1 = l1,
            weight_l2 = l2,
            batch_size=batch_size,
            dropout=dropout,
            save_every = 100
        )

    e.save(save_fn)
