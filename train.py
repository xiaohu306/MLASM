import os
import time
import shutil
import datetime
import torch
import numpy
import pickle
import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import JZK
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_i2t,shard_xattn_t2i,shard_xattn_all
import logging
import tensorboard_logger as tb_logger
gpu_id = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(opt, train_loader, model,epoch, val_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train_start()
        # print(train_data)
        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'gpu: {3}\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), gpu_id,batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


#         if model.Eiters % opt.val_step == 0:
#             validate(opt, val_loader, model)

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, region_embs, word_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    region_embs = numpy.array([region_embs[i] for i in range(0, len(region_embs), 5)])

    #     print(img_embs.shape)
    #     print(region_embs.shape)
    #     print(cap_embs.shape)
    #     print(word_embs.shape)

    start = time.time()
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, region_embs, word_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, region_embs, word_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'all':
        sims,_,_ = shard_xattn_all(img_embs, cap_embs, region_embs, word_embs, cap_lens, opt, shard_size=128)

    end = time.time()
    print("calculate similarity time:", end-start)

    (r1, r5, r10, medr, meanr) = i2t(sims, npts=img_embs.shape[0])
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(sims, npts=img_embs.shape[0],)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r_sum', r_sum, step=model.Eiters)

    return r_sum

def save_checkpoint(state, is_best, filename, prefix,date):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if is_best:
                path1 = os.path.join(prefix,date)
                if not os.path.exists(path1):
                    os.mkdir(path1)
                torch.save(state,  os.path.join(path1, filename))
            # if is_best:
            #     shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def main():

    opt = opts.parse_opt()
    #     opt.cross_attn="i2t"
    #     opt.cross_attn="t2i"
    opt.cross_attn="all"
   
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    t = datetime.datetime.now()
    logger_name = os.path.join(opt.logger_name,t.strftime("%m%d%H%M"))
    # print(logger_name)

    tb_logger.configure(logger_name, flush_secs=5)
    shutil.copyfile('model.py','{}/model.py'.format(logger_name))
    # Load Vocabulary Wrapper
    #     vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_precomp_vocab.json' % opt.data_name))
    vocab = pickle.load(open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
    print(opt)
    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt.data_path, opt)
    # dict = torch.load('')
    model = JZK(opt)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            # validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model,epoch, val_loader)

        # evaluate on validation set

        r_sum=validate(opt, val_loader, model)
        # remember best R@ sum and save checkpoint
        is_best = True
        best_rsum = max(r_sum, best_rsum)

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}_rsum={}.pth.tar'.format(epoch,r_sum), prefix=opt.model_name,date = t.strftime("%m%d%H%M"))

if __name__ == '__main__':
    main()
