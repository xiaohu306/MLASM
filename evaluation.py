import os
gpu_id = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
import sys
from data import get_test_loader
import time
import pickle
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import JZK, xattn_score_i2t1,xattn_score_t2i1,cosine_sim
from collections import OrderedDict
import time
from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    region_embs = None
    word_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0

    for i, (images,  region_feat, targets, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
    with torch.no_grad():
        for i, (images,  region_feat,  targets, lengths, ids) in enumerate(data_loader):
            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            img_emb, cap_emb, region_emb, word_emb, cap_len,_= model.forward_emb(images,  region_feat,  targets, lengths)

            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
                region_embs = np.zeros((len(data_loader.dataset), region_emb.size(1), region_emb.size(2)))
                word_embs = np.zeros((len(data_loader.dataset), max_n_word, word_emb.size(2)))
                cap_lens = [0] * len(data_loader.dataset)
            # cache embeddings
            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
            region_embs[ids] = region_emb.data.cpu().numpy().copy()
            word_embs[ids,:max(lengths),:] = word_emb.data.cpu().numpy().copy()
            for j, nid in enumerate(ids):
                cap_lens[nid] = cap_len[j]

            # measure accuracy and record loss
            #         model.forward_loss(img_emb, cap_emb, cap_len)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                    i, len(data_loader), batch_time=batch_time,
                    e_log=str(model.logger)))
            del images,  region_feat,  targets
    return img_embs, cap_embs, region_embs, word_embs, cap_lens

def evalrank(model_path, data_path=None,feature_path=None, split='dev', fold5=False):

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    # if data_path is not None:

    # load vocabulary used by the model
    #     vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_precomp_vocab.json' % opt.data_name))
    vocab = pickle.load(open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    # construct model
    model = JZK(opt)
    print(opt)
    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers,opt.data_path, opt)

    print('Computing results...')
    img_embs, cap_embs, region_embs, word_embs, cap_lens = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))


    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        region_embs = np.array([region_embs[i] for i in range(0, len(region_embs), 5)])
        start = time.time()

        if opt.cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, region_embs, word_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, region_embs, word_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'all':
            sims,sims_l,sims_g = shard_xattn_all(img_embs, cap_embs, region_embs, word_embs, cap_lens, opt, shard_size=128)
        np.save('sims_f30k_full.npy',sims)
        np.save('sims_l_f30k_full.npy',sims_l)
        np.save('sims_g_f30k_full.npy',sims_g)
        end = time.time()
        print("calculate similarity time:", end-start)
        i=0
        name=['l+g','l','g']
        for sims in [sims,sims_l,sims_g]:
            print('----- {} ---'.format(name[i]))
            i=i+1
            r, rt = i2t(sims, npts=img_embs.shape[0], return_ranks=True)
            ri, rti = t2i( sims,npts=img_embs.shape[0], return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)
            #             print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            #             print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            region_embs_shard = region_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            word_embs_shard = word_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            start = time.time()
            
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, region_embs_shard, word_embs_shard, cap_lens_shard, opt, shard_size=128)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, region_embs_shard, word_embs_shard, cap_lens_shard, opt, shard_size=128)
            elif opt.cross_attn == 'all':
                sims,sims_l,sims_g = shard_xattn_all(img_embs_shard, cap_embs_shard, region_embs_shard, word_embs_shard, cap_lens_shard, opt, shard_size=128)
            if i==0:
                np.save('sims_coco_full.npy',sims)
            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t( sims, npts=img_embs_shard.shape[0],return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(sims,npts=img_embs_shard.shape[0], return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    # torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

def shard_xattn_all(img_embs, cap_embs, region_embs, word_embs, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(region_embs)-1)/shard_size) + 1
    n_cap_shard = int((len(word_embs)-1)/shard_size) + 1

    sims_local = np.zeros((len(region_embs), len(word_embs)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(region_embs))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_all batch (%d,%d)' % (i,j))
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(word_embs))
                # im = Variable(torch.from_numpy(img_embs[im_start:im_end])).float().cuda()
                re = Variable(torch.from_numpy(region_embs[im_start:im_end])).float().cuda()
                # ca = Variable(torch.from_numpy(cap_embs[cap_start:cap_end])).float().cuda()
                wo = Variable(torch.from_numpy(word_embs[cap_start:cap_end])).float().cuda()
                l = caplens[cap_start:cap_end]
                sim1 = xattn_score_i2t1(re, wo, l, opt)
                sim2 = xattn_score_t2i1(re, wo, l, opt)
#                 sim_local = 0.5*(sim1+sim2)
                sim_local = 0.5*sim1+0.5*sim2
                #                 print(sim_local.shape,sim_local)
                sims_local[im_start:im_end, cap_start:cap_end] = sim_local.data.cpu().numpy()
        sys.stdout.write('\n')

    #     sims_global = 0
    #     opt.ratio = 0.8
    sims_global = np.dot(img_embs,cap_embs.T)
    d = opt.ratio*sims_local + (1-opt.ratio)*sims_global
    #     loca = sims_global
    return d,sims_local,sims_global

def shard_xattn_i2t(img_embs, cap_embs, region_embs, word_embs, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(region_embs)-1)/shard_size) + 1
    n_cap_shard = int((len(word_embs)-1)/shard_size) + 1

    sims_local = np.zeros((len(region_embs), len(word_embs)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(region_embs))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(word_embs))
                # im = Variable(torch.from_numpy(img_embs[im_start:im_end])).float().cuda()
                re = Variable(torch.from_numpy(region_embs[im_start:im_end])).float().cuda()
                # ca = Variable(torch.from_numpy(cap_embs[cap_start:cap_end])).float().cuda()
                wo = Variable(torch.from_numpy(word_embs[cap_start:cap_end])).float().cuda()
                l = caplens[cap_start:cap_end]
                sim_local = xattn_score_i2t(re, wo, l, opt)
                sims_local[im_start:im_end, cap_start:cap_end] = sim_local.data.cpu().numpy()
        sys.stdout.write('\n')
    # img_emb_new = img_embs[0:img_embs.size(0):5]
    #     sims_global = 0
    sims_global = np.dot(img_embs,cap_embs.T)
    d = opt.ratio*sims_local + (1-opt.ratio)*sims_global
    #     d = 0*sims_local + 1*sims_global
    return d

def shard_xattn_t2i(img_embs, cap_embs, region_embs, word_embs, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(region_embs)-1)/shard_size) + 1
    n_cap_shard = int((len(word_embs)-1)/shard_size) + 1

    sims_local = np.zeros((len(region_embs), len(word_embs)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(region_embs))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(word_embs))
                # im = Variable(torch.from_numpy(img_embs[im_start:im_end])).float().cuda()
                re = Variable(torch.from_numpy(region_embs[im_start:im_end])).float().cuda()
                # ca = Variable(torch.from_numpy(cap_embs[cap_start:cap_end])).float().cuda()
                wo = Variable(torch.from_numpy(word_embs[cap_start:cap_end])).float().cuda()
                l = caplens[cap_start:cap_end]
                sim_local = xattn_score_t2i(re, wo, l, opt)
                sims_local[im_start:im_end, cap_start:cap_end] = sim_local.data.cpu().numpy()
        sys.stdout.write('\n')
    # img_emb_new = img_embs[0:img_embs.size(0):5]
    sims_global = np.dot(img_embs,cap_embs.T)
    #     sims_global = 0
    d = opt.ratio*sims_local + (1-opt.ratio)*sims_global
    return d




def i2t(sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i( sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

if __name__ == '__main__':
#     evalrank('/data1/CMIR/new_new_new_jzk_master/run/f30k/checkpoint/04110925/checkpoint_18_rsum=466.9625246548324.pth.tar',
#              data_path='/data1/data', split="test", fold5=False)
    
    evalrank('/data1/CMIR/new_new_new_jzk_master/run/coco/checkpoint/04111439/checkpoint_14_rsum=501.65999999999997.pth.tar',
             data_path='/data1/data/vlp', split="test", fold5=True)
