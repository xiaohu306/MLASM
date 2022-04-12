import argparse
import pickle
import torchtext
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        self.cnn = self.get_cnn(cnn_type, True)
        print('finetune:', finetune)

        for param in self.cnn.parameters():
            param.requires_grad = finetune

        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            # print(self.cnn.module.fc.in_features)
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()
        self.fc_attn_i = nn.Linear(1024,1024)
        self.fusion = Fusion()
        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)

        if torch.cuda.is_available():
            model.cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImage, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        r = np.sqrt(6.) / np.sqrt(self.fc_attn_i.in_features +
                                  self.fc_attn_i.out_features)
        self.fc_attn_i.weight.data.uniform_(-r, r)
        self.fc_attn_i.bias.data.fill_(0)

    def forward(self, images, local_image):
        """Extract image feature vectors."""
        features = self.cnn(images)
        features = l2norm(features)
        # linear projection to the joint embedding space
        features = self.fc(features)
        # normalization in the joint embedding space
        features_1 = self.fc_attn_i(local_image)
        
        # features = l2norm(local_imgae)
        features = l2norm(self.fusion(features,features_1))
        return features
    
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.f_size = 1024
        self.gate0 = nn.Linear(self.f_size*2, self.f_size)
#         self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion0 = nn.Linear(self.f_size, self.f_size)
        self.fusion1 = nn.Linear(self.f_size, self.f_size)

    def forward(self, vec1, vec2):
        vec = torch.cat((vec1,vec2),dim=1)
        features_1 = self.gate0(vec)
#         features_2 = self.gate1(vec2)
        t = torch.sigmoid(features_1)
        f = t * vec1 + (1 - t) * vec2
        return f

class EncoderRegion(nn.Module):
    def __init__(self, opt):
        super(EncoderRegion, self).__init__()
        self.fc_region = nn.Linear(2048, opt.embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_region.in_features +
                                  self.fc_region.out_features)
        self.fc_region.weight.data.uniform_(-r, r)
        self.fc_region.bias.data.fill_(0)

    def forward(self, region_feat):
        region_feat = self.fc_region(region_feat)
        region_feat = l2norm(region_feat, dim=-1)
        return region_feat


class EncoderWord(nn.Module):

    def __init__(self, opt):
        super(EncoderWord, self).__init__()
        self.embed_size = opt.embed_size
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        # caption embedding
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)
        vocab = pickle.load(open('vocab/'+opt.data_name+'_vocab.pkl', 'rb'))
        word2idx = vocab.word2idx
        # self.init_weights()
        self.init_weights('glove', word2idx, opt.word_dim)
        self.dropout = nn.Dropout(0.1)


    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # return out
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = l2norm(cap_emb, dim=-1)
        cap_emb_mean = torch.mean(cap_emb, 1)
        cap_emb_mean = l2norm(cap_emb_mean)

        return cap_emb, cap_emb_mean



class EncoderText(nn.Module):
    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.sa = TextSA(opt.embed_size, 0.4)
        self.fc_text = nn.Linear(1024,1024)
        self.init_weights()
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_text.in_features +
                                  self.fc_text.out_features)
        self.fc_text.weight.data.uniform_(-r, r)
        self.fc_text.bias.data.fill_(0)
    def forward(self, word_emb):
        # word_emb_mean = torch.mean(word_emb, 1)
        # cap_emb = self.sa(word_emb, word_emb_mean)
        word_emb = l2norm(self.fc_text(word_emb))
        return word_emb


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    weiContext_i = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext = func_attention(cap_i_expand, images, opt, smooth=9.)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)

        row_sim = row_sim.mean(dim=1, keepdim=True)

        similarities.append(row_sim)

        weiContext = weiContext.mean(dim=1, keepdim=True)

        weiContext_i.append(weiContext)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    weiContext_i = torch.cat(weiContext_i, 1)
    weiContext_i = [weiContext_i[i, i, :].view(1, 1024) for i in range(n_image)]
    weiContext_i = torch.cat(weiContext_i, 0)

    return similarities,weiContext_i


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    weiContext_t = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext = func_attention(images, cap_i_expand, opt, smooth=4.)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)
        similarities.append(row_sim)
        weiContext = weiContext.mean(dim=1, keepdim=True)
        weiContext_t.append(weiContext)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    weiContext_t = torch.cat(weiContext_t, 1)
    weiContext_t = [weiContext_t[i, i, :].view(1, 1024) for i in range(n_image)]
    weiContext_t = torch.cat(weiContext_t, 0)
    return similarities, weiContext_t


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):

    def __init__(self, opt, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        #         self.net_type = opt.type
        self.margin = margin
        #         self.margin = 0.2
        self.opt = opt
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s, region_feats, word_feats, length, sims_local):

        scores_global = self.sim(im, s)

        scores_local = sims_local

        scores = self.opt.ratio * scores_local + (1 - self.opt.ratio) * scores_global
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class JZK(object):

    def __init__(self, opt, pre_scan=False):
        #         self.net_type = opt.type
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.region_enc = EncoderRegion(opt)
        self.cap_enc = EncoderText(opt)
        self.word_enc = EncoderWord(opt)
        # self.label_enc = EncoderLabel(opt)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.cap_enc.cuda()
            self.region_enc.cuda()
            self.word_enc.cuda()
            # self.label_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt, margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)


        params = list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        params += list(self.word_enc.parameters())
        params += list(self.region_enc.parameters())
        params += list(self.cap_enc.parameters())


        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)


        self.Eiters = 0

    def state_dict(self):
        # state_dict = [self.img_enc.state_dict(), self.cap_enc.state_dict(), self.label_enc.state_dict(),
        #               self.region_enc.state_dict(), self.word_enc.state_dict()]
        state_dict = [self.img_enc.state_dict(), self.cap_enc.state_dict(),
                      self.region_enc.state_dict(), self.word_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.cap_enc.load_state_dict(state_dict[1])
        # self.label_enc.load_state_dict(state_dict[2])
        self.region_enc.load_state_dict(state_dict[2])
        self.word_enc.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.cap_enc.train()
        # self.label_enc.train()
        self.region_enc.train()
        self.word_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.cap_enc.eval()
        # self.label_enc.eval()
        self.region_enc.eval()
        self.word_enc.eval()

    def forward_emb(self, images, region_feat,  captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)

        region_feat = Variable(region_feat)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            region_feat = region_feat.cuda()

        # Forward

        region_emb = self.region_enc(region_feat)
        word_emb, _ = self.word_enc(captions, lengths)
        sims_local, attn_txt, attn_img = self.local_sim(region_emb,word_emb,lengths)
        img_emb = self.img_enc(images, attn_img)
        cap_emb = self.cap_enc(attn_txt)
        # img_label, cap_label, label = self.label_enc(img_emb, cap_emb, region_emb, region_cls, word_emb, lengths)
        return img_emb, cap_emb, region_emb, word_emb, lengths, sims_local


    def forward_loss(self, img_emb, cap_emb, region_emb, word_emb, lengths, sims_local, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, region_emb, word_emb, lengths, sims_local)
        self.logger.update('Loss', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, region_feat, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, region_emb, word_emb, lengths, sims_local = self.forward_emb(images, region_feat, captions,
                                                                                       lengths)
        # measure accuracy and record loss
        self.optimizer.zero_grad()


        loss = self.forward_loss(img_emb, cap_emb, region_emb, word_emb, lengths, sims_local)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()


    def local_sim(self, region_emb, word_emb, length):
        attn_i = None
        attn_t = None
        scores = None
        if self.opt.cross_attn == 't2i':
            scores, attn_i = xattn_score_t2i(region_emb, word_emb, length, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores, attn_t = xattn_score_i2t(region_emb, word_emb, length, self.opt)
        elif self.opt.cross_attn == 'all':
            score1, attn_t = xattn_score_i2t(region_emb, word_emb, length, self.opt)
            score2, attn_i = xattn_score_t2i(region_emb, word_emb, length, self.opt)
            scores = 0.5 * (score1 + score2)
        return scores, attn_t, attn_i
def xattn_score_t2i1(images, captions, cap_lens, opt):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext = func_attention(cap_i_expand, images, opt, smooth=9.)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_i2t1(images, captions, cap_lens, opt):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext = func_attention(images, cap_i_expand, opt, smooth=4.)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities

