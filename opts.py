import argparse

def parse_opt():
#     data_name='f30k'
    data_name='coco'
    finetune=False
    if data_name=='f30k':
        data_path = '/data1/data'
        feature_path = '/data1/data/vlp/flickr30k/region_feat_gvd_wo_bgd/trainval/'
    else:
        data_path ='/data1/data/vlp'
        feature_path = '/data1/data/vlp/coco/region_feat_gvd_wo_bgd/feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval'
    epoch=20
    lr=0.0005
    lr_update=10
    bz=128
    path=''
#     path='/data1/CMIR/new_new_jzk_master/run/f30k/checkpoint/12231255/checkpoint_6_rsum=450.51282051282055.pth.tar'
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default=0.8)
    parser.add_argument('--logger_name', default='./run/{}/checkpoint'.format(data_name))
    parser.add_argument('--data_name', default='{}'.format(data_name))
    parser.add_argument('--model_name', default='./run/{}/checkpoint'.format(data_name))
    parser.add_argument('--data_path', default=data_path)
    parser.add_argument('--feature_path', default=feature_path)
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.') 
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=epoch, type=int,              
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=bz, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=lr, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=lr_update, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=20, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=10, type=int,
                        help='Number of steps to run validation.')
#     parser.add_argument('--resume', default='/data1/CMIR/new_jzk_master/run/f30k/checkpoint/11190314/checkpoint_28_rsum=368.65877712031556.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', default=path, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', default=True,
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', default=finetune,
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet152',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_false',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    opt = parser.parse_args()
    return opt