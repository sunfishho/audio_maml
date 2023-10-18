
import  torch, os
import  numpy as np
from    birdCallNShot import BirdCallNShot
import  argparse
from    meta import Meta
import pdb

import datetime

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    SPEC_SHAPE = (28, 28)

    config = [
        ('conv2d', [32, 1, 7, 7, 1, 3]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [16, 32, 5, 5, 1, 2]),
        ('relu', [True]),
        ('bn', [16]),
        ('conv2d', [8, 16, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [8]),
        ('flatten', []),
        ('linear', [args.n_way, 8 * SPEC_SHAPE[0] * SPEC_SHAPE[1]])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = BirdCallNShot('birdCall',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       SPEC_SHAPE = SPEC_SHAPE)
    
    train_accs_list = []
    val_accs_list = []
    test_accs_list = []

    # for step in range(args.epoch):
    for step in range(1001):
        
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        
        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)
        val_accs_list.append(accs[-1])

        if step % 10 == 0:
            torch.save(maml.net.state_dict(), "onn_birdcall_lr1mlr0.01shot10.pth")
            print('LR = 1, MetaLR = 0.01, 10-shot step:', step, '\ttraining acc:', accs)
            print(f'current time: {datetime.datetime.now()}')

        if step % 200 == 0 and step > 0:
            accs = []
            for _ in range( 1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetuning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            test_accs_list.append(accs[-1])
            print('Test acc:', accs)
            print(f"test_accs_list: {test_accs_list}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=10)

    args = argparser.parse_args()
    
    main(args)
