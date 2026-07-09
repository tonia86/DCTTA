import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=1, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=1,help="Batch size to use per GPU")
# parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--fisher_ratio', type=float, default=0.6, help='threshold of stochastic restoration')
parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/data/nsh/DATA/Kodak24_cropped/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/data/nsh/DATA/Rain100H/LQ',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/data/nsh/DATA/O-HAZY_resize/',
                    help='where training images of dehazing saves.')




parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="none",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")

###############
parser.add_argument('--iterations', type=int, default=1, help='the number of iterations to adapt on each test image')
###############
parser.add_argument('--crop', type=int, default = 0, help='crop or ont')
parser.add_argument('--patch_size', type=int, default=320 , help='patchsize of input.')
# parser.add_argument('--patch_size', type=int, default=256, help='patchsize of input.')
# parser.add_argument('--patch_size', type=int, default=256, help='patchsize of input.')
parser.add_argument('--compute_fisher', type=int, default=1, help='')############################
parser.add_argument('--teacher_weight', type=float, default=5, help='the weight of the teacher degradation loss')
parser.add_argument('--save_results', action='store_true', help='save output results')
parser.add_argument('--save_dir', type=str, default="experiment/", help='checkpoint save path')
options = parser.parse_args()

