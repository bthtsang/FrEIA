import configargparse
import torch
import torch.nn as nn
import os
import numpy as np

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# model summary
from torchinfo import summary

# input config file parser
def parse_model_args():
    parser = configargparse.ArgParser()
    # Config and input file locations
    parser.add('-c', required=True, is_config_file=True, 
                   help='config file path')
    parser.add('--input_db', type=str, 
                   default="/home/bthtsang/Research/ML/FrEIA/iip_lcs.npz",
                   help='path to input light curve database')

    # data handles for physical parameters
    parser.add('--lc_dh', type=str, default="iip_lcs", 
                  help='dataset handle for light curves in the input npz file')
    parser.add('--Eexp_dh', type=str, default="Eexp_arr", 
                  help='dataset handle for E_exp in the input npz file')
    parser.add('--MNi_dh', type=str, default="MNi_arr",
                  help='dataset handle for MNi in the input npz file')
    parser.add('--Mej_dh', type=str, default="Mej_arr", 
                  help='dataset handle for Mej in the input npz file')
    parser.add('--Rp_dh', type=str, default="Rp_arr", 
                  help='dataset handle for Rp in the input npz file')

    # Physical parameter options
    parser.add('--input_dims', type=int, default=2, 
                  help='input dimension/#parameters, e.g., Mej, Eexp, R_p')
    parser.add('--param_list', type=int, nargs="+", default=[0, 1],
                  help='physical parameters to include: Eexp, MNi, Mej, R_p')
    parser.add('--lc_noise', type=float, default=0.0,
                  help='artificial noise to impose on light curves')

    # Network setting
    parser.add('--cond_dims', type=int, default=256,
                  help='input light curve dimensions')
    parser.add('--num_aff_coupl_blks', type=int, default=3, 
                  help='number of affine blocks') 
    parser.add('--filter_sizes', type=int, nargs="+",
                  help='filter sizes for conv2d layers')
    parser.add('--kernel_sizes', type=int, nargs="+", 
                  help='kernel sizes for conv2d layers')
    parser.add('--maxpool_sizes', type=int, nargs="+",
                  help='maxpooling sizes for conv2d layers')
    parser.add('--strides', type=int, nargs="+",
                  help='strides for conv2d layers')
    parser.add('--batch_norm', action='store_true',
                  help='whether to use batch normalization')
    parser.add('--subnet_nlayers', type=int, default=3, 
                  help='number of layers in each affine block')
    parser.add('--subnet_nunits', type=int, default=256, 
                  help='number of neurons in each subnet layer')

    # network training parameters
    parser.add('--optimizer', type=str, default="adam",
                  help='optimizer: adam or sgd')
    parser.add('--seed', type=int, default=42,
                  help='random seed for torch random number generator')
    parser.add('--batch_size', type=int, default=32, 
                  help='batch size for training')
    parser.add('--val_split_fac', type=float, default=0.2, 
                  help='fraction of data allocated for validation')
    parser.add('--lr', type=float, default=1.0e-4, 
                  help='constant learning rate')
    parser.add('--n_epochs', type=int, default=200,
                  help='number of epochs to train for')
    parser.add('--loss', type=str, default="categorical_crossentropy",
                  help='loss function type')

    # output setting
    parser.add('--output_dir', type=str, default="outputs",
                  help='path to all output files')
    parser.add('--chk_dir', type=str, default="checkpoints",
                  help='name of output checkpoint dir')
    parser.add('--chk_interval', type=int, default=10, 
                  help='frequency to save checkpoint')

    # posterior estimation setting
    parser.add('--n_samples', type=int, default=128,
                  help='number of sample for each posterior performance measure')

    args = parser.parse_args()
    return args

def prep_directories(args):
    """ Prepare log and checkpoint directories """
    runID = get_runid(args)
    print (f"Unique run ID: {runID}")

    # create output directory and parse for trained model
    log_dir = os.path.join(os.getcwd(), args.output_dir, runID)
    chk_dir = os.path.join(os.getcwd(), args.output_dir, runID, args.chk_dir)
    if os.path.exists(log_dir):
        print (f"Log dir {log_dir} already exists")
    else:
        print (f"Creating log dir: {log_dir}")
        os.makedirs(log_dir)
    # create also the checkpoint directory
    os.makedirs(chk_dir, exist_ok=True)

    return runID 

def get_runid(args):
    """ Generate a unique run ID from arguments """
    # Input and network setting
    runid = f"in{args.input_dims}_cond{args.cond_dims}_\
              affx{args.num_aff_coupl_blks}_\
              subnet{args.subnet_nlayers}x{args.subnet_nunits}"

    if (args.batch_norm):
      runid += "_batchnorm"

    # training parameters
    runid += f"_bs{args.batch_size}_epochs{args.n_epochs}_\
               opt{args.optimizer}_lr{args.lr:.1e}_nsamples{args.n_samples}"

    return runid

def load_iip_data(args):
    data = np.load(args.input_db)
    print(f"Successfully loaded database {args.input_db}")
    lcs  = data[args.lc_dh]
    num_lcs = lcs.shape[0]
    print(f"Number of samples = {num_lcs}")

    # dictionary to get data handle
    param_name = {0:"E_exp", 1:"M_Ni", 2:"M_ej", 3:"R_p"}
    param_map = {0:args.Eexp_dh, 1:args.MNi_dh, 2:args.Mej_dh, 3:args.Rp_dh}

    # check input consistency
    assert (len(args.param_list) == args.input_dims)
    print ("Physical parameters included as inputs:")
    for pid in args.param_list:
        print(f"> {param_name[pid]}")
    print (f"Total number of input parameters = {args.input_dims}")

    x = np.zeros((num_lcs, args.input_dims))
    # go over param_list to assemble input parameters
    for pid in args.param_list:
        param = data[param_map[pid]]

        # check parameter list's length
        param_len = param.shape[0]
        assert(num_lcs == param_len)

        # store input parameters
        x[:, pid] = param

    noise = np.zeros(lcs.shape)
    # optionally impose noise on light curves
    if (args.lc_noise > 0.0):
        noise = np.random.rand(lcs.shape[0], lcs.shape[1])*args.lc_noise
        lcs = lcs + noise
    y = lcs # conditional input/light curves

    # convert to float
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # convert to torch tensor
    X = torch.from_numpy(x)
    Y = torch.from_numpy(y)

    return X, Y

class inn4iip(nn.Module):
    def __init__(self, args):
        super(inn4iip, self).__init__()

        # set up subnetwork
        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, args.subnet_nunits),
                                 nn.ReLU(),
                                 nn.Linear(args.subnet_nunits, c_out))

        # define input node (input progenitor parameters)
        in1 = Ff.InputNode(args.input_dims, name="Input")  # 1D vector of stellar params 
        cond = Ff.ConditionNode(args.cond_dims, name="LC Condition")

        # Layer collection
        inn_layers = [in1, cond]
        out_now = in1.out0
        # general version, construct INN with a loop
        # connect input to a series of random permutation and
        # conditional affine coupling blocks
        for n_cb in range(args.num_aff_coupl_blks):
            perm = Ff.Node(out_now, Fm.PermuteRandom, {}, name=f"RandomPermutation{n_cb+1}")
            affine = Ff.Node(perm, Fm.GLOWCouplingBlock,
                                module_args={'subnet_constructor':subnet_fc,
                                             'clamp':2.0},
                                conditions=cond, name=f"GLOW Coupling Block{n_cb+1}")
            inn_layers.extend([perm, affine])
            out_now = affine
        output = Ff.OutputNode(affine, name="Output")
        inn_layers.append(output) # append final output layer

        inn = Ff.GraphINN(inn_layers)

        self.net = inn


    def forward(self, x, c):
        latent, L_jac = self.net(x, c=c)
        return latent, L_jac

    def inverse(self, z, c):
        x, L_jac = self.net(z, rev=True, c=c)
        return x, L_jac

    def save_weights(self, args):
        """
        Helper function for saving the weights of the INN model
        """
        from shutil import copyfile

        chk_name = "inn_latest"
        backup_name = "inn_backup"

        runID = get_runid(args)

        chk_path = os.path.join(os.getcwd(), args.output_dir, runID,
                                 args.chk_dir, chk_name)
        chk_backup_path = os.path.join(os.getcwd(), args.output_dir, runID,
                                        args.chk_dir, backup_name)

        if os.path.exists(chk_path):
            copyfile(chk_path, chk_backup_path)
        torch.save(self.state_dict(), chk_path)
        return self

def train(runID, model, dataset, device, args):
    """ function to perform network training """
    torch.manual_seed(args.seed)

    # setting up training optimizer for the INN
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            #pin_memory=False,
        )

    txt_file_name = os.path.join(os.getcwd(), args.output_dir, 
                                 runID, "training.log")
    txt_file = open(txt_file_name, 'w')
    outputs = []
    for epoch in range(args.n_epochs):

        for (x_train, y_train) in train_loader:
            # forward pass through INN
            z, log_jac_det = model(x_train.to(device), c=y_train.to(device))

            # compute negative log-likelihood with std normal prior
            loss = 0.5*torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean() / args.input_dims

            # back-propagate the update the weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} | Loss = {loss:.5f}") 
        txt_file.write(f"{epoch+1} {loss}\n")

        if epoch % args.chk_interval == 0 and (epoch > 0):
            model.save_weights(args)
    txt_file.close()

def main():
    args = parse_model_args()
    runID = prep_directories(args)

    # load up data
    X, Y = load_iip_data(args)

    num_samples = X.shape[0] # total number samples
    shuffled_ids = np.arange(num_samples)
    np.random.shuffle(shuffled_ids)

    # splitting into training and validation dataset
    num_valid = int(num_samples*args.val_split_fac)
    valid_ids = shuffled_ids[:num_valid]
    train_ids = shuffled_ids[num_valid:]
    print("Splitting into training and validation datasets..")
    print(f"Train: {len(train_ids)}; Valid: {len(valid_ids)}")

    # setting up GPU device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train = X[train_ids]
    Y_train = Y[train_ids]
    # combine input and condition into one dataset
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    X_valid = X[valid_ids]
    Y_valid = Y[valid_ids]

    # create a new INN
    inn = inn4iip(args).to(device)

    # output model config details
    model_stats = summary(inn, input_size=[(args.batch_size, args.input_dims),
                                           (args.batch_size, args.cond_dims)])
    model_sum_str = str(model_stats)
    model_sum_file_name = os.path.join(os.getcwd(), args.output_dir,
                                       runID, "model_summary.txt")
    model_sum_file = open(model_sum_file_name, 'w')
    model_sum_file.write(model_sum_str+"\n")
    model_sum_file.close()

    # perform actual INN training
    train(runID, inn, dataset, device, args)

    # check training results
    # 1. Normal distribution for latent variable
    #    Output the latent z after the forward pass
    #    Discard the jacobian loss
    z_valid, _ = inn(X_valid.to(device), c=Y_valid.to(device))
    z_valid = z_valid.detach().cpu().numpy()

    # 2. Inference from trained network
    y_valid_repeated = torch.repeat_interleave(Y_valid, args.n_samples, dim=0)
    #  - randomly sample in latent space;
    z_rand  = torch.randn(num_valid*args.n_samples, args.input_dims)
    #  - output X_test from inverse of network conditioned on LCs
    x_pred, _ = inn.inverse(z_rand.to(device), c=y_valid_repeated.to(device))

    # detach and save
    x_pred = x_pred.detach().cpu().numpy()
    x_pred = x_pred.reshape(num_valid, args.n_samples, args.input_dims)
    x_gt  = X_valid.detach().cpu().numpy()

    output_npz_name = os.path.join(os.getcwd(), args.output_dir, runID, "zs_and_xs.npz")
    np.savez(output_npz_name, valid_ids=valid_ids, z_valid=z_valid,
                              x_pred=x_pred, x_gt=x_gt)

# Execute main() function
if __name__ == '__main__':
    main()
