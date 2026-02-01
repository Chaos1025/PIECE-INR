# A demo script to run the reconstruction pipeline.
# Make sure to replace the placeholders with your actual data and parameters.

# PSF shape used for reconstruction,
#   default (50, 500, 500)
psf_z="<psf z length>"
psf_y="<psf y length>"
psf_x="<psf x length>"
# Filename of the wide-field stack, should be located under the root_dir
#   replace <measurement> with your measurement file name
wide_filed="<measurement>.tif"
# Activation function for INR
#   options: "ReLU"/"SIREN"/"WIRE"/"HashGrid"
act_func="ReLU"
# Encoding method, only works when `act_func`="ReLU"
#   options: "cartesian"/"radial_cartesian"/"gaussian"/"spherical"/"PIEE"
encoding="PIEE"
# Output intensity upper bound of the network
#   empirically 20 or 50
max_val=50 
# Loss weight for hessian term, empirically from 1e-4 to 1e-3
hessian=5e-4
# z-scale for hessian term
hess_zscale=1
# Loss weight for G-FDMAE term, empirically from 1e-3 to 5e-3
gfdmae=4e-3
# Iteration number for pretraining and training
pretrain_iter=1000
train_iter=2000


# Replace <data_name> with your dataset name
# Replace <your_exp_name> with your experiment name
#   results will be stored under "exp/<your_exp_name>"
python main.py \
--psf_generation "External" \
--config "source/<data_name>/config.yaml" \
--psf_shape ${psf_z} ${psf_y} ${psf_x} \
--root_dir "./source/<data_name>/" \
--data_stack_name ${wide_filed} \
--init_stack_name ${wide_filed} \
--exp_name "<your_exp_name>" \
--gpu_list 0 1 2 \
--hessian_weight ${hessian} \
--hessian_z_scale ${hess_zscale} \
--fdmae_loss_weight ${gfdmae} \
--pretraining_num_iter ${pretrain_iter} \
--training_num_iter ${train_iter} \
--encoding_option ${encoding} \
--zenith_encoding_angle 45 \
--radial_encoding_angle 9 \
--radial_encoding_depth 6 \
--axial_pad_length 0 \
--lateral_pad_length 20 \
--loading_pretrained_model "True" \
--saving_model "True" \
--log_option "True" \
--nerf_max_val ${max_val}