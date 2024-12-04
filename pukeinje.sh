python main.py \
--psf_generation="External" \
--psf_dz=0.5 \
--psf_dx=0.1083 \
--psf_dy=0.1083 \
--n_detection=0.9 \
--emission_wavelength=0.525 \
--n_obj=1.518 \
--n_immersion=1.5 \
--psf_shape 100 180 180 \
--working_type="solver" \
--source_dir="./source/" \
--data_stack_name="measurement.tif" \
--init_stack_name="measurement.tif" \
--psf_name="psf.tif" \
--root_dir="./source/pukeinje/" \
--ref_name="sample.tif" \
--exp_name="pukeinje" \
--gpu_list 0 1 2 3 \
--data_fidelity_term="mse" \
--tv_loss_weight 0 0 0 \
--l1_weight=0 \
--hessian_weight=5e-4 \
--hessian_z_scale=1 \
--fdmae_loss_weight=5e-3 \
--projection_type="max" \
--nerf_num_layers=3 \
--nerf_num_filters=64 \
--pretraining_num_iter=2000 \
--training_num_iter=2000 \
--encoding_option="PISE" \
--zenith_encoding_angle=45 \
--cartesian_encoding_depth=9 \
--radial_encoding_angle=9 \
--radial_encoding_depth=6 \
--lateral_view=100 \
--lateral_overlap=20 \
--axial_pad_length=20 \
--lateral_pad_length=20 \
--row_picker \
--col_picker \
--pure_background_mean_gate=0 \
--pure_background_variance_gate=0 \
--mask_mode="smooth" \
--loading_pretrained_model="True" \
--saving_model="True" \
--log_option="False" \
--nerf_max_val=50 