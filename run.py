import fp_gan_nn

#nn = fp_gan_nn.fp_gan_nn(debug=True, np_x_dim=8, np_y_dim=8, sample_label="fullds", restore_checkpoint="./checkpoints/state9990.ckpt")
nn = fp_gan_nn.fp_gan_nn(debug=False, np_x_dim=8, np_y_dim=8, sample_label="fullds")

nn.train_all(10000)


print("done.")
