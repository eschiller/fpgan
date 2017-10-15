import fp_gan_nn

nn = fp_gan_nn.fp_gan_nn(debug=True, np_x_dim=32, np_y_dim=32, sample_label="fullds_datatools_test", sample_data=True)

nn.train_all(20000)


print("done.")
