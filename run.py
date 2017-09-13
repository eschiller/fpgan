import fp_gan_nn

nn = fp_gan_nn.fp_gan_nn(debug=True, sample_label="sl_test")

nn.train_all(10000)


print("done.")