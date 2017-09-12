import fp_gan_nn

nn = fp_gan_nn.fp_gan_nn(debug=True)

nn.train_all(10000)


print("done.")