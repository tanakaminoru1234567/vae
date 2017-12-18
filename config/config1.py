class Config(object):
    def __init__(self):
        # image width
        self.w = 3
        # moving square's width
        self.sw = 2
        # number of generating sequence
        self.n_seq = 32
        # max time step of generating sequence
        self.T = 14
        # show width in cv2
        self.w_width = 200
        # numbering moving square
        self.order = (0, 3, 6, 7, 8, 5, 2, 1)
        # id for gpu
        self.gpu = -1
        # train max time step
        self.tsize = 7
        # train print frequency
        self.print_freq = 10
        # number of filters for convolution
        self.n_filter = 4
        # kernel size
        self.ksize = 2
        # stride width of kernel
        self.stride = 1
        # whether to use convolution model
        self.conv = False

        # train batchsize
        self.batchsize = 32
        # train iteration
        self.n_iter = 7000
        # dimension of latent variables
        self.n_latent = 2
        # number of units in hidden layers
        self.n_hidden = 10

        # test batchsize
        self.test_batchsize = 1
        # saving model name
        self.model_name = 'model/vae1.model'
        # test showing max time step
        self.test_maxt = 9
        # test number of samples for showing latent variables' distribution
        self.n_plot_sample = 50
