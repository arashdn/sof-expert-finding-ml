

import numpy


wc = numpy.load("./save/wc_N25_bias.npy")

numpy.savetxt("./data/wc_export_heat.csv", wc, delimiter=",")
