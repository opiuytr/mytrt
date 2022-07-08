import torch
import struct
# download model form: http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth
mnist_model_dict = torch.load("mnist-b07bb66b.pth")



f = open("mnist.weights", 'w')
f.write("{}\n".format(len(mnist_model_dict.keys())))
for k,v in mnist_model_dict.items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
f.close()
