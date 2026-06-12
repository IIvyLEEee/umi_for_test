import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

class Generator:
    def __init__(self, seed: np.int32):
        # initialize the regsiters
        self.regs = np.zeros(32, np.int8)
        for i in range(len(self.regs)):
            self.regs[i] = (seed >> i) & 1

        # LTB of Gaussian ICDF
        H = np.arange(4/256, 4+4/256, 4/128)
        self.threshold = st.norm.cdf(H) - 0.5
        for i in range(128):
            self.threshold[i] = int(self.threshold[i] * (2 ** 32))
        # for i in range(16):
        #     print(bin(int(self.threshold[i+96])))
        # for i in range(127):
        #     self.threshold[i] = int((self.threshold[i] + self.threshold[i+1])/2)
        # self.threshold[127] = int((self.threshold[127]+2**31)/2)
    
    def step(self):
        '''
            implementation of LFSR
            (32-primitive polynomial: x^32+x^7+x^5+x^3+x^2+x+1
        '''
        temp = self.regs[31] ^ self.regs[30] ^ self.regs[29] ^ self.regs[27] ^ self.regs[25] ^ self.regs[0]
        for i in range(len(self.regs)-1):
            self.regs[i] = self.regs[i+1]
        self.regs[31] = temp
    
    def readout(self):
        data = 0
        for i in range(len(self.regs)):
            data += self.regs[i] * (1 << i)
        return data
    
    def uniform(self, shape):
        n = 1
        # print(shape)
        for i in shape:
            n *= i
        res = np.zeros(n)
        for i in range(n):
            self.step()
            res[i] = self.readout()
        return res.reshape(shape)
    
    def gaussian(self, shape):
        n = 1
        for i in shape:
            n *= i
        res = np.zeros(n)
        for i in range(n):
            self.step()
            u = self.readout()
            # print("readout: {}".format(u))
            sign = u >> 31
            # print("sign: {}".format(sign))
            bias = u & (2**31 - 1)
            # print("bias: {}".format(bias))
            value = self.encoder(bias)/32
            # print("value: {}".format(value))
            res[i] = value if not sign else -value
        return res.reshape(shape)
    
    def encoder(self, bias):
        if bias > self.threshold[127]:
            return 127
        for i in range(128):
            if bias <= self.threshold[i]:
                return i

if __name__ == "__main__":
    a = Generator(42)
    data = a.gaussian((100000,))
    # print(a.gaussian((40,)))

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=256, edgecolor='black')
    plt.title('Data Distribution Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('./test.png')
