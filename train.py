from SignalIntegrity.Lib.SParameters.Devices import TLineTwoPortRLGC
from Network import Net
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt

amp_factor = 100

# N = 1025
# batch = 30000
# freq=np.linspace(1e6,1e12,N)

N = 129
batch = 300000
freq=np.linspace(1e6,2e11,N)

# Ts = 1/299792548/(2*(N-1))
# Fs = 1/Ts
# tdesc = TimeDescriptor(0,2*(N-1),Fs)
# Generate random input data and desired output data
omega = 2 * np.pi * freq
# enforce causality by Hilbert Transform
mode = 2
mode_list = ['Hilbert 1', 'Hilbert 2', 'EvenFunction', 'None']

freq = list(freq)

def ForceCausal(_sp,_fr,_Td):
    phase_shift = 1j * omega * _Td

    fr_array = np.array(_fr.Response()) * np.exp(phase_shift)
    fr_real = np.real(np.array(_fr.Response() * np.exp(phase_shift)))

    for k in range(len(fr_real)):
        _fr[k] = fr_real[k]

    ir_even = _fr.ImpulseResponse()  # N+1 --> 5001
    new_ir = copy.deepcopy(ir_even)
    t = ir_even.Times('ns')  # K+1
    ir_even = np.array(ir_even)

    if mode_list[mode - 1] == mode_list[2]:  # even function
        for k in range(len(t)):
            new_ir[k] = ir_even[k] + np.sign(t[k]) * ir_even[k]
        print(new_ir)
        new_fr = new_ir.FrequencyResponse()
        frv = np.array(new_fr.Response())
    elif mode_list[mode - 1] == mode_list[1]:  # Hilbert 2
        fr_amp = abs(fr_array)
        for k in range(len(fr_amp)):
            _fr[k] = fr_amp[k]
        new_ir = _fr.ImpulseResponse()
        _ir = copy.deepcopy(new_ir)
        for k in range(len(t)):
            _ir[k] = new_ir[k] * np.sign(t[k])

        fr_phi = 1j * np.array(_ir.FrequencyResponse().Response())
        frv = fr_amp * np.exp(-1j*fr_phi)
    elif mode_list[mode - 1] == mode_list[0]: # Hilbert 1
        for k in range(len(t)):
            new_ir[k] = 1j * ir_even[k] * np.sign(t[k])
        fr_imag = np.array(new_ir.FrequencyResponse().Response())
        frv = fr_real - 1j * fr_imag

    phase_shift = -phase_shift

    for n in range(len(frv)):
        _fr[n] = frv[n] * np.exp(phase_shift[n])
    return _fr

input_data = np.load('input_data_256.npy')*amp_factor
ground_truth = np.load('ground_truth_256.npy')*amp_factor

X = torch.tensor(input_data, dtype=torch.float32).cuda()
Y = torch.tensor(ground_truth, dtype=torch.float32).cuda()
print('input_tensor shape:', X.shape)
print('ground_truth_tensor shape:', Y.shape)

# model = Net((N-1)*2,(N-1)*4,(N-1)*6,(N-1)*4,(N-1)*2).cuda()
model = Net((N-1)*2,(N-1)*24, (N-1)*2).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_func = torch.nn.MSELoss()

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
torch_data = GetLoader(X, Y)

loss_list = []
B = 400 # batch size
datas = DataLoader(torch_data, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
for step, (input,groundtruth) in enumerate(datas):
    prediction = model(input)
    loss = loss_func(prediction, groundtruth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)
    if step%10==0:
        print('Training step',step/(batch/B)*100,'%')

path='model_param.pt'
torch.save(model.state_dict(), path)
print('Model saved:',path)




# Confirm that it works
R = np.random.uniform(0,0)
Rse = np.random.uniform(0.001,0.001)
L = np.random.uniform(1e-9,5e-8)
G = np.random.uniform(0,0)
C = np.random.uniform(1e-12,5e-11)
df = np.random.uniform(0.00,0.00)
Tdelay = np.sqrt(L*C)
tl = TLineTwoPortRLGC(freq,R,Rse,L,G,C,df)
fr = tl.FrequencyResponse(2, 1)
ir = fr.ImpulseResponse()
data = np.array(ir).astype(float)*amp_factor


assert len(data) == 2*(N-1), "Input must be length {}, but got {}".format(2*(N-1),len(data))
input = torch.unsqueeze(torch.tensor(data, dtype=torch.float32),0).cuda()
output =  torch.squeeze(model(input)).detach().cpu().data

ir_ANN = output
ir_causal = np.array(ForceCausal(tl,fr,Tdelay).ImpulseResponse())*amp_factor
print(f'ANN matches FFT: {np.allclose(ir_ANN, ir_causal)}')

fig, ax = plt.subplots()
ax.plot(ir_ANN,label='ANN output')
ax.plot(ir_causal,label='Hilbert method output')
ax.plot(data,label='non-causal input')
plt.legend()
plt.show()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.imshow(model.cpu().layer1.weight.data, cmap='coolwarm')
ax2.imshow(model.cpu().layer2.weight.data, cmap='coolwarm')

fig, ax = plt.subplots()
ax.plot(loss_list)
ax.set_yscale('log')
ax.set_title('loss')

plt.show()