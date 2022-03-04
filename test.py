from SignalIntegrity.Lib.SParameters.Devices import TLineTwoPortRLGC
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from Network import Net

amp_factor = 100
# N = 1025
# batch = 30000
# freq=np.linspace(1e6,1e12,N)

N = 129
batch = 300000
freq=np.linspace(1e6,2e11,N)
omega = 2 * np.pi * freq
freq = list(freq)

# enforce causality by Hilbert Transform
mode = 2
mode_list = ['Hilbert 1', 'Hilbert 2', 'EvenFunction', 'None']
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




# model = Net((N-1)*2,(N-1)*10, (N-1)*2).cuda()
path = 'model_param.pt'
model = Net((N-1)*2,(N-1)*24, (N-1)*2)
# model.cuda()
model.load_state_dict(torch.load(path, map_location='cpu'))
model.eval()

# Heat map of neuron weights
fig, ax = plt.subplots()
ax.imshow(model.layer1.weight.data, cmap='coolwarm')
fig, ax = plt.subplots()
ax.imshow(model.layer2.weight.data, cmap='coolwarm')
plt.show()

# Confirm that it works
def ANN(_x):
    if len(_x) != 2*(N-1):
        raise ValueError('Input must be length {}, but got {}'.format(2*(N-1),len(_x)))
    input = torch.unsqueeze(torch.tensor(_x, dtype=torch.float32),0)
    pred = model(input)
    return torch.squeeze(pred).detach().cpu()

for i in range(10):
    R = np.random.uniform(0, 0)
    Rse = np.random.uniform(0.001, 0.001)
    L = np.random.uniform(1e-9, 5e-8)
    G = np.random.uniform(0, 0)
    C = np.random.uniform(1e-12, 5e-11)
    df = np.random.uniform(0.00, 0.00)
    Tdelay = np.sqrt(L * C)
    tl = TLineTwoPortRLGC(freq, R, Rse, L, G, C, df)
    fr = tl.FrequencyResponse(2, 1)
    ir = fr.ImpulseResponse()
    data = np.array(ir).astype(float) * amp_factor

    assert len(data) == 2 * (N - 1), "Input must be length {}, but got {}".format(2 * (N - 1), len(data))
    input = torch.unsqueeze(torch.tensor(data, dtype=torch.float32), 0)
    output = torch.squeeze(model(input)).detach().cpu().data

    ir_ANN = output
    ir_causal = np.array(ForceCausal(tl, fr, Tdelay).ImpulseResponse()) * amp_factor
    print(f'ANN matches FFT: {np.allclose(ir_ANN, ir_causal)}')
    fig, ax = plt.subplots()
    ax.plot(ir_ANN,label='ANN output')
    ax.plot(ir_causal,label='Hilbert method output')
    ax.plot(data,label='non-causal input')
    plt.legend()
    plt.show()