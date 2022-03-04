from SignalIntegrity.Lib.SParameters.Devices import TLineTwoPortRLGC
from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor
import copy
import numpy as np
import matplotlib.pyplot as plt

max_tdelay = 0.15/299792548
print('max_tdelay',max_tdelay/1e-9)
N = 129
batch = 300000
freq=np.linspace(1e6,2e11,N)
# Ts = max_tdelay/(2*(N-1))
# Fs = 1/Ts

R_list = np.random.uniform(0,0,batch)
Rse_list = np.random.uniform(0.001,0.001,batch)
L_list = np.random.uniform(1e-9,5e-8,batch)
G_list = np.random.uniform(0,0,batch)
C_list = np.random.uniform(1e-12,5e-11,batch)
df_list = np.random.uniform(0.00,0.00,batch)
Td_List = np.sqrt(L_list*C_list)
# tdesc = TimeDescriptor(0,2*(N-1),Fs)
# print(freq)
# print(np.array(tdesc.FrequencyList()))
# Generate random input data and desired output data
omega = 2 * np.pi * freq
# enforce causality by Hilbert Transform
mode = 2
mode_list = ['Hilbert 1', 'Hilbert 2', 'EvenFunction', 'None']

freq = list(freq)
impulse_data = []
causal_impulse_data = []


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


# Training data generation
for mc_index in range(batch):
    if mc_index%500==0:
        print('Generating data',mc_index/batch*100,'%')
    R = R_list[mc_index]
    Rse = Rse_list[mc_index]
    L = L_list[mc_index]
    G = G_list[mc_index]
    C = C_list[mc_index]
    df = df_list[mc_index]
    Tdelay = Td_List[mc_index]
    # print('delay (ns)',Tdelay/1e-9)
    tl = TLineTwoPortRLGC(freq,R,Rse,L,G,C,df)
    fr = tl.FrequencyResponse(2, 1)
    ir = fr.ImpulseResponse()
    impulse_data.append(ir)  # input data
    fr = ForceCausal(tl,fr,Tdelay)
    ir_causal = fr.ImpulseResponse()
    plt.plot(ir.Times('ns'),ir)
    plt.plot(ir_causal.Times('ns'),ir_causal)
    plt.show()
    causal_impulse_data.append(ir_causal)

input_data = np.array(impulse_data)
ground_truth = np.array(causal_impulse_data)

np.save('input_data_256.npy', input_data)
np.save('ground_truth_256.npy', ground_truth)

print('input_data shape:', input_data.shape)
print('ground_truth shape:', ground_truth.shape)