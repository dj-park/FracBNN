#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, time
import numpy as np

#from pynq import Xlnk
from pynq import Overlay
import pynq


# In[2]:


overlay = Overlay("./design_1_wrapper.bit")
# overlay?
FracNet = overlay.FracNet_T_0
# timer = overlay.axi_timer_0


# In[3]:


image_thermo = pynq.allocate(shape=(3,32,32), dtype=np.uint64)
result = pynq.allocate(shape=(10), dtype=np.float32)

import numpy as np
images = np.load('conv1_input_uint64.npy')

num_tests = 1000
with open('labels.bin', 'rb') as f:
    content = f.read()
print(len(content))

# for i in overlay.ip_dict:
#     print(i)
# print(overlay.FracNet_T_0.s_axi_CTRL.register_map)
# print(overlay.FracNet_T_0.s_axi_control.register_map)

labels = np.ndarray((num_tests,))
for i in range(num_tests):
    labels[i] = content[i]


# In[4]:


FracNet.s_axi_control.register_map.image_r_1.image_r = image_thermo.device_address
FracNet.s_axi_control.register_map.output_r_1.output_r  = result.device_address


# In[5]:



from time import perf_counter

t = 0
correct = 0
for i in range(num_tests):
    np.copyto(image_thermo, images[i])

    idle = 0
    FracNet.s_axi_CTRL.register_map.CTRL.AP_START = 1

    ts = perf_counter()
    while idle == 0:
        idle = FracNet.s_axi_CTRL.register_map.CTRL.AP_IDLE

    tt = perf_counter()
    t += tt - ts
    
    pred = np.argmax(result)
    if pred == labels[i]:
        correct += 1
    #else:
    #    print("pred: " + str(pred) + ", correct: " + str(labels[i]))

print('Latency: %.4f ms'%(t/num_tests*1000))
print('Throughput: %.4f fps'%(1/(t/num_tests)))
print('Accuracy: %.1f%%'%(correct/num_tests*100))


# In[ ]:




