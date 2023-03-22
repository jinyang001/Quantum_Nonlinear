import numpy as np
from qiskit import QuantumCircuit, IBMQ, transpile, assemble
import matplotlib.pyplot as plt
import time

state = np.array([1, 2, 3, 4])
state = np.array([1, 2, 3, 4,5,6,7,8])
state_64 = np.array([-0.4242, -0.4242, -0.4242, -0.4242, -0.3554,  0.1293,  0.1421, -0.2450,
        -0.4242, -0.4242, -0.4242, -0.4242,  0.0288,  0.7081,  0.5333, -0.2450,
        -0.4242, -0.1658, -0.1658, -0.2941,  0.5977,  0.9608,  0.2961, -0.3819,
        -0.3502,  0.2712,  0.2712,  0.0724,  1.0972,  1.3601,  0.4309, -0.0759,
        -0.1291,  0.7948,  1.1603,  1.1094,  1.6731,  1.5339,  0.5127, -0.0546,
        -0.1187,  0.6603,  1.4066,  1.5048,  1.4583,  0.7774,  0.1509, -0.2928,
        -0.3188,  0.1236,  0.8595,  0.9180,  0.4335, -0.3395, -0.4242, -0.4242,
        -0.4242, -0.4242, -0.1772, -0.1135, -0.1743, -0.4242, -0.4242, -0.4242])
state_64 =    np.array([[-0.4242, -0.4242, -0.4242, -0.4242, -0.3554,  0.1293,  0.1421, -0.2450],
                        [-0.4242, -0.4242, -0.4242, -0.4242,  0.0288,  0.7081,  0.5333, -0.2450],
                        [-0.4242, -0.1658, -0.1658, -0.2941,  0.5977,  0.9608,  0.2961, -0.3819],
                        [-0.3502,  0.2712,  0.2712,  0.0724,  1.0972,  1.3601,  0.4309, -0.0759],
                        [-0.1291,  0.7948,  1.1603,  1.1094,  1.6731,  1.5339,  0.5127, -0.0546],
                        [-0.1187,  0.6603,  1.4066,  1.5048,  1.4583,  0.7774,  0.1509, -0.2928],
                        [-0.3188,  0.1236,  0.8595,  0.9180,  0.4335, -0.3395, -0.4242, -0.4242],
                        [-0.4242, -0.4242, -0.1772, -0.1135, -0.1743, -0.4242, -0.4242, -0.4242]])
m = np.asmatrix(state_64)

listarray=[]

for i in range(0,8, 2):
        for j in range(0,8, 2):
                temp=[]
                for k in range(2):
                        temp.append(m[i+k, j])
                        temp.append(m[i+k, j+1])


                listarray.append(temp)
                print(i,j)



print(listarray)
print(len(listarray))

n = 65
#state = range(1,n)
state1 = listarray[0]
state1 = state1 / np.linalg.norm(state1)

state2 = listarray[1]
state2 = state2 / np.linalg.norm(state2)

state3 = listarray[2]
state3 = state3 / np.linalg.norm(state3)

state4 = listarray[3]
state4 = state4 / np.linalg.norm(state4)

state5 = listarray[4]
state5 = state5 / np.linalg.norm(state5)

state6 = listarray[5]
state6 = state6 / np.linalg.norm(state6)

state7 = listarray[6]
state7 = state7 / np.linalg.norm(state7)

state8 = listarray[7]
state8 = state8 / np.linalg.norm(state8)

state9 = listarray[8]
state9 = state9 / np.linalg.norm(state9)

state10 = listarray[9]
state10 = state10 / np.linalg.norm(state10)

state11 = listarray[10]
state11 = state11 / np.linalg.norm(state11)

state12 = listarray[11]
state12 = state12 / np.linalg.norm(state12)

state13 = listarray[12]
state13 = state13 / np.linalg.norm(state13)

state14 = listarray[13]
state14 = state14 / np.linalg.norm(state14)

state15 = listarray[14]
state15 = state15 / np.linalg.norm(state15)

state16 = listarray[15]
state16 = state16 / np.linalg.norm(state16)

state = state_64
state = state / np.linalg.norm(state)

# listarray=listarray / np.linalg.norm(listarray)


for i in range(16):
    listarray[i]=listarray[i] / np.linalg.norm(listarray[i])
print(state5)
print(listarray[4])


print(np.square(state1))
tempss=[np.square(state1),np.square(state2),np.square(state3),np.square(state4)
        ,np.square(state5),np.square(state6),np.square(state7),np.square(state8)
        ,np.square(state9),np.square(state10),np.square(state11),np.square(state12)
        ,np.square(state13),np.square(state14),np.square(state15),np.square(state16)]
tempss=np.array(tempss)
print(tempss)
print(tempss.flatten().tolist())

# sss

# Get backend for experiment
TOKEN='' # use the token from IBM
IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-lanl', group='lanl', project='quantum-optimiza')
#provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('simulator_statevector')

for i in range(16):
    print(i)
    circuit = QuantumCircuit(2)

    circuit.initialize(listarray[i], circuit.qubits)

    circuit.measure_all()

    # prepare the circuit for the backend
    mapped_circuit = transpile(circuit, backend=backend)
    qobj = assemble(mapped_circuit, backend=backend, shots=1000)

    # execute the circuit
    job = backend.run(qobj)

    time.sleep(5)
    # circuit.draw(output='mpl')
print('done')



print(circuit.decompose(reps=100).depth())
circuit.decompose(reps=100).draw(output='mpl')
plt.show()