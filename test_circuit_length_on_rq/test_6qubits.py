import numpy as np
from qiskit import QuantumCircuit, IBMQ, transpile, assemble
import matplotlib.pyplot as plt
import qiskit
# state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
state = np.array([1, 2, 3, 4])
state = np.array([1, 2, 3, 4,5,6,7,8])
state_64_original = np.array([-0.4242, -0.4242, -0.4242, -0.4242, -0.3554,  0.1293,  0.1421, -0.2450,
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
# print(m)
# print(m[3,1])

listarray=[]

# print(listarray)

for i in range(0,8, 4):
        for j in range(0,8, 4):
                temp=[]
                for k in range(4):
                        temp.append(m[i+k, j])
                        temp.append(m[i+k, j+1])
                        temp.append(m[i + k, j+2])
                        temp.append(m[i + k, j+3])

                listarray.append(temp)
                #print(i,j)



# print(listarray)
# print(len(listarray))

# for i in range(0,16):



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

state = state_64_original
state = state / np.linalg.norm(state)
TOKEN='' # use the token from IBM
IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-lanl', group='lanl', project='quantum-optimiza')
#provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibm_geneva')

circuit = QuantumCircuit(6)
circuit.initialize(state, circuit.qubits)
circuit.measure_all()

# prepare the circuit for the backend
mapped_circuit = transpile(circuit, backend=backend)
qobj = assemble(mapped_circuit, backend=backend, shots=1000)

# execute the circuit
job = backend.run(qobj)
# time.sleep(10)
# circuit.draw(output='mpl')

print('done')
print(circuit.decompose(reps=100).depth())
circuit.decompose(reps=100).draw(output='mpl')
plt.show()