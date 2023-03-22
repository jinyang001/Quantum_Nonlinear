from qiskit import IBMQ
TOKEN='' # use the token from IBM
IBMQ.save_account(TOKEN, overwrite=True)
IBMQ.load_account() # Load account from disk
print(IBMQ.providers())    # List all available providers