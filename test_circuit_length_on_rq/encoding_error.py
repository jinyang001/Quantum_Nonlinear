import torch
#amp
x=torch.tensor([-0.2985, -0.1706, -0.5186,-0.1122,-0.5158, 0.0191, -0.0735,0.0946])

x2=torch.tensor([-0.2026,-0.0588, 0.0688,0.3433,-0.1448, -0.1035,-0.2144, 0.0322])

#our
x=torch.tensor([-0.5497, -0.0997, -0.0599, -0.9305,  0.2242, -0.5268,  0.0962,  0.7642,
        -0.7132, -0.7488, -0.6424,  0.6370,  0.3540, -0.2754,  0.0580, -0.1894])
x2=torch.tensor([-0.4131,  0.0234, -0.0303, -0.8586,  0.2124, -0.5103,  0.0806,  0.7349,
        -0.6714, -0.6868, -0.4656,  0.5190,  0.3516, -0.2537,  0.0649, -0.1921])
#angle
x=torch.tensor([0.5938, 0.7884, 0.0770, 0.4723,
        0.5390, 0.5199, 0.3892, 0.7544])


x2=torch.tensor([0.5156, 0.4780, 0.3574, 0.7441,
        0.6013, 0.7476, 0.0750, 0.4712])

# x=torch.pow(x,2)
y = torch.sub(x,x2)
y=torch.pow(y,2)
print(torch.mean(y))