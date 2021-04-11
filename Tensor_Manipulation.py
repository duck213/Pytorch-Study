import numpy as np
import torch

# 1D Array with NumPy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print('Rank of t: ', t.ndim)  # 1
print('Shape of t: ', t.shape)  # (7,)
print('t[0] t[1] t[-1]', t[0], t[1], t[-1])  # 0.0 1.0 6.0
print('t[2:5] t[4:-1]', t[2:5], t[4:-1])  # [2. 3. 4.] [4. 5.]
print('t[:2] t[3:]', t[:2], t[3:])  # [0. 1.] [3. 4. 5. 6.]

# 2D Array with NumPy
tt = np.array([[1., 2., 3.], [4., 5., 6.],[7., 8., 9.],[10., 11., 12.]])
print('Rank of tt: ', tt.ndim)  # Rank of tt:  2
print('Shape of tt: ', tt.shape)  # Shape of t:  (4,3)


# 1D Array with PyTorch
tch = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(tch.dim()) # 1
print(tch.shape)  # torch.Size([7])
print(tch.size())  # torch.Size([7])
print(tch[0], tch[1], tch[-1])  # tensor(0.) tensor(1.) tensor(6.)
print(tch[2:5], tch[4:-1])  # tensor([2., 3., 4.]) tensor([4., 5.])
print(tch[:2], tch[3:])  # tensor([0., 1.]) tensor([3., 4., 5., 6.])


# 2D Array with PyTorch
tDtch = torch.FloatTensor([[1.,2.,3],[4.,5.,6],[7.,8.,9],[10.,11.,12.]])
print(tDtch.dim())  # 2
print(tDtch.shape)  # torch.Size([4, 3])
print(tDtch.size())  # torch.Size([4, 3])
print(tDtch[:,1])  # torch.Size([4])
print(tDtch[:,1].size())  # torch.Size([4])
print(tDtch[:,:-1])  # tensor([[ 1.,  2.], [ 4.,  5.], [ 7.,  8.], [10., 11.]])


# Broadcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)  # tensor([[5., 5.]])

m3 = torch.FloatTensor([[1, 2]])
m4 = torch.FloatTensor([[3]])
print(m3 + m4)  # tensor([[4., 5.]])

m5 = torch.FloatTensor([[1, 2]])
m6 = torch.FloatTensor([[3], [4]])
print(m5 + m6)  # tensor([[4., 5.], [5., 6.]])

# multiplication vs matrix multiplication
m7 = torch.FloatTensor([[1,2],[3,4]])
m8 = torch.FloatTensor([[1],[2]])
print('shape of matrix 7: ', m7.shape)  # 2 x 2
print('shape of matrix 8: ', m8.shape)  # 2 x 1
print(m7.matmul(m8))  # 2 x 1, tensor([[ 5.], [11.]])

print(m7 * m8)  # 2 x 2, tensor([[1., 2.], [6., 8.]])
print(m7.mul(m8))  # 2 x 2, tensor([[1., 2.], [6., 8.]])

# mean
mean = torch.FloatTensor([[1, 2]])
print(mean.mean()) # tensor(1.5000)
'''
long_mean = torch.LongTensor([[1, 2]])
try:
    print(long_mean.mean())
except Exception as exc:
    print(exc)
'''

# mean
tm = torch.FloatTensor([[1, 2],[3, 4]])
print(tm.mean())  # tensor(2.5000)
print(tm.mean(dim=0))  #  tensor([2., 3.])
print(tm.mean(dim=1))  # tensor([1.5000, 3.5000])
print(tm.mean(dim=-1))  # tensor([1.5000, 3.5000])

# sum
print(tm.sum())  # tensor([4., 6.])
print(tm.sum(dim=0))  # tensor([2., 3.])
print(tm.sum(dim=1))  # tensor([3., 7.])
print(tm.sum(dim=-1))  # tensor([3., 7.])

# max, argmax
print(tm.max())  # tensor(4.)
print(tm.max(dim=0))
'''
torch.return_types.max(
values=tensor([3., 4.]), 
indices=tensor([1, 1]))
'''
print('Max: ', tm.max(dim=0)[0])  # Max: tensor([3., 4.])
print('Argmax: ', tm.max(dim=0)[1])  # Argmax: tensor([1, 1])

# view
view_t = np.array([[[0, 1, 2],[3, 4, 5]],[[6, 7, 8],[9, 10, 11]]])
ft = torch.FloatTensor(view_t)
print(ft.shape)  # torch.Size([2, 2, 3])
print(ft.view([-1,3]))
'''
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
'''
print(ft.view([-1,3]).shape)  # torch.Size([4, 3])
print(ft.view([-1,1,3]))
'''
tensor([[[ 0.,  1.,  2.]],

        [[ 3.,  4.,  5.]],

        [[ 6.,  7.,  8.]],

        [[ 9., 10., 11.]]])
'''
print(ft.view([-1,1,3]).shape)  # torch.Size([4, 1, 3])

# squeeze
sq = torch.FloatTensor([[0],[1],[2]])
print(sq.shape)  # torch.Size([3, 1])
print(sq.squeeze())  # tensor([0., 1., 2.])
print(sq.squeeze().shape)  # torch.Size([3])

# unsqueeze
unsq = torch.FloatTensor([0,1,2])
print(unsq.shape)  # torch.Size([3])
print(unsq.unsqueeze(0))  # tensor([[0., 1., 2.]])
print(unsq.unsqueeze(0).shape)  # torch.Size([1, 3])
print(unsq.view(1,-1)) # tensor([[0., 1., 2.]])
print(unsq.view(1,-1).shape) # torch.Size([1, 3])
print(unsq.unsqueeze(1))
'''
tensor([[0.],
        [1.],
        [2.]])
'''
print(unsq.unsqueeze(1).shape)  # torch.Size([3, 1])
print(unsq.unsqueeze(-1))
'''
tensor([[0.],
        [1.],
        [2.]])
'''
print(unsq.unsqueeze(-1).shape)  # torch.Size([3, 1])

#scatter
lt = torch.LongTensor([[0],[1],[2],[0]])
one_hot = torch.zeros(4,3)
one_hot.scatter_(1, lt, 1)
print(one_hot)
'''
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.]])
'''

# casting
cst = torch.LongTensor([1,2,3,4])
print(lt.float()) # tensor([[0.],[1.],[2.],[0.]])
bt = torch.ByteTensor([True, False, False, True])
print(bt)  # tensor([1, 0, 0, 1], dtype=torch.uint8)
print(bt.long())  # tensor([1, 0, 0, 1])
print(bt.float())  # tensor([1., 0., 0., 1.])

# concatenation
ccn1 = torch.FloatTensor([[1,2],[3,4]])
ccn2 = torch.FloatTensor([[5,6],[7,8]])
print(torch.cat([ccn1,ccn2], dim=0))
'''
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
'''
print(torch.cat([ccn1,ccn2], dim=1))
'''
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
'''

# stacking
stack_x = torch.FloatTensor([1,4])
stack_y = torch.FloatTensor([2,5])
stack_z = torch.FloatTensor([3,6])

print(torch.stack([stack_x,stack_y,stack_z]))
'''
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
'''
print(torch.stack([stack_x,stack_y,stack_z], dim=1))
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]])
'''
print(torch.stack([stack_x.unsqueeze(0),stack_y.unsqueeze(0),stack_z.unsqueeze(0)], dim=0))
'''
tensor([[[1., 4.]],

        [[2., 5.]],

        [[3., 6.]]])
'''

# Ones and Zeros Like
one_zeros = torch.FloatTensor([[0,1,2],[2,1,0]])
print(torch.ones_like(one_zeros))
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]])
'''
print(torch.zeros_like(one_zeros))
'''
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''
# In-place Operation
inplace = torch.FloatTensor([[1,2],[3,4]])
print(inplace.mul(2.))
'''
tensor([[2., 4.],
        [6., 8.]])
'''
print(inplace)
'''
tensor([[1., 2.],
        [3., 4.]])
'''
print(inplace.mul_(2.))
'''
tensor([[2., 4.],
        [6., 8.]])
'''
print(inplace)
'''
tensor([[2., 4.],
        [6., 8.]])
'''
# Zip
for x,y in zip([1,2,3],[4,5,6]):
    print(x,y)
# 1 4
# 2 5
# 3 6

for x,y,z in zip([1,2,3],[4,5,6],[7,8,9]):
    print(x,y,z)
# 1 4 7
# 2 5 8
# 3 6 9
