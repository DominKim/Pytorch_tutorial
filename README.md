# Pytorch_tutorial
## Pytorch_tutorial

### 01
- torch.FloatTensor(x) : 가장 많이 쓰이며, 32bit
- torch.LongTensor(x) : 정수를 담는다, 주로 index를 담을때 많이 쓰임
- torch.ByteTensor(x) : 0, 1을 담을때 쓰이며, torch.BooleanTensor(x)로 대체
- torch.from_numpy(x) : numpy를 토치 텐서로 변환
- torch.numpy() : tensor를 numpy로 변화

### 02. tensor_operations
- tensor 사칙연산 : element wise 연산수행
- element wise operation : 연산이 텐서의 각 요소에 독립적으로 이루어진다는 뜻
``` python3
a = torch.FloatTensor([[1, 2],
                       [3, 4]])
b = torch.FloatTensor([[2, 2],
                       [3, 3]])
a == b
# a==b output
tensor([[False,  True],
        [ True, False]])
```
- inplacee opeartions : 객체에 연산을 덮어 씌우는 것
- ex) a.mul_(b)
- Dimension Reducing Opertations : x.sum(), x.mean()
- ex) x.sum(dim = 0) : 0번째 디멘젼이 사라진다.

### 03. tensor_manipulations
- x.reshape() : change Tensor Shape
- x.view() : low-level change tensor shape(contiguous + view = reshape)
- x.squeeze() : Remove any dimension which has only one element
- x.unsqueeze() : Insert dimension at certain index

### 04. tensor_slicing_and_concat
- x.split(4, dim = 0) : Split tensor to desirable shapes
- x.chunk(3, dim = 0) : Split tensor to number of chunks
- x.index_select : Select elements by using dimension index
- x.concat() : Concatenation of multiple tensors in the list.
- x.stack() : Stacking of multiple tensors in the list

### 05. tensor_useful_methods
- x.expand() : copy the given tensor and concat those at desired dimension
- torch.randperm() : 랜덤한 순열을 만드는 거, Random Permutation
- x.argmax() : Return index of maximum values
- torch.topk(x, k = 1, dim=-1) : Return tuploe of top-k values and indices
- x.masked_fill(maxk, value = -1) : fill the value if element of mask is True
- torch.ones_like(x) : 크기와 디바이스도 같이 복사

## Linear Layer
### 01. matrix_multiplication
- torch.matmul(x, y) : 행렬곱
- torch.bmm(x, y) : 배치 행렬곱
- * bmm은 batch size도 동일해야 된다.

## loss_function
### 02. mse
- torch.nn.functional.mse_loss : 함수를 구현
- torch.nn.MSELoss() : 객체를 구현

## Gradient Descent
### 05. auto_grad
- x.grad : 미분 값 출력