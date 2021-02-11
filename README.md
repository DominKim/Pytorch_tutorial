# Pytorch_tutorial
## Pytorch_tutorial

## Our objective
- 세상에 존재하는 알 수 없는 함수 f를 근사하는 함술 f를 찾고 싶다.
- 따라서 함수 f의 동작을 정의하는 파라미터를 잘 조절해야 한다.
- 손실한수는 파라미터에 따른 함수 f의 동작의 오류의 크기를 반환한다.
- 따라서 손실함수를 최소화 하는 파라미터를 찾으면 된다.

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
- The gradient vector can be interpreted as the "direction and rate of fastest increase".
- x.grad : 미분 값 출력

## Logistic_Regression
### 02. activation_function
- torch.randn() : 정규분포의 랜덤한 수
$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$
$$
\text{tanh}(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$
### 06. logistic_regression
- 변수별 histogram을 통해 분류 정도 확인 가능
``` python3
for c in cols[:-1]:
    sns.histplot(df, x=c, hue=cols[-1], bins=50, stat='probability')
    plt.show()
```

## Stochastic Gradient Descent
- 배치 사이즈에 따라 모델의 학습 속도에 영향을 끼친다. 따라서 적절한 배치 사이즈를 사용하는 것 이 좋다.

## Overfitting
- Train_Loss and Valid_Loss가 같이 줄어 들면 bias나 noise없이 잘 학습 하고 있다.
``` python3
scaler = StandardScaler()
scaler.fit(x[0].numpy()) # You must fit with train data only.

x[0] = torch.from_numpy(scaler.transform(x[0].numpy())).float()
x[1] = torch.from_numpy(scaler.transform(x[1].numpy())).float()
x[2] = torch.from_numpy(scaler.transform(x[2].numpy())).float()
```

## DNN 
- Precision : 내가 postive라고 예측한것 중에서 맞은 것
- Recall : 실제 potive 중 내가 맞춘 비율, 회수율
- AUROC : 두 클래스의 분포 간의 분리 정도를 나타낼 수 있는 metric(같은 accuracy라도 분리 정도에 따라 강인함(robustness)이 다를 수 있다. )
- NLL Loss(Negative Log Likelihood) with Log-Softmax

## Regularizations
- Batch Normalization : 빠른 학습과 높은 성능 보장!
- 단, RNN에는 사용할 수 없음(Layer Normalization 사용)
- model.train(), model.eval() 엄청 중요하다. Regularization을 turn on / off하기 위해