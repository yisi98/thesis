Ê¼
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ÉÀ

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
: *
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:@*
dtype0

batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_9/gamma

/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:@*
dtype0

batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_9/beta

.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:@*
dtype0

!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_9/moving_mean

5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_9/moving_variance

9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
:@*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes	
:*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes	
:*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:*
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:*
dtype0
u
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_14/bias
n
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes	
:*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
 *
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	f*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:f*
dtype0
|
training_4/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_4/Adam/iter
u
(training_4/Adam/iter/Read/ReadVariableOpReadVariableOptraining_4/Adam/iter*
_output_shapes
: *
dtype0	

training_4/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_4/Adam/beta_1
y
*training_4/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_4/Adam/beta_1*
_output_shapes
: *
dtype0

training_4/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_4/Adam/beta_2
y
*training_4/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_4/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_4/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_4/Adam/decay
w
)training_4/Adam/decay/Read/ReadVariableOpReadVariableOptraining_4/Adam/decay*
_output_shapes
: *
dtype0

training_4/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_4/Adam/learning_rate

1training_4/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_4/Adam/learning_rate*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
¨
"training_4/Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_4/Adam/conv2d_10/kernel/m
¡
6training_4/Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_10/kernel/m*&
_output_shapes
: *
dtype0

 training_4/Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_4/Adam/conv2d_10/bias/m

4training_4/Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_10/bias/m*
_output_shapes
: *
dtype0
¨
"training_4/Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *3
shared_name$"training_4/Adam/conv2d_11/kernel/m
¡
6training_4/Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_11/kernel/m*&
_output_shapes
:  *
dtype0

 training_4/Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_4/Adam/conv2d_11/bias/m

4training_4/Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_11/bias/m*
_output_shapes
: *
dtype0
²
-training_4/Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-training_4/Adam/batch_normalization_8/gamma/m
«
Atraining_4/Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_8/gamma/m*
_output_shapes
: *
dtype0
°
,training_4/Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,training_4/Adam/batch_normalization_8/beta/m
©
@training_4/Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp,training_4/Adam/batch_normalization_8/beta/m*
_output_shapes
: *
dtype0
¨
"training_4/Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"training_4/Adam/conv2d_12/kernel/m
¡
6training_4/Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_12/kernel/m*&
_output_shapes
: @*
dtype0

 training_4/Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_4/Adam/conv2d_12/bias/m

4training_4/Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_12/bias/m*
_output_shapes
:@*
dtype0
²
-training_4/Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-training_4/Adam/batch_normalization_9/gamma/m
«
Atraining_4/Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_9/gamma/m*
_output_shapes
:@*
dtype0
°
,training_4/Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,training_4/Adam/batch_normalization_9/beta/m
©
@training_4/Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp,training_4/Adam/batch_normalization_9/beta/m*
_output_shapes
:@*
dtype0
©
"training_4/Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"training_4/Adam/conv2d_13/kernel/m
¢
6training_4/Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_13/kernel/m*'
_output_shapes
:@*
dtype0

 training_4/Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_4/Adam/conv2d_13/bias/m

4training_4/Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_13/bias/m*
_output_shapes	
:*
dtype0
µ
.training_4/Adam/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.training_4/Adam/batch_normalization_10/gamma/m
®
Btraining_4/Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp.training_4/Adam/batch_normalization_10/gamma/m*
_output_shapes	
:*
dtype0
³
-training_4/Adam/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-training_4/Adam/batch_normalization_10/beta/m
¬
Atraining_4/Adam/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_10/beta/m*
_output_shapes	
:*
dtype0
ª
"training_4/Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_4/Adam/conv2d_14/kernel/m
£
6training_4/Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_14/kernel/m*(
_output_shapes
:*
dtype0

 training_4/Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_4/Adam/conv2d_14/bias/m

4training_4/Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_14/bias/m*
_output_shapes	
:*
dtype0
µ
.training_4/Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.training_4/Adam/batch_normalization_11/gamma/m
®
Btraining_4/Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp.training_4/Adam/batch_normalization_11/gamma/m*
_output_shapes	
:*
dtype0
³
-training_4/Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-training_4/Adam/batch_normalization_11/beta/m
¬
Atraining_4/Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_11/beta/m*
_output_shapes	
:*
dtype0

 training_4/Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *1
shared_name" training_4/Adam/dense_4/kernel/m

4training_4/Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_4/kernel/m* 
_output_shapes
:
 *
dtype0

training_4/Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training_4/Adam/dense_4/bias/m

2training_4/Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_4/bias/m*
_output_shapes	
:*
dtype0

 training_4/Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f*1
shared_name" training_4/Adam/dense_5/kernel/m

4training_4/Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_5/kernel/m*
_output_shapes
:	f*
dtype0

training_4/Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*/
shared_name training_4/Adam/dense_5/bias/m

2training_4/Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_5/bias/m*
_output_shapes
:f*
dtype0
¨
"training_4/Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"training_4/Adam/conv2d_10/kernel/v
¡
6training_4/Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_10/kernel/v*&
_output_shapes
: *
dtype0

 training_4/Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_4/Adam/conv2d_10/bias/v

4training_4/Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_10/bias/v*
_output_shapes
: *
dtype0
¨
"training_4/Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *3
shared_name$"training_4/Adam/conv2d_11/kernel/v
¡
6training_4/Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_11/kernel/v*&
_output_shapes
:  *
dtype0

 training_4/Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" training_4/Adam/conv2d_11/bias/v

4training_4/Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_11/bias/v*
_output_shapes
: *
dtype0
²
-training_4/Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-training_4/Adam/batch_normalization_8/gamma/v
«
Atraining_4/Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_8/gamma/v*
_output_shapes
: *
dtype0
°
,training_4/Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,training_4/Adam/batch_normalization_8/beta/v
©
@training_4/Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp,training_4/Adam/batch_normalization_8/beta/v*
_output_shapes
: *
dtype0
¨
"training_4/Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"training_4/Adam/conv2d_12/kernel/v
¡
6training_4/Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_12/kernel/v*&
_output_shapes
: @*
dtype0

 training_4/Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_4/Adam/conv2d_12/bias/v

4training_4/Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_12/bias/v*
_output_shapes
:@*
dtype0
²
-training_4/Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-training_4/Adam/batch_normalization_9/gamma/v
«
Atraining_4/Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_9/gamma/v*
_output_shapes
:@*
dtype0
°
,training_4/Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,training_4/Adam/batch_normalization_9/beta/v
©
@training_4/Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp,training_4/Adam/batch_normalization_9/beta/v*
_output_shapes
:@*
dtype0
©
"training_4/Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"training_4/Adam/conv2d_13/kernel/v
¢
6training_4/Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_13/kernel/v*'
_output_shapes
:@*
dtype0

 training_4/Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_4/Adam/conv2d_13/bias/v

4training_4/Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_13/bias/v*
_output_shapes	
:*
dtype0
µ
.training_4/Adam/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.training_4/Adam/batch_normalization_10/gamma/v
®
Btraining_4/Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp.training_4/Adam/batch_normalization_10/gamma/v*
_output_shapes	
:*
dtype0
³
-training_4/Adam/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-training_4/Adam/batch_normalization_10/beta/v
¬
Atraining_4/Adam/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_10/beta/v*
_output_shapes	
:*
dtype0
ª
"training_4/Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"training_4/Adam/conv2d_14/kernel/v
£
6training_4/Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOp"training_4/Adam/conv2d_14/kernel/v*(
_output_shapes
:*
dtype0

 training_4/Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_4/Adam/conv2d_14/bias/v

4training_4/Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOp training_4/Adam/conv2d_14/bias/v*
_output_shapes	
:*
dtype0
µ
.training_4/Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.training_4/Adam/batch_normalization_11/gamma/v
®
Btraining_4/Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp.training_4/Adam/batch_normalization_11/gamma/v*
_output_shapes	
:*
dtype0
³
-training_4/Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-training_4/Adam/batch_normalization_11/beta/v
¬
Atraining_4/Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp-training_4/Adam/batch_normalization_11/beta/v*
_output_shapes	
:*
dtype0

 training_4/Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *1
shared_name" training_4/Adam/dense_4/kernel/v

4training_4/Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_4/kernel/v* 
_output_shapes
:
 *
dtype0

training_4/Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training_4/Adam/dense_4/bias/v

2training_4/Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_4/bias/v*
_output_shapes	
:*
dtype0

 training_4/Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f*1
shared_name" training_4/Adam/dense_5/kernel/v

4training_4/Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_5/kernel/v*
_output_shapes
:	f*
dtype0

training_4/Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*/
shared_name training_4/Adam/dense_5/bias/v

2training_4/Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_5/bias/v*
_output_shapes
:f*
dtype0

NoOpNoOp
õ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¯
value¤B  B

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
R
&	variables
'regularization_losses
(trainable_variables
)	keras_api
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api

.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
h

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
R
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api

Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
h

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
R
c	variables
dregularization_losses
etrainable_variables
f	keras_api

gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
R
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
h

tkernel
ubias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
R
z	variables
{regularization_losses
|trainable_variables
}	keras_api
l

~kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
ý
	iter
beta_1
beta_2

decay
learning_ratem÷mø mù!mú/mû0mü7mý8mþBmÿCmJmKmUmVm]m^mhmimtmum~mmvv v!v/v0v7v8vBvCvJvKvUvVv]v^vhvivtvuv ~v¡v¢
æ
0
1
 2
!3
/4
05
16
27
78
89
B10
C11
D12
E13
J14
K15
U16
V17
W18
X19
]20
^21
h22
i23
j24
k25
t26
u27
~28
29
 
¦
0
1
 2
!3
/4
05
76
87
B8
C9
J10
K11
U12
V13
]14
^15
h16
i17
t18
u19
~20
21
²
layer_metrics
	variables
regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
 layer_regularization_losses
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
²
layer_metrics
	variables
regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
 layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
²
layer_metrics
"	variables
#regularization_losses
$trainable_variables
metrics
non_trainable_variables
layers
 layer_regularization_losses
 
 
 
²
layer_metrics
&	variables
'regularization_losses
(trainable_variables
metrics
non_trainable_variables
layers
  layer_regularization_losses
 
 
 
²
¡layer_metrics
*	variables
+regularization_losses
,trainable_variables
¢metrics
£non_trainable_variables
¤layers
 ¥layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

/0
01
12
23
 

/0
01
²
¦layer_metrics
3	variables
4regularization_losses
5trainable_variables
§metrics
¨non_trainable_variables
©layers
 ªlayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
²
«layer_metrics
9	variables
:regularization_losses
;trainable_variables
¬metrics
­non_trainable_variables
®layers
 ¯layer_regularization_losses
 
 
 
²
°layer_metrics
=	variables
>regularization_losses
?trainable_variables
±metrics
²non_trainable_variables
³layers
 ´layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3
 

B0
C1
²
µlayer_metrics
F	variables
Gregularization_losses
Htrainable_variables
¶metrics
·non_trainable_variables
¸layers
 ¹layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
²
ºlayer_metrics
L	variables
Mregularization_losses
Ntrainable_variables
»metrics
¼non_trainable_variables
½layers
 ¾layer_regularization_losses
 
 
 
²
¿layer_metrics
P	variables
Qregularization_losses
Rtrainable_variables
Àmetrics
Ánon_trainable_variables
Âlayers
 Ãlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
W2
X3
 

U0
V1
²
Älayer_metrics
Y	variables
Zregularization_losses
[trainable_variables
Åmetrics
Ænon_trainable_variables
Çlayers
 Èlayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
²
Élayer_metrics
_	variables
`regularization_losses
atrainable_variables
Êmetrics
Ënon_trainable_variables
Ìlayers
 Ílayer_regularization_losses
 
 
 
²
Îlayer_metrics
c	variables
dregularization_losses
etrainable_variables
Ïmetrics
Ðnon_trainable_variables
Ñlayers
 Òlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

h0
i1
j2
k3
 

h0
i1
²
Ólayer_metrics
l	variables
mregularization_losses
ntrainable_variables
Ômetrics
Õnon_trainable_variables
Ölayers
 ×layer_regularization_losses
 
 
 
²
Ølayer_metrics
p	variables
qregularization_losses
rtrainable_variables
Ùmetrics
Únon_trainable_variables
Ûlayers
 Ülayer_regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
 

t0
u1
²
Ýlayer_metrics
v	variables
wregularization_losses
xtrainable_variables
Þmetrics
ßnon_trainable_variables
àlayers
 álayer_regularization_losses
 
 
 
²
âlayer_metrics
z	variables
{regularization_losses
|trainable_variables
ãmetrics
änon_trainable_variables
ålayers
 ælayer_regularization_losses
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1
 

~0
1
µ
çlayer_metrics
	variables
regularization_losses
trainable_variables
èmetrics
énon_trainable_variables
êlayers
 ëlayer_regularization_losses
 
 
 
µ
ìlayer_metrics
	variables
regularization_losses
trainable_variables
ímetrics
înon_trainable_variables
ïlayers
 ðlayer_regularization_losses
SQ
VARIABLE_VALUEtraining_4/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_4/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_4/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_4/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_4/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

ñ0
8
10
21
D2
E3
W4
X5
j6
k7

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

W0
X1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
I

òtotal

ócount
ô
_fn_kwargs
õ	variables
ö	keras_api
QO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

ò0
ó1

õ	variables

VARIABLE_VALUE"training_4/Adam/conv2d_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_11/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_11/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_8/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,training_4/Adam/batch_normalization_8/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_12/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_12/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_9/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,training_4/Adam/batch_normalization_9/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_13/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_13/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.training_4/Adam/batch_normalization_10/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_10/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_14/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_14/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.training_4/Adam/batch_normalization_11/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_11/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/dense_4/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining_4/Adam/dense_4/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/dense_5/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining_4/Adam/dense_5/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_11/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_11/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_8/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,training_4/Adam/batch_normalization_8/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_12/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_12/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_9/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,training_4/Adam/batch_normalization_9/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_13/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_13/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.training_4/Adam/batch_normalization_10/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_10/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"training_4/Adam/conv2d_14/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/conv2d_14/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.training_4/Adam/batch_normalization_11/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_4/Adam/batch_normalization_11/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/dense_4/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining_4/Adam/dense_4/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE training_4/Adam/dense_5/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEtraining_4/Adam/dense_5/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_10_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿdd
ò
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_10_inputconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_12060
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
þ#
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp(training_4/Adam/iter/Read/ReadVariableOp*training_4/Adam/beta_1/Read/ReadVariableOp*training_4/Adam/beta_2/Read/ReadVariableOp)training_4/Adam/decay/Read/ReadVariableOp1training_4/Adam/learning_rate/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp6training_4/Adam/conv2d_10/kernel/m/Read/ReadVariableOp4training_4/Adam/conv2d_10/bias/m/Read/ReadVariableOp6training_4/Adam/conv2d_11/kernel/m/Read/ReadVariableOp4training_4/Adam/conv2d_11/bias/m/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp@training_4/Adam/batch_normalization_8/beta/m/Read/ReadVariableOp6training_4/Adam/conv2d_12/kernel/m/Read/ReadVariableOp4training_4/Adam/conv2d_12/bias/m/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp@training_4/Adam/batch_normalization_9/beta/m/Read/ReadVariableOp6training_4/Adam/conv2d_13/kernel/m/Read/ReadVariableOp4training_4/Adam/conv2d_13/bias/m/Read/ReadVariableOpBtraining_4/Adam/batch_normalization_10/gamma/m/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_10/beta/m/Read/ReadVariableOp6training_4/Adam/conv2d_14/kernel/m/Read/ReadVariableOp4training_4/Adam/conv2d_14/bias/m/Read/ReadVariableOpBtraining_4/Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_11/beta/m/Read/ReadVariableOp4training_4/Adam/dense_4/kernel/m/Read/ReadVariableOp2training_4/Adam/dense_4/bias/m/Read/ReadVariableOp4training_4/Adam/dense_5/kernel/m/Read/ReadVariableOp2training_4/Adam/dense_5/bias/m/Read/ReadVariableOp6training_4/Adam/conv2d_10/kernel/v/Read/ReadVariableOp4training_4/Adam/conv2d_10/bias/v/Read/ReadVariableOp6training_4/Adam/conv2d_11/kernel/v/Read/ReadVariableOp4training_4/Adam/conv2d_11/bias/v/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp@training_4/Adam/batch_normalization_8/beta/v/Read/ReadVariableOp6training_4/Adam/conv2d_12/kernel/v/Read/ReadVariableOp4training_4/Adam/conv2d_12/bias/v/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp@training_4/Adam/batch_normalization_9/beta/v/Read/ReadVariableOp6training_4/Adam/conv2d_13/kernel/v/Read/ReadVariableOp4training_4/Adam/conv2d_13/bias/v/Read/ReadVariableOpBtraining_4/Adam/batch_normalization_10/gamma/v/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_10/beta/v/Read/ReadVariableOp6training_4/Adam/conv2d_14/kernel/v/Read/ReadVariableOp4training_4/Adam/conv2d_14/bias/v/Read/ReadVariableOpBtraining_4/Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpAtraining_4/Adam/batch_normalization_11/beta/v/Read/ReadVariableOp4training_4/Adam/dense_4/kernel/v/Read/ReadVariableOp2training_4/Adam/dense_4/bias/v/Read/ReadVariableOp4training_4/Adam/dense_5/kernel/v/Read/ReadVariableOp2training_4/Adam/dense_5/bias/v/Read/ReadVariableOpConst*^
TinW
U2S	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_13246
¥
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/biastraining_4/Adam/itertraining_4/Adam/beta_1training_4/Adam/beta_2training_4/Adam/decaytraining_4/Adam/learning_ratetotal_4count_4"training_4/Adam/conv2d_10/kernel/m training_4/Adam/conv2d_10/bias/m"training_4/Adam/conv2d_11/kernel/m training_4/Adam/conv2d_11/bias/m-training_4/Adam/batch_normalization_8/gamma/m,training_4/Adam/batch_normalization_8/beta/m"training_4/Adam/conv2d_12/kernel/m training_4/Adam/conv2d_12/bias/m-training_4/Adam/batch_normalization_9/gamma/m,training_4/Adam/batch_normalization_9/beta/m"training_4/Adam/conv2d_13/kernel/m training_4/Adam/conv2d_13/bias/m.training_4/Adam/batch_normalization_10/gamma/m-training_4/Adam/batch_normalization_10/beta/m"training_4/Adam/conv2d_14/kernel/m training_4/Adam/conv2d_14/bias/m.training_4/Adam/batch_normalization_11/gamma/m-training_4/Adam/batch_normalization_11/beta/m training_4/Adam/dense_4/kernel/mtraining_4/Adam/dense_4/bias/m training_4/Adam/dense_5/kernel/mtraining_4/Adam/dense_5/bias/m"training_4/Adam/conv2d_10/kernel/v training_4/Adam/conv2d_10/bias/v"training_4/Adam/conv2d_11/kernel/v training_4/Adam/conv2d_11/bias/v-training_4/Adam/batch_normalization_8/gamma/v,training_4/Adam/batch_normalization_8/beta/v"training_4/Adam/conv2d_12/kernel/v training_4/Adam/conv2d_12/bias/v-training_4/Adam/batch_normalization_9/gamma/v,training_4/Adam/batch_normalization_9/beta/v"training_4/Adam/conv2d_13/kernel/v training_4/Adam/conv2d_13/bias/v.training_4/Adam/batch_normalization_10/gamma/v-training_4/Adam/batch_normalization_10/beta/v"training_4/Adam/conv2d_14/kernel/v training_4/Adam/conv2d_14/bias/v.training_4/Adam/batch_normalization_11/gamma/v-training_4/Adam/batch_normalization_11/beta/v training_4/Adam/dense_4/kernel/vtraining_4/Adam/dense_4/bias/v training_4/Adam/dense_5/kernel/vtraining_4/Adam/dense_5/bias/v*]
TinV
T2R*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_13499×«
¦	
º
D__inference_conv2d_10_layer_call_and_return_conditional_losses_11315

inputs*
&conv2d_readvariableop_conv2d_10_kernel)
%biasadd_readvariableop_conv2d_10_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_10_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
©
E
)__inference_flatten_2_layer_call_fn_12909

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_116972
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
´
B__inference_dense_5_layer_call_and_return_conditional_losses_12963

inputs(
$matmul_readvariableop_dense_5_kernel'
#biasadd_readvariableop_dense_5_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_5_kernel*
_output_shapes
:	f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:f*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_10987

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_10979

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬	
º
D__inference_conv2d_13_layer_call_and_return_conditional_losses_11526

inputs*
&conv2d_readvariableop_conv2d_13_kernel)
%biasadd_readvariableop_conv2d_13_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_13_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_13_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ç
´
B__inference_dense_4_layer_call_and_return_conditional_losses_11715

inputs(
$matmul_readvariableop_dense_4_kernel'
#biasadd_readvariableop_dense_4_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_4_kernel* 
_output_shapes
:
 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	

6__inference_batch_normalization_10_layer_call_fn_12709

inputs 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_115612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ

::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
Ð
ä
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_11075

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
º
D__inference_conv2d_11_layer_call_and_return_conditional_losses_12395

inputs*
&conv2d_readvariableop_conv2d_11_kernel)
%biasadd_readvariableop_conv2d_11_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿbb :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
Õ
×
,__inference_sequential_2_layer_call_fn_12332

inputs
conv2d_10_kernel
conv2d_10_bias
conv2d_11_kernel
conv2d_11_bias
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
conv2d_12_kernel
conv2d_12_bias
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_13_kernel
conv2d_13_bias 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
conv2d_14_kernel
conv2d_14_bias 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
dense_4_kernel
dense_4_bias
dense_5_kernel
dense_5_bias
identity¢StatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_kernelconv2d_10_biasconv2d_11_kernelconv2d_11_biasbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancedense_4_kerneldense_4_biasdense_5_kerneldense_5_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_119022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ì

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10939

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	

5__inference_batch_normalization_8_layer_call_fn_12457

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_113852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs

é
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12880

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
ä
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_10966

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_11_layer_call_fn_11208

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_112052
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12700

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ

:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs

g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_11197

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_12938

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
´
B__inference_dense_4_layer_call_and_return_conditional_losses_12919

inputs(
$matmul_readvariableop_dense_4_kernel'
#biasadd_readvariableop_dense_4_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_4_kernel* 
_output_shapes
:
 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì

P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_11048

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½

Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12682

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ

::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
Ë
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_12943

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

'__inference_dense_4_layer_call_fn_12926

inputs
dense_4_kernel
dense_4_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_117152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_11096

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_11579

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ

:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
ì

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12484

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12430

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs


Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12808

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

6__inference_batch_normalization_11_layer_call_fn_12889

inputs 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_116492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_11266

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
c
G__inference_activation_4_layer_call_and_return_conditional_losses_11354

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 
 
_user_specified_nameinputs
Æª
Ë
G__inference_sequential_2_layer_call_and_return_conditional_losses_12182

inputs4
0conv2d_10_conv2d_readvariableop_conv2d_10_kernel3
/conv2d_10_biasadd_readvariableop_conv2d_10_bias4
0conv2d_11_conv2d_readvariableop_conv2d_11_kernel3
/conv2d_11_biasadd_readvariableop_conv2d_11_biasD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance4
0conv2d_12_conv2d_readvariableop_conv2d_12_kernel3
/conv2d_12_biasadd_readvariableop_conv2d_12_biasD
@batch_normalization_9_readvariableop_batch_normalization_9_gammaE
Abatch_normalization_9_readvariableop_1_batch_normalization_9_beta[
Wbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meana
]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance4
0conv2d_13_conv2d_readvariableop_conv2d_13_kernel3
/conv2d_13_biasadd_readvariableop_conv2d_13_biasF
Bbatch_normalization_10_readvariableop_batch_normalization_10_gammaG
Cbatch_normalization_10_readvariableop_1_batch_normalization_10_beta]
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanc
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance4
0conv2d_14_conv2d_readvariableop_conv2d_14_kernel3
/conv2d_14_biasadd_readvariableop_conv2d_14_biasF
Bbatch_normalization_11_readvariableop_batch_normalization_11_gammaG
Cbatch_normalization_11_readvariableop_1_batch_normalization_11_beta]
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanc
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance0
,dense_4_matmul_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias0
,dense_5_matmul_readvariableop_dense_5_kernel/
+dense_5_biasadd_readvariableop_dense_5_bias
identity¢%batch_normalization_10/AssignNewValue¢'batch_normalization_10/AssignNewValue_1¢%batch_normalization_11/AssignNewValue¢'batch_normalization_11/AssignNewValue_1¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢$batch_normalization_9/AssignNewValue¢&batch_normalization_9/AssignNewValue_1»
conv2d_10/Conv2D/ReadVariableOpReadVariableOp0conv2d_10_conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOpÂ
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
paddingVALID*
strides
2
conv2d_10/Conv2D°
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/conv2d_10_biasadd_readvariableop_conv2d_10_bias*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
conv2d_10/Relu»
conv2d_11/Conv2D/ReadVariableOpReadVariableOp0conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:  *
dtype02!
conv2d_11/Conv2D/ReadVariableOpØ
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingVALID*
strides
2
conv2d_11/Conv2D°
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
conv2d_11/BiasAdd
activation_4/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
activation_4/ReluË
max_pooling2d_8/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPoolÉ
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOpÎ
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ö
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_8/MaxPool:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_8/FusedBatchNormV3µ
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValueË
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1»
conv2d_12/Conv2D/ReadVariableOpReadVariableOp0conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: @*
dtype02!
conv2d_12/Conv2D/ReadVariableOpæ
conv2d_12/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
conv2d_12/Conv2D°
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp°
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_12/ReluÈ
max_pooling2d_9/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPoolÉ
$batch_normalization_9/ReadVariableOpReadVariableOp@batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpÎ
&batch_normalization_9/ReadVariableOp_1ReadVariableOpAbatch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ö
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_9/MaxPool:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_9/FusedBatchNormV3µ
$batch_normalization_9/AssignNewValueAssignVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValueË
&batch_normalization_9/AssignNewValue_1AssignVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1¼
conv2d_13/Conv2D/ReadVariableOpReadVariableOp0conv2d_13_conv2d_readvariableop_conv2d_13_kernel*'
_output_shapes
:@*
dtype02!
conv2d_13/Conv2D/ReadVariableOpç
conv2d_13/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_13/Conv2D±
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/ReluË
max_pooling2d_10/MaxPoolMaxPoolconv2d_13/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPoolÎ
%batch_normalization_10/ReadVariableOpReadVariableOpBbatch_normalization_10_readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02'
%batch_normalization_10/ReadVariableOpÓ
'batch_normalization_10/ReadVariableOp_1ReadVariableOpCbatch_normalization_10_readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02)
'batch_normalization_10/ReadVariableOp_1
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_10/MaxPool:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_10/FusedBatchNormV3½
%batch_normalization_10/AssignNewValueAssignVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_10/AssignNewValueÓ
'batch_normalization_10/AssignNewValue_1AssignVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_10/AssignNewValue_1½
conv2d_14/Conv2D/ReadVariableOpReadVariableOp0conv2d_14_conv2d_readvariableop_conv2d_14_kernel*(
_output_shapes
:*
dtype02!
conv2d_14/Conv2D/ReadVariableOpè
conv2d_14/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_14/Conv2D±
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp/conv2d_14_biasadd_readvariableop_conv2d_14_bias*
_output_shapes	
:*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp±
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_14/ReluË
max_pooling2d_11/MaxPoolMaxPoolconv2d_14/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolÎ
%batch_normalization_11/ReadVariableOpReadVariableOpBbatch_normalization_11_readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02'
%batch_normalization_11/ReadVariableOpÓ
'batch_normalization_11/ReadVariableOp_1ReadVariableOpCbatch_normalization_11_readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_11/MaxPool:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_11/FusedBatchNormV3½
%batch_normalization_11/AssignNewValueAssignVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*l
_classb
`^loc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValueÓ
'batch_normalization_11/AssignNewValue_1AssignVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*r
_classh
fdloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_2/Const«
flatten_2/ReshapeReshape+batch_normalization_11/FusedBatchNormV3:y:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flatten_2/Reshape­
dense_4/MatMul/ReadVariableOpReadVariableOp,dense_4_matmul_readvariableop_dense_4_kernel* 
_output_shapes
:
 *
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul©
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout_2/dropout/Const¤
dropout_2/dropout/MulMuldense_4/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mulz
dropout_2/dropout/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÓ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2"
 dropout_2/dropout/GreaterEqual/yç
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Cast£
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul_1¬
dense_5/MatMul/ReadVariableOpReadVariableOp,dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes
:	f*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
dense_5/MatMul¨
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:f*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
dense_5/BiasAdd
activation_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
activation_5/Softmax¶
IdentityIdentityactivation_5/Softmax:softmax:0&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ð
ä
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12574

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò	

5__inference_batch_normalization_8_layer_call_fn_12520

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_109662
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_11157

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ä
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11403

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00 :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
àd
þ
G__inference_sequential_2_layer_call_and_return_conditional_losses_11990

inputs
conv2d_10_conv2d_10_kernel
conv2d_10_conv2d_10_bias
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias5
1batch_normalization_9_batch_normalization_9_gamma4
0batch_normalization_9_batch_normalization_9_beta;
7batch_normalization_9_batch_normalization_9_moving_mean?
;batch_normalization_9_batch_normalization_9_moving_variance
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias7
3batch_normalization_10_batch_normalization_10_gamma6
2batch_normalization_10_batch_normalization_10_beta=
9batch_normalization_10_batch_normalization_10_moving_meanA
=batch_normalization_10_batch_normalization_10_moving_variance
conv2d_14_conv2d_14_kernel
conv2d_14_conv2d_14_bias7
3batch_normalization_11_batch_normalization_11_gamma6
2batch_normalization_11_batch_normalization_11_beta=
9batch_normalization_11_batch_normalization_11_moving_meanA
=batch_normalization_11_batch_normalization_11_moving_variance
dense_4_dense_4_kernel
dense_4_dense_4_bias
dense_5_dense_5_kernel
dense_5_dense_5_bias
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallµ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_113152#
!conv2d_10/StatefulPartitionedCallÙ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_113372#
!conv2d_11/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_113542
activation_4/PartitionedCall
max_pooling2d_8/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_108782!
max_pooling2d_8/PartitionedCall¤
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_114032/
-batch_normalization_8/StatefulPartitionedCallå
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_114382#
!conv2d_12/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_109872!
max_pooling2d_9/PartitionedCall¤
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:01batch_normalization_9_batch_normalization_9_gamma0batch_normalization_9_batch_normalization_9_beta7batch_normalization_9_batch_normalization_9_moving_mean;batch_normalization_9_batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_114912/
-batch_normalization_9/StatefulPartitionedCallæ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_115262#
!conv2d_13/StatefulPartitionedCall
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110962"
 max_pooling2d_10/PartitionedCall±
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1157920
.batch_normalization_10/StatefulPartitionedCallç
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_116142#
!conv2d_14/StatefulPartitionedCall
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_112052"
 max_pooling2d_11/PartitionedCall±
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1166720
.batch_normalization_11/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_116972
flatten_2/PartitionedCall¼
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_117152!
dense_4/StatefulPartitionedCallü
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_117442
dropout_2/PartitionedCall»
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_117672!
dense_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_117842
activation_5/PartitionedCall³
IdentityIdentity%activation_5/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs


)__inference_conv2d_11_layer_call_fn_12402

inputs
conv2d_11_kernel
conv2d_11_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_kernelconv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_113372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿbb ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
z

G__inference_sequential_2_layer_call_and_return_conditional_losses_12297

inputs4
0conv2d_10_conv2d_readvariableop_conv2d_10_kernel3
/conv2d_10_biasadd_readvariableop_conv2d_10_bias4
0conv2d_11_conv2d_readvariableop_conv2d_11_kernel3
/conv2d_11_biasadd_readvariableop_conv2d_11_biasD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance4
0conv2d_12_conv2d_readvariableop_conv2d_12_kernel3
/conv2d_12_biasadd_readvariableop_conv2d_12_biasD
@batch_normalization_9_readvariableop_batch_normalization_9_gammaE
Abatch_normalization_9_readvariableop_1_batch_normalization_9_beta[
Wbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meana
]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance4
0conv2d_13_conv2d_readvariableop_conv2d_13_kernel3
/conv2d_13_biasadd_readvariableop_conv2d_13_biasF
Bbatch_normalization_10_readvariableop_batch_normalization_10_gammaG
Cbatch_normalization_10_readvariableop_1_batch_normalization_10_beta]
Ybatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanc
_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance4
0conv2d_14_conv2d_readvariableop_conv2d_14_kernel3
/conv2d_14_biasadd_readvariableop_conv2d_14_biasF
Bbatch_normalization_11_readvariableop_batch_normalization_11_gammaG
Cbatch_normalization_11_readvariableop_1_batch_normalization_11_beta]
Ybatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanc
_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance0
,dense_4_matmul_readvariableop_dense_4_kernel/
+dense_4_biasadd_readvariableop_dense_4_bias0
,dense_5_matmul_readvariableop_dense_5_kernel/
+dense_5_biasadd_readvariableop_dense_5_bias
identity»
conv2d_10/Conv2D/ReadVariableOpReadVariableOp0conv2d_10_conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOpÂ
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
paddingVALID*
strides
2
conv2d_10/Conv2D°
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/conv2d_10_biasadd_readvariableop_conv2d_10_bias*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp°
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
conv2d_10/Relu»
conv2d_11/Conv2D/ReadVariableOpReadVariableOp0conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:  *
dtype02!
conv2d_11/Conv2D/ReadVariableOpØ
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingVALID*
strides
2
conv2d_11/Conv2D°
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp/conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp°
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
conv2d_11/BiasAdd
activation_4/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
activation_4/ReluË
max_pooling2d_8/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPoolÉ
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_8/ReadVariableOpÎ
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02(
&batch_normalization_8/ReadVariableOp_1
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1è
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_8/MaxPool:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3»
conv2d_12/Conv2D/ReadVariableOpReadVariableOp0conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: @*
dtype02!
conv2d_12/Conv2D/ReadVariableOpæ
conv2d_12/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
conv2d_12/Conv2D°
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp/conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp°
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_12/ReluÈ
max_pooling2d_9/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPoolÉ
$batch_normalization_9/ReadVariableOpReadVariableOp@batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_9/ReadVariableOpÎ
&batch_normalization_9/ReadVariableOp_1ReadVariableOpAbatch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_9/ReadVariableOp_1
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1è
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_9/MaxPool:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3¼
conv2d_13/Conv2D/ReadVariableOpReadVariableOp0conv2d_13_conv2d_readvariableop_conv2d_13_kernel*'
_output_shapes
:@*
dtype02!
conv2d_13/Conv2D/ReadVariableOpç
conv2d_13/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_13/Conv2D±
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp/conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes	
:*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp±
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_13/ReluË
max_pooling2d_10/MaxPoolMaxPoolconv2d_13/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPoolÎ
%batch_normalization_10/ReadVariableOpReadVariableOpBbatch_normalization_10_readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02'
%batch_normalization_10/ReadVariableOpÓ
'batch_normalization_10/ReadVariableOp_1ReadVariableOpCbatch_normalization_10_readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02)
'batch_normalization_10/ReadVariableOp_1
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype028
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02:
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ô
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_10/MaxPool:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_10/FusedBatchNormV3½
conv2d_14/Conv2D/ReadVariableOpReadVariableOp0conv2d_14_conv2d_readvariableop_conv2d_14_kernel*(
_output_shapes
:*
dtype02!
conv2d_14/Conv2D/ReadVariableOpè
conv2d_14/Conv2DConv2D+batch_normalization_10/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_14/Conv2D±
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp/conv2d_14_biasadd_readvariableop_conv2d_14_bias*
_output_shapes	
:*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp±
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_14/ReluË
max_pooling2d_11/MaxPoolMaxPoolconv2d_14/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPoolÎ
%batch_normalization_11/ReadVariableOpReadVariableOpBbatch_normalization_11_readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02'
%batch_normalization_11/ReadVariableOpÓ
'batch_normalization_11/ReadVariableOp_1ReadVariableOpCbatch_normalization_11_readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpYbatch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ô
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_11/MaxPool:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_2/Const«
flatten_2/ReshapeReshape+batch_normalization_11/FusedBatchNormV3:y:0flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
flatten_2/Reshape­
dense_4/MatMul/ReadVariableOpReadVariableOp,dense_4_matmul_readvariableop_dense_4_kernel* 
_output_shapes
:
 *
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul©
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAdd
dropout_2/IdentityIdentitydense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Identity¬
dense_5/MatMul/ReadVariableOpReadVariableOp,dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes
:	f*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldropout_2/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
dense_5/MatMul¨
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:f*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
dense_5/BiasAdd
activation_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
activation_5/Softmaxr
IdentityIdentityactivation_5/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd:::::::::::::::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Õ
c
G__inference_activation_4_layer_call_and_return_conditional_losses_12407

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 
 
_user_specified_nameinputs
¦	
º
D__inference_conv2d_12_layer_call_and_return_conditional_losses_11438

inputs*
&conv2d_readvariableop_conv2d_12_kernel)
%biasadd_readvariableop_conv2d_12_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
¯	
º
D__inference_conv2d_14_layer_call_and_return_conditional_losses_12783

inputs*
&conv2d_readvariableop_conv2d_14_kernel)
%biasadd_readvariableop_conv2d_14_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_14_kernel*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_14_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ

:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
¤

P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12610

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

E
)__inference_dropout_2_layer_call_fn_12953

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_117442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ûd

G__inference_sequential_2_layer_call_and_return_conditional_losses_11846
conv2d_10_input
conv2d_10_conv2d_10_kernel
conv2d_10_conv2d_10_bias
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias5
1batch_normalization_9_batch_normalization_9_gamma4
0batch_normalization_9_batch_normalization_9_beta;
7batch_normalization_9_batch_normalization_9_moving_mean?
;batch_normalization_9_batch_normalization_9_moving_variance
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias7
3batch_normalization_10_batch_normalization_10_gamma6
2batch_normalization_10_batch_normalization_10_beta=
9batch_normalization_10_batch_normalization_10_moving_meanA
=batch_normalization_10_batch_normalization_10_moving_variance
conv2d_14_conv2d_14_kernel
conv2d_14_conv2d_14_bias7
3batch_normalization_11_batch_normalization_11_gamma6
2batch_normalization_11_batch_normalization_11_beta=
9batch_normalization_11_batch_normalization_11_moving_meanA
=batch_normalization_11_batch_normalization_11_moving_variance
dense_4_dense_4_kernel
dense_4_dense_4_bias
dense_5_dense_5_kernel
dense_5_dense_5_bias
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¾
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_113152#
!conv2d_10/StatefulPartitionedCallÙ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_113372#
!conv2d_11/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_113542
activation_4/PartitionedCall
max_pooling2d_8/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_108782!
max_pooling2d_8/PartitionedCall¤
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_114032/
-batch_normalization_8/StatefulPartitionedCallå
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_114382#
!conv2d_12/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_109872!
max_pooling2d_9/PartitionedCall¤
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:01batch_normalization_9_batch_normalization_9_gamma0batch_normalization_9_batch_normalization_9_beta7batch_normalization_9_batch_normalization_9_moving_mean;batch_normalization_9_batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_114912/
-batch_normalization_9/StatefulPartitionedCallæ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_115262#
!conv2d_13/StatefulPartitionedCall
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110962"
 max_pooling2d_10/PartitionedCall±
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1157920
.batch_normalization_10/StatefulPartitionedCallç
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_116142#
!conv2d_14/StatefulPartitionedCall
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_112052"
 max_pooling2d_11/PartitionedCall±
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1166720
.batch_normalization_11/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_116972
flatten_2/PartitionedCall¼
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_117152!
dense_4/StatefulPartitionedCallü
dropout_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_117442
dropout_2/PartitionedCall»
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_117672!
dense_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_117842
activation_5/PartitionedCall³
IdentityIdentity%activation_5/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
)
_user_specified_nameconv2d_10_input
	

6__inference_batch_normalization_11_layer_call_fn_12898

inputs 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_116672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
é
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12754

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½

Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_11649

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_10870

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð	

5__inference_batch_normalization_9_layer_call_fn_12583

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_110482
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ	

6__inference_batch_normalization_10_layer_call_fn_12763

inputs 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_111572
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦	
º
D__inference_conv2d_10_layer_call_and_return_conditional_losses_12378

inputs*
&conv2d_readvariableop_conv2d_10_kernel)
%biasadd_readvariableop_conv2d_10_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_10_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_8_layer_call_fn_10881

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_108782
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
ä
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12502

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
å
é
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12826

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
º
D__inference_conv2d_11_layer_call_and_return_conditional_losses_11337

inputs*
&conv2d_readvariableop_conv2d_11_kernel)
%biasadd_readvariableop_conv2d_11_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿbb :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 
 
_user_specified_nameinputs
ø
à
,__inference_sequential_2_layer_call_fn_12023
conv2d_10_input
conv2d_10_kernel
conv2d_10_bias
conv2d_11_kernel
conv2d_11_bias
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
conv2d_12_kernel
conv2d_12_bias
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_13_kernel
conv2d_13_bias 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
conv2d_14_kernel
conv2d_14_bias 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
dense_4_kernel
dense_4_bias
dense_5_kernel
dense_5_bias
identity¢StatefulPartitionedCall¡	
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_kernelconv2d_10_biasconv2d_11_kernelconv2d_11_biasbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancedense_4_kerneldense_4_biasdense_5_kerneldense_5_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_119902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
)
_user_specified_nameconv2d_10_input

g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_11205

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

5__inference_batch_normalization_9_layer_call_fn_12646

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_114912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


)__inference_conv2d_13_layer_call_fn_12664

inputs
conv2d_13_kernel
conv2d_13_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_kernelconv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_115262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 

)__inference_conv2d_14_layer_call_fn_12790

inputs
conv2d_14_kernel
conv2d_14_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_kernelconv2d_14_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_116142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ

::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
½

Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_11561

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ

::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
¾
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_11697

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£f
«
G__inference_sequential_2_layer_call_and_return_conditional_losses_11793
conv2d_10_input
conv2d_10_conv2d_10_kernel
conv2d_10_conv2d_10_bias
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias5
1batch_normalization_9_batch_normalization_9_gamma4
0batch_normalization_9_batch_normalization_9_beta;
7batch_normalization_9_batch_normalization_9_moving_mean?
;batch_normalization_9_batch_normalization_9_moving_variance
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias7
3batch_normalization_10_batch_normalization_10_gamma6
2batch_normalization_10_batch_normalization_10_beta=
9batch_normalization_10_batch_normalization_10_moving_meanA
=batch_normalization_10_batch_normalization_10_moving_variance
conv2d_14_conv2d_14_kernel
conv2d_14_conv2d_14_bias7
3batch_normalization_11_batch_normalization_11_gamma6
2batch_normalization_11_batch_normalization_11_beta=
9batch_normalization_11_batch_normalization_11_moving_meanA
=batch_normalization_11_batch_normalization_11_moving_variance
dense_4_dense_4_kernel
dense_4_dense_4_bias
dense_5_dense_5_kernel
dense_5_dense_5_bias
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¾
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_113152#
!conv2d_10/StatefulPartitionedCallÙ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_113372#
!conv2d_11/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_113542
activation_4/PartitionedCall
max_pooling2d_8/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_108782!
max_pooling2d_8/PartitionedCall¢
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_113852/
-batch_normalization_8/StatefulPartitionedCallå
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_114382#
!conv2d_12/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_109872!
max_pooling2d_9/PartitionedCall¢
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:01batch_normalization_9_batch_normalization_9_gamma0batch_normalization_9_batch_normalization_9_beta7batch_normalization_9_batch_normalization_9_moving_mean;batch_normalization_9_batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_114732/
-batch_normalization_9/StatefulPartitionedCallæ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_115262#
!conv2d_13/StatefulPartitionedCall
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110962"
 max_pooling2d_10/PartitionedCall¯
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1156120
.batch_normalization_10/StatefulPartitionedCallç
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_116142#
!conv2d_14/StatefulPartitionedCall
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_112052"
 max_pooling2d_11/PartitionedCall¯
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1164920
.batch_normalization_11/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_116972
flatten_2/PartitionedCall¼
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_117152!
dense_4/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_117392#
!dropout_2/StatefulPartitionedCallÃ
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_117672!
dense_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_117842
activation_5/PartitionedCall×
IdentityIdentity%activation_5/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
)
_user_specified_nameconv2d_10_input
å
é
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_11293

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


)__inference_conv2d_12_layer_call_fn_12538

inputs
conv2d_12_kernel
conv2d_12_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_kernelconv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_114382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
Ý
×
,__inference_sequential_2_layer_call_fn_12367

inputs
conv2d_10_kernel
conv2d_10_bias
conv2d_11_kernel
conv2d_11_bias
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
conv2d_12_kernel
conv2d_12_bias
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_13_kernel
conv2d_13_bias 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
conv2d_14_kernel
conv2d_14_bias 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
dense_4_kernel
dense_4_bias
dense_5_kernel
dense_5_bias
identity¢StatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_kernelconv2d_10_biasconv2d_11_kernelconv2d_11_biasbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancedense_4_kerneldense_4_biasdense_5_kerneldense_5_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_119902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
	

6__inference_batch_normalization_10_layer_call_fn_12718

inputs 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_115792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ

::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs


Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12736

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_10/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_10/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½

Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12862

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3³
AssignNewValueAssignVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*U
_classK
IGloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_11/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÉ
AssignNewValue_1AssignVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*[
_classQ
OMloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_11/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
é
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_11184

inputs/
+readvariableop_batch_normalization_10_gamma0
,readvariableop_1_batch_normalization_10_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_11667

inputs/
+readvariableop_batch_normalization_11_gamma0
,readvariableop_1_batch_normalization_11_betaF
Bfusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanL
Hfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance
identity
ReadVariableOpReadVariableOp+readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp,readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Â
FusedBatchNormV3/ReadVariableOpReadVariableOpBfusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpÌ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ	

6__inference_batch_normalization_11_layer_call_fn_12835

inputs 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_112662
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


)__inference_conv2d_10_layer_call_fn_12385

inputs
conv2d_10_kernel
conv2d_10_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_kernelconv2d_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_113152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
f
¢
G__inference_sequential_2_layer_call_and_return_conditional_losses_11902

inputs
conv2d_10_conv2d_10_kernel
conv2d_10_conv2d_10_bias
conv2d_11_conv2d_11_kernel
conv2d_11_conv2d_11_bias5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
conv2d_12_conv2d_12_kernel
conv2d_12_conv2d_12_bias5
1batch_normalization_9_batch_normalization_9_gamma4
0batch_normalization_9_batch_normalization_9_beta;
7batch_normalization_9_batch_normalization_9_moving_mean?
;batch_normalization_9_batch_normalization_9_moving_variance
conv2d_13_conv2d_13_kernel
conv2d_13_conv2d_13_bias7
3batch_normalization_10_batch_normalization_10_gamma6
2batch_normalization_10_batch_normalization_10_beta=
9batch_normalization_10_batch_normalization_10_moving_meanA
=batch_normalization_10_batch_normalization_10_moving_variance
conv2d_14_conv2d_14_kernel
conv2d_14_conv2d_14_bias7
3batch_normalization_11_batch_normalization_11_gamma6
2batch_normalization_11_batch_normalization_11_beta=
9batch_normalization_11_batch_normalization_11_moving_meanA
=batch_normalization_11_batch_normalization_11_moving_variance
dense_4_dense_4_kernel
dense_4_dense_4_bias
dense_5_dense_5_kernel
dense_5_dense_5_bias
identity¢.batch_normalization_10/StatefulPartitionedCall¢.batch_normalization_11/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCallµ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_conv2d_10_kernelconv2d_10_conv2d_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_113152#
!conv2d_10/StatefulPartitionedCallÙ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_conv2d_11_kernelconv2d_11_conv2d_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_113372#
!conv2d_11/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_113542
activation_4/PartitionedCall
max_pooling2d_8/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_108782!
max_pooling2d_8/PartitionedCall¢
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_113852/
-batch_normalization_8/StatefulPartitionedCallå
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_12_conv2d_12_kernelconv2d_12_conv2d_12_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_114382#
!conv2d_12/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_109872!
max_pooling2d_9/PartitionedCall¢
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:01batch_normalization_9_batch_normalization_9_gamma0batch_normalization_9_batch_normalization_9_beta7batch_normalization_9_batch_normalization_9_moving_mean;batch_normalization_9_batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_114732/
-batch_normalization_9/StatefulPartitionedCallæ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_13_conv2d_13_kernelconv2d_13_conv2d_13_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_115262#
!conv2d_13/StatefulPartitionedCall
 max_pooling2d_10/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110962"
 max_pooling2d_10/PartitionedCall¯
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:03batch_normalization_10_batch_normalization_10_gamma2batch_normalization_10_batch_normalization_10_beta9batch_normalization_10_batch_normalization_10_moving_mean=batch_normalization_10_batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1156120
.batch_normalization_10/StatefulPartitionedCallç
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0conv2d_14_conv2d_14_kernelconv2d_14_conv2d_14_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_116142#
!conv2d_14/StatefulPartitionedCall
 max_pooling2d_11/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_112052"
 max_pooling2d_11/PartitionedCall¯
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:03batch_normalization_11_batch_normalization_11_gamma2batch_normalization_11_batch_normalization_11_beta9batch_normalization_11_batch_normalization_11_moving_mean=batch_normalization_11_batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1164920
.batch_normalization_11/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_116972
flatten_2/PartitionedCall¼
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_117152!
dense_4/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_117392#
!dropout_2/StatefulPartitionedCallÃ
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_117672!
dense_5/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_117842
activation_5/PartitionedCall×
IdentityIdentity%activation_5/PartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¬	
º
D__inference_conv2d_13_layer_call_and_return_conditional_losses_12657

inputs*
&conv2d_readvariableop_conv2d_13_kernel)
%biasadd_readvariableop_conv2d_13_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_13_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_13_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_9_layer_call_fn_10990

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_109872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ä
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12628

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¯	
º
D__inference_conv2d_14_layer_call_and_return_conditional_losses_11614

inputs*
&conv2d_readvariableop_conv2d_14_kernel)
%biasadd_readvariableop_conv2d_14_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_14_kernel*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_14_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ

:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_10_layer_call_fn_11099

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_110962
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_11473

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ä
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_11491

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð	

5__inference_batch_normalization_8_layer_call_fn_12511

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_109392
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	

5__inference_batch_normalization_9_layer_call_fn_12637

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_114732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò

'__inference_dense_5_layer_call_fn_12970

inputs
dense_5_kernel
dense_5_bias
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_117672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò	

5__inference_batch_normalization_9_layer_call_fn_12592

inputs
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_110752
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_12904

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
H
,__inference_activation_4_layer_call_fn_12412

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_113542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_10878

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä«
ö(
__inference__traced_save_13246
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop3
/savev2_training_4_adam_iter_read_readvariableop	5
1savev2_training_4_adam_beta_1_read_readvariableop5
1savev2_training_4_adam_beta_2_read_readvariableop4
0savev2_training_4_adam_decay_read_readvariableop<
8savev2_training_4_adam_learning_rate_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableopA
=savev2_training_4_adam_conv2d_10_kernel_m_read_readvariableop?
;savev2_training_4_adam_conv2d_10_bias_m_read_readvariableopA
=savev2_training_4_adam_conv2d_11_kernel_m_read_readvariableop?
;savev2_training_4_adam_conv2d_11_bias_m_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_8_gamma_m_read_readvariableopK
Gsavev2_training_4_adam_batch_normalization_8_beta_m_read_readvariableopA
=savev2_training_4_adam_conv2d_12_kernel_m_read_readvariableop?
;savev2_training_4_adam_conv2d_12_bias_m_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_9_gamma_m_read_readvariableopK
Gsavev2_training_4_adam_batch_normalization_9_beta_m_read_readvariableopA
=savev2_training_4_adam_conv2d_13_kernel_m_read_readvariableop?
;savev2_training_4_adam_conv2d_13_bias_m_read_readvariableopM
Isavev2_training_4_adam_batch_normalization_10_gamma_m_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_10_beta_m_read_readvariableopA
=savev2_training_4_adam_conv2d_14_kernel_m_read_readvariableop?
;savev2_training_4_adam_conv2d_14_bias_m_read_readvariableopM
Isavev2_training_4_adam_batch_normalization_11_gamma_m_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_11_beta_m_read_readvariableop?
;savev2_training_4_adam_dense_4_kernel_m_read_readvariableop=
9savev2_training_4_adam_dense_4_bias_m_read_readvariableop?
;savev2_training_4_adam_dense_5_kernel_m_read_readvariableop=
9savev2_training_4_adam_dense_5_bias_m_read_readvariableopA
=savev2_training_4_adam_conv2d_10_kernel_v_read_readvariableop?
;savev2_training_4_adam_conv2d_10_bias_v_read_readvariableopA
=savev2_training_4_adam_conv2d_11_kernel_v_read_readvariableop?
;savev2_training_4_adam_conv2d_11_bias_v_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_8_gamma_v_read_readvariableopK
Gsavev2_training_4_adam_batch_normalization_8_beta_v_read_readvariableopA
=savev2_training_4_adam_conv2d_12_kernel_v_read_readvariableop?
;savev2_training_4_adam_conv2d_12_bias_v_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_9_gamma_v_read_readvariableopK
Gsavev2_training_4_adam_batch_normalization_9_beta_v_read_readvariableopA
=savev2_training_4_adam_conv2d_13_kernel_v_read_readvariableop?
;savev2_training_4_adam_conv2d_13_bias_v_read_readvariableopM
Isavev2_training_4_adam_batch_normalization_10_gamma_v_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_10_beta_v_read_readvariableopA
=savev2_training_4_adam_conv2d_14_kernel_v_read_readvariableop?
;savev2_training_4_adam_conv2d_14_bias_v_read_readvariableopM
Isavev2_training_4_adam_batch_normalization_11_gamma_v_read_readvariableopL
Hsavev2_training_4_adam_batch_normalization_11_beta_v_read_readvariableop?
;savev2_training_4_adam_dense_4_kernel_v_read_readvariableop=
9savev2_training_4_adam_dense_4_bias_v_read_readvariableop?
;savev2_training_4_adam_dense_5_kernel_v_read_readvariableop=
9savev2_training_4_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_acaaf11e8aea46899846bd4f41aba984/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameâ-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*ô,
valueê,Bç,RB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¯
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*¹
value¯B¬RB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÒ'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop/savev2_training_4_adam_iter_read_readvariableop1savev2_training_4_adam_beta_1_read_readvariableop1savev2_training_4_adam_beta_2_read_readvariableop0savev2_training_4_adam_decay_read_readvariableop8savev2_training_4_adam_learning_rate_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop=savev2_training_4_adam_conv2d_10_kernel_m_read_readvariableop;savev2_training_4_adam_conv2d_10_bias_m_read_readvariableop=savev2_training_4_adam_conv2d_11_kernel_m_read_readvariableop;savev2_training_4_adam_conv2d_11_bias_m_read_readvariableopHsavev2_training_4_adam_batch_normalization_8_gamma_m_read_readvariableopGsavev2_training_4_adam_batch_normalization_8_beta_m_read_readvariableop=savev2_training_4_adam_conv2d_12_kernel_m_read_readvariableop;savev2_training_4_adam_conv2d_12_bias_m_read_readvariableopHsavev2_training_4_adam_batch_normalization_9_gamma_m_read_readvariableopGsavev2_training_4_adam_batch_normalization_9_beta_m_read_readvariableop=savev2_training_4_adam_conv2d_13_kernel_m_read_readvariableop;savev2_training_4_adam_conv2d_13_bias_m_read_readvariableopIsavev2_training_4_adam_batch_normalization_10_gamma_m_read_readvariableopHsavev2_training_4_adam_batch_normalization_10_beta_m_read_readvariableop=savev2_training_4_adam_conv2d_14_kernel_m_read_readvariableop;savev2_training_4_adam_conv2d_14_bias_m_read_readvariableopIsavev2_training_4_adam_batch_normalization_11_gamma_m_read_readvariableopHsavev2_training_4_adam_batch_normalization_11_beta_m_read_readvariableop;savev2_training_4_adam_dense_4_kernel_m_read_readvariableop9savev2_training_4_adam_dense_4_bias_m_read_readvariableop;savev2_training_4_adam_dense_5_kernel_m_read_readvariableop9savev2_training_4_adam_dense_5_bias_m_read_readvariableop=savev2_training_4_adam_conv2d_10_kernel_v_read_readvariableop;savev2_training_4_adam_conv2d_10_bias_v_read_readvariableop=savev2_training_4_adam_conv2d_11_kernel_v_read_readvariableop;savev2_training_4_adam_conv2d_11_bias_v_read_readvariableopHsavev2_training_4_adam_batch_normalization_8_gamma_v_read_readvariableopGsavev2_training_4_adam_batch_normalization_8_beta_v_read_readvariableop=savev2_training_4_adam_conv2d_12_kernel_v_read_readvariableop;savev2_training_4_adam_conv2d_12_bias_v_read_readvariableopHsavev2_training_4_adam_batch_normalization_9_gamma_v_read_readvariableopGsavev2_training_4_adam_batch_normalization_9_beta_v_read_readvariableop=savev2_training_4_adam_conv2d_13_kernel_v_read_readvariableop;savev2_training_4_adam_conv2d_13_bias_v_read_readvariableopIsavev2_training_4_adam_batch_normalization_10_gamma_v_read_readvariableopHsavev2_training_4_adam_batch_normalization_10_beta_v_read_readvariableop=savev2_training_4_adam_conv2d_14_kernel_v_read_readvariableop;savev2_training_4_adam_conv2d_14_bias_v_read_readvariableopIsavev2_training_4_adam_batch_normalization_11_gamma_v_read_readvariableopHsavev2_training_4_adam_batch_normalization_11_beta_v_read_readvariableop;savev2_training_4_adam_dense_4_kernel_v_read_readvariableop9savev2_training_4_adam_dense_4_bias_v_read_readvariableop;savev2_training_4_adam_dense_5_kernel_v_read_readvariableop9savev2_training_4_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *`
dtypesV
T2R	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ú
_input_shapesÈ
Å: : : :  : : : : : : @:@:@:@:@:@:@::::::::::::
 ::	f:f: : : : : : : : : :  : : : : @:@:@:@:@::::::::
 ::	f:f: : :  : : : : @:@:@:@:@::::::::
 ::	f:f: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
 :!

_output_shapes	
::%!

_output_shapes
:	f: 

_output_shapes
:f:

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
:  : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:-0)
'
_output_shapes
:@:!1

_output_shapes	
::!2

_output_shapes	
::!3

_output_shapes	
::.4*
(
_output_shapes
::!5

_output_shapes	
::!6

_output_shapes	
::!7

_output_shapes	
::&8"
 
_output_shapes
:
 :!9

_output_shapes	
::%:!

_output_shapes
:	f: ;

_output_shapes
:f:,<(
&
_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
:  : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: :,B(
&
_output_shapes
: @: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:-F)
'
_output_shapes
:@:!G

_output_shapes	
::!H

_output_shapes	
::!I

_output_shapes	
::.J*
(
_output_shapes
::!K

_output_shapes	
::!L

_output_shapes	
::!M

_output_shapes	
::&N"
 
_output_shapes
:
 :!O

_output_shapes	
::%P!

_output_shapes
:	f: Q

_output_shapes
:f:R

_output_shapes
: 
ì

P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12556

inputs.
*readvariableop_batch_normalization_9_gamma/
+readvariableop_1_batch_normalization_9_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_9_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_9/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_9/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

H
,__inference_activation_5_layer_call_fn_12980

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_117842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
à	

6__inference_batch_normalization_11_layer_call_fn_12844

inputs 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_112932
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
à
,__inference_sequential_2_layer_call_fn_11935
conv2d_10_input
conv2d_10_kernel
conv2d_10_bias
conv2d_11_kernel
conv2d_11_bias
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
conv2d_12_kernel
conv2d_12_bias
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_13_kernel
conv2d_13_bias 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
conv2d_14_kernel
conv2d_14_bias 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
dense_4_kernel
dense_4_bias
dense_5_kernel
dense_5_bias
identity¢StatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_kernelconv2d_10_biasconv2d_11_kernelconv2d_11_biasbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancedense_4_kerneldense_4_biasdense_5_kerneldense_5_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_119022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
)
_user_specified_nameconv2d_10_input

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_11739

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_11385

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity¢AssignNewValue¢AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueÇ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00 ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_11088

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_11744

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
c
G__inference_activation_5_layer_call_and_return_conditional_losses_11784

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
½
c
G__inference_activation_5_layer_call_and_return_conditional_losses_12975

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
â
´
B__inference_dense_5_layer_call_and_return_conditional_losses_11767

inputs(
$matmul_readvariableop_dense_5_kernel'
#biasadd_readvariableop_dense_5_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_5_kernel*
_output_shapes
:	f*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:f*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦	
º
D__inference_conv2d_12_layer_call_and_return_conditional_losses_12531

inputs*
&conv2d_readvariableop_conv2d_12_kernel)
%biasadd_readvariableop_conv2d_12_bias
identity
Conv2D/ReadVariableOpReadVariableOp&conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
ê
ï
 __inference__wrapped_model_10864
conv2d_10_inputA
=sequential_2_conv2d_10_conv2d_readvariableop_conv2d_10_kernel@
<sequential_2_conv2d_10_biasadd_readvariableop_conv2d_10_biasA
=sequential_2_conv2d_11_conv2d_readvariableop_conv2d_11_kernel@
<sequential_2_conv2d_11_biasadd_readvariableop_conv2d_11_biasQ
Msequential_2_batch_normalization_8_readvariableop_batch_normalization_8_gammaR
Nsequential_2_batch_normalization_8_readvariableop_1_batch_normalization_8_betah
dsequential_2_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meann
jsequential_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_varianceA
=sequential_2_conv2d_12_conv2d_readvariableop_conv2d_12_kernel@
<sequential_2_conv2d_12_biasadd_readvariableop_conv2d_12_biasQ
Msequential_2_batch_normalization_9_readvariableop_batch_normalization_9_gammaR
Nsequential_2_batch_normalization_9_readvariableop_1_batch_normalization_9_betah
dsequential_2_batch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_meann
jsequential_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_varianceA
=sequential_2_conv2d_13_conv2d_readvariableop_conv2d_13_kernel@
<sequential_2_conv2d_13_biasadd_readvariableop_conv2d_13_biasS
Osequential_2_batch_normalization_10_readvariableop_batch_normalization_10_gammaT
Psequential_2_batch_normalization_10_readvariableop_1_batch_normalization_10_betaj
fsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_meanp
lsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_varianceA
=sequential_2_conv2d_14_conv2d_readvariableop_conv2d_14_kernel@
<sequential_2_conv2d_14_biasadd_readvariableop_conv2d_14_biasS
Osequential_2_batch_normalization_11_readvariableop_batch_normalization_11_gammaT
Psequential_2_batch_normalization_11_readvariableop_1_batch_normalization_11_betaj
fsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_meanp
lsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance=
9sequential_2_dense_4_matmul_readvariableop_dense_4_kernel<
8sequential_2_dense_4_biasadd_readvariableop_dense_4_bias=
9sequential_2_dense_5_matmul_readvariableop_dense_5_kernel<
8sequential_2_dense_5_biasadd_readvariableop_dense_5_bias
identityâ
,sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp=sequential_2_conv2d_10_conv2d_readvariableop_conv2d_10_kernel*&
_output_shapes
: *
dtype02.
,sequential_2/conv2d_10/Conv2D/ReadVariableOpò
sequential_2/conv2d_10/Conv2DConv2Dconv2d_10_input4sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb *
paddingVALID*
strides
2
sequential_2/conv2d_10/Conv2D×
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp<sequential_2_conv2d_10_biasadd_readvariableop_conv2d_10_bias*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpä
sequential_2/conv2d_10/BiasAddBiasAdd&sequential_2/conv2d_10/Conv2D:output:05sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2 
sequential_2/conv2d_10/BiasAdd¥
sequential_2/conv2d_10/ReluRelu'sequential_2/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿbb 2
sequential_2/conv2d_10/Reluâ
,sequential_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp=sequential_2_conv2d_11_conv2d_readvariableop_conv2d_11_kernel*&
_output_shapes
:  *
dtype02.
,sequential_2/conv2d_11/Conv2D/ReadVariableOp
sequential_2/conv2d_11/Conv2DConv2D)sequential_2/conv2d_10/Relu:activations:04sequential_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingVALID*
strides
2
sequential_2/conv2d_11/Conv2D×
-sequential_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp<sequential_2_conv2d_11_biasadd_readvariableop_conv2d_11_bias*
_output_shapes
: *
dtype02/
-sequential_2/conv2d_11/BiasAdd/ReadVariableOpä
sequential_2/conv2d_11/BiasAddBiasAdd&sequential_2/conv2d_11/Conv2D:output:05sequential_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2 
sequential_2/conv2d_11/BiasAdd«
sequential_2/activation_4/ReluRelu'sequential_2/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2 
sequential_2/activation_4/Reluò
$sequential_2/max_pooling2d_8/MaxPoolMaxPool,sequential_2/activation_4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolð
1sequential_2/batch_normalization_8/ReadVariableOpReadVariableOpMsequential_2_batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype023
1sequential_2/batch_normalization_8/ReadVariableOpõ
3sequential_2/batch_normalization_8/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype025
3sequential_2/batch_normalization_8/ReadVariableOp_1©
Bsequential_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpdsequential_2_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02D
Bsequential_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp³
Dsequential_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjsequential_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02F
Dsequential_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ã
3sequential_2/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3-sequential_2/max_pooling2d_8/MaxPool:output:09sequential_2/batch_normalization_8/ReadVariableOp:value:0;sequential_2/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 25
3sequential_2/batch_normalization_8/FusedBatchNormV3â
,sequential_2/conv2d_12/Conv2D/ReadVariableOpReadVariableOp=sequential_2_conv2d_12_conv2d_readvariableop_conv2d_12_kernel*&
_output_shapes
: @*
dtype02.
,sequential_2/conv2d_12/Conv2D/ReadVariableOp
sequential_2/conv2d_12/Conv2DConv2D7sequential_2/batch_normalization_8/FusedBatchNormV3:y:04sequential_2/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
sequential_2/conv2d_12/Conv2D×
-sequential_2/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp<sequential_2_conv2d_12_biasadd_readvariableop_conv2d_12_bias*
_output_shapes
:@*
dtype02/
-sequential_2/conv2d_12/BiasAdd/ReadVariableOpä
sequential_2/conv2d_12/BiasAddBiasAdd&sequential_2/conv2d_12/Conv2D:output:05sequential_2/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2 
sequential_2/conv2d_12/BiasAdd¥
sequential_2/conv2d_12/ReluRelu'sequential_2/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
sequential_2/conv2d_12/Reluï
$sequential_2/max_pooling2d_9/MaxPoolMaxPool)sequential_2/conv2d_12/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_9/MaxPoolð
1sequential_2/batch_normalization_9/ReadVariableOpReadVariableOpMsequential_2_batch_normalization_9_readvariableop_batch_normalization_9_gamma*
_output_shapes
:@*
dtype023
1sequential_2/batch_normalization_9/ReadVariableOpõ
3sequential_2/batch_normalization_9/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_9_readvariableop_1_batch_normalization_9_beta*
_output_shapes
:@*
dtype025
3sequential_2/batch_normalization_9/ReadVariableOp_1©
Bsequential_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpdsequential_2_batch_normalization_9_fusedbatchnormv3_readvariableop_batch_normalization_9_moving_mean*
_output_shapes
:@*
dtype02D
Bsequential_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp³
Dsequential_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjsequential_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_batch_normalization_9_moving_variance*
_output_shapes
:@*
dtype02F
Dsequential_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Ã
3sequential_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3-sequential_2/max_pooling2d_9/MaxPool:output:09sequential_2/batch_normalization_9/ReadVariableOp:value:0;sequential_2/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3sequential_2/batch_normalization_9/FusedBatchNormV3ã
,sequential_2/conv2d_13/Conv2D/ReadVariableOpReadVariableOp=sequential_2_conv2d_13_conv2d_readvariableop_conv2d_13_kernel*'
_output_shapes
:@*
dtype02.
,sequential_2/conv2d_13/Conv2D/ReadVariableOp
sequential_2/conv2d_13/Conv2DConv2D7sequential_2/batch_normalization_9/FusedBatchNormV3:y:04sequential_2/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
sequential_2/conv2d_13/Conv2DØ
-sequential_2/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp<sequential_2_conv2d_13_biasadd_readvariableop_conv2d_13_bias*
_output_shapes	
:*
dtype02/
-sequential_2/conv2d_13/BiasAdd/ReadVariableOpå
sequential_2/conv2d_13/BiasAddBiasAdd&sequential_2/conv2d_13/Conv2D:output:05sequential_2/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_2/conv2d_13/BiasAdd¦
sequential_2/conv2d_13/ReluRelu'sequential_2/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/conv2d_13/Reluò
%sequential_2/max_pooling2d_10/MaxPoolMaxPool)sequential_2/conv2d_13/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingVALID*
strides
2'
%sequential_2/max_pooling2d_10/MaxPoolõ
2sequential_2/batch_normalization_10/ReadVariableOpReadVariableOpOsequential_2_batch_normalization_10_readvariableop_batch_normalization_10_gamma*
_output_shapes	
:*
dtype024
2sequential_2/batch_normalization_10/ReadVariableOpú
4sequential_2/batch_normalization_10/ReadVariableOp_1ReadVariableOpPsequential_2_batch_normalization_10_readvariableop_1_batch_normalization_10_beta*
_output_shapes	
:*
dtype026
4sequential_2/batch_normalization_10/ReadVariableOp_1®
Csequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpfsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_batch_normalization_10_moving_mean*
_output_shapes	
:*
dtype02E
Csequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp¸
Esequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_batch_normalization_10_moving_variance*
_output_shapes	
:*
dtype02G
Esequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Ï
4sequential_2/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3.sequential_2/max_pooling2d_10/MaxPool:output:0:sequential_2/batch_normalization_10/ReadVariableOp:value:0<sequential_2/batch_normalization_10/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ

:::::*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_10/FusedBatchNormV3ä
,sequential_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp=sequential_2_conv2d_14_conv2d_readvariableop_conv2d_14_kernel*(
_output_shapes
:*
dtype02.
,sequential_2/conv2d_14/Conv2D/ReadVariableOp
sequential_2/conv2d_14/Conv2DConv2D8sequential_2/batch_normalization_10/FusedBatchNormV3:y:04sequential_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
sequential_2/conv2d_14/Conv2DØ
-sequential_2/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp<sequential_2_conv2d_14_biasadd_readvariableop_conv2d_14_bias*
_output_shapes	
:*
dtype02/
-sequential_2/conv2d_14/BiasAdd/ReadVariableOpå
sequential_2/conv2d_14/BiasAddBiasAdd&sequential_2/conv2d_14/Conv2D:output:05sequential_2/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_2/conv2d_14/BiasAdd¦
sequential_2/conv2d_14/ReluRelu'sequential_2/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/conv2d_14/Reluò
%sequential_2/max_pooling2d_11/MaxPoolMaxPool)sequential_2/conv2d_14/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2'
%sequential_2/max_pooling2d_11/MaxPoolõ
2sequential_2/batch_normalization_11/ReadVariableOpReadVariableOpOsequential_2_batch_normalization_11_readvariableop_batch_normalization_11_gamma*
_output_shapes	
:*
dtype024
2sequential_2/batch_normalization_11/ReadVariableOpú
4sequential_2/batch_normalization_11/ReadVariableOp_1ReadVariableOpPsequential_2_batch_normalization_11_readvariableop_1_batch_normalization_11_beta*
_output_shapes	
:*
dtype026
4sequential_2/batch_normalization_11/ReadVariableOp_1®
Csequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpfsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_batch_normalization_11_moving_mean*
_output_shapes	
:*
dtype02E
Csequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp¸
Esequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOplsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_batch_normalization_11_moving_variance*
_output_shapes	
:*
dtype02G
Esequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ï
4sequential_2/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3.sequential_2/max_pooling2d_11/MaxPool:output:0:sequential_2/batch_normalization_11/ReadVariableOp:value:0<sequential_2/batch_normalization_11/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_11/FusedBatchNormV3
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_2/flatten_2/Constß
sequential_2/flatten_2/ReshapeReshape8sequential_2/batch_normalization_11/FusedBatchNormV3:y:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_2/flatten_2/ReshapeÔ
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp9sequential_2_dense_4_matmul_readvariableop_dense_4_kernel* 
_output_shapes
:
 *
dtype02,
*sequential_2/dense_4/MatMul/ReadVariableOpÔ
sequential_2/dense_4/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_4/MatMulÐ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp8sequential_2_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes	
:*
dtype02-
+sequential_2/dense_4/BiasAdd/ReadVariableOpÖ
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_2/dense_4/BiasAdd¨
sequential_2/dropout_2/IdentityIdentity%sequential_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_2/dropout_2/IdentityÓ
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp9sequential_2_dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes
:	f*
dtype02,
*sequential_2/dense_5/MatMul/ReadVariableOpÔ
sequential_2/dense_5/MatMulMatMul(sequential_2/dropout_2/Identity:output:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
sequential_2/dense_5/MatMulÏ
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp8sequential_2_dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:f*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOpÕ
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2
sequential_2/dense_5/BiasAddª
!sequential_2/activation_5/SoftmaxSoftmax%sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2#
!sequential_2/activation_5/Softmax
IdentityIdentity+sequential_2/activation_5/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd:::::::::::::::::::::::::::::::` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
)
_user_specified_nameconv2d_10_input
È
×
#__inference_signature_wrapper_12060
conv2d_10_input
conv2d_10_kernel
conv2d_10_bias
conv2d_11_kernel
conv2d_11_bias
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
conv2d_12_kernel
conv2d_12_bias
batch_normalization_9_gamma
batch_normalization_9_beta%
!batch_normalization_9_moving_mean)
%batch_normalization_9_moving_variance
conv2d_13_kernel
conv2d_13_bias 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
conv2d_14_kernel
conv2d_14_bias 
batch_normalization_11_gamma
batch_normalization_11_beta&
"batch_normalization_11_moving_mean*
&batch_normalization_11_moving_variance
dense_4_kernel
dense_4_bias
dense_5_kernel
dense_5_bias
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_kernelconv2d_10_biasconv2d_11_kernelconv2d_11_biasbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_varianceconv2d_12_kernelconv2d_12_biasbatch_normalization_9_gammabatch_normalization_9_beta!batch_normalization_9_moving_mean%batch_normalization_9_moving_varianceconv2d_13_kernelconv2d_13_biasbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_varianceconv2d_14_kernelconv2d_14_biasbatch_normalization_11_gammabatch_normalization_11_beta"batch_normalization_11_moving_mean&batch_normalization_11_moving_variancedense_4_kerneldense_4_biasdense_5_kerneldense_5_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_108642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf2

Identity"
identityIdentity:output:0*¨
_input_shapes
:ÿÿÿÿÿÿÿÿÿdd::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
)
_user_specified_nameconv2d_10_input
¥
b
)__inference_dropout_2_layer_call_fn_12948

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_117392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
êá
ú1
!__inference__traced_restore_13499
file_prefix%
!assignvariableop_conv2d_10_kernel%
!assignvariableop_1_conv2d_10_bias'
#assignvariableop_2_conv2d_11_kernel%
!assignvariableop_3_conv2d_11_bias2
.assignvariableop_4_batch_normalization_8_gamma1
-assignvariableop_5_batch_normalization_8_beta8
4assignvariableop_6_batch_normalization_8_moving_mean<
8assignvariableop_7_batch_normalization_8_moving_variance'
#assignvariableop_8_conv2d_12_kernel%
!assignvariableop_9_conv2d_12_bias3
/assignvariableop_10_batch_normalization_9_gamma2
.assignvariableop_11_batch_normalization_9_beta9
5assignvariableop_12_batch_normalization_9_moving_mean=
9assignvariableop_13_batch_normalization_9_moving_variance(
$assignvariableop_14_conv2d_13_kernel&
"assignvariableop_15_conv2d_13_bias4
0assignvariableop_16_batch_normalization_10_gamma3
/assignvariableop_17_batch_normalization_10_beta:
6assignvariableop_18_batch_normalization_10_moving_mean>
:assignvariableop_19_batch_normalization_10_moving_variance(
$assignvariableop_20_conv2d_14_kernel&
"assignvariableop_21_conv2d_14_bias4
0assignvariableop_22_batch_normalization_11_gamma3
/assignvariableop_23_batch_normalization_11_beta:
6assignvariableop_24_batch_normalization_11_moving_mean>
:assignvariableop_25_batch_normalization_11_moving_variance&
"assignvariableop_26_dense_4_kernel$
 assignvariableop_27_dense_4_bias&
"assignvariableop_28_dense_5_kernel$
 assignvariableop_29_dense_5_bias,
(assignvariableop_30_training_4_adam_iter.
*assignvariableop_31_training_4_adam_beta_1.
*assignvariableop_32_training_4_adam_beta_2-
)assignvariableop_33_training_4_adam_decay5
1assignvariableop_34_training_4_adam_learning_rate
assignvariableop_35_total_4
assignvariableop_36_count_4:
6assignvariableop_37_training_4_adam_conv2d_10_kernel_m8
4assignvariableop_38_training_4_adam_conv2d_10_bias_m:
6assignvariableop_39_training_4_adam_conv2d_11_kernel_m8
4assignvariableop_40_training_4_adam_conv2d_11_bias_mE
Aassignvariableop_41_training_4_adam_batch_normalization_8_gamma_mD
@assignvariableop_42_training_4_adam_batch_normalization_8_beta_m:
6assignvariableop_43_training_4_adam_conv2d_12_kernel_m8
4assignvariableop_44_training_4_adam_conv2d_12_bias_mE
Aassignvariableop_45_training_4_adam_batch_normalization_9_gamma_mD
@assignvariableop_46_training_4_adam_batch_normalization_9_beta_m:
6assignvariableop_47_training_4_adam_conv2d_13_kernel_m8
4assignvariableop_48_training_4_adam_conv2d_13_bias_mF
Bassignvariableop_49_training_4_adam_batch_normalization_10_gamma_mE
Aassignvariableop_50_training_4_adam_batch_normalization_10_beta_m:
6assignvariableop_51_training_4_adam_conv2d_14_kernel_m8
4assignvariableop_52_training_4_adam_conv2d_14_bias_mF
Bassignvariableop_53_training_4_adam_batch_normalization_11_gamma_mE
Aassignvariableop_54_training_4_adam_batch_normalization_11_beta_m8
4assignvariableop_55_training_4_adam_dense_4_kernel_m6
2assignvariableop_56_training_4_adam_dense_4_bias_m8
4assignvariableop_57_training_4_adam_dense_5_kernel_m6
2assignvariableop_58_training_4_adam_dense_5_bias_m:
6assignvariableop_59_training_4_adam_conv2d_10_kernel_v8
4assignvariableop_60_training_4_adam_conv2d_10_bias_v:
6assignvariableop_61_training_4_adam_conv2d_11_kernel_v8
4assignvariableop_62_training_4_adam_conv2d_11_bias_vE
Aassignvariableop_63_training_4_adam_batch_normalization_8_gamma_vD
@assignvariableop_64_training_4_adam_batch_normalization_8_beta_v:
6assignvariableop_65_training_4_adam_conv2d_12_kernel_v8
4assignvariableop_66_training_4_adam_conv2d_12_bias_vE
Aassignvariableop_67_training_4_adam_batch_normalization_9_gamma_vD
@assignvariableop_68_training_4_adam_batch_normalization_9_beta_v:
6assignvariableop_69_training_4_adam_conv2d_13_kernel_v8
4assignvariableop_70_training_4_adam_conv2d_13_bias_vF
Bassignvariableop_71_training_4_adam_batch_normalization_10_gamma_vE
Aassignvariableop_72_training_4_adam_batch_normalization_10_beta_v:
6assignvariableop_73_training_4_adam_conv2d_14_kernel_v8
4assignvariableop_74_training_4_adam_conv2d_14_bias_vF
Bassignvariableop_75_training_4_adam_batch_normalization_11_gamma_vE
Aassignvariableop_76_training_4_adam_batch_normalization_11_beta_v8
4assignvariableop_77_training_4_adam_dense_4_kernel_v6
2assignvariableop_78_training_4_adam_dense_4_bias_v8
4assignvariableop_79_training_4_adam_dense_5_kernel_v6
2assignvariableop_80_training_4_adam_dense_5_bias_v
identity_82¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_9è-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*ô,
valueê,Bç,RB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesµ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*¹
value¯B¬RB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÈ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*`
dtypesV
T2R	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_8_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5²
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_8_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¹
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_8_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7½
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_8_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_9_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_9_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12½
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_9_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Á
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_9_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_13_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_13_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¸
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_10_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_10_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¾
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_10_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Â
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_10_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¬
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_14_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ª
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_14_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_11_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23·
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_11_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¾
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_11_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Â
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_11_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ª
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_4_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¨
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_4_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ª
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_5_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¨
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_5_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_training_4_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_training_4_adam_beta_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32²
AssignVariableOp_32AssignVariableOp*assignvariableop_32_training_4_adam_beta_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_training_4_adam_decayIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¹
AssignVariableOp_34AssignVariableOp1assignvariableop_34_training_4_adam_learning_rateIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35£
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_4Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36£
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_4Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¾
AssignVariableOp_37AssignVariableOp6assignvariableop_37_training_4_adam_conv2d_10_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¼
AssignVariableOp_38AssignVariableOp4assignvariableop_38_training_4_adam_conv2d_10_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¾
AssignVariableOp_39AssignVariableOp6assignvariableop_39_training_4_adam_conv2d_11_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¼
AssignVariableOp_40AssignVariableOp4assignvariableop_40_training_4_adam_conv2d_11_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41É
AssignVariableOp_41AssignVariableOpAassignvariableop_41_training_4_adam_batch_normalization_8_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42È
AssignVariableOp_42AssignVariableOp@assignvariableop_42_training_4_adam_batch_normalization_8_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¾
AssignVariableOp_43AssignVariableOp6assignvariableop_43_training_4_adam_conv2d_12_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¼
AssignVariableOp_44AssignVariableOp4assignvariableop_44_training_4_adam_conv2d_12_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45É
AssignVariableOp_45AssignVariableOpAassignvariableop_45_training_4_adam_batch_normalization_9_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46È
AssignVariableOp_46AssignVariableOp@assignvariableop_46_training_4_adam_batch_normalization_9_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¾
AssignVariableOp_47AssignVariableOp6assignvariableop_47_training_4_adam_conv2d_13_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¼
AssignVariableOp_48AssignVariableOp4assignvariableop_48_training_4_adam_conv2d_13_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ê
AssignVariableOp_49AssignVariableOpBassignvariableop_49_training_4_adam_batch_normalization_10_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50É
AssignVariableOp_50AssignVariableOpAassignvariableop_50_training_4_adam_batch_normalization_10_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¾
AssignVariableOp_51AssignVariableOp6assignvariableop_51_training_4_adam_conv2d_14_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¼
AssignVariableOp_52AssignVariableOp4assignvariableop_52_training_4_adam_conv2d_14_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ê
AssignVariableOp_53AssignVariableOpBassignvariableop_53_training_4_adam_batch_normalization_11_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54É
AssignVariableOp_54AssignVariableOpAassignvariableop_54_training_4_adam_batch_normalization_11_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¼
AssignVariableOp_55AssignVariableOp4assignvariableop_55_training_4_adam_dense_4_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56º
AssignVariableOp_56AssignVariableOp2assignvariableop_56_training_4_adam_dense_4_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¼
AssignVariableOp_57AssignVariableOp4assignvariableop_57_training_4_adam_dense_5_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58º
AssignVariableOp_58AssignVariableOp2assignvariableop_58_training_4_adam_dense_5_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¾
AssignVariableOp_59AssignVariableOp6assignvariableop_59_training_4_adam_conv2d_10_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¼
AssignVariableOp_60AssignVariableOp4assignvariableop_60_training_4_adam_conv2d_10_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¾
AssignVariableOp_61AssignVariableOp6assignvariableop_61_training_4_adam_conv2d_11_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62¼
AssignVariableOp_62AssignVariableOp4assignvariableop_62_training_4_adam_conv2d_11_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63É
AssignVariableOp_63AssignVariableOpAassignvariableop_63_training_4_adam_batch_normalization_8_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64È
AssignVariableOp_64AssignVariableOp@assignvariableop_64_training_4_adam_batch_normalization_8_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¾
AssignVariableOp_65AssignVariableOp6assignvariableop_65_training_4_adam_conv2d_12_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66¼
AssignVariableOp_66AssignVariableOp4assignvariableop_66_training_4_adam_conv2d_12_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67É
AssignVariableOp_67AssignVariableOpAassignvariableop_67_training_4_adam_batch_normalization_9_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68È
AssignVariableOp_68AssignVariableOp@assignvariableop_68_training_4_adam_batch_normalization_9_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¾
AssignVariableOp_69AssignVariableOp6assignvariableop_69_training_4_adam_conv2d_13_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70¼
AssignVariableOp_70AssignVariableOp4assignvariableop_70_training_4_adam_conv2d_13_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ê
AssignVariableOp_71AssignVariableOpBassignvariableop_71_training_4_adam_batch_normalization_10_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72É
AssignVariableOp_72AssignVariableOpAassignvariableop_72_training_4_adam_batch_normalization_10_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73¾
AssignVariableOp_73AssignVariableOp6assignvariableop_73_training_4_adam_conv2d_14_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¼
AssignVariableOp_74AssignVariableOp4assignvariableop_74_training_4_adam_conv2d_14_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ê
AssignVariableOp_75AssignVariableOpBassignvariableop_75_training_4_adam_batch_normalization_11_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76É
AssignVariableOp_76AssignVariableOpAassignvariableop_76_training_4_adam_batch_normalization_11_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77¼
AssignVariableOp_77AssignVariableOp4assignvariableop_77_training_4_adam_dense_4_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78º
AssignVariableOp_78AssignVariableOp2assignvariableop_78_training_4_adam_dense_4_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79¼
AssignVariableOp_79AssignVariableOp4assignvariableop_79_training_4_adam_dense_5_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80º
AssignVariableOp_80AssignVariableOp2assignvariableop_80_training_4_adam_dense_5_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_809
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÔ
Identity_81Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_81Ç
Identity_82IdentityIdentity_81:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_82"#
identity_82Identity_82:output:0*Û
_input_shapesÉ
Æ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	

5__inference_batch_normalization_8_layer_call_fn_12466

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_114032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00 ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs

ä
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12448

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1À
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpÊ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00 :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
à	

6__inference_batch_normalization_10_layer_call_fn_12772

inputs 
batch_normalization_10_gamma
batch_normalization_10_beta&
"batch_normalization_10_moving_mean*
&batch_normalization_10_moving_variance
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_10_gammabatch_normalization_10_beta"batch_normalization_10_moving_mean&batch_normalization_10_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_111842
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
S
conv2d_10_input@
!serving_default_conv2d_10_input:0ÿÿÿÿÿÿÿÿÿdd@
activation_50
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿftensorflow/serving/predict:¬ô
­
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
£_default_save_signature
¤__call__
+¥&call_and_return_all_conditional_losses"¿
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 102, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 102, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ª


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"	
_tf_keras_layeré{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}}
¨	

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layerç{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
×
&	variables
'regularization_losses
(trainable_variables
)	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}

*	variables
+regularization_losses
,trainable_variables
-	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ë
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3	variables
4regularization_losses
5trainable_variables
6	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layerû{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
¦	

7kernel
8bias
9	variables
:regularization_losses
;trainable_variables
<	keras_api
°__call__
+±&call_and_return_all_conditional_losses"ÿ
_tf_keras_layerå{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}

=	variables
>regularization_losses
?trainable_variables
@	keras_api
²__call__
+³&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ë
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layerû{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
§	

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}

P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
î
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
º__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layerþ{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
¨	

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layerç{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}}

c	variables
dregularization_losses
etrainable_variables
f	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
î
gaxis
	hgamma
ibeta
jmoving_mean
kmoving_variance
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"
_tf_keras_layerþ{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}}
è
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
®

tkernel
ubias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layerí{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}}}
è
z	variables
{regularization_losses
|trainable_variables
}	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
±

~kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layerì{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 102, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Þ
	variables
regularization_losses
trainable_variables
	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}

	iter
beta_1
beta_2

decay
learning_ratem÷mø mù!mú/mû0mü7mý8mþBmÿCmJmKmUmVm]m^mhmimtmum~mmvv v!v/v0v7v8vBvCvJvKvUvVv]v^vhvivtvuv ~v¡v¢"
	optimizer

0
1
 2
!3
/4
05
16
27
78
89
B10
C11
D12
E13
J14
K15
U16
V17
W18
X19
]20
^21
h22
i23
j24
k25
t26
u27
~28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
Æ
0
1
 2
!3
/4
05
76
87
B8
C9
J10
K11
U12
V13
]14
^15
h16
i17
t18
u19
~20
21"
trackable_list_wrapper
Ó
layer_metrics
	variables
regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
 layer_regularization_losses
¤__call__
£_default_save_signature
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
-
Ìserving_default"
signature_map
*:( 2conv2d_10/kernel
: 2conv2d_10/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
µ
layer_metrics
	variables
regularization_losses
trainable_variables
metrics
non_trainable_variables
layers
 layer_regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_11/kernel
: 2conv2d_11/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
µ
layer_metrics
"	variables
#regularization_losses
$trainable_variables
metrics
non_trainable_variables
layers
 layer_regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
&	variables
'regularization_losses
(trainable_variables
metrics
non_trainable_variables
layers
  layer_regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¡layer_metrics
*	variables
+regularization_losses
,trainable_variables
¢metrics
£non_trainable_variables
¤layers
 ¥layer_regularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
<
/0
01
12
23"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
µ
¦layer_metrics
3	variables
4regularization_losses
5trainable_variables
§metrics
¨non_trainable_variables
©layers
 ªlayer_regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_12/kernel
:@2conv2d_12/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
µ
«layer_metrics
9	variables
:regularization_losses
;trainable_variables
¬metrics
­non_trainable_variables
®layers
 ¯layer_regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
°layer_metrics
=	variables
>regularization_losses
?trainable_variables
±metrics
²non_trainable_variables
³layers
 ´layer_regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_9/gamma
(:&@2batch_normalization_9/beta
1:/@ (2!batch_normalization_9/moving_mean
5:3@ (2%batch_normalization_9/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
µlayer_metrics
F	variables
Gregularization_losses
Htrainable_variables
¶metrics
·non_trainable_variables
¸layers
 ¹layer_regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
+:)@2conv2d_13/kernel
:2conv2d_13/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
µ
ºlayer_metrics
L	variables
Mregularization_losses
Ntrainable_variables
»metrics
¼non_trainable_variables
½layers
 ¾layer_regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¿layer_metrics
P	variables
Qregularization_losses
Rtrainable_variables
Àmetrics
Ánon_trainable_variables
Âlayers
 Ãlayer_regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_10/gamma
*:(2batch_normalization_10/beta
3:1 (2"batch_normalization_10/moving_mean
7:5 (2&batch_normalization_10/moving_variance
<
U0
V1
W2
X3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
µ
Älayer_metrics
Y	variables
Zregularization_losses
[trainable_variables
Åmetrics
Ænon_trainable_variables
Çlayers
 Èlayer_regularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_14/kernel
:2conv2d_14/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
Élayer_metrics
_	variables
`regularization_losses
atrainable_variables
Êmetrics
Ënon_trainable_variables
Ìlayers
 Ílayer_regularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Îlayer_metrics
c	variables
dregularization_losses
etrainable_variables
Ïmetrics
Ðnon_trainable_variables
Ñlayers
 Òlayer_regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_11/gamma
*:(2batch_normalization_11/beta
3:1 (2"batch_normalization_11/moving_mean
7:5 (2&batch_normalization_11/moving_variance
<
h0
i1
j2
k3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
µ
Ólayer_metrics
l	variables
mregularization_losses
ntrainable_variables
Ômetrics
Õnon_trainable_variables
Ölayers
 ×layer_regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ølayer_metrics
p	variables
qregularization_losses
rtrainable_variables
Ùmetrics
Únon_trainable_variables
Ûlayers
 Ülayer_regularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
": 
 2dense_4/kernel
:2dense_4/bias
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
µ
Ýlayer_metrics
v	variables
wregularization_losses
xtrainable_variables
Þmetrics
ßnon_trainable_variables
àlayers
 álayer_regularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
âlayer_metrics
z	variables
{regularization_losses
|trainable_variables
ãmetrics
änon_trainable_variables
ålayers
 ælayer_regularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
!:	f2dense_5/kernel
:f2dense_5/bias
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
¸
çlayer_metrics
	variables
regularization_losses
trainable_variables
èmetrics
énon_trainable_variables
êlayers
 ëlayer_regularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ìlayer_metrics
	variables
regularization_losses
trainable_variables
ímetrics
înon_trainable_variables
ïlayers
 ðlayer_regularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_4/Adam/iter
 : (2training_4/Adam/beta_1
 : (2training_4/Adam/beta_2
: (2training_4/Adam/decay
':% (2training_4/Adam/learning_rate
 "
trackable_dict_wrapper
(
ñ0"
trackable_list_wrapper
X
10
21
D2
E3
W4
X5
j6
k7"
trackable_list_wrapper
®
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper


òtotal

ócount
ô
_fn_kwargs
õ	variables
ö	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total_4
:  (2count_4
 "
trackable_dict_wrapper
0
ò0
ó1"
trackable_list_wrapper
.
õ	variables"
_generic_user_object
::8 2"training_4/Adam/conv2d_10/kernel/m
,:* 2 training_4/Adam/conv2d_10/bias/m
::8  2"training_4/Adam/conv2d_11/kernel/m
,:* 2 training_4/Adam/conv2d_11/bias/m
9:7 2-training_4/Adam/batch_normalization_8/gamma/m
8:6 2,training_4/Adam/batch_normalization_8/beta/m
::8 @2"training_4/Adam/conv2d_12/kernel/m
,:*@2 training_4/Adam/conv2d_12/bias/m
9:7@2-training_4/Adam/batch_normalization_9/gamma/m
8:6@2,training_4/Adam/batch_normalization_9/beta/m
;:9@2"training_4/Adam/conv2d_13/kernel/m
-:+2 training_4/Adam/conv2d_13/bias/m
;:92.training_4/Adam/batch_normalization_10/gamma/m
::82-training_4/Adam/batch_normalization_10/beta/m
<::2"training_4/Adam/conv2d_14/kernel/m
-:+2 training_4/Adam/conv2d_14/bias/m
;:92.training_4/Adam/batch_normalization_11/gamma/m
::82-training_4/Adam/batch_normalization_11/beta/m
2:0
 2 training_4/Adam/dense_4/kernel/m
+:)2training_4/Adam/dense_4/bias/m
1:/	f2 training_4/Adam/dense_5/kernel/m
*:(f2training_4/Adam/dense_5/bias/m
::8 2"training_4/Adam/conv2d_10/kernel/v
,:* 2 training_4/Adam/conv2d_10/bias/v
::8  2"training_4/Adam/conv2d_11/kernel/v
,:* 2 training_4/Adam/conv2d_11/bias/v
9:7 2-training_4/Adam/batch_normalization_8/gamma/v
8:6 2,training_4/Adam/batch_normalization_8/beta/v
::8 @2"training_4/Adam/conv2d_12/kernel/v
,:*@2 training_4/Adam/conv2d_12/bias/v
9:7@2-training_4/Adam/batch_normalization_9/gamma/v
8:6@2,training_4/Adam/batch_normalization_9/beta/v
;:9@2"training_4/Adam/conv2d_13/kernel/v
-:+2 training_4/Adam/conv2d_13/bias/v
;:92.training_4/Adam/batch_normalization_10/gamma/v
::82-training_4/Adam/batch_normalization_10/beta/v
<::2"training_4/Adam/conv2d_14/kernel/v
-:+2 training_4/Adam/conv2d_14/bias/v
;:92.training_4/Adam/batch_normalization_11/gamma/v
::82-training_4/Adam/batch_normalization_11/beta/v
2:0
 2 training_4/Adam/dense_4/kernel/v
+:)2training_4/Adam/dense_4/bias/v
1:/	f2 training_4/Adam/dense_5/kernel/v
*:(f2training_4/Adam/dense_5/bias/v
î2ë
 __inference__wrapped_model_10864Æ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *6¢3
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd
þ2û
,__inference_sequential_2_layer_call_fn_11935
,__inference_sequential_2_layer_call_fn_12023
,__inference_sequential_2_layer_call_fn_12332
,__inference_sequential_2_layer_call_fn_12367À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_12297
G__inference_sequential_2_layer_call_and_return_conditional_losses_11846
G__inference_sequential_2_layer_call_and_return_conditional_losses_11793
G__inference_sequential_2_layer_call_and_return_conditional_losses_12182À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_10_layer_call_fn_12385¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_10_layer_call_and_return_conditional_losses_12378¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_11_layer_call_fn_12402¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_11_layer_call_and_return_conditional_losses_12395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_4_layer_call_fn_12412¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_4_layer_call_and_return_conditional_losses_12407¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_max_pooling2d_8_layer_call_fn_10881à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
²2¯
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_10870à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
5__inference_batch_normalization_8_layer_call_fn_12457
5__inference_batch_normalization_8_layer_call_fn_12511
5__inference_batch_normalization_8_layer_call_fn_12520
5__inference_batch_normalization_8_layer_call_fn_12466´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12502
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12430
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12484
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12448´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_12_layer_call_fn_12538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_12_layer_call_and_return_conditional_losses_12531¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
/__inference_max_pooling2d_9_layer_call_fn_10990à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
²2¯
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_10979à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
5__inference_batch_normalization_9_layer_call_fn_12583
5__inference_batch_normalization_9_layer_call_fn_12592
5__inference_batch_normalization_9_layer_call_fn_12646
5__inference_batch_normalization_9_layer_call_fn_12637´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12556
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12610
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12628
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12574´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_13_layer_call_fn_12664¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_13_layer_call_and_return_conditional_losses_12657¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_10_layer_call_fn_11099à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_11088à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_batch_normalization_10_layer_call_fn_12718
6__inference_batch_normalization_10_layer_call_fn_12772
6__inference_batch_normalization_10_layer_call_fn_12709
6__inference_batch_normalization_10_layer_call_fn_12763´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12682
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12700
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12754
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12736´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_14_layer_call_fn_12790¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_14_layer_call_and_return_conditional_losses_12783¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_11_layer_call_fn_11208à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_11197à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
6__inference_batch_normalization_11_layer_call_fn_12835
6__inference_batch_normalization_11_layer_call_fn_12898
6__inference_batch_normalization_11_layer_call_fn_12889
6__inference_batch_normalization_11_layer_call_fn_12844´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12880
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12862
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12826
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12808´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_flatten_2_layer_call_fn_12909¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_2_layer_call_and_return_conditional_losses_12904¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_4_layer_call_fn_12926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_4_layer_call_and_return_conditional_losses_12919¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_dropout_2_layer_call_fn_12948
)__inference_dropout_2_layer_call_fn_12953´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_2_layer_call_and_return_conditional_losses_12943
D__inference_dropout_2_layer_call_and_return_conditional_losses_12938´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_dense_5_layer_call_fn_12970¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_5_layer_call_and_return_conditional_losses_12963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_5_layer_call_fn_12980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_5_layer_call_and_return_conditional_losses_12975¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:B8
#__inference_signature_wrapper_12060conv2d_10_inputÄ
 __inference__wrapped_model_10864 !/01278BCDEJKUVWX]^hijktu~@¢=
6¢3
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd
ª ";ª8
6
activation_5&#
activation_5ÿÿÿÿÿÿÿÿÿf³
G__inference_activation_4_layer_call_and_return_conditional_losses_12407h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ`` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ`` 
 
,__inference_activation_4_layer_call_fn_12412[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ`` 
ª " ÿÿÿÿÿÿÿÿÿ`` £
G__inference_activation_5_layer_call_and_return_conditional_losses_12975X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 {
,__inference_activation_5_layer_call_fn_12980K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "ÿÿÿÿÿÿÿÿÿfÉ
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12682tUVWX<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ


 É
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12700tUVWX<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ


 î
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12736UVWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_12754UVWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¡
6__inference_batch_normalization_10_layer_call_fn_12709gUVWX<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p
ª "!ÿÿÿÿÿÿÿÿÿ

¡
6__inference_batch_normalization_10_layer_call_fn_12718gUVWX<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p 
ª "!ÿÿÿÿÿÿÿÿÿ

Æ
6__inference_batch_normalization_10_layer_call_fn_12763UVWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_10_layer_call_fn_12772UVWXN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12808hijkN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12826hijkN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12862thijk<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 É
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_12880thijk<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_batch_normalization_11_layer_call_fn_12835hijkN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_11_layer_call_fn_12844hijkN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¡
6__inference_batch_normalization_11_layer_call_fn_12889ghijk<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¡
6__inference_batch_normalization_11_layer_call_fn_12898ghijk<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÆ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12430r/012;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00 
 Æ
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12448r/012;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00 
 ë
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12484/012M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ë
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_12502/012M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
5__inference_batch_normalization_8_layer_call_fn_12457e/012;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p
ª " ÿÿÿÿÿÿÿÿÿ00 
5__inference_batch_normalization_8_layer_call_fn_12466e/012;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p 
ª " ÿÿÿÿÿÿÿÿÿ00 Ã
5__inference_batch_normalization_8_layer_call_fn_12511/012M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
5__inference_batch_normalization_8_layer_call_fn_12520/012M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ë
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12556BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12574BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Æ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12610rBCDE;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Æ
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_12628rBCDE;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batch_normalization_9_layer_call_fn_12583BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_9_layer_call_fn_12592BCDEM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
5__inference_batch_normalization_9_layer_call_fn_12637eBCDE;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
5__inference_batch_normalization_9_layer_call_fn_12646eBCDE;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@´
D__inference_conv2d_10_layer_call_and_return_conditional_losses_12378l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿbb 
 
)__inference_conv2d_10_layer_call_fn_12385_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd
ª " ÿÿÿÿÿÿÿÿÿbb ´
D__inference_conv2d_11_layer_call_and_return_conditional_losses_12395l !7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿbb 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ`` 
 
)__inference_conv2d_11_layer_call_fn_12402_ !7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿbb 
ª " ÿÿÿÿÿÿÿÿÿ`` ´
D__inference_conv2d_12_layer_call_and_return_conditional_losses_12531l787¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ..@
 
)__inference_conv2d_12_layer_call_fn_12538_787¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00 
ª " ÿÿÿÿÿÿÿÿÿ..@µ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_12657mJK7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_13_layer_call_fn_12664`JK7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_14_layer_call_and_return_conditional_losses_12783n]^8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ


ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_14_layer_call_fn_12790a]^8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ


ª "!ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_4_layer_call_and_return_conditional_losses_12919^tu0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_4_layer_call_fn_12926Qtu0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_5_layer_call_and_return_conditional_losses_12963]~0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 {
'__inference_dense_5_layer_call_fn_12970P~0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿf¦
D__inference_dropout_2_layer_call_and_return_conditional_losses_12938^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¦
D__inference_dropout_2_layer_call_and_return_conditional_losses_12943^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dropout_2_layer_call_fn_12948Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ~
)__inference_dropout_2_layer_call_fn_12953Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿª
D__inference_flatten_2_layer_call_and_return_conditional_losses_12904b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_flatten_2_layer_call_fn_12909U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ î
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_11088R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_10_layer_call_fn_11099R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_11197R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_11_layer_call_fn_11208R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_10870R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_8_layer_call_fn_10881R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_10979R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_9_layer_call_fn_10990R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÝ
G__inference_sequential_2_layer_call_and_return_conditional_losses_11793 !/01278BCDEJKUVWX]^hijktu~H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 Ý
G__inference_sequential_2_layer_call_and_return_conditional_losses_11846 !/01278BCDEJKUVWX]^hijktu~H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 Ô
G__inference_sequential_2_layer_call_and_return_conditional_losses_12182 !/01278BCDEJKUVWX]^hijktu~?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 Ô
G__inference_sequential_2_layer_call_and_return_conditional_losses_12297 !/01278BCDEJKUVWX]^hijktu~?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 µ
,__inference_sequential_2_layer_call_fn_11935 !/01278BCDEJKUVWX]^hijktu~H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd
p

 
ª "ÿÿÿÿÿÿÿÿÿfµ
,__inference_sequential_2_layer_call_fn_12023 !/01278BCDEJKUVWX]^hijktu~H¢E
>¢;
1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd
p 

 
ª "ÿÿÿÿÿÿÿÿÿf«
,__inference_sequential_2_layer_call_fn_12332{ !/01278BCDEJKUVWX]^hijktu~?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p

 
ª "ÿÿÿÿÿÿÿÿÿf«
,__inference_sequential_2_layer_call_fn_12367{ !/01278BCDEJKUVWX]^hijktu~?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p 

 
ª "ÿÿÿÿÿÿÿÿÿfÚ
#__inference_signature_wrapper_12060² !/01278BCDEJKUVWX]^hijktu~S¢P
¢ 
IªF
D
conv2d_10_input1.
conv2d_10_inputÿÿÿÿÿÿÿÿÿdd";ª8
6
activation_5&#
activation_5ÿÿÿÿÿÿÿÿÿf