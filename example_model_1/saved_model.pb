͌
��
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
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:	*
dtype0
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
�
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate
�
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
training/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*-
shared_nametraining/Adam/dense/kernel/m
�
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
_output_shapes

:	*
dtype0
�
training/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining/Adam/dense/bias/m
�
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
_output_shapes
:*
dtype0
�
training/Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*/
shared_name training/Adam/dense_1/kernel/m
�
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*
_output_shapes

:	*
dtype0
�
training/Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nametraining/Adam/dense_1/bias/m
�
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
_output_shapes
:	*
dtype0
�
training/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*-
shared_nametraining/Adam/dense/kernel/v
�
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
_output_shapes

:	*
dtype0
�
training/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining/Adam/dense/bias/v
�
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
_output_shapes
:*
dtype0
�
training/Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*/
shared_name training/Adam/dense_1/kernel/v
�
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*
_output_shapes

:	*
dtype0
�
training/Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nametraining/Adam/dense_1/bias/v
�
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
_output_shapes
:	*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer_with_weights-1
layer-0
layer-1
	optimizer
	keras_api
trainable_variables
	variables
regularization_losses

signatures
h

	kernel

bias
	keras_api
trainable_variables
	variables
regularization_losses
h

kernel
bias
	keras_api
trainable_variables
	variables
regularization_losses
�
iter

beta_1

beta_2
	decay
learning_rate	m/
m0m1m2	v3
v4v5v6
�

layers
trainable_variables
non_trainable_variables
layer_metrics
metrics
layer_regularization_losses
regularization_losses
	variables

	0

1
2
3

	0

1
2
3
 
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
�

layers
trainable_variables
 non_trainable_variables
!layer_metrics
"metrics
#layer_regularization_losses
regularization_losses
	variables

	0

1

	0

1
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
�

$layers
trainable_variables
%non_trainable_variables
&layer_metrics
'metrics
(layer_regularization_losses
regularization_losses
	variables

0
1

0
1
 
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 

)0
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
D
	*total
	+count
,
_fn_kwargs
-	keras_api
.	variables
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

.	variables

*0
+1
��
VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*'
_output_shapes
:���������	*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference_signature_wrapper_386
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*%
f R
__inference__traced_save_557
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/bias/mtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/dense_1/kernel/vtraining/Adam/dense_1/bias/v*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_restore_626��
�
�
(__inference_sequential_layer_call_fn_375
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_bias*
Tin	
2*
Tout
2*'
_output_shapes
:���������	*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_3682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
%__inference_dense_1_layer_call_fn_463

inputs
dense_1_kernel
dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_336
input_1
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2912
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3162!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�9
�
__inference__traced_save_557
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop=
9savev2_training_adam_dense_1_kernel_m_read_readvariableop;
7savev2_training_adam_dense_1_bias_m_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop=
9savev2_training_adam_dense_1_kernel_v_read_readvariableop;
7savev2_training_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5e37583ecf0449489895c168ed96788c/part2	
Const_1�
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop9savev2_training_adam_dense_1_kernel_m_read_readvariableop7savev2_training_adam_dense_1_bias_m_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop9savev2_training_adam_dense_1_kernel_v_read_readvariableop7savev2_training_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapest
r: :	::	:	: : : : : : : :	::	:	:	::	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:

_output_shapes
: 
�
~
#__inference_dense_layer_call_fn_456

inputs
dense_kernel

dense_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_dense_layer_call_and_return_conditional_losses_449

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity��
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	:::O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_368

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2912
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3162!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_403

inputs,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias
identity��
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:	*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense/Sigmoid�
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes

:	*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Sigmoid:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:	*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	:::::O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_dense_layer_call_and_return_conditional_losses_291

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity��
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	:::O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense_1_layer_call_and_return_conditional_losses_316

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity��
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_420

inputs,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias
identity��
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:	*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense/Sigmoid�
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes

:	*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Sigmoid:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:	*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
dense_1/BiasAddl
IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	:::::O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_sequential_layer_call_fn_429

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_bias*
Tin	
2*
Tout
2*'
_output_shapes
:���������	*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_3492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_sequential_layer_call_fn_438

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_bias*
Tin	
2*
Tout
2*'
_output_shapes
:���������	*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_3682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_349

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2912
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3162!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense_1_layer_call_and_return_conditional_losses_473

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity��
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_326
input_1
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2912
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*'
_output_shapes
:���������	*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_3162!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_sequential_layer_call_fn_356
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_bias*
Tin	
2*
Tout
2*'
_output_shapes
:���������	*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_3492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
__inference__wrapped_model_276
input_17
3sequential_dense_matmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias;
7sequential_dense_1_matmul_readvariableop_dense_1_kernel:
6sequential_dense_1_biasadd_readvariableop_dense_1_bias
identity��
&sequential/dense/MatMul/ReadVariableOpReadVariableOp3sequential_dense_matmul_readvariableop_dense_kernel*
_output_shapes

:	*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMulinput_1.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/dense/BiasAdd�
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential/dense/Sigmoid�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes

:	*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp�
sequential/dense_1/MatMulMatMulsequential/dense/Sigmoid:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
sequential/dense_1/MatMul�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:	*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
sequential/dense_1/BiasAddw
IdentityIdentity#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	:::::P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
!__inference_signature_wrapper_386
input_1
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1dense_kernel
dense_biasdense_1_kerneldense_1_bias*
Tin	
2*
Tout
2*'
_output_shapes
:���������	*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__wrapped_model_2762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������	::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�X
�

__inference__traced_restore_626
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias)
%assignvariableop_4_training_adam_iter+
'assignvariableop_5_training_adam_beta_1+
'assignvariableop_6_training_adam_beta_2*
&assignvariableop_7_training_adam_decay2
.assignvariableop_8_training_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count4
0assignvariableop_11_training_adam_dense_kernel_m2
.assignvariableop_12_training_adam_dense_bias_m6
2assignvariableop_13_training_adam_dense_1_kernel_m4
0assignvariableop_14_training_adam_dense_1_bias_m4
0assignvariableop_15_training_adam_dense_kernel_v2
.assignvariableop_16_training_adam_dense_bias_v6
2assignvariableop_17_training_adam_dense_1_kernel_v4
0assignvariableop_18_training_adam_dense_1_bias_v
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_training_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp'assignvariableop_5_training_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_training_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_training_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_training_adam_dense_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_training_adam_dense_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp2assignvariableop_13_training_adam_dense_1_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_training_adam_dense_1_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_training_adam_dense_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_training_adam_dense_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp2assignvariableop_17_training_adam_dense_1_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp0assignvariableop_18_training_adam_dense_1_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19�
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
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
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������	;
dense_10
StatefulPartitionedCall:0���������	tensorflow/serving/predict:�`
�
layer_with_weights-0
layer_with_weights-1
layer-0
layer-1
	optimizer
	keras_api
trainable_variables
	variables
regularization_losses

signatures
7__call__
8_default_save_signature
*9&call_and_return_all_conditional_losses"�
_tf_keras_sequential�{"class_name": "Sequential", "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "name": "sequential", "backend": "tensorflow", "dtype": "float32", "expects_training_arg": true, "input_spec": {"class_name": "InputSpec", "config": {"min_ndim": 2, "shape": null, "ndim": null, "dtype": null, "max_ndim": null, "axes": {"-1": 9}}}, "trainable": true, "is_graph_network": true, "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"use_bias": true, "kernel_regularizer": null, "bias_regularizer": null, "name": "dense", "activation": "sigmoid", "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "trainable": true, "units": 18}}, {"class_name": "Dense", "config": {"use_bias": true, "kernel_regularizer": null, "bias_regularizer": null, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "activation": "linear", "name": "dense_1", "trainable": true, "units": 9}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}, "batch_input_shape": null, "keras_version": "2.3.0-tf", "training_config": {"weighted_metrics": null, "metrics": ["mae"], "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"amsgrad": false, "beta_2": 0.9990000128746033, "learning_rate": 0.0010000000474974513, "decay": 0.0, "epsilon": 1e-07, "name": "Adam", "beta_1": 0.8999999761581421}}, "loss": "mse", "loss_weights": null}, "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Dense", "config": {"use_bias": true, "kernel_regularizer": null, "bias_regularizer": null, "name": "dense", "activation": "sigmoid", "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "trainable": true, "units": 18}}, {"class_name": "Dense", "config": {"use_bias": true, "kernel_regularizer": null, "bias_regularizer": null, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "activation": "linear", "name": "dense_1", "trainable": true, "units": 9}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}}}
�

	kernel

bias
	keras_api
trainable_variables
	variables
regularization_losses
:__call__
*;&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"stateful": false, "config": {"use_bias": true, "kernel_regularizer": null, "bias_regularizer": null, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "trainable": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "activation": "sigmoid", "name": "dense", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 18}, "batch_input_shape": null, "class_name": "Dense", "dtype": "float32", "name": "dense", "expects_training_arg": false, "input_spec": {"class_name": "InputSpec", "config": {"min_ndim": 2, "shape": null, "ndim": null, "dtype": null, "max_ndim": null, "axes": {"-1": 9}}}, "trainable": true}
�

kernel
bias
	keras_api
trainable_variables
	variables
regularization_losses
<__call__
*=&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"stateful": false, "config": {"use_bias": true, "kernel_regularizer": null, "bias_regularizer": null, "dtype": "float32", "activity_regularizer": null, "bias_constraint": null, "trainable": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "activation": "linear", "name": "dense_1", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 9}, "batch_input_shape": null, "class_name": "Dense", "dtype": "float32", "name": "dense_1", "expects_training_arg": false, "input_spec": {"class_name": "InputSpec", "config": {"min_ndim": 2, "shape": null, "ndim": null, "dtype": null, "max_ndim": null, "axes": {"-1": 18}}}, "trainable": true}
�
iter

beta_1

beta_2
	decay
learning_rate	m/
m0m1m2	v3
v4v5v6"
	optimizer
�

layers
trainable_variables
non_trainable_variables
layer_metrics
metrics
layer_regularization_losses
regularization_losses
	variables
7__call__
8_default_save_signature
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
,
>serving_default"
signature_map
:	2dense/kernel
:2
dense/bias
�

layers
trainable_variables
 non_trainable_variables
!layer_metrics
"metrics
#layer_regularization_losses
regularization_losses
	variables
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
 :	2dense_1/kernel
:	2dense_1/bias
�

$layers
trainable_variables
%non_trainable_variables
&layer_metrics
'metrics
(layer_regularization_losses
regularization_losses
	variables
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
)0"
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
�
	*total
	+count
,
_fn_kwargs
-	keras_api
.	variables"�
_tf_keras_metric�{"dtype": "float32", "class_name": "MeanMetricWrapper", "name": "mae", "config": {"dtype": "float32", "name": "mae", "fn": "mean_absolute_error"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
-
.	variables"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
,:*	2training/Adam/dense/kernel/m
&:$2training/Adam/dense/bias/m
.:,	2training/Adam/dense_1/kernel/m
(:&	2training/Adam/dense_1/bias/m
,:*	2training/Adam/dense/kernel/v
&:$2training/Adam/dense/bias/v
.:,	2training/Adam/dense_1/kernel/v
(:&	2training/Adam/dense_1/bias/v
�2�
(__inference_sequential_layer_call_fn_375
(__inference_sequential_layer_call_fn_429
(__inference_sequential_layer_call_fn_438
(__inference_sequential_layer_call_fn_356�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_276�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������	
�2�
C__inference_sequential_layer_call_and_return_conditional_losses_420
C__inference_sequential_layer_call_and_return_conditional_losses_336
C__inference_sequential_layer_call_and_return_conditional_losses_403
C__inference_sequential_layer_call_and_return_conditional_losses_326�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_dense_layer_call_fn_456�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_dense_layer_call_and_return_conditional_losses_449�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_1_layer_call_fn_463�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dense_1_layer_call_and_return_conditional_losses_473�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0B.
!__inference_signature_wrapper_386input_1�
__inference__wrapped_model_276k	
0�-
&�#
!�
input_1���������	
� "1�.
,
dense_1!�
dense_1���������	�
@__inference_dense_1_layer_call_and_return_conditional_losses_473\/�,
%�"
 �
inputs���������
� "%�"
�
0���������	
� x
%__inference_dense_1_layer_call_fn_463O/�,
%�"
 �
inputs���������
� "����������	�
>__inference_dense_layer_call_and_return_conditional_losses_449\	
/�,
%�"
 �
inputs���������	
� "%�"
�
0���������
� v
#__inference_dense_layer_call_fn_456O	
/�,
%�"
 �
inputs���������	
� "�����������
C__inference_sequential_layer_call_and_return_conditional_losses_326g	
8�5
.�+
!�
input_1���������	
p

 
� "%�"
�
0���������	
� �
C__inference_sequential_layer_call_and_return_conditional_losses_336g	
8�5
.�+
!�
input_1���������	
p 

 
� "%�"
�
0���������	
� �
C__inference_sequential_layer_call_and_return_conditional_losses_403f	
7�4
-�*
 �
inputs���������	
p

 
� "%�"
�
0���������	
� �
C__inference_sequential_layer_call_and_return_conditional_losses_420f	
7�4
-�*
 �
inputs���������	
p 

 
� "%�"
�
0���������	
� �
(__inference_sequential_layer_call_fn_356Z	
8�5
.�+
!�
input_1���������	
p

 
� "����������	�
(__inference_sequential_layer_call_fn_375Z	
8�5
.�+
!�
input_1���������	
p 

 
� "����������	�
(__inference_sequential_layer_call_fn_429Y	
7�4
-�*
 �
inputs���������	
p

 
� "����������	�
(__inference_sequential_layer_call_fn_438Y	
7�4
-�*
 �
inputs���������	
p 

 
� "����������	�
!__inference_signature_wrapper_386v	
;�8
� 
1�.
,
input_1!�
input_1���������	"1�.
,
dense_1!�
dense_1���������	