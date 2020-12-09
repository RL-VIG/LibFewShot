# LibFewShot

python文件说明

Train_Fewshot_DN4:                  通过Episodic Training的方式来训练一个DN4，并测试DN4
Train_Fewshot_ProtoNet:             通过Episodic Training的方式来训练一个ProtoNet，并测试ProtoNet
Train_BaseClassifier_Test_LR:       先训练一个64-classes classifier，然后在测试阶段fine-tune 一个LR分类器进行分类
Train_BaseClassifier_Test_ProtoNet: 先训练一个64-classes classifier，然后在测试阶段使用ProtoNet的分类器进行分类

Test_Fewshot: 　         直接使用预训练的网络进行Fewshot模型的测试
Test_with_FineTune_LR: 　使用预训练的网络进行测试，在测试阶段的时候fine-tune一个LR分类器

