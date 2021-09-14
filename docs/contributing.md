# 贡献代码
Feel free to contribute classifiers, backbones , functions and any enhancement.

## 增加一个方法/特性或修复一个错误
我们建议你按照以下工作流添加一个方法、增加`LibFewShot`的功能或者修订一个补丁。
1. fork `main`分支的最新`LibFewShot`代码
2. checkout 一个新的分支，分支的名字应该能够直观表现本次contribution的主要内容，如`add-method-demo`或者`fix-doc-typos`
3. commit
4. create a PR

注意，如果你添加了一个方法，你必须要做的是
1. 测试该方法的表现是否正常
2. 提供你复现该方法使用的config文件，以及对应的在miniImageNet上的1-shot和5-shot精度。
如果你有更多精力，我们还希望你提供
3. 在其它数据集（如*tiered*ImageNet）上的1-shot和5-shot精度
4. 对应的`model_best.pth`文件（以zip文件格式）
我们会在`README`和其它显眼的地方感谢您的贡献。

## 使用 `pre-commit` 检查代码
在提交你的代码之前，你的代码必须要能够通过 [black](https://github.com/psf/black) 的格式化以及 [flake8](https://github.com/PyCQA/flake8)
，我们使用 [pre-commit](https://pre-commit.com/) 进行测试和自动修订：
1. 首先安装 `pre-commit`
```shell
cd <path-to-LibFewShot>
pip install pre-commit
```
2. run `pre-commit install`
3. run `pre-commit run --all-files`
4. 根据pre-commit给出的warning修改代码
5.
## PR style
你的PR请求标题应该如下：
```text
[Method] XXXX XXXX
# 或者
[Feature] XXXX XXXX
# 或者
[FIX] XXXX XXXX
```
PR请求的正文内容应该使用英文/中文简要描述本次PR的主要内容。
