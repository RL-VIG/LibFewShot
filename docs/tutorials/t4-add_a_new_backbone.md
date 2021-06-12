# Add a new backbone

如果想在`LibFewShot`中添加一个新的`backbone`，可以将所有与`backbone`有关的文件放到`core/model/backbone/`目录下，例如添加ResNet网络到`LibFewShot`中，需要将代码写入`core/model/backbone/resnet.py`中，并且在`resnet.py`中提供一个能够生成`backbone`的`class`或者是`function`。例如`resnet.py`文件：

```python
...

class ResNet(nn.Module):
	def __init(self,...):
...

def ResNet18():
	model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
	return model
```

之后为了能够从`backbone`包中调用到`ResNet18`这个`function`，需要修改`/core/model/backbone/__init__.py`文件，添加如下一行代码

```python
...

from resnet import ResNet18
```

这样一个新的`backbone`就添加完成了。

这个新加入的`backbone`和以前的`backbone`是同样的使用方式。举个例子，要将`ResNet18`替换为`DN4`的`backbone`，只需要在`config/dn4.yaml`中将修改`backbone`字段如下：

```yaml
# arch info
backbone:
  name: resnet18
  kwargs:
    avg_pool: False
    is_flatten: False
```

即可完成替换。

