# Reproducible configs and checkpoints

This folder contains:
+ Reproducible config of Table.1 in the paper of `LibFewShot`.
+ Reproducible config of Table.2 in the paper of `LibFewShot`.

## Reproduction results on miniImageNet
(Results may different from the paper. Here are up-to-date results with checkpoints and configs.)
<table>
  <tr>
      <td rowspan="2">Method</td>
      <td rowspan="2">Embed.</td>
      <td colspan="2">5-way 1-shot</td>
      <td colspan="2">5-way 5-shot</td>
  </tr>
  <tr>
      <td>Reported</td>
      <td>Ours</td>
      <td>Reported</td>
      <td>Ours</td>
  </tr>
  <tr>
      <td rowspan="2"> <a href="./Baseline">Baseline</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet18</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td rowspan="2"><a href="./Baseline++">Baseline++</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet18</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./RFS/">RFS-simple</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./RFS/">RFS-distill</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./SKD">SKD-GEN0</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./SKD">SKD-GEN1</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./RENet/">RENet</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./MAML">MAML</a></td>
      <td>Conv32F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./Versa">Versa</a></td>
      <td>Conv64F†</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td rowspan="2"><a href="./R2D2">R2D2</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>Conv64F‡</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./ANIL">ANIL</a></td>
      <td>Conv32F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
    <tr>
      <td rowspan="3"><a href="./BOIL/">BOIL</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet12(wo LSC)</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./MTL">MTL</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./Proto/">ProtoNet†</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./RelationNet">RelationNet</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./CovaMNet">CovaMNet</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td rowspan="2"><a href="./DN4">DN4</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet12†</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td><a href="./CAN">CAN</a></td>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td rowspan="2"><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet12</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
</table>


## The overview picture of the SOTAs

### Conv64F
<table>
    <tr>
        <td rowspan="2">Method</td>
        <td rowspan="2">Venue</td>
        <td rowspan="2">Type</td>
        <td colspan="2">miniImageNet</td>
        <td colspan="2">tieredImageNet</td>
    </tr>
    <tr>
        <td>1-shot</td>
        <td>5-shot</td>
        <td>1-shot</td>
        <td>5-shot</td>
    </tr>
    <tr>
        <td><a href="./Baseline">Baseline</a></td>
        <td>ICLR’19</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-distill</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="SKD">SKD-GEN1</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./MAML">MAML</a></td>
        <td>ICML’17</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./Versa">Versa</a></td>
        <td>NeurIPS’18</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./R2D2">R2D2</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./LEO">LEO</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./MTL">MTL</a></td>
        <td>CVPR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./ANIL">ANIL</a></td>
        <td>ICLR’20</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./BOIL">BOIL</a></td>
        <td>ICLR’21</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./ProtoNet">ProtoNet</a></td>
        <td>NeurIPS’17</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./CovaMNet">CovaMNet</a></td>
        <td>AAAI’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./DN4">DN4</a></td>
        <td>CVPR’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./CAN">CAN</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td>Metric</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
</table>

### ResNet12
<table>
    <tr>
        <td rowspan="2">Method</td>
        <td rowspan="2">Venue</td>
        <td rowspan="2">Type</td>
        <td colspan="2">miniImageNet</td>
        <td colspan="2">tieredImageNet</td>
    </tr>
    <tr>
        <td>1-shot</td>
        <td>5-shot</td>
        <td>1-shot</td>
        <td>5-shot</td>
    </tr>
    <tr>
        <td><a href="./Baseline">Baseline</a></td>
        <td>ICLR’19</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-distill</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="SKD">SKD-GEN1</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./Versa">Versa</a></td>
        <td>NeurIPS’18</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./R2D2">R2D2</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./LEO">LEO</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./MTL">MTL</a></td>
        <td>CVPR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./ANIL">ANIL</a></td>
        <td>ICLR’20</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./BOIL">BOIL</a></td>
        <td>ICLR’21</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./ProtoNet">ProtoNet</a></td>
        <td>NeurIPS’17</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./CovaMNet">CovaMNet</a></td>
        <td>AAAI’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./DN4">DN4</a></td>
        <td>CVPR’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./CAN">CAN</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td>Metric</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
</table>

### ResNet18
<table>
    <tr>
        <td rowspan="2">Method</td>
        <td rowspan="2">Venue</td>
        <td rowspan="2">Type</td>
        <td colspan="2">miniImageNet</td>
        <td colspan="2">tieredImageNet</td>
    </tr>
    <tr>
        <td>1-shot</td>
        <td>5-shot</td>
        <td>1-shot</td>
        <td>5-shot</td>
    </tr>
    <tr>
        <td><a href="./Baseline">Baseline</a></td>
        <td>ICLR’19</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-distill</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="SKD">SKD-GEN1</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td>Fine-tuning</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
        <tr>
        <td><a href="./Versa">Versa</a></td>
        <td>NeurIPS’18</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./R2D2">R2D2</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./LEO">LEO</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./MTL">MTL</a></td>
        <td>CVPR’19</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./ANIL">ANIL</a></td>
        <td>ICLR’20</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./BOIL">BOIL</a></td>
        <td>ICLR’21</td>
        <td>Meta</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./ProtoNet">ProtoNet</a></td>
        <td>NeurIPS’17</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./CovaMNet">CovaMNet</a></td>
        <td>AAAI’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./DN4">DN4</a></td>
        <td>CVPR’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./CAN">CAN</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td>Metric</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
</table>
