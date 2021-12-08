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
      <td> 42.11 </td>
      <td> 42.34 ± 0.31 </td>
      <td> 62.53 </td>
      <td> 62.18 ± 0.30 </td>
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
      <td> 48.24 </td>
      <td> 46.21 </td>
      <td> 66.43 </td>
      <td> 65.18 </td>
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
      <td> 62.02 ± 0.63 </td>
      <td> 62.80 ± 0.52 </td>
      <td> 79.64 ± 0.44 </td>
      <td> 79.57± 0.39 </td>
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
      <td> 67.60 ± 0.44 </td>
      <td> 66.83 ± 0.36 </td>
      <td> 82.58 ± 0.30 </td>
      <td> 82.13 ± 0.26 </td>
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
      <td rowspan="2"><a href="./BOIL/">BOIL</a></td>
      <td>Conv64F</td>
      <td> 49.61 ± 0.16 </td>
      <td> 48.00 ± 0.36 </td>
      <td> 66.45 ± 0.37 </td>
      <td> - </td>
  </tr>
  <tr>
      <td>ResNet12(wo LSC)</td>
      <td> - </td>
      <td> 52.75 ± 0.37 </td>
      <td> 71.30 ± 0.28 </td>
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
  <tr>
      <td><a href="./Negative_Margin">Negative_Margin</a></td>
      <td>ResNet12</td>
      <td> 63.85 ± 0.81 </td>
      <td> 63.28 ± 0.36 </td>
      <td> 81.57 ± 0.56 </td>
      <td> 81.24 ± 0.26 </td>
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
        <td> 44.90 ± 0.32 </td>
        <td> 63.96 ± 0.30 </td>
        <td> 48.20 ± 0.35 </td>
        <td> 68.96 ± 0.33 </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td>Fine-tuning</td>
        <td> 48.86 ± 0.35 </td>
        <td> 63.29 ± 0.30 </td>
        <td> 55.94 ± 0.39 </td>
        <td> 73.80 ± 0.32 </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> 47.97 ± 0.33 </td>
        <td> 65.88 ± 0.30 </td>
        <td> 52.21 ± 0.37 </td>
        <td> 71.82 ± 0.32 </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-distill</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> </td>
        <td>  </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> 48.14 ± 0.33  </td>
        <td> 66.36 ± 0.29 </td>
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
        <td> 57.62 ± 0.36 </td>
        <td> 74.14 ± 0.27 </td>
        <td> 61.62 ± 0.40 </td>
        <td> 76.74 ± 0.33 </td>
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
        <td> 47.05 ± 0.35 </td>
        <td> 68.56 ± 0.16 </td>
        <td> 46.11 ± 0.39 </td>
        <td> 70.07 ± 0.34 </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> 51.52 ± 0.37 </td>
        <td> 66.76 ± 0.30 </td>
        <td> 54.37 ± 0.44 </td>
        <td> 71.93 ± 0.35 </td>
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
        <td> 55.88 ± 0.38 </td>
        <td> 70.98 ± 0.3 </td>
        <td> 55.96 ± 0.42 </td>
        <td> 70.52 ± 0.35 </td>
    </tr>
    <tr>
      <td><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td>Metric</td>
      <td> - </td>
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
        <td> 56.39 ± 0.36 </td>
        <td> 76.18 ± 0.27 </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td>Fine-tuning</td>
        <td> 56.75 ± 0.38 </td>
        <td> 66.36 ± 0.29 </td>
        <td> 65.95 ± 0.42 </td>
        <td> 82.25 ± 0.31 </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td>Fine-tuning</td>
        <td> 60.96 ± 0.35 </td>
        <td> 77.36 ± 0.27 </td>
        <td> 70.55 ± 0.42 </td>
        <td> 84.74 ± 0.29 </td>
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
        <td> 66.40 ± 0.36 </td>
        <td> 83.06 ± 0.24 </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="SKD">SKD-GEN1</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> 67.35 ± 0.37 </td>
        <td> 83.31 ± 0.24 </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td>Fine-tuning</td>
        <td> 64.81 ± 0.37 </td>
        <td> 79.90 ± 0.27 </td>
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
        <td> 59.52 ± 0.39 </td>
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
        <td> 54.25 ± 0.37 </td>
        <td> 74.65 ± 0.29 </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> 55.22 ± 0.39  </td>
        <td> 69.25 ± 0.31 </td>
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
        <td> 59.82 ± 0.38 </td>
        <td> 76.54 ± 0.29 </td>
        <td> 70.46 ± 0.43 </td>
        <td> 84.50 ± 0.30 </td>
    </tr>
    <tr>
      <td><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td>Metric</td>
      <td> - </td>
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
        <td> 54.11 ± 0.35 </td>
        <td> 74.44 ± 0.29 </td>
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
        <td> 61.65 ± 0.37 </td>
        <td> 76.60 ± 0.28 </td>
        <td> 69.14 ± 0.42 </td>
        <td> 83.21 ± 0.31 </td>
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
        <td> 66.18 ± 0.37 </td>
        <td> 82.21 ± 0.24 </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
        <td><a href="SKD">SKD-GEN1</a></td>
        <td>arXiv’20</td>
        <td>Fine-tuning</td>
        <td> 66.70 ± 0.37 </td>
        <td> 82.60 ±  0.24 </td>
        <td>  </td>
        <td>  </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td>Fine-tuning</td>
        <td> 62.86 ± 0.37 </td>
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
        <td> 53.98 ± 0.37 </td>
        <td> 71.27 ± 0.31 </td>
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
        <td><a href="./CAN">CAN(!re-run,h2=11!)</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> 60.78 ± 0.40  </td>
        <td> 75.05 ± 0.29 </td>
        <td> 71.70 ± 0.43 </td>
        <td> 84.61 ± 0.37 </td>
    </tr>
    <tr>
      <td><a href="./DSN">DSN</a></td>
      <td>Conv64F</td>
      <td>Metric</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
  </tr>
</table>
