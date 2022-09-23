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
      <td> 42.34 </td>
      <td> 62.53 </td>
      <td> 62.18 </td>
  </tr>
  <tr>
      <td>ResNet18</td>
      <td> 51.75 </td>
      <td> 51.18 </td>
      <td> 74.27 </td>
      <td> 74.06 </td>
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
      <td> 51.87 </td>
      <td> 53.60 </td>
      <td> 75.68 </td>
      <td> 73.63 </td>
  </tr>
  <tr>
      <td><a href="./RFS/">RFS-simple</a></td>
      <td>ResNet12</td>
      <td> 62.02 </td>
      <td> 62.80 </td>
      <td> 79.64 </td>
      <td> 79.57 </td>
  </tr>
  <tr>
      <td><a href="./RFS/">RFS-distill</a></td>
      <td>ResNet12</td>
      <td> 64.82 </td>
      <td> 63.44 </td>
      <td> 82.14 </td>
      <td> 80.17 </td>
  </tr>
  <tr>
      <td><a href="./SKD">SKD-GEN0</a></td>
      <td>ResNet12</td>
      <td> 65.93 </td>
      <td> 66.40 </td>
      <td> 83.15 </td>
      <td> 83.06 </td>
  </tr>
  <tr>
      <td><a href="./SKD">SKD-GEN1</a></td>
      <td>ResNet12</td>
      <td> 67.04 </td>
      <td> 67.35 </td>
      <td> 83.54 </td>
      <td> 80.30 </td>
  </tr>
    <tr>
      <td><a href="./NegCos">NegCos</a></td>
      <td>ResNet12</td>
      <td> 63.85 </td>
      <td> 63.28 </td>
      <td> 81.57 </td>
      <td> 81.24 </td>
  </tr>
  <tr>
      <td><a href="./MAML">MAML</a></td>
      <td>Conv32F</td>
      <td> 48.70 </td>
      <td> 47.41 </td>
      <td> 63.11 </td>
      <td> 65.24 </td>
  </tr>
  <tr>
      <td><a href="./Versa">Versa</a></td>
      <td>Conv64F†</td>
      <td> 53.40 </td>
      <td> 51.92 </td>
      <td> 67.37 </td>
      <td> 66.26 </td>
  </tr>
  <tr>
      <td rowspan="2"><a href="./R2D2">R2D2</a></td>
      <td>Conv64F</td>
      <td> 49.50 </td>
      <td> 47.57 </td>
      <td> 65.40 </td>
      <td> 66.68 </td>
  </tr>
  <tr>
      <td>Conv64F‡</td>
      <td> 51.80 </td>
      <td> 55.53 </td>
      <td> 68.40 </td>
      <td> 70.79 </td>
  </tr>
  <tr>
      <td><a href="./ANIL">ANIL</a></td>
      <td>Conv32F</td>
      <td> 46.70 </td>
      <td> 48.44 </td>
      <td> 61.50 </td>
      <td> 64.35 </td>
  </tr>
    <tr>
      <td rowspan="2"><a href="./BOIL/">BOIL</a></td>
      <td>Conv64F</td>
      <td> 49.61 </td>
      <td> 48.00 </td>
      <td> 66.45 </td>
      <td> 64.39 </td>
  </tr>
  <tr>
      <td>ResNet12**</td>
      <td> - </td>
      <td> 58.87 </td>
      <td> 71.30 </td>
      <td> 72.88 </td>
  </tr>
  <tr>
      <td><a href="./MTL">MTL</a></td>
      <td>ResNet12</td>
      <td> 60.20 </td>
      <td> 60.20 </td>
      <td> 74.30 </td>
      <td> 75.86 </td>
  </tr>
  <tr>
      <td><a href="./Proto/">ProtoNet†</a></td>
      <td>Conv64F</td>
      <td> 46.14 </td>
      <td> 46.30 </td>
      <td> 65.77 </td>
      <td> 66.24 </td>
  </tr>
  <tr>
      <td><a href="./RelationNet">RelationNet</a></td>
      <td>Conv64F</td>
      <td> 50.44 </td>
      <td> 51.75 </td>
      <td> 65.32 </td>
      <td> 66.77 </td>
  </tr>
  <tr>
      <td><a href="./CovaMNet">CovaMNet</a></td>
      <td>Conv64F</td>
      <td> 51.19 </td>
      <td> 53.36 </td>
      <td> 67.65 </td>
      <td> 68.17 </td>
  </tr>
  <tr>
      <td rowspan="2"><a href="./DN4">DN4</a></td>
      <td>Conv64F</td>
      <td> 51.24 </td>
      <td> 51.95 </td>
      <td> 71.02 </td>
      <td> 71.42 </td>
  </tr>
  <tr>
      <td>ResNet12†</td>
      <td> 54.37 </td>
      <td> 57.76 </td>
      <td> 74.44 </td>
      <td> 77.57 </td>
  </tr>
  <tr>
      <td><a href="./CAN">CAN</a></td>
      <td>ResNet12</td>
      <td> 63.85 </td>
      <td> 66.62 </td>
      <td> 79.44 </td>
      <td> 78.96 </td>
  </tr>
  <tr>
      <td><a href="./RENet/">RENet</a></td>
      <td>ResNet12</td>
      <td> 67.60 </td>
      <td> 66.83 </td>
      <td> 82.58 </td>
      <td> 82.13 </td>
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
        <td> Non-episodic</td>
        <td> 44.90  </td>
        <td> 63.96  </td>
        <td> 48.20  </td>
        <td> 68.96  </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td> Non-episodic</td>
        <td> 48.86  </td>
        <td> 63.29  </td>
        <td> 55.94  </td>
        <td> 73.80  </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td> Non-episodic</td>
        <td> 47.97  </td>
        <td> 65.88  </td>
        <td> 52.21  </td>
        <td> 71.82  </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>BMVC'20</td>
        <td> Non-episodic</td>
        <td> 48.14  </td>
        <td> 66.36  </td>
        <td> 51.78  </td>
        <td> 70.65  </td>
    </tr>
    <tr>
        <td><a href="./NegCos">NegCos</a></td>
        <td>ECCV’20</td>
        <td> Non-episodic</td>
        <td> 47.34 </td>
        <td> 65.97 </td>
        <td> 51.21 </td>
        <td> 71.57 </td>
    </tr>
    <tr>
        <td><a href="./MAML">MAML</a></td>
        <td>ICML’17</td>
        <td>Meta</td>
        <td> 49.55  </td>
        <td> 64.92  </td>
        <td> 50.98  </td>
        <td> 67.12  </td>
    </tr>
    <tr>
        <td><a href="./Versa">Versa</a></td>
        <td>NeurIPS’18</td>
        <td>Meta</td>
        <td> 52.75  </td>
        <td> 67.40  </td>
        <td> 52.28  </td>
        <td> 69.41  </td>
    </tr>
    <tr>
        <td><a href="./R2D2">R2D2</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> 51.19  </td>
        <td> 67.29  </td>
        <td> 52.18  </td>
        <td> 69.19  </td>
    </tr>
    <tr>
        <td><a href="./LEO">LEO</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> 53.31  </td>
        <td> 67.47  </td>
        <td> 58.15  </td>
        <td> 74.21  </td>
    </tr>
    <tr>
        <td><a href="./MTL">MTL</a></td>
        <td>CVPR’19</td>
        <td>Meta</td>
        <td> 40.97 </td>
        <td> 57.12 </td>
        <td> 42.36 </td>
        <td> 64.87 </td>
    </tr>
    <tr>
        <td><a href="./ANIL">ANIL</a></td>
        <td>ICLR’20</td>
        <td>Meta</td>
        <td> 48.01  </td>
        <td> 63.88  </td>
        <td> 49.05  </td>
        <td> 66.32  </td>
    </tr>
    <tr>
        <td><a href="./BOIL">BOIL</a></td>
        <td>ICLR’21</td>
        <td>Meta</td>
        <td> 47.92  </td>
        <td> 64.39  </td>
        <td> 50.04  </td>
        <td> 65.51  </td>
    </tr>
    <tr>
        <td><a href="./ProtoNet">ProtoNet</a></td>
        <td>NeurIPS’17</td>
        <td>Metric</td>
        <td> 47.05  </td>
        <td> 68.56  </td>
        <td> 46.11  </td>
        <td> 70.07  </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> 51.52  </td>
        <td> 66.76  </td>
        <td> 54.37  </td>
        <td> 71.93  </td>
    </tr>
    <tr>
        <td><a href="./CovaMNet">CovaMNet</a></td>
        <td>AAAI’19</td>
        <td>Metric</td>
        <td> 51.59  </td>
        <td> 67.65  </td>
        <td> 51.92  </td>
        <td> 69.76  </td>
    </tr>
    <tr>
        <td><a href="./DN4">DN4</a></td>
        <td>CVPR’19</td>
        <td>Metric</td>
        <td> 54.47  </td>
        <td> 72.15  </td>
        <td> 56.07  </td>
        <td> 75.75  </td>
    </tr>
    <tr>
        <td><a href="./CAN">CAN</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> 55.88  </td>
        <td> 70.98  </td>
        <td> 55.96  </td>
        <td> 70.52  </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td> Metric</td>
        <td> 57.62  </td>
        <td> 74.14  </td>
        <td> 61.62  </td>
        <td> 76.74  </td>
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
        <td> Non-episodic</td>
        <td> 56.39  </td>
        <td> 76.18  </td>
        <td> 65.54 </td>
        <td> 83.46 </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td> Non-episodic</td>
        <td> 56.75  </td>
        <td> 66.36  </td>
        <td> 65.95  </td>
        <td> 82.25  </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td> Non-episodic</td>
        <td> 61.65  </td>
        <td> 78.88  </td>
        <td> 70.55  </td>
        <td> 84.74  </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>BMVC'20</td>
        <td> Non-episodic</td>
        <td> 66.40  </td>
        <td> 83.06  </td>
        <td> 71.90 </td>
        <td> 86.20 </td>
    </tr>
    <tr>
        <td><a href="./NegCos">NegCos</a></td>
        <td>ECCV’20</td>
        <td> Non-episodic</td>
        <td> 60.60 </td>
        <td> 78.80 </td>
        <td> 70.15 </td>
        <td> 84.94 </td>
    </tr>
    <tr>
        <td><a href="./Versa">Versa</a></td>
        <td>NeurIPS’18</td>
        <td>Meta</td>
        <td> 55.71  </td>
        <td> 70.05  </td>
        <td> 57.14  </td>
        <td> 75.48  </td>
    </tr>
    <tr>
        <td><a href="./R2D2">R2D2</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> 59.52  </td>
        <td> 74.61  </td>
        <td> 65.07  </td>
        <td> 83.04  </td>
    </tr>
    <tr>
        <td><a href="./LEO">LEO</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> 53.58  </td>
        <td> 68.24  </td>
        <td> 64.75 </td>
        <td> 81.42 </td>
    </tr>
    <tr>
        <td><a href="./MTL">MTL</a></td>
        <td>CVPR’19</td>
        <td>Meta</td>
        <td> 61.18 </td>
        <td> 79.14 </td>
        <td> 68.29 </td>
        <td> 83.77 </td>
    </tr>
    <tr>
        <td><a href="./ANIL">ANIL</a></td>
        <td>ICLR’20</td>
        <td>Meta</td>
        <td> 52.77  </td>
        <td> 68.11  </td>
        <td> 55.65  </td>
        <td> 73.53  </td>
    </tr>
    <tr>
        <td><a href="./BOIL">BOIL</a></td>
        <td>ICLR’21</td>
        <td>Meta</td>
        <td> 58.87  </td>
        <td> 72.88  </td>
        <td> 64.66  </td>
        <td> 80.38  </td>
    </tr>
    <tr>
        <td><a href="./ProtoNet">ProtoNet</a></td>
        <td>NeurIPS’17</td>
        <td>Metric</td>
        <td> 58.61  </td>
        <td> 75.02  </td>
        <td> 62.93  </td>
        <td> 83.30  </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> 55.22   </td>
        <td> 69.25  </td>
        <td> 56.86 </td>
        <td> 74.66 </td>
    </tr>
    <tr>
        <td><a href="./CovaMNet">CovaMNet</a></td>
        <td>AAAI’19</td>
        <td>Metric</td>
        <td> 56.95  </td>
        <td> 71.41  </td>
        <td> 58.49  </td>
        <td> 76.34  </td>
    </tr>
    <tr>
        <td><a href="./DN4">DN4</a></td>
        <td>CVPR’19</td>
        <td>Metric</td>
        <td> 58.68  </td>
        <td> 74.70  </td>
        <td> 64.41  </td>
        <td> 82.59  </td>
    </tr>
    <tr>
        <td><a href="./CAN">CAN</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> 59.82  </td>
        <td> 76.54  </td>
        <td> 70.46  </td>
        <td> 84.50  </td>
    </tr>
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td> Metric</td>
        <td> 64.81  </td>
        <td> 79.90  </td>
        <td> 70.14  </td>
        <td> 82.70  </td>
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
        <td> Non-episodic</td>
        <td> 54.11  </td>
        <td> 74.44  </td>
        <td> 64.65  </td>
        <td> 82.73  </td>
    </tr>
    <tr>
        <td><a href="./Baseline++">Baseline++</a></td>
        <td>ICML’19</td>
        <td> Non-episodic</td>
        <td> 52.70 </td>
        <td> 75.36 </td>
        <td> 65.85 </td>
        <td> 83.33 </td>
    </tr>
    <tr>
        <td><a href="./RFS">RFS-simple</a></td>
        <td>ECCV’20</td>
        <td> Non-episodic</td>
        <td> 61.65  </td>
        <td> 76.60  </td>
        <td> 69.14  </td>
        <td> 83.21  </td>
    </tr>
    <tr>
        <td><a href="./SKD">SKD-GEN0</a></td>
        <td>BMVC'20</td>
        <td> Non-episodic</td>
        <td> 66.18  </td>
        <td> 82.21  </td>
        <td> 70.00  </td>
        <td> 84.70  </td>
    </tr>
    <tr>
        <td><a href="./NegCos">NegCos</a></td>
        <td>ECCV’20</td>
        <td> Non-episodic</td>
        <td> 60.99 </td>
        <td> 76.30 </td>
        <td> 68.36 </td>
        <td> 83.77 </td>
    </tr>
        <tr>
        <td><a href="./Versa">Versa</a></td>
        <td>NeurIPS’18</td>
        <td>Meta</td>
        <td> 55.08  </td>
        <td> 69.16  </td>
        <td> 57.30  </td>
        <td> 75.67  </td>
    </tr>
    <tr>
        <td><a href="./R2D2">R2D2</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> 58.36  </td>
        <td> 75.69  </td>
        <td> 64.73  </td>
        <td> 83.40  </td>
    </tr>
    <tr>
        <td><a href="./LEO">LEO</a></td>
        <td>ICLR’19</td>
        <td>Meta</td>
        <td> 57.51  </td>
        <td> 69.33  </td>
        <td> 64.02  </td>
        <td> 78.89  </td>
    </tr>
    <tr>
        <td><a href="./MTL">MTL</a></td>
        <td>CVPR’19</td>
        <td>Meta</td>
        <td> 60.29 </td>
        <td> 76.25 </td>
        <td> 65.12 </td>
        <td> 79.99 </td>
    </tr>
    <tr>
        <td><a href="./ANIL">ANIL</a></td>
        <td>ICLR’20</td>
        <td>Meta</td>
        <td> 52.96  </td>
        <td> 65.88  </td>
        <td> 55.81  </td>
        <td> 73.53  </td>
    </tr>
    <tr>
        <td><a href="./BOIL">BOIL</a></td>
        <td>ICLR’21</td>
        <td>Meta</td>
        <td> 57.85  </td>
        <td> 70.84  </td>
        <td> 60.85  </td>
        <td> 77.74  </td>
    </tr>
    <tr>
        <td><a href="./ProtoNet">ProtoNet</a></td>
        <td>NeurIPS’17</td>
        <td>Metric</td>
        <td> 58.48  </td>
        <td> 75.16  </td>
        <td> 63.50  </td>
        <td> 82.51  </td>
    </tr>
    <tr>
        <td><a href="./RelationNet">RelationNet</a></td>
        <td>CVPR’18</td>
        <td>Metric</td>
        <td> 53.98  </td>
        <td> 71.27  </td>
        <td> 60.80 </td>
        <td> 77.94 </td>
    </tr>
    <tr>
        <td><a href="./CovaMNet">CovaMNet</a></td>
        <td>AAAI’19</td>
        <td>Metric</td>
        <td> 55.83  </td>
        <td> 70.97  </td>
        <td> 54.12  </td>
        <td> 73.51  </td>
    </tr>
    <tr>
        <td><a href="./DN4">DN4</a></td>
        <td>CVPR’19</td>
        <td>Metric</td>
        <td> 57.92  </td>
        <td> 75.50  </td>
        <td> 64.83  </td>
        <td> 82.77  </td>
    </tr>
    <tr>
        <td><a href="./CAN">CAN</a></td>
        <td>NeurIPS’19</td>
        <td>Metric</td>
        <td> 62.33   </td>
        <td> 77.12  </td>
        <td> 71.70  </td>
        <td> 84.61  </td>
    </tr>    
    <tr>
        <td><a href="./RENet">RENet</a></td>
        <td>ICCV’21</td>
        <td> Metric</td>
        <td> 66.21  </td>
        <td> 81.20 </td>
        <td> 71.53  </td>
        <td> 84.55 </td>
    </tr>
</table>
