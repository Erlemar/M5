Done:
- Normalization: mean+std; divide by max; log1p
very slow convergence rate and worse score

- Several block types in stacks !!!
not possible

- We observed that the interpretable model works best when weights are shared across stack, while generic model works best when none of the weights are shared.
- Try to increase train_history_modifier
- More stacks.
- share_thetas
- try to disable scale or weight in loss
- look at predictions
- rewrite the code so that it is possible to define all parameters separately for each block

---

Идеи:

- predict 1, 7 or 14 values and then predict again
- Change the block itself
- семплировать много раз из одного ряда за эпоху
мысль: при создании датасета разбить файл сразу на куски, которые можно брать\
different thetas
 
- модель на 28 дней
- модель рекурсивная
- предсказывает следующую продажу, через одну и так далее
- dense connections
- метрики:
smape
mase
mape
owa

- на вход подавать несколько backcast. Ну или просто несколько раз прогонять модель по ним
- на выход несколько периодов (1б 7б 14б 28).
- Как auxilary или реально использовать.
- разные лоссы
- several trend + seasonality. after each other. Or general between
- предсказывать 1 день вперед, 2 дня вперед и так далее
- zero and non-zero! auxilary
- add categories?
- separate models for separate categories?
---
S-width 2048
S-blocks 3
S-block-layers 4
T-width 256
T-degree 2
T-blocks 3
T-block-layers 4
Sharing STACK LEVEL

Width 512
Blocks 1
Block-layers 4
Stacks 30
Sharing NO
Lookback period 2H,3H,4H,5H,6H,7H
Batch 1024

add batch norm?
--

d:\Programs\anaconda3\envs\dl\Lib\site-packages\hydra\conf\hydra\

---
Bi-LSTM based transformer with CBR blocks(Conv1D-BN-ReLu)
Transformer-XL with CBR and WaveNet blocks.
Hybrid of both Bi-LSTM and Transformer-XL with CBR blocks.

class TimeDistributed(torch.nn.Module):
    def __init__(self, layer, time_steps, *args):        
        super(TimeDistributed, self).__init__()
        
        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):

        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([])
        for i in range(time_steps):
          output_t = self.layers[i](x[:, i, :, :, :])
          output_t  = y.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        return output

Ensembling is used by all the top entries in the M4-competition. We rely on ensembling as well
to be comparable. We found that ensembling is a much more powerful regularization technique
than the popular alternatives, e.g. dropout or L2-norm penalty. The addition of those methods
improved individual models, but was hurting the performance of the ensemble. The core property of
an ensemble is diversity. We build an ensemble using several sources of diversity. First, the ensemble
models are fit on three different metrics: sMAPE, MASE and MAPE, a version of sMAPE that has only
the ground truth value in the denominator. Second, for every horizon H, individual models are trained
on input windows of different length: 2H,3H,...,7H, for a total of six window lengths. Thus the
overall ensemble exhibits a multi-scale aspect. Finally, we perform a bagging procedure (Breiman,
1996) by including models trained with different random initializations. We use 180 total models to
report results on the test set (please refer to Appendix B for the ablation of ensemble size). We use
the median as ensemble aggregation function.



The winning M4 solution smooths the series and removes seasonality prior to fitting the data through a NN. After getting predictions from the model, the seasonality is added back. These decisions were made by the modeler due to his experiences with forecasting and NN models, and may also apply here.
