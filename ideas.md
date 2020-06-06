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
