# EEG Data Visualisation

## Overview

Dataset Used: EEG-Alcohol, which contains [EEG (Electroencephalography)](https://en.wikipedia.org/wiki/Electroencephalography) data for two groups - Alcoholic and Control Group.

In this study, there were two groups of participants, each consisting of eight individuals. The participants were fitted with 64 electrodes on their scalps to record the electrical activity of their brains. The response values were sampled at a rate of 256 Hz, with each epoch lasting 3.9 milliseconds, for a total of one second. The participants were exposed to one of two types of stimuli: a single stimulus (referred to as S1), or two stimuli (S1 and S2). The stimuli consisted of pictures of objects that were selected from the [1980 Snodgrass and Vanderwart picture set](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.294.1979&rep=rep1&type=pdf). When two stimuli were presented, they were either matched (S1 and S2 were identical) or non-matched (S1 and S2 were different). This study aimed to investigate the neural response differences between the two stimulus conditions.

The objective of the analysis is to determine whether there exists a variation in the response values to distinct stimuli between the control group and the alcoholic group. In addition, the study aims to identify the specific brain regions that are responsible for such differences, if present.

# Environment Setup
## Import


```python
import numpy as np
import pandas as pd 
import os
import random
from tqdm import tqdm
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from scipy.stats import mannwhitneyu
from pathlib import Path
import chardet
# Ignore all warnings
import warnings
# warnings.filterwarnings('ignore')

init_notebook_mode(connected=True) ## plotly init
seed = 123
random.seed = seed
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.18.0.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
print('Total amount of files in SMNI_CMI_TRAIN directory: ' + str(len(os.listdir('dataset/'))))
```

    Total amount of files in SMNI_CMI_TRAIN directory: 481



```python
filenames_list = os.listdir('dataset/') ## list of file names in the directory
EEG_data = pd.DataFrame({}) ## create an empty df that will hold data from each file

for file_name in tqdm(filenames_list):
    detected = chardet.detect(Path("dataset/" + file_name).read_bytes())
    encoding = detected.get("encoding")
    assert encoding, "Unable to detect encoding, is it a binary file?"
    temp_df = pd.read_csv('dataset/' + file_name, encoding=encoding) ## read from the file to df
    # EEG_data = EEG_data.append(temp_df) ## add the file data to the main df
    # replace frame.append with pd.concat([frame, temp_df]) if you have pandas version < 0.23
    EEG_data = pd.concat([EEG_data, temp_df], ignore_index=True) ## add the file data to the main df
    
EEG_data = EEG_data.drop(['Unnamed: 0'], axis=1) ## remove the unused column
EEG_data.loc[EEG_data['matching condition'] == 'S2 nomatch,', 'matching condition'] =  'S2 nomatch' ## remove comma sign from stimulus name
```

    100%|██████████| 481/481 [03:36<00:00,  2.22it/s]



```python
## here is how the data looks like
EEG_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial number</th>
      <th>sensor position</th>
      <th>sample num</th>
      <th>sensor value</th>
      <th>subject identifier</th>
      <th>matching condition</th>
      <th>channel</th>
      <th>name</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42.0</td>
      <td>FP1</td>
      <td>0.0</td>
      <td>-3.611</td>
      <td>c</td>
      <td>S1 obj</td>
      <td>0.0</td>
      <td>co2c0000340</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42.0</td>
      <td>FP1</td>
      <td>1.0</td>
      <td>-3.611</td>
      <td>c</td>
      <td>S1 obj</td>
      <td>0.0</td>
      <td>co2c0000340</td>
      <td>0.003906</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42.0</td>
      <td>FP1</td>
      <td>2.0</td>
      <td>-0.193</td>
      <td>c</td>
      <td>S1 obj</td>
      <td>0.0</td>
      <td>co2c0000340</td>
      <td>0.007812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42.0</td>
      <td>FP1</td>
      <td>3.0</td>
      <td>6.154</td>
      <td>c</td>
      <td>S1 obj</td>
      <td>0.0</td>
      <td>co2c0000340</td>
      <td>0.011719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42.0</td>
      <td>FP1</td>
      <td>4.0</td>
      <td>11.525</td>
      <td>c</td>
      <td>S1 obj</td>
      <td>0.0</td>
      <td>co2c0000340</td>
      <td>0.015625</td>
    </tr>
  </tbody>
</table>
</div>



## Variables

The following section provides a description of the variables used in the study. The trial number represents the order in which each trial was conducted. The position of the electrode placed on the subject's scalp was determined based on the International 10-20 system. The sample number corresponds to the discrete time points at which the electrical activity of the brain was sampled, with a value range of 0-255. The sensor value represents the electrical activity of the brain measured in microvolts. The subject identifier denotes whether the subject was part of the alcoholic or control group.

The matching condition describes the type of stimulus presented to the subject, with "S1 obj" indicating a single object being presented, "S2 match" indicating the presentation of two matching objects, and "S2 nomatch" indicating the presentation of two non-matching objects. The channel number represents the identification number assigned to each electrode, and is equivalent to the sensor position column, making one of these columns redundant. The name column represents the serial code assigned to each subject, and the time column represents the time interval measured in seconds, with its value being the inverse of the sample number.

It is worth noting that some of the sensor positions were modified to match head topography visualization, and positions labeled as X, Y, and ND were removed from the dataset since the corresponding regions could not be determined.


```python
## replace some 'sensor position' values
EEG_data.loc[EEG_data['sensor position'] == 'AF1', 'sensor position'] = 'AF3'
EEG_data.loc[EEG_data['sensor position'] == 'AF2', 'sensor position'] = 'AF4'
EEG_data.loc[EEG_data['sensor position'] == 'PO1', 'sensor position'] = 'PO3'
EEG_data.loc[EEG_data['sensor position'] == 'PO2', 'sensor position'] = 'PO4'
## remove rows with undefined positions
EEG_data = EEG_data[(EEG_data['sensor position'] != 'X') & (EEG_data['sensor position'] != 'Y') & (EEG_data['sensor position'] != 'nd')]
```

# Visualisation

In this section, one subject from each group will be randomly selected and their response values will be plotted using 3-D surface and heatmap visualizations in order to visually examine the potential differences among groups.


```python
def sample_data(stimulus, random_id=random.randint(0,7)):
    """Function merged data frame - one data frame for randomly selected subject from control group and 
    one data frame for randomly selected subject from alcoholic group"""
    ## random choose the name_id of subject from alcoholic/control group
    alcoholic_id = EEG_data['name'][(EEG_data['subject identifier'] == 'a') & 
                                    (EEG_data['matching condition'] == stimulus)].unique()[random_id]
    control_id = EEG_data['name'][(EEG_data['subject identifier'] == 'c') & 
                                  (EEG_data['matching condition'] == stimulus)].unique()[random_id]
    
    ## get min trial numbers for each group
    alcoholic_trial_number = EEG_data['trial number'][(EEG_data['name'] == alcoholic_id) & (EEG_data['matching condition'] == stimulus)].min()
    control_trial_number = EEG_data['trial number'][(EEG_data['name'] == control_id) & (EEG_data['matching condition'] == stimulus)].min()

    ## filter the EEG DF
    alcoholic_df = EEG_data[(EEG_data['name'] == alcoholic_id) & (EEG_data['trial number'] == alcoholic_trial_number)]
    control_df = EEG_data[(EEG_data['name'] == control_id) & (EEG_data['trial number'] == control_trial_number)]
    
    return alcoholic_df.append(control_df)
```


```python
sensor_positions = EEG_data[['sensor position', 'channel']].drop_duplicates().reset_index(drop=True).drop(['channel'], axis=1).reset_index(drop=False).rename(columns={'index':'channel'})['sensor position']
channels = EEG_data[['sensor position', 'channel']].drop_duplicates().reset_index(drop=True).drop(['channel'], axis=1).reset_index(drop=False).rename(columns={'index':'channel'})['channel']

def plot_3dSurface_and_heatmap(stimulus, group, df):
    
    if group == 'c':
        group_name = 'Control'
    else:
        group_name = 'Alcoholic'
        
    temp_df = pd.pivot_table(df[['channel', 'sample num', 'sensor value']][(df['subject identifier'] == group) & (df['matching condition'] == stimulus)],
                                          index='channel', columns='sample num', values='sensor value').values.tolist()
    data = [go.Surface(z=temp_df, colorscale='Bluered')]

    layout = go.Layout(
        title='<br>3d Surface and Heatmap of Sensor Values for ' + stimulus + ' Stimulus for ' + group_name + ' Group',
        width=800,
        height=900,
        autosize=False,
        margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(
            xaxis=dict(
                title='Time (sample num)',
                gridcolor='rgb(255, 255, 255)',
    #             erolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title='Channel',
                tickvals=channels,
                ticktext=sensor_positions,
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230, 230)'
            ),
            zaxis=dict(
                title='Sensor Value',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict(x=1, y=1, z=0.5),
            aspectmode = 'manual'
        )
    )

    updatemenus=list([
        dict(
            buttons=list([   
                dict(
                    args=['type', 'surface'],
                    label='3D Surface',
                    method='restyle'
                ),
                dict(
                    args=['type', 'heatmap'],
                    label='Heatmap',
                    method='restyle'
                )             
            ]),
            direction = 'left',
            pad = {'r': 10, 't': 10},
            showactive = True,
            type = 'buttons',
            x = 0.1,
            xanchor = 'left',
            y = 1.1,
            yanchor = 'top' 
        ),
    ])

    annotations = list([
        dict(text='Trace type:', x=0, y=1.085, yref='paper', align='left', showarrow=False)
    ])
    layout['updatemenus'] = updatemenus
    layout['annotations'] = annotations

    fig = dict(data=data, layout=layout)
    iplot(fig)
```

## Sample for "S1 obj"


```python
stimulus = 'S1 obj'
S1_sample_df = sample_data(stimulus=stimulus, random_id=1)
```

    /var/folders/1y/y0cjlqgn00jfgh7_fn17pwf80000gn/T/ipykernel_56059/1724492111.py:18: FutureWarning:
    
    The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    



```python
plot_3dSurface_and_heatmap(stimulus=stimulus, group='a', df=S1_sample_df)
```

```python
plot_3dSurface_and_heatmap(stimulus=stimulus, group='c', df=S1_sample_df)
```

## "S2 match" Stimulus


```python
stimulus = 'S2 match'
S2_m_sample_df = sample_data(stimulus=stimulus, random_id=1)
```

    /var/folders/1y/y0cjlqgn00jfgh7_fn17pwf80000gn/T/ipykernel_56059/1724492111.py:18: FutureWarning:
    
    The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    



```python
plot_3dSurface_and_heatmap(stimulus=stimulus, group='a', df=S2_m_sample_df)
```



```python
plot_3dSurface_and_heatmap(stimulus=stimulus, group='c', df=S2_m_sample_df)
```

## "S2 nomatch" Stimulus


```python
stimulus = 'S2 nomatch'
S2_nm_sample_df = sample_data(stimulus=stimulus, random_id=1)
```

    /var/folders/1y/y0cjlqgn00jfgh7_fn17pwf80000gn/T/ipykernel_56059/1724492111.py:18: FutureWarning:
    
    The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
    



```python
plot_3dSurface_and_heatmap(stimulus=stimulus, group='a', df=S2_nm_sample_df)
```

```python
plot_3dSurface_and_heatmap(stimulus=stimulus, group='c', df=S2_nm_sample_df)
```
